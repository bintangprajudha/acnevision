"""
AcneVision - API inferensi (FastAPI) untuk aplikasi mobile via Cloudflare Tunnel
================================================================================
Kontrak: repo acnevision-mobile `src/shared/types/api.ts` (DetectResponse, ApiError). Dokumentasi tim: docs/API.md.

Endpoint:
    POST /api/v1/detect      -> DetectResponse, jalur tamu (tanpa API key)
                                mode=capture (default): E9 + SAHI 1728, hasil resmi
                                mode=preview: E9 plain 960, cepat; ditolak 503 bila GPU sibuk (tidak mengantre)
    GET  /health, /api/v1/health -> status model & GPU (tanpa path lokal)
    POST /predict, /predict/image -> internal, header X-API-Key (env ACNEVISION_API_KEY; acak bila kosong)

Model E9 = YOLOv8l @ 960 tile 2x2, conf operasional 0.21 (dipilih di val, docs/SPEC.md). Gambar tidak disimpan.

Env (opsional):
    ACNEVISION_CORS_ORIGINS  daftar origin dipisah koma (default: WebView Capacitor + Vite dev)
    ACNEVISION_RATE_CAPTURE  "<jumlah>/<detik>" per IP, default 10/60
    ACNEVISION_RATE_PREVIEW  default 120/60
    ACNEVISION_MAX_QUEUE     capture yang boleh menunggu GPU, default 4
    ACNEVISION_DOCS=1        aktifkan /docs (Swagger); default mati karena server terbuka lewat tunnel

Usage:
    python scripts/serve.py                     # http://127.0.0.1:8000 (loopback; tunnel yang membuka ke luar)
    powershell -File scripts/run_server.ps1     # server + cloudflared tunnel
"""
import argparse
import asyncio
import hashlib
import io
import json
import logging
import math
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
from starlette.exceptions import HTTPException as StarletteHTTPException

from analysis import HAYASHI_NAMES, hayashi_grade
from common import RUNS_DIR, get_device, load_model, train_imgsz
from predict import (GREEN, add_header, build_sahi_model, draw_boxes, line_width, plain_predict,
                     sahi_predict)

E9_WEIGHTS = RUNS_DIR / "detect" / "yolov8l_960_tiles2" / "weights" / "best.pt"
E9_EVAL = RUNS_DIR / "eval" / "yolov8l_960_tiles2_sahi1728" / "metrics.json"
MODEL_NAME = "AcneVision YOLOv8l tile 2x2 (E9)"
SAHI_SLICE = 1728            # konfigurasi E9 + SAHI resmi (docs/SPEC.md)
DEFAULT_CONF = 0.21          # ambang operasional, dipilih di val (metrics.json: conf_selected_on_val)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024    # = MAX_UPLOAD_BYTES di frontend (classes.ts)
BODY_SLACK = 1024 * 1024               # ruang untuk header multipart & field form lain
MAX_PIXELS = 50_000_000                # batas resolusi (bom dekompresi / waktu SAHI)
MIN_SIDE = 32

MODES = ("capture", "preview")
SOURCES = ("web", "mobile")
INPUT_METHODS = ("camera", "upload")

# Teks dari acnevision-mobile `src/shared/constants/copy.ts` (commit 928f728). Ubah bersamaan dengan frontend.
COPY_DISCLAIMER = ("Hasil ini merupakan analisis AI dan bukan diagnosis medis. "
                   "Konsultasikan ke dokter kulit untuk penanganan yang tepat.")
COPY_ADVICE = ("Jika ada jerawat yang terasa nyeri, terasa dalam di bawah kulit, atau tidak membaik, "
               "konsultasikan ke dokter kulit.")
COPY_UPGRADE_MESSAGE = "Masuk untuk melihat analisis lengkap."
COPY_UPGRADE_UNLOCKS = ("Tingkat keparahan dari model AI", "Tingkat keyakinan AI", "Visualisasi Grad-CAM",
                        "Rekomendasi perawatan dari AI", "Simpan dan riwayat analisis")
LESION_LABEL = "Jerawat"
MSG_INVALID_IMAGE = "File harus berupa gambar JPEG, PNG, atau WEBP."   # sama dengan mock frontend
MSG_TOO_LARGE = "Ukuran file maksimal 10 MB."

log = logging.getLogger("acnevision.serve")


def _rate(env: str, default: str) -> tuple:
    n, window = os.environ.get(env, default).split("/")
    return int(n), float(window)


RATE_LIMITS = {"capture": _rate("ACNEVISION_RATE_CAPTURE", "10/60"),
               "preview": _rate("ACNEVISION_RATE_PREVIEW", "120/60")}
MAX_QUEUE = int(os.environ.get("ACNEVISION_MAX_QUEUE", "4"))
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "ACNEVISION_CORS_ORIGINS", "https://localhost,capacitor://localhost,http://localhost:5173").split(",") if o.strip()]
DOCS = os.environ.get("ACNEVISION_DOCS") == "1"

state = {}   # model, sahi_model, imgsz, device, gpu_name, sha8, api_key


# ---------------------------------------------------------------- error (format ApiError frontend)
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Optional[dict] = None,
                 headers: Optional[dict] = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details, self.headers = status, code, message, details, headers


def error_response(status: int, code: str, message: str, details=None, headers=None) -> JSONResponse:
    err = {"code": code, "message": message}
    if details:
        err["details"] = details
    return JSONResponse({"error": err}, status_code=status, headers=headers)


# ---------------------------------------------------------------- penjadwal GPU (capture didahulukan)
class GpuBusy(Exception):
    pass


class GpuScheduler:
    """Satu inferensi pada satu waktu. capture mengantre FIFO (maks `max_waiting`); preview tidak pernah
    mengantre: ditolak bila GPU sibuk ATAU ada capture menunggu, sehingga capture selalu didahulukan."""

    def __init__(self, max_waiting: int):
        self.max_waiting = max_waiting
        self.busy = False
        self.waiting = 0
        self._cond = asyncio.Condition()

    @asynccontextmanager
    async def slot(self, priority: bool):
        async with self._cond:
            if not priority:
                if self.busy or self.waiting:
                    raise GpuBusy
            else:
                if self.waiting >= self.max_waiting:
                    raise GpuBusy
                self.waiting += 1
                try:
                    await self._cond.wait_for(lambda: not self.busy)
                finally:
                    self.waiting -= 1
            self.busy = True
        try:
            yield
        finally:
            async with self._cond:
                self.busy = False
                self._cond.notify_all()


scheduler = GpuScheduler(MAX_QUEUE)
# Semua kerja GPU di SATU thread: aman walau coroutine dibatalkan saat inferensi masih berjalan.
GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


# ---------------------------------------------------------------- rate limit per IP (sliding window)
class RateLimiter:
    def __init__(self, limits: dict):
        self.limits = limits
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()
        self.last_prune = time.monotonic()

    def check(self, key: str, kind: str) -> Optional[float]:
        """Catat satu request; kembalikan detik tunggu bila kuota habis, None bila boleh."""
        n, window = self.limits[kind]
        now = time.monotonic()
        with self.lock:
            q = self.hits[(kind, key)]
            while q and now - q[0] >= window:
                q.popleft()
            if len(q) >= n:
                return window - (now - q[0])
            q.append(now)
            if now - self.last_prune > 300:   # buang IP yang sudah lama diam
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.limits[k[0]][1]]:
                    del self.hits[k]
                self.last_prune = now
        return None


limiter = RateLimiter(RATE_LIMITS)


def client_ip(request: Request) -> Optional[str]:
    """IP klien untuk rate limit. Header CF-Connecting-IP / X-Forwarded-For hanya dipercaya bila peer-nya
    loopback (cloudflared lokal). None = request lokal langsung (tes/benchmark), tidak dibatasi."""
    peer = request.client.host if request.client else ""
    try:
        loopback = ip_address(peer).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        return peer or "unknown"
    fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0]
    return fwd.strip() or None


def has_bearer(request: Request) -> bool:
    """Token TIDAK divalidasi (tugas backend tim); hanya menandai is_guest=false."""
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return scheme.lower() == "bearer" and bool(token.strip())


# ---------------------------------------------------------------- batas ukuran body (sebelum parsing form)
class _BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Tolak body > batas dengan 413 FILE_TOO_LARGE: lewat Content-Length, atau dihitung saat streaming."""

    def __init__(self, app, max_body: int):
        self.app, self.max_body = app, max_body

    async def _send_413(self, send):
        resp = error_response(413, "FILE_TOO_LARGE", MSG_TOO_LARGE)
        await send({"type": "http.response.start", "status": 413, "headers": resp.raw_headers})
        await send({"type": "http.response.body", "body": resp.body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cl = dict(scope["headers"]).get(b"content-length", b"")
        if cl.isdigit() and int(cl) > self.max_body:
            return await self._send_413(send)

        received, too_large, started = 0, False, False

        async def limited_receive():
            nonlocal received, too_large
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > self.max_body:
                    too_large = True
                    raise _BodyTooLarge
            return msg

        async def guarded_send(msg):
            nonlocal started
            if too_large:          # ganti respons apa pun (mis. 400 parsing) dengan 413
                if msg["type"] == "http.response.start" and not started:
                    started = True
                    await self._send_413(send)
                return
            started = started or msg["type"] == "http.response.start"
            await send(msg)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            if not started:
                await self._send_413(send)


# ---------------------------------------------------------------- gambar
def sniff_type(data: bytes) -> Optional[str]:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> array BGR dengan orientasi EXIF diterapkan (cv2.imdecode, sama dengan jalur file SAHI/Ultralytics)."""
    if sniff_type(data) is None:
        raise ApiError(400, "INVALID_IMAGE", MSG_INVALID_IMAGE)
    try:
        with Image.open(io.BytesIO(data)) as im:      # header saja, tanpa decode piksel
            w, h = im.size
    except Exception:
        raise ApiError(400, "INVALID_IMAGE", MSG_INVALID_IMAGE)
    if w * h > MAX_PIXELS:
        raise ApiError(400, "INVALID_IMAGE", f"Resolusi gambar terlalu besar (maks {MAX_PIXELS // 1_000_000} MP).",
                       {"width": w, "height": h})
    if min(w, h) < MIN_SIDE:
        raise ApiError(400, "INVALID_IMAGE", f"Gambar terlalu kecil (min {MIN_SIDE} px).", {"width": w, "height": h})
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ApiError(400, "INVALID_IMAGE", MSG_INVALID_IMAGE)
    return img


# ---------------------------------------------------------------- inferensi (dipanggil di GPU_EXECUTOR)
def infer(img_bgr: np.ndarray, engine: str, conf: float = DEFAULT_CONF):
    """engine 'sahi' (capture) atau 'plain' (preview) -> (boxes xyxy piksel, scores, ms)."""
    t0 = time.perf_counter()
    if engine == "sahi":
        sm = state["sahi_model"]
        sm.confidence_threshold = conf
        try:
            boxes, scores = sahi_predict(sm, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), SAHI_SLICE)
        finally:
            sm.confidence_threshold = DEFAULT_CONF
    else:
        boxes, scores = plain_predict(state["model"], img_bgr, conf, state["imgsz"], state["device"])
    return np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float), (time.perf_counter() - t0) * 1000


async def run_on_gpu(img_bgr, engine: str, priority: bool, conf: float = DEFAULT_CONF):
    """Masuk penjadwal GPU lalu inferensi. Kembalikan (boxes, scores, infer_ms, queue_ms)."""
    tq = time.perf_counter()
    try:
        async with scheduler.slot(priority):
            queue_ms = (time.perf_counter() - tq) * 1000
            loop = asyncio.get_running_loop()
            boxes, scores, ms = await loop.run_in_executor(GPU_EXECUTOR, infer, img_bgr, engine, conf)
    except GpuBusy:
        raise ApiError(503, "SERVER_BUSY", "Server sedang memproses gambar lain. Coba lagi sebentar lagi.",
                       {"retry_after_s": 1}, {"Retry-After": "1"})
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise ApiError(503, "SERVER_BUSY", "Memori GPU penuh. Coba lagi sebentar lagi.",
                       {"retry_after_s": 2}, {"Retry-After": "2"})
    return boxes, scores, ms, queue_ms


def severity_from_count(n: int) -> Optional[str]:
    """Estimasi Hayashi dari jumlah lesi (ambang analysis.HAYASHI_UPPER = classes.ts); None bila 0 lesi."""
    return None if n <= 0 else HAYASHI_NAMES[int(hayashi_grade(n))]


def lesions_payload(boxes: np.ndarray, scores: np.ndarray, w: int, h: int) -> list:
    """Kotak piksel (di-clip ke gambar), idx 0..n-1 urut pusat kotak atas->bawah lalu kiri->kanan."""
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    order = np.lexsort(((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2))
    return [{"idx": i,
             "bbox": {"x1": round(float(x1), 2), "y1": round(float(y1), 2),
                      "x2": round(float(x2), 2), "y2": round(float(y2), 2)},
             "det_confidence": round(float(s), 4),
             "label": LESION_LABEL}
            for i, ((x1, y1, x2, y2), s) in enumerate(zip(b[order], scores[order]))]


def model_version(engine: str) -> str:
    desc = f"SAHI {SAHI_SLICE}" if engine == "sahi" else f"plain {state['imgsz']}"
    return f"{state['sha8']} · {desc} · conf {DEFAULT_CONF}"


# ---------------------------------------------------------------- aplikasi
app = FastAPI(title="AcneVision", docs_url="/docs" if DOCS else None, redoc_url=None,
              openapi_url="/openapi.json" if DOCS else None)
app.add_middleware(BodyLimitMiddleware, max_body=MAX_UPLOAD_BYTES + BODY_SLACK)
# CORS ditambahkan terakhir = lapisan terluar, sehingga respons 413/4xx/5xx tetap membawa header CORS.
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Authorization", "Content-Type"],
                   expose_headers=["Retry-After", "Server-Timing"], max_age=600)


@app.exception_handler(ApiError)
async def _api_error(_request, exc: ApiError):
    return error_response(exc.status, exc.code, exc.message, exc.details, exc.headers)


@app.exception_handler(RequestValidationError)
async def _validation_error(_request, exc: RequestValidationError):
    fields = sorted({str(e["loc"][-1]) for e in exc.errors() if e.get("loc")})
    return error_response(400, "VALIDATION_ERROR", "Permintaan tidak valid.", {"fields": fields})


_HTTP_CODES = {400: ("VALIDATION_ERROR", "Permintaan tidak valid."),
               401: ("UNAUTHORIZED", "Silakan masuk terlebih dahulu."),
               404: ("NOT_FOUND", "Endpoint tidak tersedia di server model ini."),
               405: ("METHOD_NOT_ALLOWED", "Metode tidak diizinkan."),
               413: ("FILE_TOO_LARGE", MSG_TOO_LARGE)}


@app.exception_handler(StarletteHTTPException)
async def _http_error(_request, exc: StarletteHTTPException):
    code, msg = _HTTP_CODES.get(exc.status_code, ("INTERNAL_ERROR", "Terjadi kesalahan pada server."))
    return error_response(exc.status_code, code, msg, headers=getattr(exc, "headers", None))


@app.get("/health")
@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "model": {"name": MODEL_NAME, "weights_sha256": state["sha8"], "imgsz": state["imgsz"], "conf": DEFAULT_CONF},
        "modes": {"capture": f"SAHI {SAHI_SLICE}", "preview": f"plain {state['imgsz']}"},
        "gpu": {"device": state["gpu_name"], "busy": scheduler.busy, "waiting_captures": scheduler.waiting,
                "max_queue": scheduler.max_waiting},
        "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_pixels": MAX_PIXELS,
                   "rate_per_ip": {k: {"requests": n, "window_s": w} for k, (n, w) in RATE_LIMITS.items()}},
    }


@app.post("/api/v1/detect")
async def detect(request: Request, image: UploadFile = File(...), source: str = Form("web"),
                 input_method: str = Form("upload"), mode: str = Form("capture")):
    t0 = time.perf_counter()
    for name, value, allowed in (("mode", mode, MODES), ("source", source, SOURCES),
                                 ("input_method", input_method, INPUT_METHODS)):
        if value not in allowed:
            raise ApiError(400, "VALIDATION_ERROR", f"{name} harus salah satu dari: {', '.join(allowed)}.",
                           {"field": name, "allowed": list(allowed)})
    ip = client_ip(request)
    if ip is not None:
        wait = limiter.check(ip, mode)
        if wait is not None:
            sec = max(1, math.ceil(wait))
            raise ApiError(429, "RATE_LIMITED", "Terlalu banyak permintaan. Coba lagi sebentar lagi.",
                           {"retry_after_s": sec, "mode": mode}, {"Retry-After": str(sec)})

    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "FILE_TOO_LARGE", MSG_TOO_LARGE)
    try:
        img = await run_in_threadpool(decode_image, data)
        decode_ms = (time.perf_counter() - t0) * 1000
        engine = "sahi" if mode == "capture" else "plain"
        boxes, scores, infer_ms, queue_ms = await run_on_gpu(img, engine, priority=(mode == "capture"))
    except ApiError as e:
        log.info("detect mode=%s source=%s input=%s status=%d code=%s", mode, source, input_method, e.status, e.code)
        raise
    except Exception:
        log.exception("detect gagal")
        raise ApiError(500, "INTERNAL_ERROR", "Terjadi kesalahan pada server.")

    h, w = img.shape[:2]
    guest = not has_bearer(request)
    lesions = lesions_payload(boxes, scores, w, h)
    body = {
        "detection_id": str(uuid.uuid4()),
        "is_guest": guest,
        "image": {"width": int(w), "height": int(h)},
        "lesions": lesions,
        "summary": {"total_lesions": len(lesions), "severity": severity_from_count(len(lesions)),
                    "severity_is_estimate": True},
        "timing_ms": {"detection": int(round(infer_ms))},
        "model_info": {"detection": {"name": MODEL_NAME, "version": model_version(engine)}},
        "disclaimer": COPY_DISCLAIMER,
        "advice": COPY_ADVICE,
        "upgrade": {"message": COPY_UPGRADE_MESSAGE, "unlocks": list(COPY_UPGRADE_UNLOCKS)} if guest else None,
    }
    total_ms = (time.perf_counter() - t0) * 1000
    log.info("detect mode=%s source=%s input=%s guest=%s img=%dx%d lesi=%d decode=%.0fms queue=%.0fms "
             "infer=%.0fms total=%.0fms", mode, source, input_method, guest, w, h, len(lesions), decode_ms,
             queue_ms, infer_ms, total_ms)
    timing = f"decode;dur={decode_ms:.0f}, queue;dur={queue_ms:.0f}, infer;dur={infer_ms:.0f}, total;dur={total_ms:.0f}"
    return JSONResponse(body, headers={"Server-Timing": timing})


# ---------------------------------------------------------------- endpoint internal (API key)
def check_key(request: Request):
    key = request.headers.get("x-api-key", "")
    if not key or not secrets.compare_digest(key, state["api_key"]):
        raise ApiError(401, "UNAUTHORIZED", "API key salah atau tidak ada.")


async def _predict_internal(file: UploadFile, mode: str, conf: float):
    if mode not in ("plain", "sahi"):
        raise ApiError(400, "VALIDATION_ERROR", "mode harus 'plain' atau 'sahi'.", {"field": "mode"})
    if not 0.0 < conf < 1.0:
        raise ApiError(400, "VALIDATION_ERROR", "conf harus di antara 0 dan 1.", {"field": "conf"})
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "FILE_TOO_LARGE", MSG_TOO_LARGE)
    img = await run_in_threadpool(decode_image, data)
    boxes, scores, ms, _ = await run_on_gpu(img, mode, priority=True, conf=conf)
    return img, boxes, scores, ms


@app.post("/predict", dependencies=[Depends(check_key)])
async def predict(mode: str = Form("plain"), conf: float = Form(DEFAULT_CONF), image: UploadFile = File(...)):
    img, boxes, scores, ms = await _predict_internal(image, mode, conf)
    return {
        "count": int(len(boxes)), "mode": mode, "conf": conf, "inference_ms": round(ms, 1),
        "image": {"width": img.shape[1], "height": img.shape[0]},
        "detections": [{"box": [round(float(v), 1) for v in b], "score": round(float(s), 3)}
                       for b, s in zip(boxes, scores)],
    }


@app.post("/predict/image", dependencies=[Depends(check_key)])
async def predict_image(mode: str = Form("plain"), conf: float = Form(DEFAULT_CONF),
                        image: UploadFile = File(...)):
    img, boxes, scores, ms = await _predict_internal(image, mode, conf)
    draw_boxes(img, boxes, GREEN, line_width(img))
    out = add_header(img, [(f"Acne: {len(boxes)}", (255, 255, 255))])
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"X-Count": str(len(boxes)), "X-Inference-Ms": f"{ms:.0f}"})


# ---------------------------------------------------------------- startup
def file_sha8(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def load_models(weights: Path):
    state["device"] = get_device()
    state["gpu_name"] = torch.cuda.get_device_name(0) if state["device"] != "cpu" else "cpu"
    state["model"] = load_model(weights)
    state["imgsz"] = train_imgsz(state["model"])
    state["sha8"] = file_sha8(weights)
    state["sahi_model"] = build_sahi_model(weights, DEFAULT_CONF, state["imgsz"], state["device"])


def warm_up():
    """Satu inferensi plain + SAHI pada gambar sintetis seukuran foto ACNE04 (CUDA init, alokasi memori)."""
    rng = np.random.default_rng(0)
    img = rng.integers(90, 170, size=(3456, 3112, 3), dtype=np.uint8)
    for engine in ("plain", "sahi", "plain"):
        _, _, ms = infer(img, engine)
        log.info("warm-up %s: %.0f ms", engine, ms)


def check_conf():
    try:
        conf = json.loads(E9_EVAL.read_text(encoding="utf-8")).get("conf_selected_on_val")
    except (OSError, ValueError):
        log.warning("Tidak bisa membaca %s untuk memeriksa conf operasional", E9_EVAL)
        return
    if conf != DEFAULT_CONF:
        log.warning("PERINGATAN: conf server %.2f berbeda dengan conf_selected_on_val %s", DEFAULT_CONF, conf)


def main():
    parser = argparse.ArgumentParser(description="AcneVision - API inferensi untuk aplikasi mobile")
    parser.add_argument("--weights", type=Path, default=E9_WEIGHTS)
    parser.add_argument("--host", default="127.0.0.1", help="Biarkan loopback; tunnel yang membuka ke luar")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    state["api_key"] = os.environ.get("ACNEVISION_API_KEY") or secrets.token_urlsafe(24)
    check_conf()
    t0 = time.perf_counter()
    load_models(args.weights)
    # Warm-up di thread GPU yang sama dengan inferensi request
    GPU_EXECUTOR.submit(warm_up).result()
    log.info("Model %s (sha256 %s) | imgsz=%d | device=%s | siap dalam %.1f s", args.weights.name, state["sha8"],
             state["imgsz"], state["gpu_name"], time.perf_counter() - t0)
    log.info("CORS: %s | rate/IP: %s | antrean capture maks %d", ", ".join(CORS_ORIGINS), RATE_LIMITS, MAX_QUEUE)
    log.info("API key internal (/predict, header X-API-Key): %s", state["api_key"])
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
