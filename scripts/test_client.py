"""
AcneVision - Uji server API (scripts/serve.py) dari sisi klien
==============================================================
Server harus sudah berjalan. Hanya memakai gambar VAL (data/images/val), bukan test set.
Bentuk respons divalidasi terhadap `DetectResponse` / `ApiError` di acnevision-mobile `src/shared/types/api.ts`.

Usage:
    python scripts/test_client.py
    python scripts/test_client.py --url http://127.0.0.1:8000 \
        --reference runs/serve/consistency/predictions.json   # dari predict.py --sahi --slice 1728 --conf 0.21 --save-json
"""
import argparse
import io
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROJECT_ROOT, RUNS_DIR  # noqa: E402

VAL_DIR = PROJECT_ROOT / "data" / "images" / "val"
DETECT_KEYS = {"detection_id", "is_guest", "image", "lesions", "summary", "timing_ms", "model_info",
               "disclaimer", "advice", "upgrade"}
SEVERITIES = ("mild", "moderate", "severe", "very_severe")
MIN_CONF = 0.21


def estimate_severity(total: int):
    """Salinan estimateSeverity() di acnevision-mobile src/shared/constants/classes.ts."""
    if total <= 0:
        return None
    if total <= 5:
        return "mild"
    if total <= 20:
        return "moderate"
    if total <= 50:
        return "severe"
    return "very_severe"


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_detect(body) -> list:
    """Cek nama field, tipe, dan nullability terhadap DetectResponse. Kembalikan daftar pelanggaran."""
    e = []
    if not isinstance(body, dict):
        return ["respons bukan objek"]
    if set(body) != DETECT_KEYS:
        e.append(f"key tidak cocok: kurang {DETECT_KEYS - set(body)}, lebih {set(body) - DETECT_KEYS}")
        return e
    if not isinstance(body["detection_id"], str) or not body["detection_id"]:
        e.append("detection_id harus string")
    if not isinstance(body["is_guest"], bool):
        e.append("is_guest harus boolean")
    img = body["image"]
    if not (isinstance(img, dict) and set(img) == {"width", "height"} and all(isinstance(img[k], int) and img[k] > 0
                                                                            for k in img)):
        e.append(f"image tidak valid: {img}")
        return e
    W, H = img["width"], img["height"]
    les = body["lesions"]
    if not isinstance(les, list):
        return e + ["lesions harus array"]
    for i, l in enumerate(les):
        if not isinstance(l, dict) or set(l) != {"idx", "bbox", "det_confidence", "label"}:
            e.append(f"lesion {i}: key {set(l) if isinstance(l, dict) else type(l)}")
            continue
        if not isinstance(l["idx"], int) or l["idx"] != i:
            e.append(f"lesion {i}: idx {l['idx']} (harus {i}, mulai 0)")
        b = l["bbox"]
        if not isinstance(b, dict) or set(b) != {"x1", "y1", "x2", "y2"} or not all(_num(v) for v in b.values()):
            e.append(f"lesion {i}: bbox {b}")
        elif not (0 <= b["x1"] <= b["x2"] <= W and 0 <= b["y1"] <= b["y2"] <= H):
            e.append(f"lesion {i}: bbox di luar gambar {b} ({W}x{H})")
        if not _num(l["det_confidence"]) or not (MIN_CONF - 1e-4 <= l["det_confidence"] <= 1):
            e.append(f"lesion {i}: det_confidence {l['det_confidence']}")
        if l["label"] != "Jerawat":
            e.append(f"lesion {i}: label {l['label']!r}")
    s = body["summary"]
    if not isinstance(s, dict) or set(s) != {"total_lesions", "severity", "severity_is_estimate"}:
        e.append(f"summary key {s}")
    else:
        if s["total_lesions"] != len(les):
            e.append(f"total_lesions {s['total_lesions']} != {len(les)}")
        if s["severity"] is not None and s["severity"] not in SEVERITIES:
            e.append(f"severity {s['severity']!r}")
        if s["severity"] != estimate_severity(len(les)):
            e.append(f"severity {s['severity']} != estimateSeverity({len(les)}) = {estimate_severity(len(les))}")
        if s["severity_is_estimate"] is not True:
            e.append("severity_is_estimate harus true")
    t = body["timing_ms"]
    if not (isinstance(t, dict) and set(t) == {"detection"} and _num(t["detection"]) and t["detection"] >= 0):
        e.append(f"timing_ms {t}")
    m = body["model_info"]
    if not (isinstance(m, dict) and set(m) == {"detection"} and isinstance(m["detection"], dict)
            and set(m["detection"]) == {"name", "version"} and all(isinstance(v, str) for v in m["detection"].values())):
        e.append(f"model_info {m}")
    for k in ("disclaimer", "advice"):
        if not isinstance(body[k], str) or not body[k]:
            e.append(f"{k} harus string")
    u = body["upgrade"]
    if u is not None and not (isinstance(u, dict) and set(u) == {"message", "unlocks"} and isinstance(u["message"], str)
                              and isinstance(u["unlocks"], list) and all(isinstance(x, str) for x in u["unlocks"])):
        e.append(f"upgrade {u}")
    return e


def is_api_error(resp, code: str) -> bool:
    try:
        err = resp.json()["error"]
    except (ValueError, KeyError, TypeError):
        return False
    return err.get("code") == code and isinstance(err.get("message"), str) and set(err) <= {"code", "message", "details"}


def jpeg(path: Path, short_side=None, orientation=None, quality=92) -> bytes:
    im = Image.open(path).convert("RGB")
    if short_side:
        s = short_side / min(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    kw = {}
    if orientation:
        ex = Image.Exif()
        ex[0x0112] = orientation
        kw["exif"] = ex.tobytes()
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, **kw)
    return buf.getvalue()


class Tester:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.results = []

    def detect(self, data: bytes, mode="capture", headers=None, filename="val.jpg", ctype="image/jpeg", fields=None):
        form = {"source": "mobile", "input_method": "camera", "mode": mode}
        form.update(fields or {})
        files = {"image": (filename, data, ctype)} if data is not None else None
        return requests.post(f"{self.url}/api/v1/detect", files=files, data=form, headers=headers, timeout=180)

    def check(self, name: str, ok: bool, detail: str = ""):
        self.results.append({"test": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def main():
    parser = argparse.ArgumentParser(description="AcneVision - uji klien server API")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--reference", type=Path, default=None, help="predictions.json dari predict.py --save-json")
    parser.add_argument("--reference-src", type=Path, default=RUNS_DIR / "serve" / "consistency_src")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # konsol Windows (cp1252) tidak bisa mencetak Δ
    T = Tester(args.url)

    vals = sorted(VAL_DIR.glob("*.jpg"))
    sizes = {p: Image.open(p).size for p in vals}
    std = next(p for p in vals if sizes[p] == (3112, 3456))           # ukuran ACNE04 paling umum
    big = max(vals, key=lambda p: sizes[p][0] * sizes[p][1])           # 4912x3264: SAHI paling lama
    small = jpeg(std, short_side=320)

    # 1. health
    for path in ("/health", "/api/v1/health"):
        r = requests.get(T.url + path, timeout=10)
        T.check(f"GET {path}", r.status_code == 200 and r.json().get("status") == "ok", f"HTTP {r.status_code}")
    health = requests.get(T.url + "/health", timeout=10).json()
    dumped = json.dumps(health)
    T.check("health tanpa path lokal", not any(s in dumped for s in ("\\\\", "runs/", "best.pt")), dumped[:160])

    # 2. capture (resolusi penuh)
    t0 = time.perf_counter()
    r = T.detect(std.read_bytes(), "capture")
    ms = (time.perf_counter() - t0) * 1000
    body = r.json()
    errs = validate_detect(body) if r.status_code == 200 else [f"HTTP {r.status_code} {r.text[:200]}"]
    T.check("capture: bentuk DetectResponse", not errs, "; ".join(errs[:5]) or
            f"{std.name}: {body['summary']['total_lesions']} lesi, {body['summary']['severity']}, "
            f"server {body['timing_ms']['detection']} ms, klien {ms:.0f} ms")
    if not errs:
        T.check("capture: ukuran gambar asli", (body["image"]["width"], body["image"]["height"]) == sizes[std],
                f"{body['image']} vs {sizes[std]}")
        T.check("capture: tamu (is_guest=true, upgrade ada)", body["is_guest"] is True and body["upgrade"] is not None)
        T.check("capture: model_info SAHI 1728", "SAHI 1728" in body["model_info"]["detection"]["version"],
                body["model_info"]["detection"]["version"])

    # 3. preview (frame ~720p)
    r = T.detect(jpeg(std, short_side=720), "preview")
    body = r.json()
    errs = validate_detect(body) if r.status_code == 200 else [f"HTTP {r.status_code} {r.text[:200]}"]
    T.check("preview: bentuk DetectResponse", not errs, "; ".join(errs[:5]) or
            f"{body['image']['width']}x{body['image']['height']}: {body['summary']['total_lesions']} lesi, "
            f"server {body['timing_ms']['detection']} ms")
    if not errs:
        T.check("preview: model_info plain", "plain" in body["model_info"]["detection"]["version"],
                body["model_info"]["detection"]["version"])

    # 4. Bearer token -> bukan tamu (token tidak divalidasi)
    r = T.detect(small, "preview", headers={"Authorization": "Bearer token-uji"})
    b = r.json()
    T.check("Bearer: is_guest=false, upgrade=null", r.status_code == 200 and b["is_guest"] is False
            and b["upgrade"] is None and not validate_detect(b), f"HTTP {r.status_code}")

    # 5. preview saat GPU sibuk -> 503 cepat; capture yang berjalan tetap selesai
    holder = {}
    th = threading.Thread(target=lambda: holder.setdefault("r", T.detect(big.read_bytes(), "capture")))
    th.start()
    busy, t_end = False, time.time() + 10
    while time.time() < t_end and not busy:
        busy = requests.get(T.url + "/health", timeout=5).json()["gpu"]["busy"]
        if not busy:
            time.sleep(0.02)
    t0 = time.perf_counter()
    r = T.detect(small, "preview")
    ms = (time.perf_counter() - t0) * 1000
    th.join()
    T.check("preview saat GPU sibuk -> 503 SERVER_BUSY cepat",
            busy and r.status_code == 503 and is_api_error(r, "SERVER_BUSY") and ms < 500
            and r.headers.get("Retry-After") == "1", f"busy={busy}, HTTP {r.status_code}, {ms:.0f} ms")
    rc = holder.get("r")
    T.check("capture bersamaan tetap 200", rc is not None and rc.status_code == 200,
            f"HTTP {rc.status_code if rc is not None else '-'}")

    # 6. file bukan gambar / terlalu besar
    r = T.detect(b"ini bukan gambar " * 64, "capture", filename="palsu.jpg")
    T.check("bukan gambar -> 400 INVALID_IMAGE", r.status_code == 400 and is_api_error(r, "INVALID_IMAGE"),
            f"HTTP {r.status_code} {r.text[:120]}")
    r = T.detect(small[:200], "capture")   # header JPEG valid, data terpotong
    T.check("JPEG rusak -> 400 INVALID_IMAGE", r.status_code == 400 and is_api_error(r, "INVALID_IMAGE"),
            f"HTTP {r.status_code}")
    over = b"\xff\xd8\xff" + b"\0" * (10 * 1024 * 1024 - 2)   # 10 MiB + 1 byte
    r = T.detect(over, "capture")
    T.check("file 10 MiB + 1 B -> 413 FILE_TOO_LARGE", r.status_code == 413 and is_api_error(r, "FILE_TOO_LARGE"),
            f"HTTP {r.status_code}")
    r = T.detect(b"\xff\xd8\xff" + b"\0" * (12 * 1024 * 1024), "capture")
    T.check("file 12 MiB -> 413 (ditolak sebelum parsing)", r.status_code == 413 and is_api_error(r, "FILE_TOO_LARGE"),
            f"HTTP {r.status_code}")

    # 7. validasi field & endpoint tak tersedia
    r = T.detect(small, "live")
    T.check("mode tidak dikenal -> 400 VALIDATION_ERROR", r.status_code == 400 and is_api_error(r, "VALIDATION_ERROR"),
            f"HTTP {r.status_code}")
    r = T.detect(None, "capture")
    T.check("tanpa field image -> 400 VALIDATION_ERROR", r.status_code == 400 and is_api_error(r, "VALIDATION_ERROR"),
            f"HTTP {r.status_code}")
    r = requests.post(T.url + "/api/v1/analyze", files={"image": ("a.jpg", small, "image/jpeg")}, timeout=30)
    T.check("/api/v1/analyze -> 404 NOT_FOUND (format ApiError)", r.status_code == 404 and is_api_error(r, "NOT_FOUND"),
            f"HTTP {r.status_code}")
    r = requests.post(T.url + "/predict", files={"image": ("a.jpg", small, "image/jpeg")}, timeout=30)
    T.check("/predict tanpa API key -> 401 UNAUTHORIZED", r.status_code == 401 and is_api_error(r, "UNAUTHORIZED"),
            f"HTTP {r.status_code}")

    # 8. EXIF orientation diterapkan (orientation 6 = putar 90 derajat)
    data = jpeg(std, short_side=720, orientation=6)
    w0, h0 = Image.open(io.BytesIO(data)).size
    r = T.detect(data, "preview")
    b = r.json()
    T.check("EXIF orientation 6 -> width/height tertukar", r.status_code == 200 and
            (b["image"]["width"], b["image"]["height"]) == (h0, w0) and not validate_detect(b),
            f"file {w0}x{h0} -> respons {b.get('image')}")

    # 9. CORS
    pre = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization"}
    for origin, allowed in (("https://localhost", True), ("capacitor://localhost", True),
                            ("http://localhost:5173", True), ("https://contoh-asing.example", False)):
        r = requests.options(T.url + "/api/v1/detect", headers={"Origin": origin, **pre}, timeout=10)
        got = r.headers.get("access-control-allow-origin") == origin
        T.check(f"CORS preflight {origin} -> {'diizinkan' if allowed else 'ditolak'}", got == allowed,
                f"HTTP {r.status_code}, ACAO={r.headers.get('access-control-allow-origin')}")
    r = T.detect(small, "preview", headers={"Origin": "https://localhost"})
    T.check("CORS pada respons POST", r.headers.get("access-control-allow-origin") == "https://localhost"
            and "Server-Timing" in r.headers.get("access-control-expose-headers", ""), r.headers.get("Server-Timing", ""))

    # 10. ambang Hayashi server vs frontend
    import serve   # noqa: E402  (hanya fungsi, model tidak dimuat)
    bad = [n for n in range(0, 301) if serve.severity_from_count(n) != estimate_severity(n)]
    T.check("ambang Hayashi server == classes.ts (0..300)", not bad, f"berbeda pada {bad[:10]}" if bad else
            "batas 0/1/5/6/20/21/50/51 cocok")

    # 11. konsistensi capture vs predict.py --sahi --slice 1728 --conf 0.21
    if args.reference:
        ref = json.loads(args.reference.read_text(encoding="utf-8"))
        meta = ref["meta"]
        T.check("referensi = E9 SAHI 1728 conf 0.21", meta["mode"] == "sahi" and meta["slice"] == 1728
                and abs(meta["conf"] - 0.21) < 1e-9 and "yolov8l_960_tiles2" in meta["weights"].replace("\\", "/"),
                json.dumps(meta))
        for name, rv in ref["images"].items():
            r = T.detect((args.reference_src / name).read_bytes(), "capture")
            b = r.json()
            W, H = b["image"]["width"], b["image"]["height"]
            rb = np.asarray(rv["boxes"], float).reshape(-1, 4)
            clipped = np.clip(rb, 0, [W, H, W, H])
            n_clip = int((np.abs(clipped - rb) > 0).any(axis=1).sum())
            sb = np.array([[l["bbox"][k] for k in ("x1", "y1", "x2", "y2")] for l in b["lesions"]]).reshape(-1, 4)
            ss = np.array([l["det_confidence"] for l in b["lesions"]])
            o_ref, o_srv = np.lexsort(clipped.T[::-1]), np.lexsort(sb.T[::-1])
            same_n = len(rb) == len(sb)
            dbox = float(np.abs(clipped[o_ref] - sb[o_srv]).max()) if same_n and len(sb) else 0.0
            dscore = float(np.abs(np.asarray(rv["scores"])[o_ref] - ss[o_srv]).max()) if same_n and len(sb) else 0.0
            T.check(f"konsistensi {name}: capture == predict.py",
                    same_n and (W, H) == (rv["width"], rv["height"]) and dbox <= 0.01 and dscore <= 1e-4,
                    f"jumlah {len(sb)} vs {len(rb)}, maks |Δkotak| {dbox:.4f} px, maks |Δskor| {dscore:.1e}, "
                    f"kotak ter-clip {n_clip}")

    # 12. rate limit per IP (IP simulasi lewat CF-Connecting-IP dari loopback) - dijalankan terakhir
    lim = health["limits"]["rate_per_ip"]["preview"]["requests"]
    hdr = {"CF-Connecting-IP": "203.0.113.7"}
    codes = []
    for _ in range(lim + 1):
        rr = T.detect(small, "preview", headers=hdr)
        codes.append(rr.status_code)
        if rr.status_code == 429:
            break
    T.check(f"rate limit preview: request ke-{lim + 1} -> 429 RATE_LIMITED",
            codes[:-1].count(200) == lim and codes[-1] == 429 and is_api_error(rr, "RATE_LIMITED")
            and rr.headers.get("Retry-After", "").isdigit(),
            f"{codes.count(200)} x 200 lalu HTTP {codes[-1]}, Retry-After {rr.headers.get('Retry-After')}")
    rr = T.detect(small, "capture", headers=hdr)
    T.check("rate limit capture terpisah dari preview", rr.status_code == 200, f"HTTP {rr.status_code}")

    n_fail = sum(not r["pass"] for r in T.results)
    print(f"\nRingkasan: {len(T.results) - n_fail}/{len(T.results)} PASS")
    out = RUNS_DIR / "serve" / f"test_client_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"url": T.url, "results": T.results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Hasil: {out}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
