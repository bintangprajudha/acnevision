"""
AcneVision - API inferensi (FastAPI) untuk klien mobile via tunnel
==================================================================
Model dimuat sekali; endpoint:
    GET  /health         -> status model (tanpa API key)
    POST /predict        -> JSON: jumlah lesi + kotak (piksel gambar asli)
    POST /predict/image  -> JPEG beranotasi (kotak hijau, jumlah di header)

Form field: image (gambar), mode (plain | sahi), conf (default 0.21 = ambang operasional E9).
Autentikasi: header `X-API-Key` (dari env ACNEVISION_API_KEY; dibuat acak bila kosong).

Usage:
    python scripts/serve.py
    python scripts/serve.py --weights runs/detect/yolov8l_960/weights/best.pt --port 8000
Tunnel (di terminal lain): cloudflared tunnel --url http://localhost:8000
"""
import argparse
import os
import secrets
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from ultralytics import YOLO

from common import MAX_DET, RUNS_DIR, get_device, train_imgsz
from predict import GREEN, add_header, draw_boxes, line_width

E9_WEIGHTS = RUNS_DIR / "detect" / "yolov8l_960_tiles2" / "weights" / "best.pt"
SAHI_SLICE = 1728          # konfigurasi E9 + SAHI terbaik (docs/STATE.md)
DEFAULT_CONF = 0.21        # ambang operasional dari val (docs/STATE.md)
MAX_UPLOAD_MB = 20

app = FastAPI(title="AcneVision")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
state = {}                 # model, sahi_model, imgsz, device, api_key, weights
gpu_lock = threading.Lock()   # satu inferensi pada satu waktu (VRAM 8GB)


def check_key(key: str = Security(api_key_header)):
    if not key or not secrets.compare_digest(key, state["api_key"]):
        raise HTTPException(401, "API key salah atau tidak ada")


def load_models(weights: Path, with_sahi: bool):
    state["device"] = get_device()
    state["model"] = YOLO(str(weights))
    state["imgsz"] = train_imgsz(state["model"])
    state["weights"] = str(weights)
    if with_sahi:
        from sahi import AutoDetectionModel
        state["sahi_model"] = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(weights), confidence_threshold=DEFAULT_CONF,
            image_size=state["imgsz"], device="cuda:0" if state["device"] != "cpu" else "cpu",
        )


def run_inference(img_bgr, mode: str, conf: float):
    """Kembalikan (boxes xyxy, scores) dengan protokol yang sama seperti scripts/predict.py."""
    if mode == "sahi":
        from sahi.predict import get_sliced_prediction
        sm = state.get("sahi_model")
        if sm is None:
            raise HTTPException(400, "Server dijalankan tanpa SAHI (--no-sahi)")
        sm.confidence_threshold = conf
        res = get_sliced_prediction(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), sm, slice_height=SAHI_SLICE, slice_width=SAHI_SLICE,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2, postprocess_type="NMS",
            postprocess_match_metric="IOU", postprocess_match_threshold=0.5, verbose=0,
        )
        preds = res.object_prediction_list
        return (np.array([p.bbox.to_xyxy() for p in preds]).reshape(-1, 4),
                np.array([p.score.value for p in preds]))
    r = state["model"].predict(img_bgr, conf=conf, imgsz=state["imgsz"], max_det=MAX_DET,
                               device=state["device"], verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


async def _predict(file, mode, conf):
    if mode not in ("plain", "sahi"):
        raise HTTPException(400, "mode harus 'plain' atau 'sahi'")
    if not 0.0 < conf < 1.0:
        raise HTTPException(400, "conf harus di antara 0 dan 1")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Gambar melebihi {MAX_UPLOAD_MB} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # EXIF orientation diterapkan
    if img is None:
        raise HTTPException(400, "File bukan gambar yang valid")
    t0 = time.perf_counter()
    with gpu_lock:
        boxes, scores = run_inference(img, mode, conf)
    return img, boxes, scores, (time.perf_counter() - t0) * 1000


@app.get("/health")
def health():
    return {"status": "ok", "weights": state["weights"], "imgsz": state["imgsz"],
            "sahi": "sahi_model" in state, "default_conf": DEFAULT_CONF}


@app.post("/predict", dependencies=[Depends(check_key)])
async def predict(mode: str = Form("plain"), conf: float = Form(DEFAULT_CONF), image: UploadFile = File(...)):
    img, boxes, scores, ms = await _predict(image, mode, conf)
    return {
        "count": int(len(boxes)), "mode": mode, "conf": conf, "inference_ms": round(ms, 1),
        "image": {"width": img.shape[1], "height": img.shape[0]},
        "detections": [{"box": [round(float(v), 1) for v in b], "score": round(float(s), 3)}
                       for b, s in zip(boxes, scores)],
    }


@app.post("/predict/image", dependencies=[Depends(check_key)])
async def predict_image(mode: str = Form("plain"), conf: float = Form(DEFAULT_CONF),
                        image: UploadFile = File(...)):
    img, boxes, scores, ms = await _predict(image, mode, conf)
    draw_boxes(img, boxes, GREEN, line_width(img))
    out = add_header(img, [(f"Acne: {len(boxes)}", (255, 255, 255))])
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"X-Count": str(len(boxes)), "X-Inference-Ms": f"{ms:.0f}"})


def main():
    parser = argparse.ArgumentParser(description="AcneVision - API inferensi")
    parser.add_argument("--weights", type=Path, default=E9_WEIGHTS)
    parser.add_argument("--host", default="127.0.0.1", help="Biarkan loopback; tunnel yang membuka ke luar")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-sahi", action="store_true", help="Hanya mode plain (hemat VRAM)")
    args = parser.parse_args()

    state["api_key"] = os.environ.get("ACNEVISION_API_KEY") or secrets.token_urlsafe(24)
    load_models(args.weights, with_sahi=not args.no_sahi)
    print(f"Model: {args.weights} | imgsz={state['imgsz']} | device={state['device']} | SAHI={not args.no_sahi}")
    print(f"API key (header X-API-Key): {state['api_key']}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
