"""
AcneVision - FastAPI Production Model Server
Serves YOLOv8 acne detection & spot counting model over REST API.
Optimized for high-throughput GPU inference on NVIDIA RTX 4060.
"""
import io
import time
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "yolov8l_training" / "weights" / "best.pt"
FALLBACK_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "baseline" / "weights" / "best.pt"

# Initialize FastAPI App
app = FastAPI(
    title="AcneVision AI Server",
    description="REST API for facial acne detection, localization, and automated spot counting.",
    version="1.0.0",
)

# Enable CORS for web clients / mobile frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
device_name = "CPU"


@app.on_event("startup")
def load_model():
    """Load model into GPU VRAM on startup."""
    global model, device_name
    print("=" * 60)
    print("AcneVision Server - Initializing AI Model")
    print("=" * 60)

    target_weights = DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else FALLBACK_WEIGHTS
    if not target_weights.exists():
        print(f"[WARNING] Weights not found at {DEFAULT_WEIGHTS} or {FALLBACK_WEIGHTS}.")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print(f"Loading weights from: {target_weights}")
    print(f"Target compute device: {device_name}")
    
    model = YOLO(str(target_weights))
    # Warmup inference
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model(dummy, device=device, verbose=False)
    print("[SUCCESS] Model loaded and warmed up.")
    print("=" * 60)


class DetectionBox(BaseModel):
    box_xyxy: List[float]
    confidence: float
    class_name: str


class DetectionResponse(BaseModel):
    status: str
    acne_count: int
    inference_time_ms: float
    device: str
    detections: List[DetectionBox]


@app.get("/")
def root():
    return {
        "name": "AcneVision AI Server",
        "status": "online",
        "docs_url": "/docs",
        "device": device_name,
    }


@app.get("/health")
def health():
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total"
    
    return {
        "status": "healthy" if model is not None else "degraded",
        "device": device_name,
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory": gpu_mem,
        "model_loaded": model is not None,
    }


def read_image_from_upload(upload_file: UploadFile) -> np.ndarray:
    """Decode uploaded image bytes into OpenCV BGR numpy array."""
    contents = upload_file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file or unsupported format.")
    return img


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(..., description="Image file (JPG/PNG)"),
    conf: float = Query(0.25, ge=0.05, le=0.95, description="Confidence threshold"),
):
    """
    Detect acne spots on uploaded image and return bounding boxes + total count as JSON.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    img = read_image_from_upload(file)

    t0 = time.time()
    results = model(img, conf=conf, imgsz=640, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
    inference_ms = (time.time() - t0) * 1000

    result = results[0]
    boxes = result.boxes
    detections = []

    for box in boxes:
        xyxy = [round(float(coord), 2) for coord in box.xyxy[0].tolist()]
        confidence = round(float(box.conf[0]), 3)
        detections.append(
            DetectionBox(
                box_xyxy=xyxy,
                confidence=confidence,
                class_name="acne",
            )
        )

    return DetectionResponse(
        status="success",
        acne_count=len(detections),
        inference_time_ms=round(inference_ms, 2),
        device=device_name,
        detections=detections,
    )


@app.post("/detect/visualize")
async def detect_visualize(
    file: UploadFile = File(..., description="Image file (JPG/PNG)"),
    conf: float = Query(0.25, ge=0.05, le=0.95, description="Confidence threshold"),
):
    """
    Detect acne spots and return annotated image with bounding boxes & count overlay directly.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    img = read_image_from_upload(file)

    results = model(img, conf=conf, imgsz=640, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
    result = results[0]
    acne_count = len(result.boxes)

    # Render bounding boxes
    annotated = result.plot()

    # Overlay badge on top-left
    label_text = f"Acne Spots: {acne_count}"
    cv2.putText(
        annotated,
        label_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    # Encode to JPEG in memory
    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    # Default host 0.0.0.0 enables access across local network (LAN)
    uvicorn.run("scripts.server:app", host="0.0.0.0", port=8000, reload=False)
