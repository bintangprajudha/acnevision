"""
AcneVision - Utilitas bersama untuk script train/evaluate/predict.
"""
import re
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_RUN = "yolov8l_960"
DEFAULT_WEIGHTS = RUNS_DIR / "detect" / DEFAULT_RUN / "weights" / "best.pt"

# Batas deteksi per gambar. Default Ultralytics (300) memotong jumlah lesi:
# test set memiliki gambar dengan 434 lesi.
MAX_DET = 1000

# Nama file ACNE04 diawali level keparahan Hayashi, mis. "levle2_15.jpg"
_SEVERITY_RE = re.compile(r"^levle(\d)_")


def load_model(path):
    """Muat model Ultralytics dengan kelas yang benar.

    RT-DETR butuh kelas `RTDETR` (loss, validator, dan postprocess berbeda dari YOLO); kelas `YOLO`
    akan salah memakai pipeline YOLO. Deteksi via nama file (bobot pretrained, mis. rtdetr-l.pt)
    atau via kelas model di checkpoint (best.pt hasil training).
    """
    from ultralytics import RTDETR, YOLO
    path = str(path)
    if "rtdetr" in Path(path).name.lower():
        return RTDETR(path)
    model = YOLO(path)
    if type(model.model).__name__ == "RTDETRDetectionModel":
        return RTDETR(path)
    return model


def get_device():
    """Return device index GPU pertama, atau 'cpu' bila CUDA tidak tersedia."""
    return 0 if torch.cuda.is_available() else "cpu"


def severity_from_name(name: str) -> int:
    """Ambil level keparahan (0-3) dari nama file, -1 bila tidak dikenali."""
    m = _SEVERITY_RE.match(name)
    return int(m.group(1)) if m else -1


def train_imgsz(model, default: int = 640) -> int:
    """Ambil imgsz yang dipakai saat training dari checkpoint, agar inferensi konsisten."""
    args = getattr(model, "ckpt", None) or {}
    return int(args.get("train_args", {}).get("imgsz", default))
