"""
AcneVision - YOLOv8l Training Script (Resumable with Safe Worker Settings)
Trains high-capacity YOLOv8l model on ACNE04-v2 dataset.
Configured with workers=2 to prevent system RAM spikes during image decoding.
"""
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
LAST_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "yolov8l_training" / "weights" / "last.pt"

def main():
    print("=" * 60)
    print("AcneVision - YOLOv8l Training")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Compute Device: {device_name}")

    if LAST_WEIGHTS.exists():
        print(f"Found existing checkpoint: {LAST_WEIGHTS}")
        print("Resuming training from checkpoint...")
        model = YOLO(str(LAST_WEIGHTS))
        results = model.train(
            resume=True,
            workers=2,
            batch=8,
            device=device,
        )
    else:
        print("Starting training from yolov8l.pt backbone...")
        model = YOLO("yolov8l.pt")
        results = model.train(
            data=str(DATA_YAML),
            epochs=50,
            imgsz=640,
            batch=8,
            device=device,
            project=str(PROJECT_ROOT / "runs" / "detect"),
            name="yolov8l_training",
            patience=15,
            save=True,
            plots=True,
            verbose=True,
            close_mosaic=5,
            workers=2,
        )

    print("\nYOLOv8l training complete.")
    print(f"Results saved in: {PROJECT_ROOT / 'runs' / 'detect' / 'yolov8l_training'}")

if __name__ == "__main__":
    main()
