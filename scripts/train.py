"""
AcneVision - Baseline YOLOv8s Training Script
Trains YOLOv8s on ACNE04-v2 single-class dataset on NVIDIA RTX 4060 GPU.
"""
from ultralytics import YOLO
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"

def main():
    print("=" * 60)
    print("AcneVision - Starting Baseline Training (YOLOv8s)")
    print("=" * 60)
    
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # Load pretrained YOLOv8s model
    model = YOLO("yolov8s.pt")
    
    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="baseline",
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        close_mosaic=5,
        workers=4,
    )
    
    print("\nBaseline training complete.")
    print(f"Results saved in: {PROJECT_ROOT / 'runs' / 'detect' / 'baseline'}")

if __name__ == "__main__":
    main()
