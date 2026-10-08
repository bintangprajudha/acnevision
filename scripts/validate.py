"""
AcneVision - Baseline Validation Script
Evaluates trained YOLOv8s baseline model on the test split.
"""
from ultralytics import YOLO
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
WEIGHTS_PATH = PROJECT_ROOT / "runs" / "detect" / "baseline" / "weights" / "best.pt"

def main():
    print("=" * 60)
    print("AcneVision - Validating Model on Test Split")
    print("=" * 60)
    
    if not WEIGHTS_PATH.exists():
        print(f"Error: Model weights not found at {WEIGHTS_PATH}")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(WEIGHTS_PATH))
    
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device=device,
        plots=True,
        save_json=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="test_evaluation",
    )
    
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION METRICS")
    print("=" * 60)
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print(f"mAP@50:       {metrics.box.map50:.4f}")
    print(f"mAP@50-95:    {metrics.box.map:.4f}")
    print(f"Inference ms: {metrics.speed['inference']:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
