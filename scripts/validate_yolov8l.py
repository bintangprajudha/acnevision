"""
AcneVision - Model Validation on Test Split
Evaluates trained YOLOv8l model on the unseen test split.
"""
from ultralytics import YOLO
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
V8L_BEST = PROJECT_ROOT / "runs" / "detect" / "yolov8l_training" / "weights" / "best.pt"

def main():
    print("=" * 60)
    print("AcneVision - Validating YOLOv8l on Test Split")
    print("=" * 60)

    if not V8L_BEST.exists():
        print(f"Error: Model weights not found at {V8L_BEST}")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(V8L_BEST))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=8,
        device=device,
        plots=True,
        save_json=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="test_evaluation_v8l",
    )

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION METRICS (YOLOv8l)")
    print("=" * 60)
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print(f"mAP@50:       {metrics.box.map50:.4f}")
    print(f"mAP@50-95:    {metrics.box.map:.4f}")
    print(f"Inference ms: {metrics.speed['inference']:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
