"""
AcneVision - ONNX Export Script
Exports trained PyTorch YOLOv8s model to ONNX format for mobile deployment.
"""
from ultralytics import YOLO
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PT = PROJECT_ROOT / "runs" / "detect" / "baseline" / "weights" / "best.pt"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_ONNX = MODELS_DIR / "acnevision.onnx"

def main():
    print("=" * 60)
    print("AcneVision - Exporting Model to ONNX")
    print("=" * 60)
    print(f"Loading weights from: {BEST_PT}")
    
    model = YOLO(str(BEST_PT))
    
    # Export to ONNX (FP32, opset 12, dynamic axes=False for standard mobile runtime)
    exported_file = model.export(
        format="onnx",
        imgsz=640,
        half=False,
        simplify=True,
        opset=12,
    )
    
    print(f"Exported to: {exported_file}")
    shutil.copy2(exported_file, str(TARGET_ONNX))
    print(f"[SUCCESS] Final ONNX saved to: {TARGET_ONNX}")
    print(f"Size: {TARGET_ONNX.stat().st_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
