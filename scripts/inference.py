"""
AcneVision - End-to-End Inference & Spot Counting Script
Detects acne spots on facial images and reports the count.
Supports either PyTorch (.pt) or exported ONNX (.onnx) model.
"""
import argparse
import time
import os
import cv2
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "acnevision.onnx"
FALLBACK_MODEL = PROJECT_ROOT / "runs" / "detect" / "baseline" / "weights" / "best.pt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="AcneVision Acne Spot Detector & Counter")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, default="", help="Path to model file (.onnx or .pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        if DEFAULT_MODEL.exists():
            model_path = str(DEFAULT_MODEL)
        elif FALLBACK_MODEL.exists():
            model_path = str(FALLBACK_MODEL)
        else:
            print(f"Error: No model found. Please export to {DEFAULT_MODEL} or train first.")
            return

    print("=" * 60)
    print("AcneVision - Facial Acne Detection & Counting")
    print("=" * 60)
    print(f"Model:      {model_path}")
    print(f"Input:      {args.image}")
    print(f"Threshold:  {args.conf}")

    model = YOLO(model_path)
    
    t0 = time.time()
    results = model(args.image, conf=args.conf, imgsz=640)
    inference_time = (time.time() - t0) * 1000

    result = results[0]
    boxes = result.boxes
    acne_count = len(boxes)

    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    print(f"Total Acne Spots Detected: {acne_count}")
    print(f"Total Inference Time:      {inference_time:.2f} ms")
    if acne_count > 0:
        confidences = [f"{float(box.conf[0]):.2f}" for box in boxes]
        print(f"Confidence range:          {min(confidences)} - {max(confidences)}")
    print("=" * 60)

    # Save visual result
    img_name = Path(args.image).name
    save_path = OUTPUTS_DIR / f"annotated_{img_name}"
    annotated_img = result.plot()
    
    # Overlay total count text on top left
    cv2.putText(
        annotated_img,
        f"Acne Spots: {acne_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(save_path), annotated_img)
    print(f"Visualized detection saved to: {save_path}")

if __name__ == "__main__":
    main()
