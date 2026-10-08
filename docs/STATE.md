# AcneVision - Project State

## Current Phase: Complete & Ready for Serving (YOLOv8l)

### Completed Milestones
- [x] **Plan 1.1: Environment Setup & Dependencies**
  - Python virtual environment `.venv/` configured
  - PyTorch 2.6.0 + CUDA 12.4 (NVIDIA GeForce RTX 4060, 8GB VRAM) verified
  - Ultralytics YOLOv8, OpenCV, FastAPI, Uvicorn, dan dependensi terpasang
- [x] **Plan 1.2: Dataset Preparation & YOLO Conversion**
  - ACNE04-v2 annotations (32,443 anotasi pada 1,204 citra) dimuat
  - Citra ACNE04 diekstrak dari dataset HuggingFace, difilter 1,204 citra v2
  - Konversi format lingkaran ke bounding box YOLO (normalized [0, 1]) dengan boundary clipping
  - Dataset split: 963 train (25,521 boxes), 120 val (3,366 boxes), 121 test (3,507 boxes)
  - `data/data.yaml` siap pakai
- [x] **Plan 2.1: Baseline YOLOv8s Training & Evaluation**
  - Model YOLOv8s dilatih sebagai baseline (Test mAP@50: 35.82%, Precision: 47.57%, Recall: 38.13%)
- [x] **Plan 3.1: High-Capacity Model (YOLOv8l) Training & Evaluation**
  - Model YOLOv8l (43.6M parameter) dilatih 50 epoch pada GPU RTX 4060
  - Hasil evaluasi pada Test Split (121 citra test, 3,507 jerawat):
    - **Precision**: **56.15%** (+8.58% peningkatan)
    - **Recall**: **43.51%** (+5.38% peningkatan)
    - **mAP@50**: **43.73%** (+7.91% peningkatan)
    - **mAP@50-95**: **15.81%** (+2.80% peningkatan)
    - **Inference Speed**: ~22.88 ms (~43 FPS di GPU)
  - Bobot model terbaik tersimpan di: `runs/detect/yolov8l_training/weights/best.pt`
- [x] **Plan 4.1: Production REST API Server**
  - FastAPI server (`scripts/server.py`) otomatis memuat bobot `yolov8l` terbaik
  - Endpoint JSON: `POST /detect`
  - Endpoint Visual Image: `POST /detect/visualize`
  - Swagger Docs: `http://localhost:8000/docs`
