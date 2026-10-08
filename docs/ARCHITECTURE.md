# AcneVision - AI Architecture (REVISED)

> Updated berdasarkan riset mendalam repo ACNE04-v2 (2026-10-01)

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                    AcneVision Pipeline               │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │  ACNE04  │   │ ACNE04-v2│   │  Convert Script  │ │
│  │  Images  │ + │   JSON   │ → │ Circle → YOLO BB │ │
│  │ (1,204)  │   │ (32,443) │   │                  │ │
│  └──────────┘   └──────────┘   └────────┬─────────┘ │
│                                         │            │
│                               ┌─────────▼─────────┐ │
│                               │  YOLOv8s Training  │ │
│                               │  (Ultralytics)     │ │
│                               └─────────┬─────────┘ │
│                                         │            │
│                               ┌─────────▼─────────┐ │
│                               │   ONNX Export      │ │
│                               │   (Mobile Ready)   │ │
│                               └─────────┬─────────┘ │
│                                         │            │
│                               ┌─────────▼─────────┐ │
│                               │  Inference Script  │ │
│                               │  Detect + Count    │ │
│                               └───────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## 2. Data Pipeline

### Sumber Data
- **Gambar**: Dataset ACNE04 asli (1,457 gambar, tapi hanya 1,204 digunakan)
  - Sumber: HuggingFace `Layered-Labs/ACNE04` atau Kaggle
  - Resolusi asli: ~3456×3112 pixels
- **Anotasi**: ACNE04-v2 (`Acne04-v2_annotations.json`)
  - Sumber: https://github.com/AIpourlapeau/acne04v2
  - Format: **Lingkaran** `{center: [cx, cy], radius: r}` (BUKAN bounding box)
  - 32,443 anotasi total, ~27 lesi per gambar

### Konversi Circle → YOLO Bounding Box
```
Anotasi lingkaran:  center=[cx, cy], radius=r
Gambar:             width=W, height=H

YOLO format (normalized):
  x_center = cx / W
  y_center = cy / H
  w = min(2*r, min(cx+r, W) - max(cx-r, 0)) / W
  h = min(2*r, min(cy+r, H) - max(cy-r, 0)) / H

Output: "0 x_center y_center w h"  (class 0 = acne)
```

### Data Split
- 80% train (~960 gambar) / 10% val (~120) / 10% test (~120)
- Split berdasarkan IMAGE (bukan annotation) — fixed seed=42
- Tidak ada split bawaan dari dataset

## 3. Model Architecture
- **Base**: YOLOv8s (small) — pretrained pada COCO
- **Head**: Single-class detection head (1 class: `acne`)
- **Input**: 640×640 RGB (baseline), potensial naik ke 1280 jika small objects hilang
- **Framework**: Ultralytics + PyTorch + CUDA

### Small Object Considerations
- Lesi jerawat berukuran 10-30px pada gambar 3456×3112
- Saat di-resize ke 640×640, lesi menjadi ~2-5px — sangat kecil
- Strategi mitigasi:
  1. Mosaic augmentation (menggabungkan 4 gambar, membantu small object)
  2. Pertimbangkan `imgsz=1280` jika baseline mAP rendah
  3. Close mosaic di akhir training untuk stabilitas

## 4. Serving Architecture (FastAPI Server)
- **Framework**: FastAPI + Uvicorn (ASGI)
- **Host**: `0.0.0.0:8000` (dapat diakses via LAN / reverse proxy Nginx)
- **Inference Runtime**: PyTorch native dengan akselerasi NVIDIA CUDA (RTX 4060)
- **Latensi**: ~7.3ms inference per request
- **Endpoints**:
  - `GET /health`: Health check & status alokasi VRAM GPU
  - `POST /detect`: Menerima upload citra wajah (`multipart/form-data`), mengembalikan JSON berisi total jerawat (`acne_count`), bounding boxes, dan waktu inferensi.
  - `POST /detect/visualize`: Menerima upload citra wajah, mengembalikan gambar JPEG beranotasi bounding box + badge hitungan.
  - `GET /docs`: Dokumentasi interaktif Swagger UI.

## 5. Inference Pipeline
```python
Input: Gambar wajah (JPG/PNG)
  → Resize ke 640×640
  → YOLOv8s inference
  → Post-processing (NMS, confidence threshold)
  → Output: {
      "acne_count": int,
      "detections": [{x, y, w, h, confidence}, ...],
      "inference_time_ms": float
    }
```
