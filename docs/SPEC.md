# AcneVision - AI Specification (FINALIZED)

> **Status: FINALIZED** — Disepakati pada 2026-10-01

## 1. Overview
AcneVision adalah sistem AI berbasis **YOLOv8 Object Detection** yang mendeteksi lokasi jerawat pada wajah dan menghitung jumlahnya secara otomatis. Model dirancang untuk di-*serve* pada **komputer/server** via **FastAPI REST API**, memanfaatkan akselerasi GPU (NVIDIA RTX 4060) untuk throughput tinggi dan latensi rendah (~7ms per request).

## 2. Core Objectives
- Mendeteksi setiap titik lesi jerawat pada gambar wajah menggunakan bounding box.
- Menghitung jumlah total jerawat yang terdeteksi per gambar.
- Menyediakan endpoint REST API (`/detect`, `/detect/visualize`) untuk integrasi client (Web, Mobile App, Frontend).

## 3. Dataset
- **Sumber**: ACNE04-v2 (GitHub - AIpourlapeau / MICCAI 2024)
  - URL: https://github.com/AIpourlapeau/acne04v2
  - 1.204 gambar, 32.443 anotasi bounding box
  - Lisensi: Academic / Research Use
- **Label**: 1 kelas — `acne`
- **Format Anotasi**: Konversi dari COCO JSON ke YOLOv8 `.txt` (script disediakan repo)

## 4. Model
- **Arsitektur**: YOLOv8l (Large variant — ~43.7M parameters)
  - Kapasitas representasi fitur jauh lebih tinggi untuk mendeteksi lesi jerawat kecil dan padat
  - Pretrained weights: `yolov8l.pt` dari Ultralytics
- **Input Size**: 640×640 RGB
- **Output**: Bounding boxes + confidence scores untuk setiap jerawat terdeteksi
- **Batch Size**: 8 (dioptimalkan untuk 8GB VRAM RTX 4060)
- **Deployment**: Server REST API (FastAPI) dengan akselerasi GPU native

## 5. Metrik Evaluasi
| Metrik | Deskripsi |
|--------|-----------|
| **Precision** | Ketepatan prediksi (berapa banyak prediksi yang benar) |
| **Recall** | Kelengkapan deteksi (berapa banyak jerawat yang berhasil ditemukan) |
| **mAP@50** | Mean Average Precision pada IoU threshold 0.50 |
| **mAP@50-95** | Mean Average Precision pada IoU 0.50–0.95 (metrik utama) |
| **Confusion Matrix** | Matriks true/false positive/negative |
| **Inference Speed** | Waktu inferensi per gambar (ms) |

## 6. Hardware
- **Training**: NVIDIA RTX 4060 (8 GB VRAM) — lokal
- **Python**: 3.10.11
- **Framework**: Ultralytics YOLOv8 (PyTorch)

## 7. Development Phases (GSD Methodology)
1. **Dataset Preparation** — Download ACNE04-v2, konversi ke format YOLOv8
2. **Baseline Training** — Train YOLOv8s dengan default hyperparameters
3. **Tuning & Optimization** — Augmentasi, hyperparameter tuning
4. **Final Evaluation & Export** — Evaluasi test set, export ONNX

## 8. Catatan Dataset Klasifikasi
Dataset klasifikasi yang sudah ada (`AcneDataset/`, `train_clean.csv`, dll.)
**TIDAK digunakan** untuk training detection. Disimpan sebagai referensi atau
untuk proyek klasifikasi terpisah di masa depan.
