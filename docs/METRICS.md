# AcneVision - Model Benchmark & Comparison

> Tanggal Evaluasi: 2026-10-01  
> Dataset: ACNE04-v2 Test Set (121 Citra Wajah Unseen, 3.507 Titik Jerawat)  
> Hardware: NVIDIA GeForce RTX 4060 (8.0 GB VRAM)  
> Input Size: 640x640 RGB

---

## 📊 Hasil Evaluasi Komparatif (Test Set)

| Metrik Evaluasi | YOLOv8s (Baseline) | YOLOv8l (High-Capacity) | Peningkatan Absolut | Peningkatan Relatif |
| :--- | :---: | :---: | :---: | :---: |
| **Parameter** | 11.1 M | 43.6 M | +32.5 M | ~4x Capacity |
| **Precision** | 47.57% | **56.15%** | **+8.58%** | **+18.0%** |
| **Recall** | 38.13% | **43.51%** | **+5.38%** | **+14.1%** |
| **mAP@50** | 35.82% | **43.73%** | **+7.91%** | **+22.1%** |
| **mAP@50-95** | 13.01% | **15.81%** | **+2.80%** | **+21.5%** |
| **Inference Latency** | 7.29 ms | **22.88 ms** | +15.59 ms | Real-time (~43 FPS) |

---

## 🔍 Analisis Peningkatan

1. **Akurasi & Deteksi Titik Kecil (Small Objects)**:
   - Penggunaan arsitektur **YOLOv8l** menghasilkan lonjakan **mAP@50 sebesar +22.1%** (naik dari 35.82% ke 43.73%).
   - **Precision meningkat pesat ke 56.15%** (mengurangi false positive deteksi pori-pori atau noda non-jerawat).
   - **Recall meningkat ke 43.51%**, berhasil menangkap lebih banyak titik jerawat halus pada gambar wajah.

2. **Performa Server**:
   - Dengan latensi inferensi **22.88 ms**, server mampu memproses **~43 gambar per detik** secara simultan di GPU RTX 4060, sangat ideal untuk beban kerja server API production.

---

## 🗂️ File Bobot Model Tersedia

- **Model Terbaik (YOLOv8l)**: `runs/detect/yolov8l_training/weights/best.pt`
- **Model Ringan (YOLOv8s)**: `runs/detect/baseline/weights/best.pt`
