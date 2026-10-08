# AcneVision — Hasil Riset Dataset Detection

## Ringkasan

Ditemukan **5 dataset** publik dengan anotasi bounding box untuk deteksi jerawat.
Berikut perbandingannya:

## Perbandingan Dataset

| Dataset | Platform | Gambar | Anotasi BB | Kelas | Native YOLOv8? | Lisensi |
|---|---|---|---|---|---|---|
| **Acne YOLOv8** (Osmanck) | Kaggle | 927 | ~3.000+ | **1 (`acne`)** | ✅ Plug & Play | Apache 2.0 |
| **ACNE04-v2** (AcneAI) | GitHub | 1.204 | **32.443** | **1 (`acne`)** | Script included | Research |
| **acne04** (Andrei Dore) | Roboflow | 1.419 | ~18.000+ | 7 subtypes | ✅ 1-click | CC BY 4.0 |
| **DL Ensemble** (Jafri) | Kaggle | 920 | 2.847 | 6 subtypes | ✅ YOLO | Open Access |
| **Acne Detection** (Salpin) | Roboflow | 1.270 | ~8.000+ | 6 subtypes | ✅ 1-click | CC BY 4.0 |

## Rekomendasi

### 🥇 Rank 1: Acne Dataset in YOLOv8 Format (Kaggle)
- **URL**: https://www.kaggle.com/datasets/osmanck/acne-dataset-in-yolov8-format
- **Kelebihan**: 100% plug-and-play, sudah 1 kelas `acne`, sudah split train/valid/test, Apache 2.0
- **Kekurangan**: 927 gambar (relatif kecil)

### 🥈 Rank 2: ACNE04-v2 (GitHub - MICCAI 2024)
- **URL**: https://github.com/AIpourlapeau/acne04v2
- **Kelebihan**: 32.443 anotasi / 1.204 gambar (~27 spot/gambar), kualitas terbaik untuk *counting*
- **Kekurangan**: Perlu jalankan script konversi ke YOLO format

### 🥉 Rank 3: acne04 (Roboflow - Andrei Dore)
- **URL**: https://universe.roboflow.com/andrei-dore/acne04
- **Kelebihan**: 1.419 gambar, export YOLOv8 1-click, CC BY 4.0
- **Kekurangan**: 7 kelas (perlu di-merge ke 1 kelas)
