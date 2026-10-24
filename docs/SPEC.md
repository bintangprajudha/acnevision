# Spesifikasi Model AcneVision

## 1. Tujuan
Model deteksi objek yang menemukan **setiap lesi jerawat** pada foto wajah dan
menghasilkan **jumlah lesi per gambar** yang akurat.

## 2. Dataset
| Item | Nilai |
|---|---|
| Gambar | ACNE04 (Wu et al., ICCV 2019), subset 1.204 gambar, ±3456×3112 px |
| Anotasi | ACNE04-v2 (Gazeau et al., MICCAI 2024): 32.443 lesi, format lingkaran (cx, cy, r) |
| Kelas | 1: `acne` |
| Split | 963 train / 120 val / 121 test (acak, seed 42) |
| Level keparahan | Prefix nama file `levle0..3` (kriteria Hayashi); train: 371/468/105/19, test: 51/59/10/1 |
| Lisensi | Akademik / riset |

## 3. Model
| Item | Nilai |
|---|---|
| Arsitektur | YOLOv8l (43.6M parameter), pretrained COCO |
| Input | 960×960 (letterbox) |
| Config | `configs/yolov8l_960.yaml` |
| Bobot | `runs/detect/yolov8l_960/weights/best.pt` |
| Hardware | NVIDIA RTX 4060 8GB, Python 3.10, ultralytics 8.4.170 |

## 4. Protokol Evaluasi (mengikat)
1. **Deteksi**: split test, `conf=0.001`, `iou=0.7`, `max_det=1000`, `imgsz` = training.
   Metrik: Precision, Recall, mAP@50, mAP@50-95.
2. **Counting** (metrik utama secara klinis): MAE, RMSE, bias, Pearson r, keseluruhan dan per level.
   Threshold conf dipilih di **val**, lalu dilaporkan di **test**.
3. Satu-satunya implementasi: `scripts/evaluate.py`.

## 5. Target
Ambang diturunkan dari literatur di `docs/RESEARCH.md` bagian B (*opini berbasis bukti*;
metrik dari dataset/anotasi lain tidak bisa dibandingkan langsung).

| Metrik (test, conf dipilih di val, IoU 0.5) | Minimum layak | Baik |
|---|---|---|
| mAP@50 (Ultralytics) | ≥ 0.45 | ≥ 0.55 |
| Recall | ≥ 0.50 | ≥ 0.60 |
| Precision | ≥ 0.50 | ≥ 0.60 |
| MAE counting | ≤ 10 lesi/gambar | ≤ 7 |
| Pearson r (count) | ≥ 0.85 | ≥ 0.90 |
| Inferensi | < 1 s/gambar (GPU) | < 100 ms |

Siap klinis membutuhkan validasi terhadap beberapa dermatolog (ICC ≥ 0.75), bukan hanya metrik dataset.
