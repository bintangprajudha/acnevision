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
| Arsitektur | YOLOv8l (43.6M parameter), pretrained COCO; dilatih pada tile 2×2 (E9) |
| Input | 960×960 (letterbox) |
| Config | `configs/yolov8l_960_tiles2.yaml` (E9; ditetapkan sebagai model terpilih 2026-11-13, menggantikan E3 `configs/yolov8l_960.yaml`) |
| Bobot | `runs/detect/yolov8l_960_tiles2/weights/best.pt` |
| Mode inferensi | **SAHI 1728** bila akurasi diutamakan (F1 0.583, MAE 7.78, ±622 ms); **plain** bila latency < 100 ms (F1 0.528, MAE 7.60, ±70 ms). Test, conf 0.21 dari val |
| Hardware | NVIDIA RTX 4060 8GB, Python 3.10, ultralytics 8.4.170 |

## 4. Protokol Evaluasi (mengikat)
1. **Deteksi**: split test, `conf=0.001`, `iou=0.7`, `max_det=1000`, `imgsz` = training.
   Metrik: Precision, Recall, mAP@50, mAP@50-95.
2. **Counting** (metrik utama secara klinis): MAE, RMSE, bias, Pearson r, MAE/RMSE relatif (÷ rata-rata GT),
   keseluruhan dan per level. **Grading Hayashi** dari jumlah lesi (accuracy, precision, sensitivity, specificity,
   Youden; ala Wu et al. 2019). Threshold conf dipilih di **val**, lalu dilaporkan di **test**.
3. Satu-satunya implementasi: `scripts/evaluate.py`.

## 5. Target

### 5.0 Standar minimum deteksi: acuan tunggal (ditetapkan 2026-11-13)
**Acuan: Gazeau dkk., *AcneAI*, MICCAI 2024** (pembuat anotasi ACNE04-v2, jadi dataset dan anotasinya sama dengan kita).
Sumber angka: Tabel 1 di PDF lengkap (https://papers.miccai.org/miccai-2024/paper/2216_paper.pdf), terverifikasi dari teks penuh.

| Metrik | Standar minimum | Asal angka |
|---|---|---|
| Recall (TPR) | **≥ 0.54** | YOLOv8 @ 1280 AcneAI, threshold min \|FP−FN\|; AcneAI sendiri 0.52 |
| Precision | **≥ 0.54** | YOLOv8 @ 1280 AcneAI, titik operasi yang sama; AcneAI sendiri 0.52 |

Status model kita (test, conf dari val, IoU 0.5; rata-rata ± SD dari 3 seed: 42, 7, 123; `docs/EXPERIMENTS.md`):
- YOLOv8l dilatih pada tile 2×2 + SAHI 1728: recall 0.550 ± 0.021 (seed 0.557 / 0.526 / 0.567), precision 0.615 ± 0.027
  → **precision selalu di atas 0.54; recall rata-rata di atas 0.54 tetapi hanya 2 dari 3 seed** (seed 7 = 0.526).
- YOLOv8l dilatih pada tile 2×2, plain: recall 0.502 ± 0.017, precision 0.550 ± 0.020 → precision sebanding, **recall di bawah 0.54**.
- Kalimat aman untuk paper: "sebanding dengan" acuan AcneAI, bukan "melampaui".

Batasan (fakta dari paper): model AcneAI dilatih di data privat, lalu diuji pada ACNE04-v2 hasil koreksi mereka (§3.2); jumlah gambar uji
**tidak tertulis** di PDF utama (angka 1.203 dari catatan awal tidak ditemukan di teks, anggap belum terverifikasi) dan kriteria
matching prediksi-GT juga tidak dijelaskan. Kita dilatih di v2 dan diuji pada 121 gambar. "AP" di Tabel 1 didefinisikan
"jumlah prediksi benar / total prediksi", bukan COCO AP, sehingga **mAP tidak punya acuan**. FPR dan AUC dihitung per gambar lalu
diagregasi (§3.1), tanpa definisi "negatif", sehingga tidak bisa direplikasi dan tidak ditulis untuk model kita.
Paper ini juga tidak melaporkan MAE/RMSE counting maupun grading Hayashi, jadi ambang counting di bawah tetap turunan
dari literatur lain (`docs/RESEARCH.md` B, D) dan hanya indikatif.

### 5.1 Ambang lain
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
