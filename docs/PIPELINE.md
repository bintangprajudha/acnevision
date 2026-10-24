# Pipeline Model & Audit Kesesuaian Standar

> Model terpilih: **YOLOv8l @ 960px** (`runs/detect/yolov8l_960`)
> Audit: 2026-10-24 · `ultralytics 8.4.170`, `torch 2.6.0+cu124`
> Sumber dirujuk di bagian [Referensi](#referensi).

---

## 1. Alur Pipeline

```
[Data mentah]            [Konversi]                  [Training]                 [Evaluasi]                     [Prediksi]
ACNE04 images (HF)  ->  scripts/data/           ->  scripts/train.py      ->  scripts/evaluate.py       ->  scripts/predict.py
ACNE04-v2 circles       convert_to_yolo.py          configs/yolov8l_960.yaml   deteksi (test) + counting      plain / --sahi
(cx, cy, r)             circle -> bbox, split       YOLOv8l COCO pretrained    conf counting dipilih di val
                        80/10/10 seed 42
```

| Tahap | Implementasi | Detail |
|---|---|---|
| Data | `scripts/data/download_*.py` | Gambar ACNE04 (HuggingFace `Layered-Labs/ACNE04`) + anotasi lingkaran ACNE04-v2 |
| Konversi | `scripts/data/convert_to_yolo.py` | Lingkaran → bbox persegi `[cx±r, cy±r]`, clip ke batas gambar, normalisasi [0,1] |
| Split | idem | Acak per gambar, seed 42: 963 / 120 / 121 (25.521 / 3.366 / 3.507 lesi) |
| Training | `scripts/train.py` + `configs/yolov8l_960.yaml` | AdamW lr0 1e-3 cosine, 100 epoch, batch 4 (akumulasi ke nbs 64), AMP |
| Seleksi model | Ultralytics `best.pt` | Dipilih dari fitness di split **val** |
| Evaluasi | `scripts/evaluate.py` | Deteksi: `conf=0.001, iou=0.7, max_det=1000`; counting: conf dipilih di val |
| Prediksi | `scripts/predict.py` | `conf=0.25` (default), `max_det=1000`, opsi SAHI |

### Statistik ukuran lesi (train, sisi terpanjang bbox)

Gambar asli umumnya 3112×3456 px (867 gambar train; sisanya 4912×3264, 5184×3456, dan beberapa kecil);
median lesi **42 px** (min 2, maks 604; n = 25.521).
Setelah letterbox ke `imgsz` (skala per gambar = `imgsz / sisi terpanjang`; dihitung ulang 2026-10-24
oleh `scripts/diagrams.py`, menggantikan angka aproksimasi sebelumnya yang tidak dapat direproduksi):

| imgsz | Median lesi | < 16 px | < 8 px |
|---|---|---|---|
| 640 | 7.8 px | 91.5% | 52.4% |
| **960** | **11.7 px** | **73.5%** | **21.0%** |
| 1280 | 15.6 px | 52.4% | 6.9% |

Artinya, pada 960px mayoritas lesi masih "tiny". Resolusi input adalah tuas
terbesar yang tersisa (lihat Roadmap).

---

## 2. Audit vs Standar

Legenda: ✅ sesuai · 🔧 sudah diperbaiki di audit ini · ⚠️ risiko/perlu eksperimen

| # | Aspek | Status | Temuan | Dasar |
|---|---|---|---|---|
| 1 | Format label | ✅ | YOLO txt ternormalisasi, 1 kelas, 5 kolom, tidak ada label kosong | [cfg] |
| 2 | Transfer learning | ✅ | Mulai dari bobot COCO `yolov8l.pt` | [tips] |
| 3 | Optimizer & LR | ✅ | AdamW eksplisit dengan `lr0=0.001` sesuai rekomendasi Adam (1E-3); jika `optimizer=auto`, `lr0` diabaikan | [cfg] |
| 4 | Pemisahan data | ✅ | `best.pt` dipilih dari val, test hanya untuk laporan | praktik ML umum |
| 5 | Reproducibility | ✅ | `seed=42`, `deterministic=True`, semua hyperparameter di config YAML | [cfg] |
| 6 | **`copy_paste=0.3`** | 🔧 | Hanya didukung task *segment/semantic/obb*; di 8.4.170 `CopyPaste.__call__` langsung return bila tidak ada segmen. Pada run ini **no-op**. Config diset `0.0` (hasil identik) | [cfg], `ultralytics/data/augment.py` |
| 7 | **Protokol evaluasi** | 🔧 | Script lama memakai `conf=0.25, iou=0.6`. Standar val: `conf=0.001`, `iou=0.7`; conf tinggi menurunkan mAP. Sekarang dipaksa lewat `evaluate.py` | [val], [cfg] |
| 8 | **`max_det`** | 🔧 | Default predict 300; satu gambar test punya **434 lesi**, sehingga hitungan terpotong. Sekarang `MAX_DET=1000` di semua script | [cfg] |
| 9 | **Threshold counting** | 🔧 | `conf=0.25` dipilih arbitrer. Sekarang dipilih di val (MAE minimum), hasilnya **0.21**: MAE test 9.85 → **8.69**, bias −8.0 → −3.0 | `runs/eval/yolov8l_960/metrics.json` |
| 10 | `imgsz` eval = training | 🔧 | Script lama di-hardcode 640 untuk model 960. Sekarang dibaca dari checkpoint (`train_imgsz`) | [val] |
| 11 | Resolusi input | ⚠️ | 73.5% lesi < 16 px di 960. Paper ACNE04-v2 (AcneAI, MICCAI 2024) membandingkan dengan YOLOv8 pada **1280×1280** | [acneai] |
| 12 | Jumlah epoch | ⚠️ | Best di epoch 98/100, kurva masih naik. Ultralytics menyarankan 300 epoch sebagai titik awal | [tips], `results.csv` |
| 13 | Split tidak terstratifikasi | ⚠️ | Split acak per gambar. Test hanya berisi **1 gambar level 3**, sehingga metrik per-level-3 tidak bermakna | `counts_test.csv` |
| 14 | Kebocoran antar-pasien | ⚠️ *(belum terverifikasi)* | ACNE04 tidak menyediakan ID pasien di sumber yang dibaca. Bila satu pasien punya beberapa foto, split per gambar bisa bocor | [acne04v2], [ldl] |
| 15 | Anotasi lingkaran → bbox | ⚠️ | Paper ACNE04-v2 menganotasi dengan lingkaran dan **mengevaluasi counting/AUC, bukan mAP**. Ambiguitas lokalisasi menekan mAP@50-95, sehingga metrik counting + mAP@50 lebih representatif | [acneai] |
| 16 | `flipud=0.5` | ⚠️ | Wajah tidak pernah terbalik saat inferensi; lesi bersifat lokal sehingga mungkin aman, tetapi belum diuji (kandidat ablation) | [cfg] *(opini)* |
| 17 | Dokumentasi lama | 🔧 | Docs/docstring mengklaim head P2 + 1280px, padahal run memakai `yolov8l.pt` @960 tanpa P2. Sudah dikoreksi | `args.yaml` |

---

## 3. Hasil Evaluasi Standar (model terpilih)

Sumber: `runs/eval/yolov8l_960/metrics.json` (perintah: `python scripts/evaluate.py`).

**Deteksi, split test** (121 gambar, 3.507 lesi, `imgsz=960`, `conf=0.001`, `iou=0.7`, `max_det=1000`):

| Precision | Recall | mAP@50 | mAP@50-95 | Inferensi |
|---|---|---|---|---|
| 56.31% | 48.10% | **49.80%** | 19.14% | 45.4 ms/img (RTX 4060) |

P/R dilaporkan Ultralytics pada confidence F1-maksimum.

**Counting, split test:**

| Threshold | MAE | RMSE | Bias | Pearson r | Total pred / GT |
|---|---|---|---|---|---|
| conf 0.25 (lama) | 9.85 | 24.20 | −8.03 | 0.910 | 2.535 / 3.507 |
| **conf 0.21 (dipilih di val)** | **8.69** | **20.18** | **−3.00** | 0.910 | 3.144 / 3.507 |

Per level (conf 0.21): level 0 MAE 5.27 (n=51) · level 1 MAE 8.46 (n=59) ·
level 2 MAE 9.50 (n=10) · level 3 MAE 188 (n=1, 434 lesi GT, hanya 246 terdeteksi).
Under-count pada wajah sangat padat adalah kelemahan utama.

---

## 4. Kesimpulan Audit

Pipeline **secara garis besar sudah sesuai praktik standar Ultralytics**:
format data, transfer learning, optimizer, dan pemisahan val/test. Ada empat
penyimpangan yang memengaruhi angka (copy_paste no-op, protokol eval
non-standar, `max_det` memotong hitungan, `imgsz` eval salah). Keempatnya
**sudah diperbaiki** di script baru. Sisa risiko (resolusi, epoch, stratifikasi,
kemungkinan kebocoran pasien) menjadi agenda eksperimen di `ROADMAP.md`.

---

## Referensi

- [cfg] Ultralytics, *Configuration*: https://docs.ultralytics.com/usage/cfg/ (diakses 2026-10-24)
- [val] Ultralytics, *Model Validation*: https://docs.ultralytics.com/modes/val/
- [tips] Ultralytics, *Model Training Tips*: https://docs.ultralytics.com/guides/model-training-tips/
- [acne04v2] ACNE04-v2 repo: https://github.com/AIpourlapeau/acne04v2
- [acneai] Gazeau et al., *AcneAI*, MICCAI 2024: https://link.springer.com/chapter/10.1007/978-3-031-72086-4_7
- [ldl] Wu et al., ICCV 2019 (ACNE04): https://github.com/xpwu95/ldl
