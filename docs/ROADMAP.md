# Roadmap Pengembangan Model

> Fokus: menaikkan **recall** dan akurasi **counting** YOLOv8l tanpa melanggar protokol evaluasi.
> Urutan berdasarkan dampak yang diharapkan ÷ biaya. Setiap item = 1 config baru + 1 baris di `EXPERIMENTS.md`.

## Target (dari SPEC §5)

Kandidat terbaik saat ini: **E4 + SAHI 1728** (test, conf 0.22 dari val; `runs/eval/yolov8l_960_tiles2_ft_sahi1728/metrics.json`).

| Metrik (test) | E4 + SAHI | Minimum | Baik |
|---|---|---|---|
| AP@50 | 0.550 | ≥ 0.45 ✅ | ≥ 0.55 |
| Recall | 0.536 | ≥ 0.50 ✅ | ≥ 0.60 |
| Precision | 0.611 | ≥ 0.50 ✅ | ≥ 0.60 ✅ |
| MAE counting | 8.29 | ≤ 10 ✅ | ≤ 7 |
| Pearson r | 0.943 | ≥ 0.85 ✅ | ≥ 0.90 ✅ |

## Selesai (2026-10-24)
- SAHI di test set → signifikan positif untuk E3 (lihat `REPORT.md` §5.5)
- Fine-tune berbasis tile 2×2 (E4, 10 epoch dari E3) → positif kecil, tidak signifikan
- TTA → negatif; baseline default (B0) → deteksi setara E3

## Antrian Eksperimen

| # | Eksperimen | Hipotesis | Biaya | Dasar |
|---|---|---|---|---|
| 1 | ~~**Training tile dari nol**~~ **selesai (E9, kandidat terbaik)**; fine-tune 60 epoch: E8 | E4 hanya 10 epoch fine-tune; resolusi efektif terbukti tuas utama | Tinggi (±5 j) | REPORT §8 |
| 2 | **imgsz 1280**, batch 2 (dicoba E12: ΔF1 val +0.004 n.s., mAP val +0.013; tidak lolos) | 52% lesi < 16 px (vs 73.5% di 960); latency lebih rendah dari SAHI | Tinggi (VRAM) | Audit #11 |
| 3 | **YOLOv8l-P2** (`yolov8l-p2.yaml` + transfer bobot) (dicoba E10: ΔF1 val −0.018 signifikan, training 21,6 jam; ditolak) | Head stride-4 untuk lesi tiny (FN dominan) | Tinggi | REPORT §7.2 |
| 4 | **Multi-seed (≥ 3)** untuk kandidat | Selisih antar varian kecil; perlu estimasi variasi seed | Sedang | REPORT §9 |
| 5 | **Epoch 200–300** (config E3) | E3 best di epoch 98/100, belum konvergen | Sedang | Audit #12 |
| 6 | **Ablation augmentasi** (`flipud=0`, `mixup=0`) | Memisahkan efek; E3 tidak lebih baik dari default untuk deteksi | Sedang | B0 vs E3 |
| 7 | **Pasang `albumentations`** (terpasang 2.0.8; augmentasi noise/kecerahan E11: ΔF1 val −0.008 / −0.015 SAHI; ditolak) | Augmentasi bawaan selama ini tidak aktif | Rendah | RESEARCH A2 |
| 8 | **Split terstratifikasi + cek duplikat/pasien** | Level 3 test n=1; risiko kebocoran | Sedang | Audit #13–14 |
| 9 | **Experiment tracking MLflow** (skill `mlflow`) | Perbandingan run otomatis | Rendah | – |

## Rencana Benchmark (dari riset 2026-11-13, `RESEARCH.md` D)

**Keputusan pengguna 2026-11-13: pembanding YOLO hanya keluarga YOLOv8** (B0, E3, E4, E12 @ 1280, E9 multi-seed). RT-DETR-l (B2) **dibatalkan** oleh pengguna 2026-11-13 (tidak diperlukan sebagai pembanding). YOLO11 (E6/E7, B1) tidak masuk tabel utama.

| Prioritas | Langkah | Status |
|---|---|---|
| P0 | Publikasikan daftar split dan aturan konversi lingkaran→box (`splits/`) | Selesai |
| P1a | ~~**B1**: YOLO11n @ 640~~ **tidak dipakai di tabel utama** (keputusan 2026-11-13: paper hanya memakai keluarga YOLOv8). B1 di split kita (baseline internal; acuan kasar 0.573 dari SAT-AcneDet yang belum terverifikasi), `scripts/run_baseline_b1.ps1` | Siap, pengguna menjalankan |
| P1b | YOLOv8l @ 1280 (= E12, setup mirip AcneAI): evaluasi **test** sekali, `run_paper_baselines.ps1 -Parts e12` | Siap |
| P1c | ~~**B2** RT-DETR-l @ 960, `configs/rtdetr_l_960.yaml`, `-Parts rtdetr`~~ | Dibatalkan 2026-11-13 (keputusan pengguna; training dihentikan di awal, tidak dicatat sebagai hasil) |
| P1d | Multi-seed E9 (seed 7, 123), `configs/yolov8l_960_tiles2_s{7,123}.yaml`, `-Parts seeds` | Siap (±11 jam/seed) |
| P1e | Faster R-CNN (pembanding klasik literatur jerawat) | Perlu keputusan: butuh pipeline baru di luar Ultralytics |
| P2 | Bootstrap berpasangan untuk baseline; laporkan MAE, RMSE, r, MAE relatif, grade Hayashi dari count | Antri |
| P3 | Protokol sekunder gaya AcneAI (sapuan conf 0.10–0.50, iou 0.6; **di luar aturan evaluasi proyek**, diberi label sekunder); opsional jembatan ke label v1 / 5-fold | Opsional |

Desain pembanding paper (3 tingkat: terkontrol di split kita / literatur dikutip / jembatan v1 5-fold; metode dari dataset lain hanya lewat replikasi di split kita atau dikutip di Related Work): `C:\Users\User\.claude\plans\jika-saya-ingin-membuat-immutable-starlight.md`.

## Di luar cakupan saat ini
Serving/API, ONNX/TensorRT export, aplikasi klien, grading keparahan sebagai produk.
Script lama tersedia di git history (commit `a56fc3c`) dan di `_to_delete/`.
