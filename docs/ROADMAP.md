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
| 1 | **Training tile dari nol**, 100+ epoch | E4 hanya 10 epoch fine-tune; resolusi efektif terbukti tuas utama | Tinggi (±5 j) | REPORT §8 |
| 2 | **imgsz 1280**, batch 2 | 52% lesi < 16 px (vs 73.5% di 960); latency lebih rendah dari SAHI | Tinggi (VRAM) | Audit #11 |
| 3 | **YOLOv8l-P2** (`yolov8l-p2.yaml` + transfer bobot) | Head stride-4 untuk lesi tiny (FN dominan) | Tinggi | REPORT §7.2 |
| 4 | **Multi-seed (≥ 3)** untuk kandidat | Selisih antar varian kecil; perlu estimasi variasi seed | Sedang | REPORT §9 |
| 5 | **Epoch 200–300** (config E3) | E3 best di epoch 98/100, belum konvergen | Sedang | Audit #12 |
| 6 | **Ablation augmentasi** (`flipud=0`, `mixup=0`) | Memisahkan efek; E3 tidak lebih baik dari default untuk deteksi | Sedang | B0 vs E3 |
| 7 | **Pasang `albumentations`** (CLAHE/Blur default aktif) | Augmentasi bawaan selama ini tidak aktif | Rendah | RESEARCH A2 |
| 8 | **Split terstratifikasi + cek duplikat/pasien** | Level 3 test n=1; risiko kebocoran | Sedang | Audit #13–14 |
| 9 | **Experiment tracking MLflow** (skill `mlflow`) | Perbandingan run otomatis | Rendah | – |

## Di luar cakupan saat ini
Serving/API, ONNX/TensorRT export, aplikasi klien, grading keparahan sebagai produk.
Script lama tersedia di git history (commit `a56fc3c`) dan di `_to_delete/`.
