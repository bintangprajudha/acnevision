# Indeks Script

Jalankan dari **root proyek** dengan `.venv\Scripts\python.exe`. Skrip `.ps1` menjalankan rantai perintah dan
menulis progres ke `runs\chain.log` (log tiap langkah: `runs\chain_<langkah>.log`).

## Inti (entry point)

| Script | Fungsi |
|---|---|
| `train.py` | Training dari config YAML (`--cfg`), auto-resume; smoke test: `--epochs 1 --fraction 0.02` |
| `evaluate.py` | **Satu-satunya** evaluasi resmi: deteksi, counting, grading Hayashi; mode `plain`, `tta`, `sahi` (`--slice`) |
| `predict.py` | Visualisasi prediksi untuk gambar/folder; opsi `--sahi` |
| `common.py`, `analysis.py` | Pustaka bersama: path, pemuat model, matching TP/FP/FN, AP, grading Hayashi |

## Analisis & laporan

| Script | Fungsi |
|---|---|
| `bootstrap.py` | Uji signifikansi paired bootstrap antar varian (CI 95%) |
| `report.py` | Grafik dan `docs/report/summary.json` untuk `docs/REPORT.md` |
| `export_ranking.py` | Peringkat model dalam CSV: `docs/report/ranking_models.csv` |
| `diagrams.py` | Diagram metode untuk presentasi |

## Data (`data/`)

| Script | Fungsi |
|---|---|
| `download_dataset.py`, `download_and_extract_images.py` | Unduh anotasi ACNE04-v2 dan gambar ACNE04 |
| `convert_to_yolo.py` | Lingkaran → kotak YOLO; split 963/120/121 (seed 42) |
| `preprocess.py` | Potong train/val menjadi tile (`--grid 2 --overlap 0.1`) |

## Rantai eksperimen (`.ps1`)

| Script | Isi |
|---|---|
| `run_paper_baselines.ps1` | Pembanding paper: `-Parts e12`, `rtdetr` (dibatalkan), `seeds` (E9 seed 7 dan 123) |
| `run_e8_e9.ps1` | E8 lalu E9 (training tile), evaluasi plain + SAHI |
| `run_stage1.ps1` | Screening E10, E11, E12 (hanya val) |
| `run_experiments.ps1`, `run_e5.ps1`, `run_yolo11.ps1` | Rantai eksperimen awal (E4, E5, E6/E7) |
| `run_baseline_b1.ps1` | B1 YOLO11n @ 640 (dihentikan) |
| `rerun_evals.ps1` | Evaluasi ulang semua varian agar `metrics.json` memuat metrik terbaru |

## Lain-lain

| Script | Fungsi |
|---|---|
| `verify_env.py` | Cek PyTorch dan CUDA |
| `serve.py` | API inferensi FastAPI (di luar cakupan pengembangan model saat ini) |

Folder `scripts/runs/` dan `scripts/weights/` adalah sisa unduhan otomatis Ultralytics (di-ignore git), bukan bagian kode.
