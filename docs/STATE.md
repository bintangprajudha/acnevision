# Status Proyek

> Terakhir diperbarui: 2026-11-13

## Fase saat ini: Pengembangan Model

- **Model terpilih (SPEC)**: **E9** (YOLOv8l @ 960, tile 2×2 dari COCO), `runs/detect/yolov8l_960_tiles2/weights/best.pt`; ditetapkan pengguna 2026-11-13 (sebelumnya E3). Mode: **E9 + SAHI 1728** (training tile dari COCO), `runs/detect/yolov8l_960_tiles2/weights/best.pt`
  (test, conf 0.21 dari val): F1 0.583 · Recall 0.557 · Precision 0.611 · AP@50 0.562 · MAE 7.78 · r 0.961 · 622 ms/gambar.
  Bobot yang sama dalam mode plain (±70 ms): F1 0.528 · MAE 7.60. Sebelumnya E4 + SAHI (F1 0.571, MAE 8.29).
- **Laporan lengkap**: `docs/REPORT.md`

## Selesai
- [x] Setup environment (.venv, PyTorch 2.6 + CUDA 12.4, RTX 4060)
- [x] Dataset ACNE04-v2 → format YOLO (963/120/121)
- [x] Eksperimen E1 (YOLOv8s 640), E2 (YOLOv8l 640), E3 (YOLOv8l 960)
- [x] Audit pipeline vs standar Ultralytics/literatur (`PIPELINE.md`), 5 perbaikan diterapkan
- [x] Restrukturisasi: config YAML, `train.py` / `evaluate.py` / `predict.py` terpadu
- [x] Confusion matrix pada conf operasional (sel TN = N/A), visualisasi bbox tanpa label (`predict.py --compare`)
- [x] Riset peningkatan performa & ambang metrik minimum (`RESEARCH.md`, `SPEC.md` §5)
- [x] B0 (baseline default), E3 + TTA/SAHI, E4 (tile fine-tune), B0 + SAHI, dievaluasi dengan protokol yang sama
- [x] Prediksi seluruh 121 gambar test (B0 & E4 + SAHI), laporan `REPORT.md`
- [x] Diagram metode (`scripts/diagrams.py`), rangkuman `PRESENTASI.md`, deck 17 slide (3 varian: B0, E3, E4 + SAHI) dibangun ulang (`presentasi/build_deck.js`; v1 di `presentasi/arsip/`)
- [x] Notebook ringkasan `notebooks/AcneVision_Ringkasan.ipynb` (25 grafik/gambar, dieksekusi di CPU; prediksi test E3 sudah ada di `runs/predict/yolov8l_960_plain/test/`)
- [x] `rerun_evals.ps1` selesai (2026-10-24 21:26): mAP@50-95 internal test B0 0.1923, E3 0.1863, E4 + SAHI 0.2046
- [x] Predict E3 pada test (`runs/predict/yolov8l_960_plain/test/`, conf 0.21)
- [x] Metrik wajib deck: P, R, mAP@50, mAP@50-95, confusion matrix, kecepatan inferensi; `evaluate.py` kini menulis `map50_95_test_internal` (IoU 0.50:0.05:0.95, berlaku untuk SAHI; selisih vs Ultralytics ≤ 0.0024 pada prediksi yang sama)
- [x] Deck dibangun ulang 2026-10-25: 17 slide, 4 model (B0, E3, E5 + SAHI, E4 + SAHI), slide 7 = diagram cara keempat model dibuat
- [x] `scripts/bootstrap.py` (reprodusibel, identik dengan `bootstrap.json` lama) + pasangan 2×2
- [x] E5 (tile-FT dari B0, kontrol 2×2) selesai 2026-10-25: E5+SAHI F1 0.554 (> B0+SAHI 0.525, < E4+SAHI 0.571, keduanya signifikan); `REPORT.md` §5.6, notebook bagian 5.6
- [x] Statistik ukuran lesi dihitung ulang secara eksak (960 px: median 11.7 px, 73.5% < 16 px)
- [x] E6/E7 = YOLO11l dengan resep E3/E4 (`scripts/run_yolo11.ps1`): lebih efisien (25,3 jt vs 43,6 jt parameter, 61 vs 70 ms plain) tetapi F1 lebih rendah (E7+SAHI 0.552 vs E4+SAHI 0.571, ΔF1 −0.019 signifikan); `REPORT.md` §5.8. Deck & notebook belum memuat E6/E7
- [x] E8 (fine-tune tile 60 ep) dan E9 (tile dari COCO, 100 ep) selesai 2026-10-29: E9+SAHI terbaik (F1 0.583, recall +0.020 vs E4+SAHI signifikan); E8/E9 plain > E3 signifikan (F1, MAE). `REPORT.md` §5.9. Deck & notebook belum memuat E8/E9
- [x] Riset paper YOLO pada ACNE04 (`RESEARCH.md` bagian C): tidak ada bobot terbuka yang layak; ide P2 / noise / 1280 masuk tahap 1. `train.py` kini mendukung YAML bawaan Ultralytics, `pretrained`, dan blok `albumentations`; `evaluate.py --val-only`; `bootstrap.py --split val`

## Berikutnya
- [x] Standar minimum deteksi ditetapkan 2026-11-13 (`docs/SPEC.md` §5.0): acuan tunggal AcneAI (Gazeau, MICCAI 2024), recall ≥ 0.54 dan precision ≥ 0.54; tile 2×2 + SAHI memenuhi, plain belum (recall 0.5056)
- [x] Pembanding paper disiapkan 2026-11-13: `scripts/run_paper_baselines.ps1` (E12 test, E9 seed 7/123; B2 RT-DETR-l dibatalkan pengguna 2026-11-13); `common.load_model` memilih kelas RTDETR/YOLO otomatis (train/evaluate/predict)
- [x] Keputusan pengguna 2026-11-13: paper memakai keluarga **YOLOv8** saja; YOLO11 (E6/E7, B1) tidak masuk tabel utama
- [x] E9 seed 7 dan 123 selesai 2026-11-14 (`-Parts seeds`); SAHI F1 0.580 ± 0.003, recall 0.550 ± 0.021, MAE 8.02 ± 0.33 (n = 3; `docs/EXPERIMENTS.md`)
- [x] Tahap 1 screening ide paper selesai 2026-10-30 (val saja): E10 (P2) dan E11 (noise) lebih buruk, E12 (1280) ΔF1 +0.004 n.s. → tidak ada yang lolos kriteria; tahap 2 tidak dijalankan. `EXPERIMENTS.md` E10–E12
- [x] Keputusan pengguna 2026-11-13: E9 ditetapkan sebagai model terpilih (SAHI untuk akurasi, plain untuk latency < 100 ms); `SPEC.md` diperbarui
- [x] Riset benchmark pembanding 2026-11-13 (`RESEARCH.md` bagian D, laporan `reports/Benchmark deteksi jerawat ACNE04.md`): belum ada benchmark ACNE04-v2 yang sebanding penuh; SAT-AcneDet (mAP@50 0.608, baseline YOLO11n 0.573) **tidak terverifikasi**, jangan dikutip sampai PDF terbaca
- [x] P0: daftar split `splits/acne04v2_split.csv` + `splits/README.md`; config B1 `configs/yolo11n_640.yaml` + `scripts/run_baseline_b1.ps1` disiapkan (belum dijalankan)
- [ ] **Pengguna menjalankan B1**: `powershell -ExecutionPolicy Bypass -File scripts\run_baseline_b1.ps1` (±1 jam), (B1 sedang berjalan sejak 12:41); dipakai sebagai baseline internal, angka 0.573 SAT-AcneDet hanya acuan kasar
- [ ] Roadmap: training tile dari nol / imgsz 1280 dengan epoch lebih panjang, head P2, multi-seed

## Catatan
- Bobot awal COCO ada di `pretrained/yolov8l.pt`; bobot hasil training per run di `runs/detect/<run>/weights/`.
