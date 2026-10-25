# Status Proyek

> Terakhir diperbarui: 2026-10-25

## Fase saat ini: Pengembangan Model

- **Model terpilih (SPEC)**: YOLOv8l @ 960 (E3), `runs/detect/yolov8l_960/weights/best.pt`
- **Kandidat terbaik**: **E4 + SAHI 1728**, `runs/detect/yolov8l_960_tiles2_ft/weights/best.pt`
  (test, conf 0.22 dari val): F1 0.571 · Recall 0.536 · Precision 0.611 · MAE 8.29 · r 0.943 · 628 ms/gambar.
  Memenuhi semua ambang minimum `SPEC.md`. **Menunggu keputusan pengguna** untuk ditetapkan sebagai model terpilih.
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

## Berikutnya
- [ ] **Pengguna (GPU)**: E6/E7 = YOLO11l dengan resep E3/E4 (`scripts\run_yolo11.ps1`, ±3-3,5 jam), lalu agen memperbarui report/bootstrap/EXPERIMENTS
- [ ] Keputusan pengguna: tetapkan E4 + SAHI (akurasi) atau E3 plain (latency) sebagai model terpilih
- [ ] Roadmap: training tile dari nol / imgsz 1280 dengan epoch lebih panjang, head P2, multi-seed

## Catatan
- Bobot awal COCO ada di `pretrained/yolov8l.pt`; bobot hasil training per run di `runs/detect/<run>/weights/`.
