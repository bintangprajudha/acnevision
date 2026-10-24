# Status Proyek

> Terakhir diperbarui: 2026-10-24

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
- [x] `rerun_evals.ps1` selesai (2026-10-24 21:26): mAP@50-95 internal test B0 0.1923, E3 0.1863, E4 + SAHI 0.2046
- [ ] **Pengguna**: predict E3 pada test (`predict.py --source data\images	est --compare --conf 0.21`) untuk contoh prediksi 3 varian
- [x] Metrik wajib deck: P, R, mAP@50, mAP@50-95, confusion matrix, kecepatan inferensi; `evaluate.py` kini menulis `map50_95_test_internal` (IoU 0.50:0.05:0.95, berlaku untuk SAHI; selisih vs Ultralytics ≤ 0.0024 pada prediksi yang sama)
- [ ] **Pengguna**: jalankan `scripts
erun_evals.ps1` (GPU) agar mAP@50-95 semua varian terisi, lalu build ulang deck
- [x] Statistik ukuran lesi dihitung ulang secara eksak (960 px: median 11.7 px, 73.5% < 16 px)

## Berikutnya
- [ ] Keputusan pengguna: tetapkan E4 + SAHI (akurasi) atau E3 plain (latency) sebagai model terpilih
- [ ] Roadmap: training tile dari nol / imgsz 1280 dengan epoch lebih panjang, head P2, multi-seed

## Catatan
- Bobot awal COCO ada di `pretrained/yolov8l.pt`; bobot hasil training per run di `runs/detect/<run>/weights/`.
