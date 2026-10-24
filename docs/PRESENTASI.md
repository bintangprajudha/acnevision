# AcneVision: Rangkuman Development & Konten Presentasi

> Tanggal: 2026-10-24. Dokumen ini adalah **sumber konten** untuk
> `presentasi/AcneVision_Presentasi.pptx` (dibangun ulang oleh `presentasi/build_deck.js`).
> Detail teknis & tabel lengkap: `docs/REPORT.md`. Semua angka berasal dari
> `runs/eval/<run>/metrics.json`, `docs/report/summary.json`, dan `docs/report/bootstrap.json`
> (split **test**, 121 gambar, 3.507 lesi; conf operasional dipilih di **val**; matching IoU 0.5),
> kecuali disebutkan lain. Figur: `docs/report/figures/` (dibuat oleh `scripts/report.py` dan
> `scripts/diagrams.py`).

---

## A. Rangkuman Development

### A1. Timeline

| Tahap | Isi | Hasil kunci |
|---|---|---|
| Setup | `.venv` Python 3.10, PyTorch 2.6 + CUDA 12.4, RTX 4060 8 GB, `ultralytics 8.4.170` | – |
| Data | ACNE04 + anotasi ACNE04-v2 → YOLO; split 963/120/121 (seed 42) | 32.443 lesi, 1 kelas |
| E1 | YOLOv8s @640 (dokumentasi lama) | mAP@50 35.82% |
| E2 | YOLOv8l @640, default | mAP@50 43.73% |
| E3 | YOLOv8l @960, AdamW + loss & augmentasi tuned, 100 epoch | mAP@50 49.80%, MAE 8.69 |
| Audit | 17 poin vs standar Ultralytics; 5 perbaikan (protokol eval, `max_det`, threshold di val, `imgsz` eval, `copy_paste` no-op) | `PIPELINE.md` |
| Riset | Teknik peningkatan + ambang metrik dari literatur | `RESEARCH.md`, `SPEC.md` §5 |
| B0 | Baseline YOLOv8l @960 setelan default, 100 epoch | mAP@50 49.97% |
| Inferensi | E3 + TTA, E3 + SAHI 1728, B0 + SAHI | SAHI signifikan, TTA negatif |
| E4 | Fine-tune E3 pada tile 2×2 (10 epoch) + SAHI | **F1 0.571, MAE 8.29** (terbaik) |
| Evaluasi ulang | Semua varian dievaluasi ulang di GPU idle, dengan confusion matrix baru, jenis FP, latency, dan bootstrap | `REPORT.md` |

### A2. Keputusan penting
1. **Model YOLOv8l @960**: kompromi akurasi vs VRAM 8 GB (batch 4).
2. **Protokol evaluasi tunggal** (`evaluate.py`): `conf=0.001`, `iou=0.7`, `max_det=1000`;
   threshold counting dipilih di val, dan test hanya untuk laporan.
3. **Confusion matrix pada conf operasional**; sel background–background = N/A (TN tidak terdefinisi).
4. **Baseline = YOLOv8l default @960**. Anggaran GPU ≤ 4 jam, sehingga E4 berupa fine-tune, bukan training dari nol.
5. Visualisasi bbox **tanpa label teks**; jumlah lesi di strip header di luar foto.

### A3. Temuan utama
1. **Resolusi efektif = tuas terbesar.** SAHI menaikkan F1 +0.047 dan recall +0.046 pada E3 (signifikan).
2. Tuning augmentasi/hyperparameter (E3) **tidak** lebih baik dari default untuk deteksi
   (mAP@50 0.4980 vs 0.4997). Namun hanya E3 yang mendapat manfaat dari SAHI.
3. Fine-tune tile (E4) memberi tambahan kecil, tidak signifikan (ΔF1 +0.010 vs E3 + SAHI).
4. TTA memperburuk hasil (AP@50 0.467 vs 0.491).
5. Perbedaan MAE antar varian **belum signifikan** dengan 121 gambar test.
6. Statistik ukuran lesi dihitung ulang secara eksak: pada 960 px median **11.7 px**, **73.5%** < 16 px
   (angka lama 13 px / 65% tidak dapat direproduksi dan sudah dikoreksi di semua docs).
7. SAHI 1728 pada foto 3112×3456 membuat 9 slice, 3 di antaranya nyaris duplikat (geser 1 px).
   *Opini*: ini peluang memangkas latency ±1/3 tanpa mengubah akurasi.

### A4. Inventaris figur

| Figur | Isi | Pembuat |
|---|---|---|
| `pipeline.png` | Alur 6 tahap | `diagrams.py` |
| `lesion_size_hist.png` | Distribusi ukuran lesi pada 960 px | `diagrams.py` |
| `tiling_example.png` | Tile 2×2 + label pada satu tile | `diagrams.py` |
| `sahi_example.png` | Grid slice SAHI 1728 px | `diagrams.py` |
| `experiment_tree.png` | Silsilah varian + F1/MAE | `diagrams.py` |
| `eval_protocol.png` | Alur protokol evaluasi | `diagrams.py` |
| `bootstrap_ci.png` | Forest plot ΔF1 / ΔMAE (CI 95%) | `diagrams.py` |
| `detection_metrics.png`, `counting_metrics.png`, `count_scatter.png`, `mae_per_level.png`, `training_curves.png` | Grafik hasil | `report.py` |
| `cm_*.png` | Confusion matrix per varian | `evaluate.py` → `report.py` |
| `examples/*_{e4sahi,b0}.jpg` | Contoh prediksi (TP biru, FP kuning, FN merah) | `predict.py --compare` |
| `presentasi/assets/image-5-{1,2,3}.jpg` | Foto asli → lingkaran → bbox | deck v1 |

---

## B. Konten per Slide (17 slide, 3 varian)

Format tiap slide: **pesan utama** · isi · visual · *notes* (speaker notes ada di deck).

**Varian yang dibandingkan di deck**: B0 baseline (default Ultralytics), E3 tuned (plain), E4 + SAHI
(terbaik). Empat varian lain (E3 + TTA, E3 + SAHI, B0 + SAHI, E4 plain) hanya dibahas di `REPORT.md`
dan disebut singkat di notes.

**Metrik wajib**: Precision, Recall, mAP@50, mAP@50-95, confusion matrix, kecepatan inferensi
(slide 3, 9, 10, 11, 12, 15). mAP@50 dan mAP@50-95 memakai definisi internal (`scripts/analysis.py`:
matching greedy, interpolasi 101 titik, IoU 0.50 dan 0.50:0.05:0.95) agar SAHI sebanding dengan mode plain.
Pada prediksi yang sama, selisihnya terhadap Ultralytics ≤ 0.0002 (mAP@50) dan ≤ 0.0024 (mAP@50-95).
Nilai mAP@50-95 dari `scripts/rerun_evals.ps1` (2026-10-24 21:26); evaluasi ulang tidak mengubah P/R/TP/FP/FN, latency bergeser < 4 ms.

Angka: split test (121 gambar, 3.507 lesi), conf dipilih di val, matching IoU 0.5,
sumber `runs/eval/<run>/metrics.json`.

| | B0 baseline | E3 tuned | E4 + SAHI |
|---|---|---|---|
| conf (val) | 0.26 | 0.21 | 0.22 |
| Precision | 0.572 | 0.544 | **0.611** |
| Recall | 0.476 | 0.488 | **0.536** |
| mAP@50 | 0.499 | 0.491 | **0.550** |
| mAP@50-95 | 0.192 | 0.186 | **0.205** |
| F1 | 0.520 | 0.515 | **0.571** |
| TP / FP / FN | 1.671 / 1.252 / 1.836 | 1.712 / 1.432 / 1.795 | 1.880 / 1.196 / 1.627 |
| Inferensi (median) | 69 ms | 70 ms | 626 ms |
| MAE / bias / r | 9.27 / −4.83 / 0.860 | 8.69 / −3.00 / 0.910 | **8.29** / −3.56 / **0.943** |

### 1. Judul
- **AcneVision**: deteksi dan penghitungan lesi jerawat otomatis pada foto wajah. Foto hero = anotasi GT.

### 2. Latar belakang & tujuan
- Penghitungan manual lambat dan bervariasi; tujuan: deteksi lesi, hitung lesi, standar terukur.
- *Notes*: Tan et al. 2006; target klinis ICC ≥ 0.75.

### 3. Ringkasan hasil
- **Pesan**: E4 + SAHI memenuhi semua ambang minimum; kenaikan deteksi vs B0 signifikan.
- Stat: Precision 0.611 · Recall 0.536 · mAP@50 0.550 · mAP@50-95 0.205 · 626 ms/gambar.
- vs B0: F1 +0.051 [0.037, 0.066], Recall +0.059 [0.040, 0.077] (signifikan); MAE −0.98 (tidak signifikan).

### 4. Dataset & tantangan lesi kecil
- 1.204 gambar, 32.443 lesi, split 963/120/121 (seed 42). Chart % lesi < 16 px: 91.5 / **73.5** / 52.4
  (640/960/1280); `lesion_size_hist.png` (median 11.7 px pada 960).

### 5. Pipeline & preprocessing
- `pipeline.png` + 3 gambar foto → lingkaran → bbox (`presentasi/assets/image-5-*.jpg`).

### 6. Tiling & SAHI: memperbesar lesi
- `tiling_example.png` (tile 2×2, overlap 10%, train 963 → 4.815, skala 0.53) dan `sahi_example.png`
  (9 slice 1728 px → 960, NMS IoU 0.5, skala 0.56; 3 slice nyaris duplikat).

### 7. Tiga varian yang dibandingkan
- Tabel konfigurasi B0 / E3 / E4 + SAHI: inisialisasi, data, optimizer, loss, augmentasi, epoch, mode inferensi.
- *Notes*: detail augmentasi, temuan audit (copy_paste no-op, albumentations tidak terpasang), varian lain.

### 8. Protokol evaluasi & standar minimum
- `eval_protocol.png`; confusion matrix tanpa TN; bootstrap 10.000×.
- Ambang: P ≥ 0.50 / 0.60, R ≥ 0.50 / 0.60, mAP@50 ≥ 0.45 / 0.55, MAE ≤ 10 / 7, r ≥ 0.85 / 0.90,
  inferensi < 1 s / < 100 ms (minimum / baik). *Notes*: studi literatur (RESEARCH.md B).

### 9. Metrik utama
- Tabel 3 varian: conf, Precision, Recall, mAP@50, mAP@50-95, inferensi (coral = terbaik per kolom)
  + definisi singkat tiap metrik.

### 10. Hasil deteksi
- Chart Precision / Recall / mAP@50 / mAP@50-95 (seri E4 + SAHI, B0, E3) + tabel F1.
- Selisih F1 (CI 95%): E4 + SAHI − B0 +0.051 (signifikan), E4 + SAHI − E3 +0.057 (signifikan),
  E3 − B0 −0.005 (tidak signifikan).

### 11. Confusion matrix
- Tiga matriks (`cm_yolov8l_960_default.png`, `cm_yolov8l_960.png`, `cm_yolov8l_960_tiles2_ft_sahi1728.png`);
  baris = prediksi, kolom = GT, TN = N/A. FP lokalisasi / latar: B0 657 / 595, E3 866 / 566, E4 + SAHI 500 / 696.

### 12. Kecepatan inferensi
- Bar median ms per gambar: B0 69, E3 70, E4 + SAHI 626 (±14 vs ±1,6 gambar/detik).
- *Notes*: waktu model saja (model.val, batch 4): B0 43.2 ms, E3 42.2 ms.

### 13. Hasil counting
- Chart MAE (8.29 / 8.69 / 9.27) + tabel bias & Pearson r + `count_scatter.png` (B0 vs E4 + SAHI).
- *Notes*: level 3 (434 GT): B0 173, E4 + SAHI 264; selisih MAE tidak signifikan.

### 14. Contoh prediksi (E4 + SAHI)
- `levle1_482` (terbaik), `levle1_288` (F1 terendah), `levle3_89` (terpadat); TP biru, FP kuning, FN merah.

### 15. Hasil vs standar minimum
- Tabel 7 metrik × 3 varian + ambang + status E4 + SAHI. B0 dan E3 di bawah minimum Recall;
  E4 + SAHI memenuhi semua minimum (Precision & r baik); hanya mode plain yang < 100 ms.

### 16. Kesimpulan & langkah lanjut
- Memenuhi standar · kontributor utama SAHI · trade-off 9× lebih lambat.
- Rekomendasi: E4 + SAHI (akurasi) atau E3 plain (< 100 ms). Keterbatasan & roadmap ringkas.

### 17. Terima kasih & referensi
- ACNE04, ACNE04-v2, SAHI (Akyon 2022), dokumentasi Ultralytics, Li 2023, Seité 2019, Tan 2006.
