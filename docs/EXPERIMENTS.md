# Log Eksperimen

> Aturan pencatatan ada di `docs/README.md`. Satu baris per run, termasuk run gagal.
> Metrik resmi = output `scripts/evaluate.py` pada split **test**.
> Riwayat `results.csv` / `args.yaml` run lama tersimpan di `docs/experiments/history/`.

## Ringkasan

| ID | Run | Config | imgsz | Epoch (best) | P | R | mAP@50 | mAP@50-95 | MAE count | Protokol | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | `yolov8s_640` | history/yolov8s_640_args.yaml | 640 | 32/50, berhenti sebelum selesai (26) | 47.57% | 38.13% | 35.82% | 13.01% | – | lama | arsip, bobot dikarantina |
| E2 | `yolov8l_640` | history/yolov8l_640_args.yaml | 640 | 50 (49) | 56.24% | 43.40% | 43.73% | 15.81% | 9.19* | val default, max_det 300 | arsip, bobot dikarantina |
| **E3** | **`yolov8l_960`** | `configs/yolov8l_960.yaml` | 960 | 100 (98) | 56.31% | 48.10% | 49.80% | 19.14% | 8.69 | standar (`evaluate.py`) | model terpilih (SPEC) |
| B0 | `yolov8l_960_default` | `configs/yolov8l_960_default.yaml` | 960 | 100 (80) | 57.78% | 47.05% | 49.97% | 19.31% | 9.27 | standar | baseline |
| E4 | `yolov8l_960_tiles2_ft` | `configs/yolov8l_960_tiles2_ft.yaml` | 960 | 10 (10), fine-tune dari E3 | 58.39% | 46.96% | 49.68% | 18.91% | 8.45 | standar | ditolak (plain) |
| E5 | `yolov8l_960_default_tiles2_ft` | `configs/yolov8l_960_default_tiles2_ft.yaml` | 960 | 10 (9), fine-tune dari B0 | 54.45% | 45.85% | 47.74% | 18.00% | 9.37 | standar | kontrol 2×2 (plain ditolak) |
| E6 | `yolo11l_960` | `configs/yolo11l_960.yaml` | 960 | 100, resep E3, arsitektur YOLO11l | – | – | – | – | – | standar | direncanakan (`scripts/run_yolo11.ps1`) |
| E7 | `yolo11l_960_tiles2_ft` | `configs/yolo11l_960_tiles2_ft.yaml` | 960 | 10, resep E4, fine-tune dari E6 | – | – | – | – | – | standar | direncanakan (`scripts/run_yolo11.ps1`) |

### Varian inferensi (tanpa training baru)

Matcher internal IoU 0.5, conf dipilih di val; P/R/F1 pada conf tersebut. Sumber `runs/eval/<run>/metrics.json`.

| Varian | Eval dir | conf | AP@50 internal | P | R | F1 | MAE | r | Latency | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| E3 + TTA | `yolov8l_960_tta` | 0.28 | 0.4671 | 0.5302 | 0.4534 | 0.4888 | 8.98 | 0.919 | 115 ms | ditolak |
| E3 + SAHI 1728 | `yolov8l_960_sahi1728` | 0.23 | 0.5370 | 0.5925 | 0.5341 | 0.5618 | 8.33 | 0.946 | 637 ms | diterima |
| **E4 + SAHI 1728** | `yolov8l_960_tiles2_ft_sahi1728` | 0.22 | **0.5499** | **0.6112** | **0.5361** | **0.5712** | **8.29** | 0.943 | 628 ms | ✅ **terbaik** |
| B0 + SAHI 1728 | `yolov8l_960_default_sahi1728` | 0.34 | 0.5113 | 0.5685 | 0.4876 | 0.5249 | 10.04 | 0.891 | 642 ms | ditolak |
| E5 + SAHI 1728 | `yolov8l_960_default_tiles2_ft_sahi1728` | 0.22 | 0.5333 | 0.5982 | 0.5158 | 0.5540 | 8.83 | 0.936 | 631 ms | kontrol 2×2 (< E4 + SAHI) |

Uji signifikansi (paired bootstrap) dan analisis lengkap: `docs/REPORT.md`.

\* MAE E2 diukur dengan protokol lama (conf 0.25, imgsz 640, max_det 300), jadi tidak sebanding langsung.
Angka E1 berasal dari dokumentasi lama; protokolnya tidak terdokumentasi lengkap.

## Detail

### E6 / E7: YOLO11l dengan resep E3 / E4 (direncanakan)
- **Pertanyaan**: apakah arsitektur YOLO11l lebih baik dari YOLOv8l pada pipeline yang sama?
- **Hipotesis**: YOLO11l (blok C3k2, SPPF + C2PSA; 25,4 jt vs 43,7 jt parameter) setara atau lebih baik dari YOLOv8l
  pada resep yang sama, dengan inferensi lebih cepat.
- **Perubahan**: E6 = `configs/yolov8l_960.yaml` dengan `model: pretrained/yolo11l.pt`; E7 = `configs/yolov8l_960_tiles2_ft.yaml`
  dengan init dari E6. Data, hyperparameter, augmentasi, evaluasi, dan SAHI 1728 identik (satu variabel: arsitektur).
- **Catatan**: hyperparameter dituning untuk YOLOv8l; tidak dituning ulang untuk YOLO11.
- **Jalankan**: `powershell -ExecutionPolicy Bypass -File scripts\run_yolo11.ps1` (±3-3,5 jam; `-FromE7` untuk melanjutkan dari E7).
- **Pembanding (bootstrap)**: E6 − E3, E6+SAHI − E3+SAHI, E7 − E4, E7+SAHI − E4+SAHI.

### E5: `yolov8l_960_default_tiles2_ft` (kontrol desain 2×2)
- **Pertanyaan**: E4 di-fine-tune dari E3, sehingga E4+SAHI vs B0 mencampur dua faktor (tuning E3 + tile-FT).
  E5 melengkapi desain 2×2 (init B0/E3 × tile-FT ya/tidak, semua dengan SAHI 1728).
- **Hipotesis**: tile-FT juga menaikkan F1/recall bila dimulai dari B0 (E5+SAHI > B0+SAHI).
- **Perubahan vs E4**: hanya `model` = bobot B0 (`runs/detect/yolov8l_960_default/weights/best.pt`); data, jadwal, loss, augmentasi identik.
- **Training**: 10 epoch, 66 menit (2026-10-25 00:25–01:31, `runs/chain.log`), best epoch 9 (val tile mAP@50 0.451; E4 0.460 di epoch 10).
- **Hasil (test, conf 0.22 dari val; `scripts/bootstrap.py`)**:
  - Plain: mAP@50 47.74% (B0 49.97%), F1 0.495; E5 − B0 ΔF1 −0.025 [−0.036, −0.015] → tile-FT merugikan inferensi plain.
  - SAHI: F1 0.554, recall 0.516, MAE 8.83. E5+SAHI − B0+SAHI ΔF1 +0.029 [+0.016, +0.043], ΔMAE −1.20 [−2.65, −0.14] → **hipotesis diterima**.
  - E4+SAHI − E5+SAHI ΔF1 +0.017 [+0.005, +0.030], ΔMAE −0.54 [−1.07, −0.02] → init E3 tetap lebih baik.
- **Kesimpulan**: kedua faktor berkontribusi bila memakai SAHI; E4+SAHI tetap kandidat terbaik. Bobot tile-FT hanya untuk SAHI. Satu seed.
- **Evaluasi**: `runs/eval/yolov8l_960_default_tiles2_ft{,_sahi1728}/metrics.json`; prediksi `runs/predict/yolov8l_960_default_tiles2_ft_sahi/test/`.

### E4: `yolov8l_960_tiles2_ft`
- **Hipotesis**: fine-tune pada tile resolusi penuh (lesi tampil ±2× lebih besar) + inferensi SAHI menaikkan recall lesi kecil [SAHI, Akyon 2022].
- **Perubahan vs E3**: data `data_tiles2/` (asli + tile 2×2 overlap 10%, `scripts/data/preprocess.py`), init dari E3 `best.pt`, lr0 2e-4, 10 epoch, close_mosaic 3.
- **Hasil (test)**: plain setara E3 (mAP@50 49.68%). Dengan SAHI 1728: F1 0.571, recall 0.536, MAE 8.29, varian terbaik. Selisih vs E3 + SAHI tidak signifikan (ΔF1 +0.010 [−0.004, +0.024]).
- **Kesimpulan**: diterima sebagai kandidat terbaik (bersama SAHI). Bukan training dari nol (anggaran ≤ 4 jam GPU).
- **Evaluasi**: `runs/eval/yolov8l_960_tiles2_ft{,_sahi1728}/metrics.json`

### B0: `yolov8l_960_default` (baseline)
- **Hipotesis**: setelan default Ultralytics sebagai pembanding E3.
- **Perubahan vs E3**: semua hyperparameter/augmentasi default (optimizer `auto` → AdamW lr 0.002).
- **Hasil (test)**: mAP@50 49.97% (≈ E3 49.80%); MAE 9.27 (E3 8.69, selisih tidak signifikan). Recall 0.476 < ambang minimum 0.50.
- **Kesimpulan**: tuning E3 tidak menaikkan deteksi; tetapi B0 + SAHI tidak membaik, sedangkan E3 + SAHI membaik signifikan.
- **Catatan**: training terhenti di epoch 45 (RAM sistem), lalu dilanjutkan dari `last.pt`.

### E3 + TTA / E3 + SAHI
- TTA (`augment=True`) memperburuk AP50 (0.467 vs 0.491) dan ±1.7× lebih lambat → ditolak.
- SAHI slice 1728 → 960: F1 +0.047, recall +0.046 vs E3 plain (signifikan) → diterima.

### E3: `yolov8l_960` (terpilih)
- **Hipotesis**: resolusi 960 + AdamW + bobot loss box/dfl lebih tinggi menaikkan recall lesi kecil.
- **Perubahan vs E2**: imgsz 640→960, batch 8→4, SGD(auto)→AdamW lr0 1e-3 cosine, box 7.5→10, cls 0.5→0.3, dfl 1.5→2, flipud 0.5, degrees 10, mixup 0.1, 100 epoch.
- **Hasil**: mAP@50 +6.1 poin vs E2 (49.80 vs 43.73), recall +4.7 poin. Inferensi 45 ms vs 20 ms.
- **Catatan**: `copy_paste=0.3` ternyata no-op (label bbox). Best di epoch 98, artinya belum konvergen penuh.
  Karena beberapa variabel diubah sekaligus, kontribusi masing-masing belum diketahui.
- **Evaluasi**: `runs/eval/yolov8l_960/metrics.json`

### E2: `yolov8l_640`
- YOLOv8l default hyperparameter, 640px, 50 epoch. Diukur ulang 2026-10-24 dengan argumen val default.

### E1: `yolov8s_640`
- Baseline YOLOv8s default, 640px, 50 epoch (early stop, best epoch 26).
