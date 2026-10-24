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

### Varian inferensi (tanpa training baru)

Matcher internal IoU 0.5, conf dipilih di val; P/R/F1 pada conf tersebut. Sumber `runs/eval/<run>/metrics.json`.

| Varian | Eval dir | conf | AP@50 internal | P | R | F1 | MAE | r | Latency | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| E3 + TTA | `yolov8l_960_tta` | 0.28 | 0.4671 | 0.5302 | 0.4534 | 0.4888 | 8.98 | 0.919 | 115 ms | ditolak |
| E3 + SAHI 1728 | `yolov8l_960_sahi1728` | 0.23 | 0.5370 | 0.5925 | 0.5341 | 0.5618 | 8.33 | 0.946 | 637 ms | diterima |
| **E4 + SAHI 1728** | `yolov8l_960_tiles2_ft_sahi1728` | 0.22 | **0.5499** | **0.6112** | **0.5361** | **0.5712** | **8.29** | 0.943 | 628 ms | ✅ **terbaik** |
| B0 + SAHI 1728 | `yolov8l_960_default_sahi1728` | 0.34 | 0.5113 | 0.5685 | 0.4876 | 0.5249 | 10.04 | 0.891 | 642 ms | ditolak |

Uji signifikansi (paired bootstrap) dan analisis lengkap: `docs/REPORT.md`.

\* MAE E2 diukur dengan protokol lama (conf 0.25, imgsz 640, max_det 300), jadi tidak sebanding langsung.
Angka E1 berasal dari dokumentasi lama; protokolnya tidak terdokumentasi lengkap.

## Detail

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
