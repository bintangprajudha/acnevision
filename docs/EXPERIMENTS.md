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
| E6 | `yolo11l_960` | `configs/yolo11l_960.yaml` | 960 | 100 (82), resep E3, arsitektur YOLO11l | 55.09% | 47.85% | 48.80% | 18.29% | 8.23 | standar | ditolak (< E3 pada F1/mAP) |
| E7 | `yolo11l_960_tiles2_ft` | `configs/yolo11l_960_tiles2_ft.yaml` | 960 | 10 (7), resep E4, fine-tune dari E6 | 53.55% | 45.51% | 47.07% | 17.78% | 8.75 | standar | ditolak (plain) |
| E8 | `yolov8l_960_tiles2_ft60` | `configs/yolov8l_960_tiles2_ft60.yaml` | 960 | 60 (53), resep E4, fine-tune dari E3 | 60.49% | 48.25% | 51.46% | 19.63% | 7.66 | standar | **kandidat plain** (> E3, signifikan) |
| E9 | `yolov8l_960_tiles2` | `configs/yolov8l_960_tiles2.yaml` | 960 | 100 (84), resep E3 pada data tile, init COCO | 61.55% | 46.62% | 50.40% | 18.82% | 7.60 | standar | **kandidat terbaik** (lihat E9 + SAHI) |
| E10 | `yolov8l_960_p2` | `configs/yolov8l_960_p2.yaml` | 960 | 100 (89), resep E3, head P2 | 50.3%* | 45.2%* | 45.6%* | 18.4%* | 10.03* | val (screening) | ditolak (ΔF1 −0.018 signifikan; training 21,6 jam) |
| E11 | `yolov8l_960_augnoise` | `configs/yolov8l_960_augnoise.yaml` | 960 | 100 (75), resep E3 + noise/kecerahan | 51.6%* | 45.9%* | 46.9%* | 18.0%* | 9.96* | val (screening) | ditolak (ΔF1 −0.008 plain, −0.015 SAHI signifikan) |
| E12 | `yolov8l_1280` | `configs/yolov8l_1280.yaml` | 1280 | 100 (87), resep E3, batch 2 | 52.7%* | 47.2%* | 48.7%* | 20.1%* | 9.29* | val (screening) | tidak lolos kriteria (ΔF1 +0.004, n.s.); mAP terbaik di val |

### Varian inferensi (tanpa training baru)

Matcher internal IoU 0.5, conf dipilih di val; P/R/F1 pada conf tersebut. Sumber `runs/eval/<run>/metrics.json`.

| Varian | Eval dir | conf | AP@50 internal | P | R | F1 | MAE | r | Latency | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| E3 + TTA | `yolov8l_960_tta` | 0.28 | 0.4671 | 0.5302 | 0.4534 | 0.4888 | 8.98 | 0.919 | 115 ms | ditolak |
| E3 + SAHI 1728 | `yolov8l_960_sahi1728` | 0.23 | 0.5370 | 0.5925 | 0.5341 | 0.5618 | 8.33 | 0.946 | 637 ms | diterima |
| **E4 + SAHI 1728** | `yolov8l_960_tiles2_ft_sahi1728` | 0.22 | **0.5499** | **0.6112** | **0.5361** | **0.5712** | **8.29** | 0.943 | 628 ms | ✅ **terbaik** |
| B0 + SAHI 1728 | `yolov8l_960_default_sahi1728` | 0.34 | 0.5113 | 0.5685 | 0.4876 | 0.5249 | 10.04 | 0.891 | 642 ms | ditolak |
| E5 + SAHI 1728 | `yolov8l_960_default_tiles2_ft_sahi1728` | 0.22 | 0.5333 | 0.5982 | 0.5158 | 0.5540 | 8.83 | 0.936 | 631 ms | kontrol 2×2 (< E4 + SAHI) |
| E6 + SAHI 1728 | `yolo11l_960_sahi1728` | 0.20 | 0.5173 | 0.5727 | 0.5164 | 0.5431 | 8.69 | 0.949 | 542 ms | ditolak (< E3 + SAHI) |
| E7 + SAHI 1728 | `yolo11l_960_tiles2_ft_sahi1728` | 0.20 | 0.5246 | 0.5687 | 0.5369 | 0.5524 | 8.60 | 0.928 | 690 ms | ditolak (< E4 + SAHI) |
| E8 + SAHI 1728 | `yolov8l_960_tiles2_ft60_sahi1728` | 0.25 | 0.5533 | 0.6440 | 0.5170 | 0.5736 | 8.69 | 0.954 | 627 ms | setara E4 + SAHI (recall lebih rendah) |
| **E9 + SAHI 1728** | `yolov8l_960_tiles2_sahi1728` | 0.21 | **0.5621** | 0.6110 | **0.5572** | **0.5828** | **7.78** | **0.961** | 622 ms | ✅ **terbaik baru** (menunggu keputusan pengguna) |

Uji signifikansi (paired bootstrap) dan analisis lengkap: `docs/REPORT.md`.

\* MAE E2 diukur dengan protokol lama (conf 0.25, imgsz 640, max_det 300), jadi tidak sebanding langsung.
Angka E1 berasal dari dokumentasi lama; protokolnya tidak terdokumentasi lengkap.

## Detail

### E10 / E11 / E12: screening ide paper (tahap 1, hanya train + val; selesai 2026-10-30)
- **Dasar**: riset paper YOLO pada ACNE04 (`RESEARCH.md` bagian C). Tidak ada bobot terbuka yang layak, sehingga yang
  diadopsi adalah ide resepnya. Masing-masing satu variabel vs E3 (data asli, 100 epoch):
  - **E10** head P2: `model: yolov8l-p2.yaml` + `pretrained: pretrained/yolov8l.pt` (419/725 tensor tertransfer dari COCO). Dasar: Wen 2022.
  - **E11** `albumentations`: GaussNoise std 2–6% (p 0.2) + RandomBrightnessContrast ±0.2 (p 0.3), menggantikan default
    Ultralytics (p 0.01). Rotasi dan cermin sudah ada di E3. Dasar: Zhang 2024. Paket `albumentations` 2.0.8 dipasang 2026-10-29.
  - **E12** `imgsz` 1280, `batch` 2 (terpaksa karena VRAM 8 GB; nbs 64 menjaga batch efektif). Dasar: AcneAI 2024.
- **Evaluasi**: `evaluate.py --val-only` (plain + SAHI 1728) → `runs/eval/<run>[_sahi1728]_val/metrics_val.json`.
  Test **tidak disentuh** (permintaan pengguna). Conf dipilih dan diukur di val yang sama (optimistis, tetapi setara antar-run).
- **Kriteria menang (ditetapkan sebelum hasil)**: pada mode yang sama, ΔF1 val ≥ +0.010 vs E3 dan MAE val tidak memburuk
  > 0.3. `bootstrap.py --split val` (n = 120) sebagai pendukung.
- **Tahap 2**: ide yang menang digabung ke resep E9 (tile dari COCO) → E13, dievaluasi sekali di test vs E9.
- **Jalankan**: `powershell -ExecutionPolicy Bypass -File scripts\run_stage1.ps1` (±11–12 jam; smoke test ketiganya lebih dulu).
- **Hasil** (split val, 120 gambar, IoU 0.5, conf dipilih di val; sumber `runs/eval/<run>[_sahi1728]_val/metrics_val.json`,
  bootstrap `docs/report/bootstrap_val.json`; * pada tabel utama = angka val, P/R pada conf terpilih, mAP internal):

  | Run | conf | P | R | F1 | MAE | AP50 | mAP@50-95 | ΔF1 vs E3 [95% CI] | ΔMAE vs E3 [95% CI] |
  |---|---|---|---|---|---|---|---|---|---|
  | E3 | 0.21 | 0.524 | 0.467 | 0.494 | 9.87 | 0.471 | 0.188 | – | – |
  | E10 (P2) | 0.20 | 0.503 | 0.452 | 0.476 | 10.03 | 0.456 | 0.184 | −0.018 [−0.033, −0.003]* | +0.18 [−0.93, +1.43] |
  | E11 (noise) | 0.22 | 0.516 | 0.459 | 0.486 | 9.96 | 0.469 | 0.180 | −0.008 [−0.020, +0.005] | +0.09 [−0.88, +1.03] |
  | E12 (1280) | 0.19 | 0.527 | 0.472 | 0.498 | 9.29 | 0.487 | 0.201 | +0.004 [−0.009, +0.019] | −0.57 [−1.81, +0.58] |
  | E3 + SAHI | 0.23 | 0.556 | 0.505 | 0.529 | 10.22 | 0.493 | 0.185 | – | – |
  | E10 + SAHI | 0.23 | 0.559 | 0.468 | 0.509 | 10.01 | 0.474 | 0.178 | −0.020 [−0.036, −0.006]* | −0.22 [−1.23, +0.82] |
  | E11 + SAHI | 0.25 | 0.561 | 0.475 | 0.514 | 10.21 | 0.486 | 0.178 | −0.015 [−0.027, −0.003]* | −0.02 [−0.83, +0.73] |
  | E12 + SAHI | 0.21 | 0.569 | 0.487 | 0.525 | 10.13 | 0.501 | 0.193 | −0.004 [−0.018, +0.010] | −0.09 [−1.07, +0.92] |

  (* di kolom Δ = CI 95% tidak memuat 0.)
- **Kesimpulan**: **tidak ada ide yang lolos kriteria** (ΔF1 ≥ +0.010) → tahap 2 (E13) tidak dijalankan.
  - E10 (P2) lebih buruk secara signifikan di kedua mode dan training-nya 77.627 s (±21,6 jam, vs E3 ±2,2 jam).
    Dugaan: head P2 baru tanpa bobot pretrained, ditambah VRAM 8 GB yang meluber ke shared memory.
  - E11 (noise) lebih buruk, signifikan pada SAHI (recall −0.030). Foto ACNE04 relatif bersih, jadi noise sintetis tidak membantu.
  - E12 (1280) satu-satunya yang naik: AP50 +0.016, mAP@50-95 +0.013, MAE −0.57 (plain), tetapi ΔF1 +0.004 tidak signifikan,
    dan dengan SAHI tidak ada keuntungan. Latency plain 98,5 ms (vs 69,5 ms). Training 14.395 s (±4 jam).

### E8 / E9: durasi training tile
- **Pertanyaan**: apakah E4 dibatasi oleh fine-tune yang terlalu singkat (10 epoch)?
- **Dasar**: mAP@50 val tile E4 masih naik di epoch terakhir (0.4375 di ep 5 → 0.4600 di ep 10), val box loss masih
  turun (2.539 → 2.486); pola sama pada E5 (`runs/detect/<run>/results.csv`). Tuning hyperparameter tidak dipilih karena
  E3 (tuned) vs B0 (default) tidak berbeda nyata pada plain (ΔF1 −0.005 [−0.018, +0.008]).
- **E8**: config E4 dengan `epochs` 10 → 60 (`patience` = epochs seperti E4, jadwal cosine utuh). Hipotesis: E8+SAHI > E4+SAHI.
- **E9**: config E3 dengan `data` → `data_tiles2/data.yaml` (init COCO, 100 epoch, patience 25). Hipotesis: training tile
  penuh dari COCO ≥ fine-tune dari E3. Satu epoch tile = 4.815 gambar (±5× iterasi E3 per epoch).
- `best.pt` keduanya dipilih dari val tile (600 gambar), seperti E4; conf operasional tetap dipilih di val asli oleh `evaluate.py`.
- **Jalankan**: `powershell -ExecutionPolicy Bypass -File scripts\run_e8_e9.ps1` (`-FromE9` untuk melanjutkan dari E9).
- **Training** (`runs/chain.log`, `runs/detect/<run>/results.csv`): E8 60 epoch, 387 menit, puncak mAP@50 val tile 0.4817 di
  epoch 53 (E4: 0.4600 di epoch 10); E9 100 epoch, 645 menit, puncak 0.4814 di epoch 84.
- **Hasil (test; conf dipilih di val per varian; `runs/eval/<run>/metrics.json`, `scripts/bootstrap.py`)**:
  - E9+SAHI: F1 0.583, recall 0.557, precision 0.611, AP@50 0.562, mAP@50-95 0.203, MAE 7.78, r 0.961.
    E9+SAHI − E4+SAHI: ΔF1 +0.011 [−0.002, +0.024], ΔRecall **+0.020 [+0.002, +0.037]**, ΔMAE −0.51 [−1.67, +0.39].
    E9+SAHI − B0: ΔF1 **+0.063 [+0.048, +0.078]**, ΔRecall **+0.079 [+0.056, +0.105]**, ΔMAE −1.49 [−4.31, +0.25].
  - E8+SAHI − E4+SAHI: ΔF1 +0.002 [−0.010, +0.014], ΔRecall **−0.020 [−0.036, −0.006]** (precision naik ke 0.644, conf val 0.25).
  - Plain: E8 − E4 ΔF1 **+0.024 [+0.010, +0.037]**; E8 − E3 ΔF1 **+0.019 [+0.008, +0.029]**, ΔMAE **−1.02 [−1.94, −0.22]**;
    E9 − E3 ΔF1 **+0.014 [+0.003, +0.026]**, ΔMAE **−1.09 [−2.38, −0.07]**. Latency plain tetap ±70 ms.
- **Kesimpulan**: hipotesis E9 **diterima sebagian**: E9+SAHI terbaik di F1/recall/MAE dan recall-nya signifikan di atas
  E4+SAHI, tetapi selisih F1 belum signifikan. Fine-tune lebih panjang (E8) **tidak** menaikkan E4+SAHI, tetapi tile-FT
  yang panjang tidak lagi merugikan inferensi plain (berbeda dari E4/E5): E8 dan E9 plain kini signifikan lebih baik dari E3
  pada F1 dan MAE dengan latency sama. Satu seed.
- **Evaluasi**: `runs/eval/yolov8l_960_tiles2_ft60{,_sahi1728}/`, `runs/eval/yolov8l_960_tiles2{,_sahi1728}/`; prediksi
  `runs/predict/yolov8l_960_tiles2_ft60_sahi/test/` dan `runs/predict/yolov8l_960_tiles2_sahi/test/`.

### E9 multi-seed (seed 42, 7, 123), selesai 2026-11-14
- **Perubahan**: hanya `seed` (`configs/yolov8l_960_tiles2_s7.yaml`, `_s123.yaml`); lainnya identik dengan E9. Dijalankan lewat
  `scripts/run_paper_baselines.ps1 -Parts seeds` (`runs/chain.log`). Split data tetap (seed 42).
- **Hasil (test 121 gambar, imgsz 960, conf dipilih di val per run; `runs/eval/yolov8l_960_tiles2{,_s7,_s123}{,_sahi1728}/metrics.json`)**:

| Mode | Seed | conf | Precision | Recall | F1 | AP@50 (internal) | MAE | RMSE | r |
|---|---|---|---|---|---|---|---|---|---|
| SAHI 1728 | 42 | 0.21 | 0.6110 | 0.5572 | 0.5828 | 0.5621 | 7.78 | 14.47 | 0.961 |
| SAHI 1728 | 7 | 0.24 | 0.6432 | 0.5264 | 0.5790 | 0.5632 | 8.39 | 17.44 | 0.954 |
| SAHI 1728 | 123 | 0.20 | 0.5897 | 0.5669 | 0.5781 | 0.5548 | 7.88 | 14.23 | 0.956 |
| **SAHI rata-rata ± SD** | | 0.217 | 0.615 ± 0.027 | 0.550 ± 0.021 | **0.580 ± 0.003** | 0.560 ± 0.005 | 8.02 ± 0.33 | 15.38 ± 1.79 | 0.957 ± 0.004 |
| plain | 42 | 0.21 | 0.5534 | 0.5056 | 0.5284 | 0.5046 | 7.60 | 15.19 | 0.953 |
| plain | 7 | 0.23 | 0.5677 | 0.4827 | 0.5218 | 0.5009 | 8.14 | 17.62 | 0.944 |
| plain | 123 | 0.20 | 0.5292 | 0.5164 | 0.5227 | 0.4973 | 7.88 | 15.71 | 0.942 |
| **plain rata-rata ± SD** | | 0.213 | 0.550 ± 0.020 | 0.502 ± 0.017 | **0.524 ± 0.004** | 0.501 ± 0.004 | 7.87 ± 0.27 | 16.17 ± 1.28 | 0.947 ± 0.006 |

  SD = simpangan baku sampel (n = 3, ddof = 1). Training seed 7: 100 epoch, ±650 menit.
- **Kesimpulan**: F1 sangat stabil antar seed (SD 0.003); recall, precision, dan MAE lebih bervariasi (SD recall 0.021,
  MAE 0.33), sebagian karena conf yang dipilih berbeda. Ketiga seed SAHI punya F1 (0.578–0.583) di atas E4 + SAHI (0.5712, satu seed);
  signifikansi dengan bootstrap belum dihitung. Recall SAHI rata-rata 0.550 ± 0.021: seed 7 (0.526) di bawah acuan AcneAI 0.54. YOLO11l dengan resep E3 / E4
- **Pertanyaan**: apakah arsitektur YOLO11l lebih baik dari YOLOv8l pada pipeline yang sama?
- **Hipotesis**: YOLO11l (blok C3k2, SPPF + C2PSA; 25,31 jt vs 43,63 jt parameter untuk 1 kelas) setara atau lebih baik dari YOLOv8l
  pada resep yang sama, dengan inferensi lebih cepat.
- **Perubahan**: E6 = `configs/yolov8l_960.yaml` dengan `model: pretrained/yolo11l.pt`; E7 = `configs/yolov8l_960_tiles2_ft.yaml`
  dengan init dari E6. Data, hyperparameter, augmentasi, evaluasi, dan SAHI 1728 identik (satu variabel: arsitektur).
- **Catatan**: hyperparameter dituning untuk YOLOv8l; tidak dituning ulang untuk YOLO11.
- **Jalankan**: `powershell -ExecutionPolicy Bypass -File scripts\run_yolo11.ps1` (`-FromE7` untuk melanjutkan dari E7).
- **Training** (`runs/chain.log`): E6 100 epoch, 114 menit (E3: 134 menit), puncak mAP@50 val 0.455 di epoch 82
  (E3 0.467 di epoch 98); E7 10 epoch, 55 menit, puncak val tile 0.448 di epoch 7 (E4 0.460 di epoch 10).
- **Hasil (test; conf dipilih di val per varian; `runs/eval/yolo11l_960*/metrics.json`, `scripts/bootstrap.py`)**:
  - E6 plain: mAP@50 Ultralytics 48.80% (E3 49.80%), mAP@50-95 18.29% (E3 19.14%), F1 0.508 (E3 0.515), MAE 8.23 (E3 8.69).
    E6 − E3: ΔF1 −0.007 [−0.018, +0.004], ΔMAE −0.45 [−2.13, +0.78] → tidak berbeda nyata.
  - E6+SAHI − E3+SAHI: ΔF1 **−0.019 [−0.031, −0.006]**, ΔRecall **−0.018 [−0.032, −0.005]**, ΔMAE +0.35 [−0.39, +1.07].
  - E7 − E4 (plain): ΔF1 **−0.015 [−0.028, −0.004]**; E7+SAHI − E4+SAHI: ΔF1 **−0.019 [−0.031, −0.008]**, ΔMAE +0.31 [−0.49, +1.12].
  - E7+SAHI − E6+SAHI: ΔF1 +0.010 [−0.004, +0.024] → pola tile-FT sama dengan YOLOv8l (E4 vs E3).
  - Efisiensi: latency plain median 61 ms (E3 70 ms), model saja 32.7 ms vs 43.8 ms (Ultralytics, batch 4); 86.8 vs 164.8 GFLOPs (640 px).
- **Kesimpulan**: **hipotesis ditolak** untuk akurasi. Pada resep yang sama YOLO11l lebih efisien (±58% parameter) tetapi
  F1 turun 0.015–0.019 secara signifikan pada varian SAHI/tile-FT; MAE tidak berbeda nyata. E4 + SAHI tetap kandidat terbaik.
  *Opini*: resep (loss, augmentasi, lr) dituning untuk YOLOv8l, jadi hasil ini belum menutup kemungkinan YOLO11l unggul dengan
  resep sendiri. Satu seed.
- **Evaluasi**: `runs/eval/yolo11l_960{,_sahi1728}/`, `runs/eval/yolo11l_960_tiles2_ft{,_sahi1728}/`; prediksi
  `runs/predict/yolo11l_960_plain/test/` dan `runs/predict/yolo11l_960_tiles2_ft_sahi/test/`.

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
