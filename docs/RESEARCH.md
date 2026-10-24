# Riset: Peningkatan Performa & Standar Metrik Deteksi Jerawat

> Diakses 2026-10-24 melalui Claude in Chrome / Europe PMC. Klaim kuantitatif dikutip dari sumber;
> bagian bertanda *opini* adalah penilaian penulis.

---

## A. Cara Meningkatkan Performa Model

Masalah inti proyek ini: **lesi sangat kecil dan padat**. Pada `imgsz=960`, median lesi hanya 11.7 px
dan 73.5% lesi < 16 px (lihat `PIPELINE.md`). Teknik di bawah diurutkan berdasarkan relevansinya
untuk masalah tersebut.

### A1. Resolusi efektif & slicing (prioritas tertinggi)
| Teknik | Bukti | Status di proyek |
|---|---|---|
| **Slicing Aided Hyper Inference (SAHI)**: inferensi per potongan gambar lalu digabung | Paper SAHI: AP naik **6.8% / 5.1% / 5.3%** (FCOS/VFNet/TOOD) hanya dengan sliced inference, tanpa fine-tuning [sahi] | Diuji pada E3 (tanpa training) |
| **Slicing-aided fine-tuning**: training pada tile resolusi penuh | Paper SAHI: kenaikan AP kumulatif lebih besar bila dikombinasikan dengan fine-tuning pada slice [sahi] | **E4** (tile 2×2, overlap 10%) |
| Naikkan `imgsz` (1280) | Ultralytics: "Larger sizes may improve accuracy for small objects" [val]; paper AcneAI memakai YOLOv8 @1280 [acneai] | Tidak muat dalam anggaran (VRAM 8 GB, ±2× waktu) |
| Head P2 (stride 4) | Arsitektur `yolov8-p2.yaml` tersedia di Ultralytics untuk objek kecil | Ditunda (perlu training dari awal) |

### A2. Preprocessing
| Teknik | Catatan |
|---|---|
| Tiling (A1) | Preprocessing paling relevan: mempertahankan detail resolusi asli 3456 px |
| CLAHE / normalisasi kontras | Ultralytics menyediakan `A.CLAHE(p=0.01)` lewat Albumentations, tetapi **paket `albumentations` tidak terpasang** di `.venv`, sehingga Blur/MedianBlur/ToGray/CLAHE default **tidak pernah aktif** di run mana pun (diverifikasi di `ultralytics/data/augment.py`) |
| Crop area wajah/kulit | Mengurangi FP di latar belakang; perlu detektor wajah/segmentasi kulit (skill `segment-anything`). *Opini*: dampak kecil karena FP dominan berada di kulit |

### A3. Augmentasi (Ultralytics [cfg])
| Parameter | Default | E3 | Catatan |
|---|---|---|---|
| `mosaic` | 1.0 | 1.0 | Efektif untuk objek kecil; dimatikan di epoch akhir (`close_mosaic`) |
| `copy_paste` | 0 | 0.3 → no-op | Hanya untuk task segment/semantic/obb |
| `flipud` | 0 | 0.5 | Wajah tidak pernah terbalik saat inferensi. *Opini*: kandidat ablation |
| `scale` | 0.5 | 0.5 | Variasi skala ±50% |
| `multi_scale` | 0.0 | 0.0 | Variasi `imgsz` per batch; berguna untuk inferensi multi-skala |
| `hsv_*` | 0.015/0.7/0.4 | sama | Variasi warna kulit & pencahayaan |

### A4. Hyperparameter & jadwal
| Temuan | Sumber |
|---|---|
| `optimizer=auto` mengabaikan `lr0`; pada B0 auto memilih **AdamW lr=0.002** | log training B0 |
| Ultralytics menyarankan **300 epoch** sebagai titik awal; E3 best di epoch 98/100 (belum konvergen) | [tips] |
| `model.tune()` (evolusi hyperparameter) tersedia, tetapi butuh puluhan run → di luar anggaran | [cfg] |
| Bobot loss `box`/`dfl` lebih tinggi untuk lokalisasi objek kecil (E3: 10/2.0) | praktik E3 |

### A5. Post-processing
| Teknik | Catatan |
|---|---|
| Threshold conf dipilih di val (minimisasi MAE counting) | Sudah diterapkan (`evaluate.py`): MAE 9.85 → 8.69 di E3 |
| Test-time augmentation (`augment=True`) | Flip + multi-skala saat inferensi; diuji pada E3 |
| `max_det` ≥ jumlah lesi maksimum | Sudah: 1000 (test punya gambar dengan 434 lesi) |

---

## B. Standar Metrik Minimum yang Layak

### B1. Fakta dari literatur

| Studi | Dataset / tugas | Metrik dilaporkan |
|---|---|---|
| Sangha & Rizvi 2021 (via [inass]) | ACNE04, 403 gambar, 1 kelas, YOLOv5 | mAP@50 **37.97%** |
| Zhang et al. 2024, IEEE Access (via [inass]) | ACNE04 v1, 1.457 gambar, 1 kelas, Improved-YOLOv7 | mAP@50 **83.7%** |
| Le et al. 2024 (via [inass]) | DermNet-NZ ±5.000, multi-kelas, YOLOv8 | mAP@50 78% |
| Fadhillah et al. 2026 [inass] | Roboflow 1.884 gambar (diaugmentasi → 6.116), 6 kelas, YOLOv11m | mAP@50 87.1%, mAP@50-95 64.9%, P 85%, R 82.9% |
| Li et al. 2023, Front. Med. [li2023] | Identifikasi lesi + grading | AP lesi **0.507**, recall 0.775; kappa grading 0.652–0.737 |
| Seité et al. 2019, Exp. Dermatol. [seite2019] | Smartphone, deteksi per tipe lesi | F1 **84%** (inflamasi), **61%** (non-inflamasi) |
| Gazeau et al. 2024, MICCAI (AcneAI) [acneai] | **ACNE04-v2** (dataset kita), deteksi lesi | AUC 0.79; TPR **0.52** @ FPR 0.09; ICC keparahan **0.81** |
| Tan et al. 2006 [tan2006] | 11 dermatolog, hitungan lesi | Dermatolog "tended to be reliable in acne lesion counting" (ICC); keandalan meningkat setelah pelatihan |

### B2. Kenapa angka literatur tidak bisa dibandingkan langsung
1. **Anotasi berbeda.** ACNE04-v2 menambahkan banyak lesi kecil yang terlewat di ACNE04 v1. Paper
   AcneAI menyebut bbox ACNE04 v1 "too big" dan "lesions are sometimes missed" [acneai]. mAP 80%+
   pada ACNE04 v1 diukur terhadap anotasi yang lebih longgar/jarang.
2. **Kebocoran data.** Beberapa studi mengaugmentasi dataset *sebelum* split (mis. 1.884 → 6.116 [inass]),
   yang berisiko menggelembungkan metrik.
3. **Protokol berbeda** (conf, IoU, split, max_det) dan sering tidak dilaporkan.

### B3. Ambang "layak" yang ditetapkan proyek ini (*opini berbasis bukti di atas*)

| Tingkat | mAP@50 | Recall | Precision | MAE counting | Pearson r | Dasar |
|---|---|---|---|---|---|---|
| **Minimum layak (riset)** | ≥ 0.45 | ≥ 0.50 | ≥ 0.50 | ≤ 10 | ≥ 0.85 | Melampaui AP lesi Li 2023 (0.507 multi-kelas) dan TPR AcneAI (0.52) pada dataset yang sama |
| **Baik** | ≥ 0.55 | ≥ 0.60 | ≥ 0.60 | ≤ 7 | ≥ 0.90 | Setara F1 non-inflamasi Seité 2019 (61%) |
| **Siap klinis** | – | – | – | Setara variasi antar-dermatolog | ICC ≥ 0.75 | Perlu validasi terhadap beberapa dermatolog [tan2006]; kriteria tidak bisa dicapai hanya dengan metrik dataset |

Catatan: untuk counting, **recall lebih penting daripada precision**, karena lesi terlewat menyebabkan
under-count sistematis. Target ini dicatat di `SPEC.md`.

---

## Referensi
- [sahi] Akyon et al., *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection*, ICIP 2022. https://arxiv.org/abs/2202.06934
- [cfg] Ultralytics Configuration: https://docs.ultralytics.com/usage/cfg/
- [val] Ultralytics Validation: https://docs.ultralytics.com/modes/val/
- [tips] Ultralytics Training Tips: https://docs.ultralytics.com/guides/model-training-tips/
- [acneai] Gazeau et al., *AcneAI*, MICCAI 2024. https://link.springer.com/chapter/10.1007/978-3-031-72086-4_7
- [inass] Fadhillah et al., *Deep Learning-based Multi-Class Acne Detection on Facial Images Using YOLOv11*, INASS Express 2026. https://inass-express.org/wp-content/uploads/2026/01/2026.002-2.pdf
- [li2023] Li et al., Front. Med. 2023. https://doi.org/10.3389/fmed.2023.1255704
- [seite2019] Seité et al., Exp. Dermatol. 2019. https://doi.org/10.1111/exd.14022
- [tan2006] Tan, Fung, Bulger, J. Cutan. Med. Surg. 2006;10(4):160–165. https://doi.org/10.2310/7750.2006.00044
- Zhang et al., *Automatic acne detection model based on improved YOLOv7*, IEEE Access 2024. https://ieeexplore.ieee.org/document/10810392
