# Benchmark counting & grading jerawat pada ACNE04 / ACNE04-v2

Catatan riset (2026-11-13). Konteks pengguna: YOLOv8l @ 960px pada ACNE04-v2, count = jumlah deteksi pada conf 0.21, test (121 citra) MAE 7.78, Pearson r 0.961; split 963/120/121.

Konvensi: "MSE" pada literatur ACNE04 = **root** mean squared error (RMSE), terverifikasi pada Wu et al. 2019 (lihat Q1).

---

## Q1. Wu et al. 2019 (LDL): berapa MAE/MSE counting dan akurasi grading, dengan protokol apa?

### Takeaway
Wu et al. (ICCV 2019) melaporkan **MAE 2.93 ± 0.18, "MSE" (sebenarnya RMSE) 5.42 ± 0.66** untuk counting, dan **akurasi grading 84.11%, Youden Index 75.32%**, dengan rata-rata **5-fold cross-validation** (tiap fold 80/20 = 1.165 train / 292 test, level citra, resolusi 224×224). Ini protokol "resmi" ACNE04 yang dipakai hampir semua paper lanjutan.

### Cited Findings
- Paper: Xiaoping Wu, Ni Wen, Jie Liang, Yu-Kun Lai, Dongyu She, Ming-Ming Cheng, Jufeng Yang, "Joint Acne Image Grading and Counting via Label Distribution Learning", ICCV 2019, hlm. 10642–10651 — [CVF Open Access](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)
- Dataset: "ACNE04 contains 1,457 images with 18,983 bounding boxes of lesions. For evaluating, we split the dataset into 80% training set and 20% testing set, containing 1,165 and 292 images" — [Wu 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)
- Protokol: "Notice that 5-fold cross-validation is applied for robust evaluation." Nilai tabel = rata-rata ± std antar fold. Backbone ResNet-50, input 224×224, SGD batch 32, 120 epoch — [Wu 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)
- Kode resmi melatih/menguji 5 split `NNEW_trainval_{0..4}.txt` / `NNEW_test_{0..4}.txt` (`cross_val_lists = ['0','1','2','3','4']`) — [github.com/xpwu95/LDL code/train.py](https://github.com/xpwu95/LDL/blob/master/code/train.py)
- Definisi metrik: "We adopt mean absolute error (MAE) and **root mean squared error (MSE)** to evaluate the object counting performance"; Youden Index = Sensitivity + Specificity − 1 — [Wu 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)
- Kriteria Hayashi (Tabel 1): Mild 1–5 lesi, Moderate 6–20, Severe 21–50, Very severe >50 — [Wu 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf); dikonfirmasi oleh [Prokhorov & Kalinin 2024](https://arxiv.org/html/2403.00268v1)
- Statistik split (Tabel 1, citra/lesi): Train — Mild 410/858, Moderate 506/4.547, Severe 146/3.857, Very severe 103/5.965, total 1.165/15.227. Test — Mild 103/221, Moderate 127/1.123, Severe 36/890, Very severe 26/1.522, total 292/3.756 — [Wu 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)

**Tabel 4 Wu 2019 — counting (grading diturunkan dari count via Hayashi), 5-fold mean ± std:**

| Metode | MAE | "MSE" (RMSE) | Precision | Specificity | Sensitivity | Youden | Accuracy |
|---|---|---|---|---|---|---|---|
| Faster R-CNN | 6.70 ± 0.28 | 11.51 ± 0.37 | 56.91 | 90.32 | 61.01 | 51.34 | 73.97 |
| RefineDet | 5.82 ± 0.53 | 10.14 ± 0.49 | 72.20 | 89.53 | 66.03 | 55.56 | 72.09 |
| YOLOv3 | 6.69 ± 0.28 | 11.35 ± 0.13 | 67.01 | 85.96 | 51.68 | 37.63 | 63.70 |
| MCNN (density) | 5.28 ± 0.20 | 7.76 ± 0.29 | 63.97 | 82.84 | 46.22 | 29.07 | 58.01 |
| **LDL (Ours)** | **2.93 ± 0.18** | **5.42 ± 0.66** | 84.37 | 93.80 | 81.52 | 75.32 | 84.11 |

Sumber: [Wu 2019, Tabel 4](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)

**Tabel 2 Wu 2019 — ablasi (MAE / Precision / YI / Accuracy):** SLL(C) 4.16 / 76.04 / 66.23 / 75.69; LDL(C) 3.24 / 80.39 / 70.65 / 81.09; Ours(G+C) 3.01 / 83.12 / 72.88 / 82.53; Ours 2.93 / 84.37 / 75.32 / 84.11. (Angka MAE 4.16 yang beredar di ringkasan pihak ketiga adalah baseline SLL counting, **bukan** hasil utama.) — [Wu 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)

**Pembanding manusia (Tabel 5, 700 citra acak, akurasi grading):** GD1 58.43, GD2 63.14, Derm1 75.29, Derm2 79.43 (Youden 39.38 / 55.31 / 63.22 / 70.43) — [Wu 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)

### Inferences
- "MSE" Wu pasti RMSE: secara matematis MSE ≥ MAE² (2.93² = 8.58 > 5.42), dan teks menyebut "root". Paper lanjutan (mis. Alzahrani 2022) menulis rumus MSE tanpa akar namun menyalin angka Wu, jadi istilahnya tidak konsisten di literatur.
- Rata-rata GT pada test fold Wu = 3.756/292 ≈ **12.9 lesi/citra**; MAE 2.93 ≈ 23% dari rata-rata count.
- Wu **tidak** melaporkan korelasi Pearson; r 0.961 milik pengguna tidak punya padanan langsung di Wu.

### Gaps
- Daftar file split 5-fold (`NNEW_*.txt`) ada di dalam unduhan dataset (Google Drive/Baidu), tidak diverifikasi isinya di sini.

---

## Q2. Paper lanjutan (2020–2026) yang melaporkan counting dan/atau grading pada ACNE04

### Takeaway
Hanya sedikit paper yang melaporkan **counting MAE/RMSE** pada ACNE04; sebagian besar paper lanjutan hanya melaporkan **akurasi grading** (84–87%). Angka counting terbaik yang ditemukan: **MAE 1.76 / RMSE 3.57** (Alzahrani dkk. 2022, satu split 80/20, bukan 5-fold), dan detektor YOLOv4 pada ACNE04 yang difilter: **MAE 3.49** (Wen dkk. 2022).

### Cited Findings

**Counting + grading**

| Paper | Venue/Tahun | Metode | MAE | "MSE" | Acc | Protokol | Sumber |
|---|---|---|---|---|---|---|---|
| Wu et al., Joint Acne Image Grading and Counting via LDL | ICCV 2019 | LDL multi-task, ResNet-50 | 2.93 | 5.42 (RMSE) | 84.11 | 5-fold, 1165/292 | [CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf) |
| Alzahrani, Al-Bander, Al-Nuaimy, Attention Mechanism Guided Deep Regression Model for Acne Severity Grading | Computers (MDPI) 11(3):31, 2022 | Dilated UNet density map + atensi dari box Faster R-CNN, 512×512 | **1.76** | **3.57** | **91.5** (Pre 88.25, Spe 93, Sen 83) | Split 80/20 (1165 train+val / 292 test); tidak disebut 5-fold | [DOI](https://doi.org/10.3390/computers11030031), [PDF](https://pdfs.semanticscholar.org/5941/e63070a41f48704f5c8c547ef477dda273c2.pdf) |
| Wen, Yu, Wu, Zhao, Liu, Kuang, Fan, Acne detection and severity evaluation with interpretable CNN models | Technology and Health Care 30:S143–S153, 2022 | Detektor (YOLOv4 dll.) pada ACNE04 **yang difilter** | 3.49 (terbaik); YOLOv4 3.61 / 3.65 / 3.77 pada score thr 0.25 / 0.50 / 0.75 | RMSE per level (mis. Mild MAE 1.19 / RMSE 1.83) | – | ACNE04 difilter; detail split tidak diverifikasi | [DOI](https://doi.org/10.3233/THC-228014) (angka dari snippet pencarian; teks penuh 403) |
| (anonim di abstrak) A coarse-to-fine framework for facial acne grading and lesion counting (CFGF) | Biomedical Signal Processing and Control, 2025 (S1746809425011553) | CNN backbone + coarse head + 2 cabang fine | tidak terbaca | tidak terbaca | tidak terbaca | ACNE04 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1746809425011553) (403) |

Alzahrani dkk. menyatakan: "The developed method shows MAE of 1.76, MSE of 3.57, precision of 88.35% [tabel: 88.25], specificity of 93%, sensitivity of 83%, and accuracy of 91.5%", melampaui LDL di semua metrik kecuali specificity — [Alzahrani 2022](https://pdfs.semanticscholar.org/5941/e63070a41f48704f5c8c547ef477dda273c2.pdf). Catatan: teks menyebut precision 88.35, tabel 88.25 (inkonsistensi internal).

**Grading saja (tanpa counting MAE)** — angka akurasi ACNE04 sebagaimana dikompilasi CG-HAF (Tabel 11), yang menegaskan "angka berasal dari protokol berbeda, bukan ranking terkontrol" — [arXiv 2609.31326](https://arxiv.org/html/2609.31326):

| Metode | Tahun | Acc ACNE04 |
|---|---|---|
| LDL (Wu) | 2019 | 84.11 |
| KIEGLFN (Computer Methods and Programs in Biomedicine 221:106911) | 2022 | 84.52 ([PwC](https://cs.paperswithcode.com/paper/kieglfn-a-unified-acne-grading-framework-on)) |
| Prior-Knowledge Guided (PKGF) | 2022 | 85.27 |
| AcneGrader | 2022 | 85.82 |
| DED: Diagnostic Evidence Distillation | 2023 | 86.06 ([PwC](https://ml.paperswithcode.com/paper/ded-diagnostic-evidence-distillation-for-acne)) |
| FF-PLL: Feature Feedback-Based Pseudo-Label Learning | 2025 | 87.33 ([PMC12024611](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12024611/)) |
| CG-HAF (preprint, split tetap 1049/116/292, image-level) | 2026 | 87.33; F1 0.8576; QWK 0.9173 |

- Prokhorov & Kalinin, "Improving Acne Image Grading with Label Distribution Smoothing", ISBI 2024: mengaku **tidak bisa mereproduksi persis** hasil Wu; LDL Wu yang dilatih ulang = Acc 83.70 ± 1.53, Youden 74.83 ± 4.06; metode mereka 84.11 ± 1.94, Youden 75.52 ± 3.61, MCC 76.16 (5-fold pra-definisi). Tidak ada MAE/MSE — [arXiv 2403.00268](https://arxiv.org/html/2403.00268v1)
- AcneFusionNet (proyek GitHub Technion, bukan paper peer-review): 85.86% akurasi, 5-fold dengan "official 5-fold split files"; tidak ada MAE — [GitHub](https://github.com/razanjiryis/acne-severity-assessment)
- Akurasi lain yang muncul di pencarian tetapi tidak diverifikasi dari sumber primer: AcneDGNet 91.2%, LENS-GRF (preprint 2026) 80.82% — [search; arXiv 2610.00294](https://arxiv.org/html/2610.00294)
- Zhang & Ma 99.31% memakai reformulasi 3 kelas, tidak sebanding — [CG-HAF](https://arxiv.org/html/2609.31326)

### Inferences
- Hasil Alzahrani (MAE 1.76) berasal dari satu split, bukan rata-rata 5-fold, dan belum direplikasi independen; perlakukan sebagai klaim self-reported.
- Wen dkk. (MAE 3.49 dengan YOLOv4) adalah pembanding **detection-based** terdekat dengan pendekatan pengguna, tetapi memakai ACNE04 v1 yang difilter.

### Gaps
- Angka MAE/akurasi CFGF (BSPC 2025), detail Wen 2022 (jumlah citra setelah filter, split), serta KIEGLFN/DED/PKGF/AcneGrader (penulis, protokol) tidak bisa diverifikasi dari teks penuh (403 / paywall).
- Leaderboard Papers with Code/HyperAI ACNE04 tidak bisa dimuat.

---

## Q3. SOTA per November 2026

### Takeaway
- **Counting MAE ACNE04 (v1)**: angka terendah yang ditemukan = **1.76 (RMSE 3.57)**, Alzahrani dkk. 2022 (split tunggal 80/20). Di bawah protokol resmi 5-fold, acuan tetap **LDL Wu 2019: 2.93 / 5.42**.
- **Akurasi grading ACNE04**: **~87.33%** (FF-PLL 2025; CG-HAF preprint 2026) di bawah protokol yang sebanding dengan 4 kelas Hayashi; klaim 91.5% (Alzahrani) dan 91.2% (AcneDGNet) memakai protokol/split berbeda.

### Cited Findings
- Alzahrani 2022 MAE 1.76 / MSE 3.57 / Acc 91.5 — [Computers 2022](https://doi.org/10.3390/computers11030031)
- Tabel kompilasi akurasi dengan peringatan protokol — [CG-HAF arXiv 2609.31326](https://arxiv.org/html/2609.31326)
- Benchmark counting tidak terstandar: split, pembersihan data, dan definisi metrik berbeda antar paper — [Wen 2022](https://doi.org/10.3233/THC-228014) (filter ACNE04), [Prokhorov 2024](https://arxiv.org/html/2403.00268v1) (gagal reproduksi)

### Inferences
- Tidak ada leaderboard counting resmi; "SOTA" counting bergantung pada protokol. Untuk klaim yang bisa dipertahankan di laporan, bandingkan dengan Wu 2019 Tabel 4 (5-fold) dan sebutkan Alzahrani sebagai angka terbaik yang dilaporkan dengan protokol berbeda.

### Gaps
- Tidak ditemukan paper 2023–2026 yang melaporkan counting MAE di bawah 1.76 pada ACNE04; ini bisa karena keterbatasan akses (paywall), bukan bukti ketiadaan.

---

## Q4. Apakah paper ACNE04-v2 melaporkan metrik counting/grading? Seberapa sebanding count v2 vs v1?

### Takeaway
Paper ACNE04-v2 (AcneAI, MICCAI 2024) **tidak melaporkan MAE/RMSE counting maupun akurasi Hayashi**; abstraknya hanya melaporkan **ICC 0.8** untuk klasifikasi severity pada Acne04 dan AUC 0.88 deteksi lesi inflamasi pada dataset klinis lain. Anotasi v2 jauh lebih padat (≈26.9 lesi/citra vs ≈13.0 di v1), jadi **MAE pada v2 tidak bisa dibandingkan langsung dengan MAE v1**.

### Cited Findings
- Gazeau, Nguyen, Nguyen, Lebedeva, Nguyen, To, Le Digabel, Filiol, Josse, Perlis, Wolfe, "AcneAI: A New Acne Severity Assessment Method Using Digital Images and Deep Learning", MICCAI 2024 (LNCS, hlm. 68–78). Alur: segmentasi semua lesi → skor tiap lesi → skor severity 0–100. Hasil: ICC 0.8 pada Acne04; AUC 0.88 deteksi inflamasi pada data uji klinis multi-senter — [MICCAI 2024 page](https://papers.miccai.org/miccai-2024/042-Paper2216.html)
- Reviewer mencatat split data dan penggunaan cross-validation tidak dinyatakan — [MICCAI reviews](https://papers.miccai.org/miccai-2024/042-Paper2216.html)
- ACNE04-v2: **1.204 citra, 32.443 anotasi**; subset ACNE04 rilis pertama, citra berkualitas rendah dihapus dan pakar menambahkan lesi; format mirip COCO, tiap lesi = **lingkaran (center + radius)**, bukan bounding box; tidak ada split resmi dan tidak ada angka benchmark di repo — [github.com/AIpourlapeau/acne04v2](https://github.com/AIpourlapeau/acne04v2)
- ACNE04 v1: 1.457 citra, 18.983 bounding box — [Wu 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)
- Satu-satunya paper lain yang ditemukan memakai Acne04v2: SAT-AcneDet (Ding & Wang, SPIE Vol. 14389, SHDH 2026), berbasis YOLO11n — Recall 0.518 → 0.601, mAP@50 0.573 → 0.608; tanpa metrik counting — [SPIE Proc. 14389](https://spie.org/Publications/Proceedings/Volume/14389)

### Inferences
- Kepadatan GT: v1 = 18.983/1.457 ≈ 13.0 lesi/citra; v2 = 32.443/1.204 ≈ 26.9 lesi/citra (≈2,1×). Karena error count cenderung naik dengan jumlah objek, MAE 7.78 pengguna pada v2 tidak setara "lebih buruk 2,7×" dari MAE 2.93 Wu pada v1.
- Pembanding yang lebih adil: (a) MAE relatif (MAE / rata-rata GT test), (b) mengevaluasi model pengguna dengan GT v1 pada citra yang sama, atau (c) menjalankan protokol 5-fold Wu.
- Grading Hayashi dari count v2 akan bergeser ke kelas lebih berat dibanding label severity v1 (threshold Hayashi sama, count lebih tinggi), sehingga label severity v1 tidak konsisten dengan count v2.
- Split pengguna (963/120/121 dari 1.204 citra v2) jelas berbeda dari 5-fold Wu (1.165/292 dari 1.457 citra v1); tidak ada split resmi v2.

### Gaps
- Teks penuh paper MICCAI dan supplementary tidak berhasil diambil (timeout), jadi tabel internal (mis. "AP" di Tabel 1 yang ditanyakan reviewer) tidak terverifikasi.
- Tidak ditemukan paper yang melaporkan MAE/RMSE/Pearson counting pada ACNE04-v2 (per pencarian Nov 2026).

---

## Q5. Caveat definisi ground-truth count v1 vs v2

### Takeaway
v1 count = jumlah bounding box (dan label severity langsung dari pakar, sesuai Hayashi); v2 count = jumlah lingkaran anotasi ulang yang lebih lengkap pada subset 1.204 citra. Keduanya mengukur "jumlah lesi" dengan kelengkapan anotasi yang berbeda.

### Cited Findings
- v1: bounding box lesi, 4 kelas severity Hayashi berdasarkan jumlah lesi; satu kelas lesi (tanpa tipe) — [Wu 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)
- v2: lingkaran (center, radius), lebih banyak lesi dianotasi, citra kualitas rendah dibuang — [acne04v2 repo](https://github.com/AIpourlapeau/acne04v2)
- Kualitas v1 dipermasalahkan: satu kelompok menyatakan box ACNE04 tidak berpusat pada lesi sehingga tidak memakainya untuk evaluasi counting — [Nature Sci. Reports 2025 (s41598-024-84670-z)](https://www.nature.com/articles/s41598-024-84670-z) (dari snippet pencarian, tidak dibaca penuh)
- Wen dkk. memfilter ACNE04 sebelum evaluasi; zip yang dirilis berisi 1.513 citra, bukan 1.457 — [Wen 2022](https://doi.org/10.3233/THC-228014) (snippet)

### Inferences
- Pelaporan yang disarankan: sebutkan "ACNE04-v2 (Gazeau 2024), split kustom 963/120/121, GT = jumlah lingkaran v2", sertakan rata-rata GT per citra di test, MAE, RMSE (bukan MSE), Pearson r, dan akurasi Hayashi yang diturunkan dari count; nyatakan eksplisit bahwa angka tidak sebanding langsung dengan Tabel 4 Wu 2019.

### Gaps
- Tidak ada studi yang mengukur kesepakatan count v1 vs v2 per citra; perbandingan harus dihitung sendiri dari dua file anotasi.
