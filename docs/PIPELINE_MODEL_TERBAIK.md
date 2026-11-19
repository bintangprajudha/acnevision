# Pipeline Model Terbaik: YOLOv8l Dilatih pada Tile 2×2 + SAHI

> Bahan slide. Satu bagian = satu slide. Angka: split test (121 gambar, 3.507 lesi), conf dipilih di val per run,
> **rata-rata ± simpangan baku dari 3 seed (42, 7, 123)**; rincian per seed di `docs/EXPERIMENTS.md`
> (`runs/eval/yolov8l_960_tiles2{,_s7,_s123}_sahi1728/metrics.json`).

---

## Slide 1. Ringkasan metode

- **Tugas**: mendeteksi setiap lesi jerawat pada foto wajah dan menghitung jumlahnya per gambar.
- **Masalah**: lesi sangat kecil (median 42 px pada gambar ±3112×3456, jadi ±12 px setelah diperkecil ke 960).
- **Ide**: latih pada potongan gambar (tile) resolusi penuh, lalu saat inferensi potong dan gabungkan lagi (SAHI).
- **Model**: YOLOv8l (43,6 juta parameter), pretrained COCO, input 960×960.

---

## Slide 2. Alur pipeline

```
 ACNE04 (gambar)            Anotasi ACNE04-v2           Konversi + split
 1.204 foto ±3112×3456  +   lingkaran (cx, cy, r)  ->  lingkaran -> kotak [cx±r, cy±r]
                                                        split acak seed 42: 963 / 120 / 121
                                                                 |
                    +--------------------------------------------+
                    v
        Tiling 2×2 (train & val saja)           Training YOLOv8l @ 960              Inferensi SAHI
        grid 2×2, overlap 10%, pada     ->      dari bobot COCO, 100 epoch,   ->    potong 1728 px, overlap 20%,
        resolusi asli; gambar utuh              AdamW, best.pt dipilih di val       prediksi per potongan,
        tetap dipakai: 4.815 gambar train                                           gabung dengan NMS (IoU 0.5)
                                                                                         |
                                                       Conf operasional (dipilih di val, MAE minimum) -> jumlah lesi
                                                                                         |
                                                       Grading keparahan (Hayashi) dari jumlah lesi
```

---

## Slide 3. Data

| Item | Nilai |
|---|---|
| Gambar | ACNE04 (Wu dkk., ICCV 2019), 1.204 gambar |
| Anotasi lesi | ACNE04-v2 (Gazeau dkk., MICCAI 2024), 32.443 lesi; 1 kelas (`acne`) |
| Konversi | Lingkaran → kotak persegi, dipotong ke batas gambar |
| Split | 963 train / 120 val / 121 test (acak, seed 42), tersedia di `splits/acne04v2_split.csv` |
| Kepadatan | ±27 lesi per gambar (±2× lebih padat dari label ACNE04 asli) |

Catatan: split acak per gambar, tidak terstratifikasi. Test hanya berisi 1 gambar level keparahan 3.

---

## Slide 4. Tiling untuk training

- Setiap gambar train dan val dipotong menjadi **grid 2×2 dengan overlap 10%** pada resolusi asli (`scripts/data/preprocess.py --grid 2 --overlap 0.1`).
- Kotak lesi dipotong ke tile dan dipertahankan bila **≥ 50% luasnya** berada di dalam tile.
- Gambar utuh tetap ikut: train = 963 gambar asli + tile = **4.815 gambar**; val tile = 600 gambar.
- Efeknya: lesi tampil ±2× lebih besar saat masuk jaringan 960×960.
- **Split test tidak di-tile**, dievaluasi pada gambar utuh dengan SAHI.

---

## Slide 5. Training

| Item | Nilai |
|---|---|
| Arsitektur | YOLOv8l, pretrained COCO (`yolov8l.pt`) |
| Input | 960×960, batch 4 (akumulasi gradien ke 64), AMP |
| Optimizer | AdamW, lr0 0.001, jadwal cosine, warmup 3 epoch |
| Epoch | 100, patience 25 (berhenti terbaik di epoch 84; ±645 menit) |
| Bobot loss | box 10.0, cls 0.3, dfl 2.0 (disesuaikan untuk objek kecil, 1 kelas) |
| Augmentasi | mosaic 1.0 (dimatikan 10 epoch terakhir), mixup 0.1, flip kiri-kanan dan atas-bawah 0.5, rotasi ±10°, skala 0.5, HSV |
| Seleksi model | `best.pt` dari fitness di split val tile |
| Reproducibility | seed 42, `deterministic: true`, semua parameter di `configs/yolov8l_960_tiles2.yaml` |
| Hardware | NVIDIA RTX 4060 8 GB |

---

## Slide 6. Inferensi SAHI

- Gambar utuh dipotong menjadi **potongan 1728 px dengan overlap 20%**, tiap potongan diperkecil ke 960 dan diprediksi.
- Hasil semua potongan dikembalikan ke koordinat gambar asli dan digabung dengan **NMS (IoU 0.5)** untuk menghapus duplikat di area overlap.
- **Conf operasional** dipilih di val dengan mencari nilai (0.05 sampai 0.60, langkah 0.01) yang memberi **MAE counting minimum**; untuk model ini **0.21**.
- Jumlah lesi = banyak kotak dengan skor ≥ conf. Maksimum 1000 deteksi per gambar.
- Waktu: ±622 ms per gambar (mode plain tanpa SAHI: ±70 ms).

---

## Slide 7. Protokol evaluasi

- Satu-satunya implementasi: `scripts/evaluate.py`. Conf untuk mAP 0.001, IoU NMS 0.7, `max_det` 1000.
- **Test dipakai sekali untuk laporan**; semua pemilihan (epoch, conf) memakai val.
- Metrik deteksi: Precision, Recall, F1, AP@50, mAP@50-95.
- Metrik counting: MAE, RMSE, bias, Pearson r, serta MAE dan RMSE relatif (÷ rata-rata lesi per gambar).
- Grading Hayashi dari jumlah lesi: Mild 1-5, Moderate 6-20, Severe 21-50, Very severe > 50 (accuracy, precision, sensitivity, specificity, Youden).
- Uji signifikansi: bootstrap berpasangan (CI 95%).

---

## Slide 8. Hasil model terbaik (test, SAHI 1728, rata-rata ± SD, 3 seed)

| Metrik | Rata-rata ± SD | Seed 42 (model terpilih) |
|---|---|---|
| Precision | 0.615 ± 0.027 | 0.611 |
| Recall | 0.550 ± 0.021 | 0.557 |
| F1 | **0.580 ± 0.003** | 0.583 |
| AP@50 (internal) | 0.560 ± 0.005 | 0.562 |
| MAE (lesi per gambar) | 8.02 ± 0.33 | 7.78 |
| RMSE | 15.38 ± 1.79 | 14.47 |
| Pearson r | 0.957 ± 0.004 | 0.961 |

Seed 42 saja: mAP@50-95 0.203, MAE relatif 26.8%, RMSE relatif 49.9%, akurasi Hayashi 69.42% (Youden 50.04).
Mode plain (tanpa SAHI, ±70 ms): F1 0.524 ± 0.004, MAE 7.87 ± 0.27.

---

## Slide 9. Perbandingan dengan baseline (split dan protokol sama)

| Model | Inferensi | F1 | MAE |
|---|---|---|---|
| **YOLOv8l dilatih pada tile 2×2** | **SAHI 1728** | **0.583** | 7.78 |
| YOLOv8l resep tuning, fine-tune tile 10 epoch | SAHI 1728 | 0.571 | 8.29 |
| YOLOv8l resep tuning @ 960 | SAHI 1728 | 0.562 | 8.33 |
| YOLOv8l bawaan | SAHI 1728 | 0.525 | 10.04 |
| YOLOv8l bawaan (tanpa tuning) | plain | 0.520 | 9.27 |
| YOLOv8l @ 1280 | plain | 0.510 | 9.26 |

- Dibanding YOLOv8l bawaan (plain): ΔF1 +0.063 [+0.048, +0.078], ΔRecall +0.079 [+0.056, +0.105] (bootstrap berpasangan 95%, `docs/EXPERIMENTS.md`; tile 2×2 + SAHI vs bawaan plain).
- Dibanding fine-tune tile 10 epoch: ΔRecall +0.020 [+0.002, +0.037] signifikan, tetapi ΔF1 +0.011 [−0.002, +0.024] **belum signifikan**.

---

## Slide 10. Acuan literatur (satu paper utama)

**Gazeau dkk., AcneAI, MICCAI 2024**: pembuat anotasi ACNE04-v2, Tabel 1 (YOLOv8 @ 1280 dilatih pada data privat).

| | Recall (TPR) | Precision |
|---|---|---|
| YOLOv8 @ 1280 AcneAI | 0.54 | 0.54 |
| AcneAI | 0.52 | 0.52 |
| **Model kami (SAHI, 3 seed)** | **0.550 ± 0.021** | **0.615 ± 0.027** |
| Model kami (plain, 3 seed) | 0.502 ± 0.017 | 0.550 ± 0.020 |

- **Sebanding** (precision di atas acuan di semua seed; recall di atas 0.54 pada 2 dari 3 seed, seed 7 = 0.526), **dengan catatan**: titik operasi berbeda (mereka memakai threshold |FP−FN|, kami MAE minimum di val), data latih dan data uji berbeda, jumlah gambar uji mereka tidak dilaporkan.
- Paper ini tidak melaporkan mAP, MAE, maupun grading; FPR dan AUC mereka tidak dapat direplikasi (definisi negatif tidak ditulis).

---

## Slide 11. Keterbatasan

- Hanya **3 seed**; SD recall 0.021 dan MAE 0.33 cukup lebar, sehingga selisih kecil antar model tidak boleh diklaim.
- Test hanya 121 gambar; selisih antar model teratas sebagian belum signifikan.
- Split acak per gambar: kemungkinan kebocoran antar-foto pasien yang sama belum terverifikasi.
- Pada wajah sangat padat (level 3) jumlah lesi masih kurang terhitung; hanya 1 gambar level 3 di test.
- Hanya keluarga YOLOv8; tidak ada pembanding non-YOLO.
- Grading Hayashi 69.4% di bawah LDL Wu 2019 (84.1%, dataset dan label berbeda), jadi tidak diklaim sejajar.
- SAHI ±9× lebih lambat dari plain; model plain dipakai bila latency < 100 ms dibutuhkan.

---

## File sumber

| Isi | Lokasi |
|---|---|
| Config training | `configs/yolov8l_960_tiles2.yaml` |
| Bobot | `runs/detect/yolov8l_960_tiles2/weights/best.pt` |
| Evaluasi SAHI | `runs/eval/yolov8l_960_tiles2_sahi1728/` |
| Peringkat semua model | `docs/report/ranking_models.csv` |
| Eksperimen lengkap | `docs/EXPERIMENTS.md` |
