# Panduan End-to-End: Pengembangan Model AcneVision

> Dokumen belajar untuk memahami **seluruh** alur model deteksi & penghitungan lesi jerawat,
> dari data mentah sampai metrik akhir. Semua nama file, fungsi, dan angka diambil dari kode
> dan hasil nyata di repositori ini (dibaca 2026-10-24).
>
> - **Fakta terverifikasi** = berasal dari kode/file proyek (sumber disebut).
> - **Dugaan/opini** = ditandai *(opini)*.
> - Sumber angka metrik: `runs/eval/yolov8l_960/metrics.json` (split **test**, `imgsz=960`), kecuali disebut lain.

## Daftar Isi
1. [Gambaran besar](#1-gambaran-besar)
2. [Dataset](#2-dataset)
3. [Preprocessing](#3-preprocessing)
4. [Model & transfer learning](#4-model--transfer-learning)
5. [Training](#5-training)
6. [Konsep metrik deteksi](#6-konsep-metrik-deteksi)
7. [Bagaimana proyek ini mengevaluasi](#7-bagaimana-proyek-ini-mengevaluasi)
8. [Metrik counting](#8-metrik-counting)
9. [Hasil nyata & cara membacanya](#9-hasil-nyata--cara-membacanya)
10. [Inferensi & visualisasi](#10-inferensi--visualisasi)
11. [Risiko, pelajaran, langkah berikutnya](#11-risiko-pelajaran-langkah-berikutnya)
12. [Glosarium](#12-glosarium)
13. [Referensi](#13-referensi)

---

## 1. Gambaran besar

**Masalah**: diberi foto wajah, temukan *setiap* lesi jerawat (kotak pembatas/bbox) dan hitung
jumlahnya. Ini **object detection satu kelas** (`acne`), dengan objek yang sangat kecil dan banyak
(rata-rata ±27 lesi/gambar, maksimum 434).

```
 DATA MENTAH            KONVERSI                  TRAINING               EVALUASI                 PREDIKSI
 ACNE04 (foto)    ->    convert_to_yolo.py   ->   train.py          ->    evaluate.py        ->    predict.py
 ACNE04-v2 (lingkaran   lingkaran -> bbox         + configs/*.yaml        deteksi (mAP, P, R)      gambar + kotak
 cx, cy, r)             clip + normalisasi        YOLOv8l (COCO)          counting (MAE, r)        mode plain/TTA/SAHI
                        split 80/10/10 seed 42    best.pt dari VAL        TP/FP/FN, confusion
```

| Tahap | File | Peran |
|---|---|---|
| Unduh data | `scripts/data/download_*.py` | Gambar (HuggingFace) + anotasi |
| Konversi | `scripts/data/convert_to_yolo.py` | Lingkaran → label YOLO, split |
| Tile (opsional) | `scripts/data/preprocess.py` | Potong gambar jadi tile untuk training |
| Utilitas | `scripts/common.py` | Path, `MAX_DET=1000`, level keparahan, `imgsz` dari checkpoint |
| Training | `scripts/train.py` + `configs/*.yaml` | Satu eksperimen = satu config |
| Matching & metrik | `scripts/analysis.py` | IoU, TP/FP/FN, AP50, confusion matrix |
| Evaluasi | `scripts/evaluate.py` | **Satu-satunya** protokol evaluasi resmi |
| Prediksi | `scripts/predict.py` | Inferensi & visualisasi |

Prinsip yang mengikat (AGENTS.md §2): test set hanya untuk laporan akhir; semua tuning di **val**.

---

## 2. Dataset

| Item | Nilai | Sumber |
|---|---|---|
| Gambar | ACNE04, 1.204 gambar, ±3456×3112 px | `docs/SPEC.md` |
| Anotasi | ACNE04-v2: 32.443 lesi, berbentuk **lingkaran** `(cx, cy, radius)` | `docs/SPEC.md` |
| Kelas | 1 (`acne`) | `data/data.yaml` dibuat `create_data_yaml()` |
| Level keparahan | Awalan nama file `levle0..3` (kriteria Hayashi) | `common.severity_from_name` |
| Split | 963 train / 120 val / 121 test (25.521 / 3.366 / 3.507 lesi) | `docs/PIPELINE.md` |

**Level keparahan** hanya dipakai untuk *analisis* (MAE per level), bukan sebagai label training.
Regex-nya `^levle(\d)_` (typo "levle" memang ada di dataset aslinya).

**Ukuran lesi** (train, sisi terpanjang bbox, `docs/PIPELINE.md`): median **42 px** pada gambar asli
(min 4, maks 631). Karena model menerima gambar yang diperkecil ke `imgsz`, lesi menyusut:

| imgsz | Median lesi | < 16 px | < 8 px |
|---|---|---|---|
| 640 | 7,8 px | 91,5% | 52,4% |
| **960** | **11,7 px** | **73,5%** | **21,0%** |
| 1280 | 15,6 px | 52,4% | 6,9% |

Inilah akar kesulitan proyek: pada 960 px mayoritas lesi hanya ±12 px (objek *tiny*), sehingga
mudah terlewat (recall rendah) dan sulit dilokalisasi presisi (mAP@50-95 rendah).

---

## 3. Preprocessing

### 3.1 Lingkaran → bounding box (`circle_to_yolo_bbox`)

Anotasi sumber berupa lingkaran. Detektor YOLO butuh kotak, jadi lingkaran diganti **kotak persegi
yang membungkusnya**: sisi = 2 × radius.

```
xmin = cx - r        xmax = cx + r
ymin = cy - r        ymax = cy + r
```

### 3.2 Clipping bbox ke batas gambar (inti pertanyaan Anda)

**Masalah**: lesi di tepi foto bisa punya lingkaran yang sebagian *keluar* gambar. Kotak yang
`xmin < 0` atau `xmax > lebar` tidak valid untuk training (koordinat negatif / di luar gambar,
loss dan augmentasi jadi tidak konsisten). Solusinya **clipping**: potong kotak ke area gambar.

Kode (`convert_to_yolo.py:39-42`):

```python
xmin = max(0, cx - radius)
ymin = max(0, cy - radius)
xmax = min(img_width,  cx + radius)
ymax = min(img_height, cy + radius)
```

**Contoh hitungan** (gambar 3456×3112, lesi di tepi kiri: `cx=30, cy=100, r=50`):

| Langkah | Tanpa clipping | Dengan clipping |
|---|---|---|
| xmin | 30 − 50 = **−20** (di luar gambar) | max(0, −20) = **0** |
| xmax | 80 | 80 |
| lebar kotak | 100 | 80 |
| pusat x | 30 | (0 + 80/2) = **40** |
| pusat y, tinggi | 100, 100 | 100, 100 |

Perhatikan: **pusat bergeser** (30 → 40) dan lebar menyusut (100 → 80). Itu disengaja: kotak
sekarang hanya mewakili bagian lesi yang **benar-benar tampak** di foto, dan pusat dihitung ulang
dari kotak hasil clip (`xmin + box_w/2`), bukan memakai `cx` asli. Bila tidak dihitung ulang, pusat
akan berada di luar kotak yang sudah dipotong.

*Catatan desain (opini)*: dengan lingkaran → persegi, sisi kotak bisa melampaui tepi pada 4 sisi,
jadi clipping dipakai untuk semua sisi; efeknya kecil karena lesi di tepi jarang, tetapi mencegah
label ilegal.

### 3.3 Normalisasi dan format label YOLO

YOLO memakai koordinat **relatif** terhadap ukuran gambar (0–1), agar label tidak bergantung
pada resolusi (model akan me-resize gambar):

```
x_center_norm = (xmin + box_w/2) / img_width
y_center_norm = (ymin + box_h/2) / img_height
w_norm        = box_w / img_width
h_norm        = box_h / img_height
```

Lalu diberi *clamp* ke [0, 1] sebagai pengaman numerik. Kotak berluas nol (`w<=0` atau `h<=0`)
dibuang. Satu baris label:

```
0 0.011574 0.032134 0.023148 0.032134
^ id kelas (selalu 0)  ^cx   ^cy   ^w   ^h        (6 desimal)
```

Satu file `.txt` per gambar; gambar tanpa lesi tetap dibuatkan file kosong (agar menjadi contoh
negatif yang sah).

### 3.4 Pembagian data (split)

`split_dataset()`: seed acak 42 (`RANDOM_SEED`), mengocok ID **gambar**, lalu 80% train / 10% val /
10% test. Fungsi tiap bagian:

| Split | Dipakai untuk | Aturan |
|---|---|---|
| train | Mengubah bobot (gradient) | – |
| val | Memantau training, memilih `best.pt`, memilih threshold conf | Boleh dipakai untuk tuning |
| test | **Hanya** laporan akhir | Dilarang untuk tuning |

Alasan tiga bagian: kalau model/threshold dipilih berdasarkan test, angka test menjadi terlalu
optimistis (*data leakage* lewat seleksi). Val mengambil peran "ujian latihan".

**Keterbatasan (fakta, `PIPELINE.md` audit #13–14)**: split acak per gambar, **tidak
terstratifikasi** per level (test hanya punya 1 gambar level 3), dan bila satu pasien punya beberapa
foto, foto serupa bisa tersebar di train & test (kebocoran antar-pasien; belum terverifikasi).

### 3.5 Tiling (opsional, `scripts/data/preprocess.py`)

Tujuan: membuat lesi tampak lebih besar tanpa menaikkan `imgsz` (VRAM 8 GB terbatas). Gambar
train/val dipotong menjadi grid `N×N` tile pada **resolusi asli**, ditambah gambar utuh.

Ukuran tile dihitung `tw = ceil(W / (grid − (grid−1)·overlap))`. Contoh grid 2, overlap 0,1 pada
3456×3112: `tw = ceil(3456/1,9) = 1819`, `th = ceil(3112/1,9) = 1638`. Tile kemudian di-resize ke
`imgsz`, jadi skala 960/1819 ≈ 0,53 dibanding 960/3456 ≈ 0,28, yaitu lesi tampil **±1,9× lebih besar**
(median 13 → ≈25 px; docstring menyebut ≈26).

Label per tile: bbox **di-clip ke tile**, tetapi hanya dipertahankan bila **≥ 50%** luasnya masih di
dalam tile (`MIN_VISIBLE = 0.5`). Alasan: lesi yang hanya terlihat sebagian kecil (mis. 10%) akan
berupa kotak aneh yang membingungkan model; lebih baik dibuang (dianggap latar di tile itu, dan
tertangkap utuh di tile tetangga berkat overlap). Koordinat dinormalisasi ulang terhadap tile.
Test **tidak** di-tile; evaluasi memakai gambar utuh (+ SAHI, bagian 10).

---

## 4. Model & transfer learning

**YOLOv8** (Ultralytics) adalah detektor satu tahap, *anchor-free*:

```
Gambar -> BACKBONE (CSPDarknet: ekstraksi fitur multi-skala)
       -> NECK (PAN/FPN: gabung fitur dangkal & dalam)
       -> HEAD (terpisah: kotak, kelas) pada stride 8 / 16 / 32
```

- **Stride** = berapa piksel gambar diwakili satu sel fitur. Stride 8 (terhalus) menangani objek
  kecil. Lesi 12 px ≈ 1,5 sel pada stride 8, sangat dekat batas kemampuan, itu sebabnya head
  stride-4 (P2) ada di roadmap.
- **YOLOv8l** = 43,6 juta parameter (varian besar; `yolov8s` jauh lebih kecil).
- **Transfer learning**: mulai dari bobot yang sudah dilatih di COCO (`pretrained/yolov8l.pt`).
  Model sudah paham tepi/tekstur/bentuk umum, jadi data 963 gambar cukup untuk beradaptasi. Hanya
  head kelas yang disesuaikan ke 1 kelas.
- **Letterbox**: gambar di-resize dengan menjaga rasio, lalu diberi padding menjadi kotak
  `imgsz×imgsz` (960×960).

---

## 5. Training

Semua hyperparameter ada di `configs/yolov8l_960.yaml` dan diteruskan apa adanya ke `YOLO.train()`
oleh `scripts/train.py` (yang juga resume otomatis bila `last.pt` ada). Prinsip: **satu eksperimen =
satu config**, agar hasil bisa direproduksi dan dibandingkan.

### 5.1 Jadwal & optimizer

| Parameter | Nilai | Arti |
|---|---|---|
| `epochs` | 100 | 1 epoch = seluruh data train dilihat sekali |
| `patience` | 25 | Berhenti dini bila tak membaik 25 epoch |
| `imgsz` | 960 | Resolusi input; batas aman VRAM 8 GB |
| `batch` | 4 | Gambar per langkah; `nbs=64` ⇒ gradient diakumulasi 16 langkah sebelum update (batch efektif 64) |
| `optimizer` | AdamW | Optimizer adaptif + weight decay terpisah |
| `lr0` / `lrf` | 0,001 / 0,01 | LR awal; LR akhir = lr0×lrf = 1e-5 |
| `cos_lr` | true | LR turun mengikuti kurva kosinus (halus) |
| `warmup_epochs` | 3 | LR naik bertahap di awal agar stabil |
| `weight_decay` | 0,0005 | Regularisasi L2 pencegah overfitting |
| `seed`, `deterministic` | 42, true | Reproducibility |

*Catatan*: jika `optimizer: auto`, Ultralytics mengabaikan `lr0`; karena itu AdamW ditulis eksplisit.

### 5.2 Fungsi loss (bobot di config)

Loss total = jumlah berbobot tiga komponen:

| Komponen | Bobot | Mengukur | Intuisi |
|---|---|---|---|
| `box` (CIoU) | 10,0 | Seberapa tepat kotak prediksi (tumpang tindih, jarak pusat, rasio) | Dinaikkan dari default 7,5 karena presisi lokalisasi lesi kecil kritis |
| `cls` (BCE) | 0,3 | Benar/salah "ini jerawat" | Diturunkan dari 0,5: hanya 1 kelas, klasifikasi mudah |
| `dfl` | 2,0 | Distribusi tepi kotak (lebih halus dari satu angka) | Dinaikkan dari 1,5 untuk presisi tepi |

Bobot ini hasil keputusan eksperimen E3, bersama perubahan lain. **Kontribusi masing-masing belum
terpisah** (`EXPERIMENTS.md`), jadi jangan menyimpulkan bobot mana yang paling berjasa.

### 5.3 Augmentasi (hanya saat training)

Menambah variasi data "gratis" agar model tidak menghafal 963 gambar.

| Parameter | Nilai | Efek |
|---|---|---|
| `mosaic` | 1,0 | Menempel 4 gambar jadi satu; banyak skala/konteks sekaligus |
| `close_mosaic` | 10 | 10 epoch terakhir mosaic dimatikan agar model menyesuaikan distribusi gambar asli |
| `mixup` | 0,1 | Mencampur dua gambar (peluang 10%) |
| `fliplr` / `flipud` | 0,5 / 0,5 | Pantul horizontal / vertikal (flipud belum diuji, audit #16) |
| `degrees` | 10 | Rotasi acak ±10° |
| `translate`, `scale` | 0,1; 0,5 | Geser ±10%, zoom 0,5–1,5× |
| `hsv_h/s/v` | 0,015; 0,7; 0,4 | Variasi warna, kejenuhan, kecerahan (beda pencahayaan/kulit) |
| `copy_paste` | 0,0 | **No-op** untuk label bbox (hanya jalan pada label segmentasi); run asli 0,3 tidak berpengaruh |

### 5.4 Pemilihan model akhir

Tiap epoch, Ultralytics mengevaluasi di **val** dan menghitung *fitness* (gabungan mAP, bobot besar
pada mAP@50-95). Checkpoint terbaik disimpan sebagai `best.pt`. Pada E3 best di epoch **98/100**:
kurva masih naik ⇒ model belum konvergen penuh (fakta, `PIPELINE.md` audit #12).

---

## 6. Konsep metrik deteksi

Bagian ini menjelaskan metrik dari nol.

### 6.1 IoU (Intersection over Union)

Ukuran kecocokan dua kotak:

```
IoU = luas irisan / luas gabungan      (0 = tak bersentuhan, 1 = identik)
```

Contoh: GT 10×10 px; prediksi sama ukuran tetapi bergeser 3 px.
Irisan = 7×10 = 70; gabungan = 100+100−70 = 130; **IoU = 0,54**. Bergeser 5 px ⇒ irisan 50, gabungan
150, IoU = 0,33.

**Mengapa penting untuk lesi kecil**: lesi 12 px yang meleset hanya 3 px sudah menurunkan IoU banyak
(irisan 10×13=130, gabungan 208, IoU = 0,63). Pada objek besar, 3 px tidak berarti apa-apa. Ditambah
anotasi aslinya lingkaran (batas lesi samar), IoU tinggi sulit dicapai, penyebab mAP@50-95 rendah
(bagian 9).

### 6.2 TP, FP, FN

Sebuah prediksi dicocokkan dengan GT. Dengan ambang IoU (proyek ini 0,5) dan satu GT hanya boleh
dipakai satu prediksi:

| Istilah | Arti | Contoh |
|---|---|---|
| **TP** (True Positive) | Prediksi cocok dengan GT (IoU ≥ 0,5) | Lesi nyata terdeteksi |
| **FP** (False Positive) | Prediksi tanpa GT yang cocok | Pori/bekas luka dikira jerawat; atau lesi nyata yang tidak dianotasi; atau kotak meleset |
| **FN** (False Negative) | GT yang tidak terdeteksi | Lesi terlewat |
| TN | Area benar-benar bukan lesi | **Tidak terdefinisi** pada deteksi objek (tak ada "jumlah area negatif" yang masuk akal), maka confusion matrix kita menandai sel ini N/A |

### 6.3 Confidence, threshold, NMS

- Tiap kotak prediksi punya skor **confidence** (0–1).
- **Threshold `conf`**: kotak dengan skor di bawah ini dibuang. Rendah ⇒ banyak kotak (recall naik,
  precision turun). Tinggi ⇒ sedikit kotak (precision naik, recall turun).
- **NMS** (Non-Maximum Suppression, `iou=0.7`): bila dua kotak saling tumpang tindih > 0,7, hanya
  yang skornya lebih tinggi dipertahankan, mencegah satu lesi terhitung dua kali.
- `max_det=1000`: batas jumlah kotak per gambar. Default Ultralytics 300 **memotong** gambar test
  yang punya 434 lesi, jadi diubah ke 1000 (`common.MAX_DET`).

### 6.4 Precision, Recall, F1

```
Precision = TP / (TP + FP)   "dari yang diprediksi jerawat, berapa yang benar?"
Recall    = TP / (TP + FN)   "dari semua jerawat nyata, berapa yang ketemu?"
F1        = 2·P·R / (P + R)  rata-rata harmonik (hukum keseimbangan)
```

Contoh dari hasil nyata (conf 0,21, `metrics.json › test_conf_val_tuned.detection`):
TP=1712, FP=1432, FN=1795 ⇒ Precision = 1712/3144 = **0,544**; Recall = 1712/3507 = **0,488**;
F1 = **0,515**. Artinya: model menemukan ±49% lesi, dan ±54% kotaknya benar.

Precision & Recall berlawanan. Satu pasangan nilai hanya berlaku di satu threshold; karena itu ada
AP (6.5).

### 6.5 AP dan mAP

1. Urutkan semua prediksi dari confidence tertinggi.
2. Telusuri satu per satu, hitung kumulatif TP/FP ⇒ pasangan (Recall, Precision) di tiap titik ⇒
   **kurva Precision–Recall**.
3. **AP** = luas area di bawah kurva itu (di sini: rata-rata precision pada 101 titik recall, dengan
   precision dibuat monoton turun; `analysis.ap50`).
4. **mAP** = rata-rata AP lintas kelas. Karena 1 kelas, mAP = AP.

AP tidak bergantung pada satu threshold, itu sebabnya evaluasi memakai `conf=0.001` (sangat rendah)
agar kurva tercatat sampai recall sejauh mungkin.

| Metrik | Definisi | Cara membaca |
|---|---|---|
| **mAP@50** | AP dengan syarat TP: IoU ≥ 0,5 | Ukuran "ketemu atau tidak" (longgar terhadap posisi) |
| **mAP@50-95** | Rata-rata AP pada 10 ambang IoU: 0,50; 0,55; … 0,95 | Ukuran **ketepatan kotak**; sangat keras pada objek kecil |

mAP@50-95 selalu ≤ mAP@50. Selisih besar (49,8% vs 19,1%) menandakan model *menemukan* lesi
cukup baik tetapi kotaknya tidak pas hingga IoU tinggi, wajar karena anotasi lingkaran tidak
punya batas tegas. Paper ACNE04-v2 sendiri mengevaluasi **counting/AUC**, bukan mAP (audit #15).
Karena itu mAP@50 + metrik counting lebih representatif untuk tujuan proyek *(opini berbasis audit)*.

### 6.6 Mengapa P/R Ultralytics ≠ P/R kita

Ultralytics melaporkan P dan R pada confidence yang memaksimalkan F1 (bukan 0,25 atau 0,21). Itu
sebabnya Precision 0,5631 / Recall 0,4810 (Ultralytics) berbeda dari 0,5445 / 0,4882 (matcher
proyek pada conf 0,21). Bukan inkonsistensi, tetapi **titik operasi** berbeda dan implementasi
matching sedikit berbeda. Selalu sebut sumbernya saat melaporkan angka.

---

## 7. Bagaimana proyek ini mengevaluasi

Seluruhnya di `scripts/evaluate.py` (+ `analysis.py`). Angka dari protokol lain **tidak boleh
dibandingkan** (AGENTS.md §2.3).

### 7.1 Protokol standar

| Hal | Nilai | Alasan |
|---|---|---|
| Split | test (laporan), val (tuning) | Cegah kebocoran seleksi |
| `conf` | 0,001 | Kurva PR lengkap; conf tinggi menurunkan mAP |
| `iou` NMS | 0,7 | Default validasi Ultralytics |
| `max_det` | 1000 | Hindari pemotongan gambar padat |
| `imgsz` | = training (dibaca dari checkpoint via `train_imgsz`) | Script lama hard-code 640 untuk model 960, salah |

### 7.2 Alur `evaluate.py`

1. `detection_metrics()` menjalankan `model.val(...)` ⇒ **P, R, mAP@50, mAP@50-95** resmi (Ultralytics).
2. `collect()` memprediksi semua gambar val & test sekali dengan conf sangat rendah, lalu
   `match()` memasangkan prediksi↔GT **sekali** saja.
3. **Greedy matching** (`analysis.match`): prediksi diurutkan dari skor tertinggi; masing-masing diambil
   pasangan GT belum terpakai dengan IoU terbesar; jika IoU ≥ 0,5 ⇒ TP, selain itu FP. GT tak
   terpasang ⇒ FN.
4. Semua threshold lain diturunkan dari skor tersimpan (`counts_at`, `tp_fp_fn_at`), tanpa prediksi ulang.
5. **Memilih threshold operasional di val**: coba `conf` 0,05…0,60 (langkah 0,01), pilih yang
   **MAE counting terkecil** di val ⇒ hasil **0,21**. Lalu dilaporkan di test.
6. Keluaran: `metrics.json`, confusion matrix, `counts_test.csv` per gambar.

**Mengapa threshold dipilih di val, bukan test?** Jika dipilih di test, MAE test menjadi "disetel"
terhadap data yang seharusnya hanya menguji, hasilnya bias optimistis.

**Mengapa conf 0,21 bukan 0,25?** 0,25 adalah default tanpa dasar. Efeknya nyata (test): MAE 9,85 → 8,69
dan bias −8,03 → −3,00 (kurang menghitung berkurang).

### 7.3 Confusion matrix

Matriks 2×2 (`plot_confusion`): TP, FP, FN, dan sel background–background bertanda **N/A**.
Berbeda dengan klasifikasi biasa, "kulit yang benar tidak diberi kotak" tak punya satuan hitung.

*Catatan*: kode menyediakan pecahan FP menjadi **lokalisasi** (IoU 0,1–0,5 dengan suatu GT: kotak
ada tapi meleset) vs **latar** (tanpa overlap: salah deteksi atau lesi tak dianotasi). `metrics.json`
E3 dibuat oleh versi sebelumnya sehingga **belum memuat** `fp_breakdown`; jalankan ulang
`evaluate.py` untuk mendapatkannya.

---

## 8. Metrik counting

Tujuan klinis akhirnya adalah **jumlah lesi**, jadi ini metrik utama. Untuk tiap gambar *i*:
`gt_i` = jumlah lesi GT, `pred_i` = jumlah kotak dengan skor ≥ conf; `err_i = pred_i − gt_i`.

| Metrik | Rumus | Cara membaca |
|---|---|---|
| **MAE** | rata-rata \|err\| | "Rata-rata meleset berapa lesi per gambar" (satuan lesi, mudah dipahami) |
| **RMSE** | √(rata-rata err²) | Seperti MAE tetapi **menghukum error besar** lebih berat; RMSE ≫ MAE ⇒ ada outlier |
| **Bias** | rata-rata err (bertanda) | Negatif = cenderung **under-count**, positif = over-count |
| **Pearson r** | korelasi `gt` vs `pred` | Seberapa baik urutan/proporsi tren terjaga (1 = sempurna), walau skala salah |

**Contoh kecil** (3 gambar): GT = [10, 20, 100], Pred = [8, 25, 60]; err = [−2, +5, −40].
MAE = (2+5+40)/3 = **15,7**; RMSE = √((4+25+1600)/3) = **23,3**; bias = −37/3 = **−12,3**.
RMSE jauh di atas MAE karena satu gambar padat (−40).

Hasil E3 (test, 121 gambar, conf 0,21 dipilih di val, `metrics.json › test_conf_val_tuned`):
MAE **8,69**, RMSE **20,18**, bias **−3,00**, r **0,910**, total pred 3.144 vs GT 3.507.
RMSE (20) jauh di atas MAE (8,7) ⇒ ada beberapa gambar dengan error besar, tepatnya gambar level 3.

**Mengapa MAE counting bisa baik meski recall hanya ±49%?** Karena FP dan FN saling meniadakan
saat dijumlahkan (kotak salah menutupi lesi terlewat). Jangan menganggap counting "akurat" berarti
setiap kotak benar: lihat juga P/R. *(Dugaan/penafsiran, konsisten dengan TP=1712 dari GT=3507 namun
total pred 3144.)*

**Per level keparahan** (conf 0,21, test):

| Level | n gambar | MAE | Bias | GT | Pred | Catatan |
|---|---|---|---|---|---|---|
| 0 | 51 | 5,27 | −2,37 | 865 | 744 | – |
| 1 | 59 | 8,46 | −2,29 | 1632 | 1497 | – |
| 2 | 10 | 9,50 | +8,10 | 576 | 657 | Over-count (n kecil) |
| 3 | 1 | 188,0 | −188,0 | 434 | 246 | **n=1: tidak bermakna statistik** |

Kelemahan utama (fakta): wajah sangat padat ⇒ under-count besar (recall level 3 hanya 0,32 di conf 0,21).
Karena hanya 1 gambar, jangan menggeneralisasi; itu perlu split terstratifikasi (roadmap #7).

---

## 9. Hasil nyata & cara membacanya

### 9.1 Evolusi eksperimen (`docs/EXPERIMENTS.md`)

| ID | Run | imgsz | P | R | mAP@50 | mAP@50-95 | Catatan |
|---|---|---|---|---|---|---|---|
| E1 | yolov8s_640 | 640 | 47,57% | 38,13% | 35,82% | 13,01% | Baseline, early stop; protokol lama |
| E2 | yolov8l_640 | 640 | 56,24% | 43,40% | 43,73% | 15,81% | Model lebih besar; protokol val default |
| **E3** | **yolov8l_960** | **960** | **56,31%** | **48,10%** | **49,80%** | **19,14%** | **Terpilih**; protokol standar |

Pelajaran: dari E2→E3 mAP@50 naik ±6,1 poin dan recall ±4,7 poin; resolusi (lesi lebih besar)
adalah kandidat penyebab utama, tetapi E3 mengubah *banyak variabel sekaligus* (imgsz, batch, optimizer,
bobot loss, augmentasi), sehingga **kontribusi resolusi sendiri belum terbukti** (fakta dari
`EXPERIMENTS.md`; penafsiran "resolusi dominan" adalah *opini*).

### 9.2 Posisi terhadap target (`docs/SPEC.md`, test, conf 0,001 untuk deteksi)

| Metrik | E3 | Minimum layak | Baik | Status |
|---|---|---|---|---|
| mAP@50 | 49,80% | ≥ 45% | ≥ 55% | Layak, belum baik |
| Recall (Ultralytics) | 48,10% | ≥ 50% | ≥ 60% | **Belum** minimum |
| Precision (Ultralytics) | 56,31% | ≥ 50% | ≥ 60% | Layak |
| MAE counting | 8,69 | ≤ 10 | ≤ 7 | Layak |
| Pearson r | 0,910 | ≥ 0,85 | ≥ 0,90 | **Baik** |
| mAP@50-95 | 19,14% | – | target roadmap >25% | Belum |

### 9.3 Perbaikan audit yang mengubah angka (`PIPELINE.md`)

| Masalah lama | Dampak | Perbaikan |
|---|---|---|
| `copy_paste=0.3` ternyata no-op | Klaim augmentasi menyesatkan | Config = 0,0 |
| Eval `conf=0.25, iou=0.6` | Menekan mAP, tidak standar | `conf=0.001, iou=0.7` |
| `max_det` 300 | Hitungan terpotong (gambar 434 lesi) | 1000 |
| `imgsz` eval 640 untuk model 960 | Evaluasi tidak konsisten | Dibaca dari checkpoint |
| (plus) Threshold counting arbitrer | – | Dipilih di val: 0,21 |

### 9.4 Peringatan data

- `metrics.json` mencatat `inference_ms: 120.25` (waktu inferensi Ultralytics per gambar pada run
  `model.val`), sedangkan `docs/PIPELINE.md` menulis 45,4 ms/img. Kedua angka belum direkonsiliasi;
  penyebabnya (batch, warm-up, hardware saat ukur) belum diverifikasi. Perlu diukur ulang bila latensi penting.
- Komentar di `configs/yolov8l_960.yaml` baris 4 menyebut "Test mAP@50 = 49.24%", sedangkan
  `metrics.json` = 49,80% (protokol standar). Abaikan angka di komentar itu.
- `ap50_test_internal` = 0,4909 (AP50 matcher proyek) vs mAP@50 Ultralytics 0,4980: dua implementasi
  berbeda sedikit; konsisten bila dibandingkan di dalam pendekatan yang sama saja.

---

## 10. Inferensi & visualisasi

`scripts/predict.py`:

- **`conf`** operasional default 0,25; rekomendasi pemakaian counting: **0,21** (dipilih di val).
- Mode: `plain`, `--tta` (test-time augmentation: prediksi ulang dari beberapa versi gambar lalu
  digabung), `--sahi` (**S**licing **A**ided **H**yper **I**nference).
- **SAHI**: gambar dipotong menjadi patch (`--slice 960`, overlap 20%), tiap patch diprediksi, hasil
  digabung dan diberi NMS (IoU 0,5) menghapus duplikat di area overlap. Lesi tampak lebih besar
  per patch ⇒ potensi recall naik, dengan biaya waktu inferensi. Efeknya pada model E3 **belum
  diukur** (roadmap #1).
- `--compare`: warna **biru = TP**, **kuning = FP**, **merah = FN (lesi terlewat)**; teks hitungan
  ditaruh di strip header di luar gambar agar tidak menutupi lesi; keluaran `predictions.csv`.

---

## 11. Risiko, pelajaran, langkah berikutnya

| Risiko / keterbatasan | Dampak | Status |
|---|---|---|
| 73,5% lesi < 16 px pada 960 | Recall tertahan | Roadmap #1 (SAHI), #3 (1280), #4 (tile), #5 (P2) |
| Best epoch 98/100 | Belum konvergen | Roadmap #2 (200–300 epoch) |
| Split tak terstratifikasi, 1 gambar level 3 | Metrik per-level tak andal | Roadmap #7 |
| Kemungkinan kebocoran antar-pasien | Metrik mungkin terlalu optimistis | Belum terverifikasi (tidak ada ID pasien) |
| Anotasi lingkaran → kotak | mAP@50-95 tertekan | Gunakan counting + mAP@50 sebagai utama |
| `flipud=0.5` belum diuji | Efek tak diketahui | Roadmap #6 (ablation) |
| Validasi klinis belum ada | Tak siap klinis | SPEC: butuh perbandingan ke beberapa dermatolog (ICC ≥ 0,75) |

**Rekomendasi (opini)**: urutan terefisien tetap SAHI di test (biaya rendah, tanpa training) ⇒ training lebih
lama ⇒ baru resolusi/tile. Terapkan prinsip satu variabel per eksperimen agar sebab-akibat jelas.

---

## 12. Glosarium

| Istilah | Arti singkat |
|---|---|
| bbox | Kotak pembatas objek |
| clipping | Memotong koordinat agar berada dalam batas valid (gambar/tile) |
| letterbox | Resize menjaga rasio + padding menjadi persegi |
| stride | Piksel gambar per sel fitur (8/16/32) |
| anchor-free | Prediksi kotak langsung tanpa kotak acuan (anchor) |
| transfer learning | Memulai dari bobot hasil training dataset lain (COCO) |
| epoch | Satu putaran penuh atas data train |
| gradient accumulation | Menjumlah gradien beberapa batch kecil sebelum update bobot |
| fitness | Skor gabungan mAP untuk memilih `best.pt` |
| IoU | Irisan / gabungan dua kotak |
| NMS | Menghapus kotak duplikat yang tumpang tindih |
| TP / FP / FN | Deteksi benar / salah / lesi terlewat |
| AP / mAP | Luas area kurva Precision–Recall / rata-rata lintas kelas |
| MAE / RMSE | Rata-rata error absolut / akar rata-rata error kuadrat |
| bias | Rata-rata error bertanda (negatif = kurang hitung) |
| SAHI | Inferensi per-patch lalu digabung, untuk objek kecil |
| TTA | Test-time augmentation |
| leakage | Informasi test/val "bocor" ke proses training/seleksi |

---

## 13. Referensi

- Ultralytics, *Configuration*: https://docs.ultralytics.com/usage/cfg/
- Ultralytics, *Model Validation*: https://docs.ultralytics.com/modes/val/
- Ultralytics, *Model Training Tips*: https://docs.ultralytics.com/guides/model-training-tips/
- ACNE04-v2: https://github.com/AIpourlapeau/acne04v2
- Gazeau et al., *AcneAI*, MICCAI 2024: https://link.springer.com/chapter/10.1007/978-3-031-72086-4_7
- Wu et al., ACNE04, ICCV 2019: https://github.com/xpwu95/ldl
- Akyon et al., *SAHI* (2022), dirujuk di `scripts/data/preprocess.py`

Dokumen terkait di repositori: `docs/PIPELINE.md` (audit), `docs/SPEC.md` (protokol & target),
`docs/EXPERIMENTS.md` (log run), `docs/ROADMAP.md` (eksperimen berikutnya).
