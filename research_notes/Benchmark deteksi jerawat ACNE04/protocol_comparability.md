# Protokol evaluasi, split, dan keterbandingan benchmark deteksi jerawat ACNE04 / ACNE04-v2

Konteks pengguna: YOLOv8l @ 960px, satu kelas, anotasi lingkaran ACNE04-v2 yang dikonversi ke box, split 963/120/121 (train/val/test), protokol val Ultralytics (conf=0.001, iou=0.7, max_det=1000), ditambah inferensi tiled SAHI. Lesinya sangat kecil (median ~11.7 px pada 960px; 73.5% < 16 px).
Riset dilakukan pada 2026-11-13. Batas: sekitar 22 pemanggilan alat. PDF utama MICCAI 2024 (AcneAI) akhirnya terunduh lengkap, dan Tabel 1 sudah diverifikasi (lihat bagian 2).

## 1. ACNE04 asli (Wu et al., ICCV 2019): ukuran, split, format, evaluasi, hosting

### Takeaway
ACNE04 berisi 1.457 gambar dengan 18.983 box lesi, satu kategori. Data didistribusikan lewat repo GitHub xpwu95/ldl (Baidu/Google Drive), "free for academic usage". Paper aslinya berfokus pada grading dan counting (bukan mAP deteksi), dengan 5-fold cross-validation yang sudah ditentukan (setiap fold 80/20, yaitu 1.165/292). Paper deteksi lanjutan memakai split acak 80/20 masing-masing, dengan crop 640×640, sehingga split deteksi "resmi" yang seragam tidak ada.

### Cited Findings
- Paper: Xiaoping Wu, Ni Wen, Jie Liang, Yu-Kun Lai, Dongyu She, Ming-Ming Cheng, Jufeng Yang, "Joint Acne Image Grading and Counting via Label Distribution Learning", ICCV 2019, hlm. 10642–10651. — [CVF Open Access](https://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.html); [mlanthology](https://mlanthology.org/iccv/2019/wu2019iccv-joint/)
- Hosting resmi: repo GitHub `xpwu95/ldl` (implementasi PyTorch) dengan tautan dataset Baidu (`https://pan.baidu.com/s/15JQlymnhnEmEt8Q5zpJQDw`, password `fbrm`) dan Google Drive. README: "This work is free for academic usage. For other purposes, please contact Xiaoping Wu". Tidak ada nama lisensi formal. — [GitHub xpwu95/ldl](https://github.com/xpwu95/ldl)
- 1.457 gambar, 18.983 bounding box lesi. Split 80% train / 20% test = 1.165 dan 292 gambar. Implementasi memakai "pre-defined 5-fold cross validation". Empat level keparahan berdasarkan jumlah lesi: 1–5 ringan, 6–20 sedang, 21–50 berat, >50 sangat berat. — [Prokhorov & Kalinin 2024, arXiv 2403.00268](https://arxiv.org/html/2403.00268v1)
- Jumlah lesi tidak konsisten antarsumber: suplemen MICCAI 2024 menulis 18.963 lesi untuk 1.457 gambar ACNE04, berbeda 20 dari 18.983. — [ringkasan hasil pencarian atas suplemen MICCAI 2024 #2216](https://papers.miccai.org/miccai-2024/supp/2216_supp.pdf) (teks PDF tidak bisa saya ekstrak langsung; dilaporkan oleh alat pencarian)
- Satu kategori objek, dianotasi dokter kulit profesional. Ukuran gambar sekitar 300×300 sampai 4.000×4.000 px. — [ACNet, arXiv 2105.14891](https://ar5iv.labs.arxiv.org/html/2105.14891)
- Hasil grading Wu et al.: akurasi 84,11%, Youden Index 75,32%. — [ringkasan sekunder via hasil pencarian (hyper.ai / liner)](https://hyper.ai/en/papers/joint-acne-image-grading-and-counting-via) (belum diverifikasi ke PDF asli karena CVF/Cardiff memberi HTTP 403)

### Inferences
- Wu et al. tidak menetapkan protokol mAP deteksi. ACNE04 adalah benchmark grading/counting. Setiap paper deteksi membuat split dan pre-processing sendiri.
- Format anotasi asli kemungkinan box bergaya VOC/XML. README tidak menyebutkan formatnya (lihat Gaps).

### Gaps
- Format file anotasi asli (VOC XML?) dan nama file daftar fold tidak tercantum di README. Folder `code` tidak saya periksa.
- PDF ICCV 2019 diblokir (403), jadi metrik counting asli (MAE/MSE) belum diverifikasi.

## 2. ACNE04-v2 (Gazeau et al., MICCAI 2024): judul, perubahan, split, kode, metrik, lisensi

### Takeaway
Judul paper yang tepat adalah "AcneAI: A New Acne Severity Assessment Method Using Digital Images and Deep Learning" (MICCAI 2024, LNCS hlm. 68–78). Kata "ACNE04-v2" tidak ada di judul. ACNE04-v2 adalah subset ACNE04 yang dianotasi ulang: 1.204 gambar, 32.443 lesi, berformat lingkaran (center + radius). Repo **tidak menyediakan split** dan **tidak menyatakan lisensi**. Skrip YOLO repo memakai YOLOv8x @ 1280 dan mengevaluasi mAP@0.5 (NMS IoU 0.6) dengan sweep conf 0.10–0.50. Ini protokol yang berbeda dari Ultralytics default.

### Cited Findings
- Judul: "AcneAI: A new acne severity assessment method using digital images and deep learning". Penulis: Léa Gazeau, Hang Nguyen, Zung Nguyen, Mariia Lebedeva, Thanh Nguyen, Tat-Dat To, Jimmy Le Digabel, Jérome Filiol, Gwendal Josse, Clifford Perlis, Jonathan Wolfe. — [MICCAI 2024 paper page #2216](https://papers.miccai.org/miccai-2024/042-Paper2216.html)
- DOI Springer: 10.1007/978-3-031-72086-4_7, hlm. 68–78. — [hasil pencarian / OUCI](https://ouci.dntb.gov.ua/works/4NpDmBj9)
- Metrik utama abstrak: ICC 0,8 untuk klasifikasi keparahan di Acne04, dan AUC 0,88 untuk deteksi lesi inflamasi di dataset uji klinis multisenter. Ini bukan mAP deteksi. — [MICCAI 2024 paper page](https://papers.miccai.org/miccai-2024/042-Paper2216.html)
- Model inti (menurut reviewer): UNet dengan encoder EfficientNet untuk segmentasi lesi, ditambah metode geometris untuk memisahkan lesi dan model segmentasi kulit. Paper juga membandingkan diri dengan "other object detection approaches" (Tabel 1). — [MICCAI 2024 paper page, reviews](https://papers.miccai.org/miccai-2024/042-Paper2216.html)
- Reviewer 2: "no information is provided on how the dataset was trained, if cross validation was used or division of the dataset". Reviewer juga meminta definisi "AP" di Tabel 1 dan uji statistik. Reviewer 1 menyebut angka Tabel 1 "very comparable". — [MICCAI 2024 paper page, reviews](https://papers.miccai.org/miccai-2024/042-Paper2216.html)
- Repo dataset: 1.204 gambar dan 32.443 anotasi, subset dari ACNE04 v1 ("removed low-quality images and annotated more lesions"). Format: satu JSON `Acne04-v2_annotations.json` bergaya COCO, dengan `annotations` berisi `coordinates` (pusat) dan `radius`. Tidak ada file split dan tidak ada pernyataan lisensi. Ada skrip `draw_acne_from_json.py` dan folder `yolo_scripts/`. — [GitHub AIpourlapeau/acne04v2](https://github.com/AIpourlapeau/acne04v2)
- Suplemen mencantumkan 1.203 gambar untuk Acne04 v2 (README: 1.204). Anotator menandai setiap lesi dengan lingkaran minimal yang menutup seluruh lesi. — [ringkasan hasil pencarian atas suplemen](https://papers.miccai.org/miccai-2024/supp/2216_supp.pdf)
- `yolo_scripts/train.py`: `YOLO('yolov8x.pt')`, `imgsz=1280`, `epochs=600`, `patience=200`, `data='acne_detection.yaml'` (YAML tidak disertakan, jadi split tidak diketahui). Tidak ada kode konversi lingkaran→box. — [raw train.py](https://raw.githubusercontent.com/AIpourlapeau/acne04v2/main/yolo_scripts/train.py)
- `yolo_scripts/evaluate.py`: bobot `runs/yolov8x_1280_acne_detection_23022024_2/weights/best.pt`, data `acne_clinical_data.yaml`, `imgsz=1280`, batch 6, sweep conf 0,10–0,50 (langkah 0,05), NMS `iou=0.6`, mencetak `metrics.box.map50` dan recall `metrics.box.r`. — [raw evaluate.py](https://raw.githubusercontent.com/AIpourlapeau/acne04v2/main/yolo_scripts/evaluate.py)
- Repo tidak melaporkan metrik baseline. — [GitHub AIpourlapeau/acne04v2](https://github.com/AIpourlapeau/acne04v2)

### Temuan tambahan dari PDF lengkap (diunduh setelah catatan awal)
Sumber untuk semua butir di bawah: [MICCAI 2024 Open Access PDF #2216](https://papers.miccai.org/miccai-2024/paper/2216_paper.pdf)
- Model segmentasi AcneAI adalah U-Net dengan encoder EfficientNet-B4, input 512×512. Model ini dilatih pada 901 gambar data milik sendiri ("AcneAI seg. data"), bukan pada ACNE04. ACNE04 "is not used in our training/ validation" dan hanya dipakai sebagai data uji.
- Alasan re-anotasi menurut penulis: box ACNE04 asli "not lesion-centered and/or are too big", kadang lebih dari satu lesi berada dalam satu box, dan ada lesi yang terlewat. Koreksi v2 mencakup penghapusan gambar berkualitas rendah, penambahan lingkaran, dan pemisahan lingkaran agar setiap lingkaran hanya berisi satu lesi. Penulis menulis: "we use a circle which can be converted to box".
- Bagian 4 (perbandingan): penulis melatih YOLOv8 Ultralytics dengan input 1280×1280.
- **Tabel 1 bukan mAP COCO.** "AP stands for Average precision, and corresponds to the number of correct predictions over the total number of predictions", jadi secara praktis ini precision. Threshold pertama dipilih dengan meminimalkan |FP−FN|, threshold kedua dengan memaksimalkan (TP−FP).
- Tabel 1, Performance on Acne04 v2:

| Model | Threshold | TPR | FPR | Precision | AP | AUC |
|---|---|---|---|---|---|---|
| AcneAI | 1,81 | 0,52 | 0,09 | 0,52 | 0,45 | 0,79 |
| AcneAI | 2,01 | 0,18 | 0,02 | 0,64 | (sama) | (sama) |
| YOLOv8 (dilatih pada AcneAI seg.) | 0,18 | 0,54 | 0,14 | 0,54 | 0,57 | 0,72 |
| YOLOv8 (dilatih pada AcneAI seg.) | 0,29 | 0,35 | 0,04 | 0,73 | (sama) | (sama) |

- Tabel 1, Performance on clinical data:

| Model | Threshold | TPR | FPR | Precision | AP | AUC |
|---|---|---|---|---|---|---|
| AcneAI | 1,81 | 0,46 | 0,06 | 0,46 | 0,41 | 0,88 |
| AcneAI | 2,21 | 0,27 | 0,01 | 0,67 | (sama) | (sama) |
| YOLOv8 (dilatih pada AcneAI seg.) | 0,25 | 0,33 | 0,05 | 0,34 | 0,30 | 0,88 |
| YOLOv8 (dilatih pada AcneAI seg.) | 0,40 | 0,06 | 0,00 | 0,61 | (sama) | (sama) |
| YOLOv8 (dilatih pada Acne04 v2) | 0,16 | 0,32 | 0,04 | 0,32 | 0,26 | 0,85 |
| YOLOv8 (dilatih pada Acne04 v2) | 0,41 | 0,06 | 0,00 | 0,68 | (sama) | (sama) |

- Pada ACNE04 v1, ICC severity adalah 0,81 di teks (abstrak menulis 0,8). Threshold konversi skor ke level: 17,86 / 45,46 / 78,86.
- Paper tidak menyebutkan kriteria matching (IoU vs jarak pusat), dan juga tidak menyebutkan split train/val/test ketika YOLOv8 dilatih pada Acne04 v2.

### Inferences
- **Paper v2 tidak menyediakan angka mAP deteksi ACNE04-v2 yang bisa dibandingkan.** Satu-satunya angka di v2 adalah "AP" berdefinisi precision dan TPR/FPR pada threshold tertentu, dari model yang dilatih di data lain (cross-dataset). Klaim yang paling jujur adalah "tidak ada baseline mAP terpublikasi di ACNE04-v2". Paling jauh, pengguna dapat melaporkan precision/recall pada threshold yang dipilih dengan aturan |FP−FN| sebagai perbandingan longgar.
- Dengan conf ≥ 0,10, mAP50 yang dilaporkan via skrip penulis akan berbeda (umumnya lebih rendah atau setara) dibanding conf=0.001. Kurva PR terpotong, sehingga angkanya tidak setara dengan protokol default Ultralytics.
- Split 963/120/121 milik pengguna (total 1.204, sekitar 80/10/10) adalah split buatan sendiri. Tidak ada split resmi v2 yang bisa diikuti, jadi pengguna harus merilis daftar file split-nya agar bisa direproduksi.
- Konversi lingkaran→box (misalnya box = 2r × 2r) adalah pilihan pengguna. Pilihan ini memengaruhi IoU dengan box prediksi, sehingga perlu dinyatakan eksplisit.

### Gaps
- (Terjawab) Angka Tabel 1 sudah diverifikasi dari PDF lengkap; lihat di atas.
- Saya tidak menemukan paper lanjutan kelompok yang sama (AIpourlapeau / Gazeau, 2025–2026) yang memakai ACNE04-v2 dengan protokol deteksi. Ini hanya dicari sekilas, jadi belum bisa disimpulkan bahwa paper semacam itu tidak ada.
- Paten USPTO "Compute system with acne diagnostic mechanism" muncul di hasil pencarian tetapi tidak diperiksa.

## 3. Papers with Code / leaderboard dan jebakan (Roboflow, ukuran gambar, multi-kelas)

### Takeaway
Ada entri dataset ACNE04 di Papers with Code (arsip astro.paperswithcode.com), tetapi saya tidak menemukan leaderboard mAP deteksi yang terstandar. Banyak turunan Roboflow (ACNE04 Detection, Acne04v2 Stats Model, DERMNETNZ+ACNE04) memakai split, resize, dan augmentasi sendiri, sehingga tidak dapat dibandingkan.

### Cited Findings
- Halaman dataset ACNE04 di Papers with Code (mirror astro). — [astro.paperswithcode.com/dataset/acne04](https://astro.paperswithcode.com/dataset/acne04)
- Proyek Roboflow turunan: "Acne04 Detection", "Acne04v2 Stats Model", dan "DERMNETNZ + ACNE04" (gabungan dua sumber). — [Roboflow acne04-detection](https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection); [Roboflow acne04v2-stats-model](https://universe.roboflow.com/john-hr2ni/acne04v2-stats-model); [Roboflow dermnetnz-acne04](https://universe.roboflow.com/muhammad-suzaki-zahran/dermnetnz---acne04)
- Contoh variasi protokol: crop 640×640 dengan overlap 320 px (ACNet), sedangkan YOLOv5 Sangha & Rizvi melaporkan mAP@0.5 37,97 (1 kelas) dan 26,50 (4 kelas keparahan). — [ACNet](https://ar5iv.labs.arxiv.org/html/2105.14891); [medRxiv 2021.12.05.21267310 (via hasil pencarian)](https://www.medrxiv.org/content/10.1101/2021.12.05.21267310v1)

### Inferences
- Jebakan utama:
  - Split acak yang berbeda-beda.
  - Augmentasi Roboflow ("3x") yang diterapkan sebelum split, sehingga terjadi kebocoran train→test. Ini praktik umum di Roboflow, tetapi tidak saya verifikasi pada proyek spesifik.
  - Crop/tiling yang dievaluasi per-tile, bukan per-gambar.
  - Penggunaan varian 4 kelas (berdasarkan level keparahan) sebagai "kelas".
  - Anotasi v1 dan v2 yang berbeda total (18.983 vs 32.443 lesi).
- Angka dari v1 tidak boleh dibandingkan langsung dengan v2. Ground truth v2 berisi sekitar 1,7× lebih banyak lesi untuk gambar yang lebih sedikit, sehingga recall terhadap lesi kecil menjadi jauh lebih menuntut.

### Gaps
- Saya tidak membuka halaman PwC secara penuh untuk melihat apakah ada tabel SOTA deteksi.
- Kebocoran pada proyek Roboflow tertentu belum diverifikasi.

## 4. Dataset alternatif (konteks saja)

### Takeaway
AcneSCU / ACNE-DET (276 gambar VISIA resolusi tinggi, 31.777 lesi, 10 kelas) adalah benchmark multi-kelas Sichuan University. AcneDet (1.572 foto smartphone, Faster R-CNN mAP 0,54) dan AcnePA1/2 (Sci Rep 2025) adalah dataset lain. Semuanya tidak sebanding dengan ACNE04.

### Cited Findings
- AcneSCU: 276 gambar wajah VISIA, 31.777 anotasi instance-segmentation, 10 kategori, resolusi sekitar 3128×4171 sampai 3456×5184. — [Zhang et al. 2022, arXiv 2207.03674](https://ar5iv.labs.arxiv.org/html/2207.03674)
- ACNE-DET (DSDH, Wei et al. 2023): 276 gambar, 31.777 lesi, 10 kategori. Train 241 gambar / 28.260 lesi. Pasien tidak tumpang tindih antar-split. AP (ResNet-50): Faster R-CNN 41,9; Mask R-CNN 43,3; Mask R-CNN + DSDH 45,6. — [arXiv 2301.12219 (HTML)](https://arxiv.org/html/2301.12219)
- AcneDet: 1.572 gambar smartphone (iOS/Android), Faster R-CNN mAP 0,54. — [Diagnostics 2022, PMC9406819](https://pmc.ncbi.nlm.nih.gov/articles/PMC9406819)
- AcneDGNet (Sci Rep 2025): dilatih dengan 586 gambar ACNE04, 220 AcneSCU, dan 791 AcnePA1. Kesalahan counting total 1,91±3,28. Akurasi grading 91,2% di ACNE04. Ini angka dari ringkasan sekunder. — [hyper.ai news 38397](https://hyper.ai/en/news/38397)

### Inferences
- Jumlah 276 / 31.777 yang identik menunjukkan ACNE-DET ≈ AcneSCU. Ini dugaan, belum dikonfirmasi penulis.

### Gaps
- "ACNE-ECKH" dan "Acne-1K" tidak saya temukan dalam pencarian ini.

## 5. Kemajuan deteksi objek kecil yang diterapkan ke jerawat, dengan angka di ACNE04

### Takeaway
Angka deteksi yang terdokumentasi di ACNE04 (v1) masih rendah dan berasal dari detektor dua tahap dengan crop 640:

| Metode | AP |
|---|---|
| Faster R-CNN | 10,3 |
| R-FCN | 14,0 |
| ACNet | 20,5 |
| Mask R-CNN + DSDH | 23,0 |

Angka YOLO yang ada (YOLOv5 mAP@0.5 37,97; YOLOv7 yang ditingkatkan "mAP 83,7%") memakai protokol yang tidak jelas. Saya tidak menemukan angka ACNE04 untuk YOLOv9/10/11/12, RT-DETR, DINO, kepala P2, atau SAHI dari sumber terverifikasi.

### Cited Findings
- ACNet (SMC 2021), ACNE04 dengan split acak 80/20, crop 640×640 overlap 320. mAP test: Faster R-CNN (R101) 10,3; R-FCN 14,0; Rashataprucksa et al. 14,7; ACNet 20,5. Ambang IoU tidak disebutkan. — [ACNet, arXiv 2105.14891](https://ar5iv.labs.arxiv.org/html/2105.14891)
- DSDH (Wei et al., arXiv 2023), ACNE04 dengan split acak 80/20 dan crop 640×640. Metrik "PASCAL-style AP" (IoU tidak eksplisit). APS/APM/APL memakai ambang COCO 32²/96². Maksimal 100 box per gambar, NMS 0,5, skor > 0,01, Detectron2, 13 epoch. AP ACNE04: R-FCN 14,0; Rashataprucksa 14,7; ACNet 20,5; Mask R-CNN + DSDH 23,0. Tidak ada YOLO/DETR. — [arXiv 2301.12219 HTML](https://arxiv.org/html/2301.12219)
- YOLOv5 di ACNE04: mAP@0.5 37,97 (1 kelas), 26,50 (4 kelas). — [Sangha & Rizvi, medRxiv 2021](https://www.medrxiv.org/content/10.1101/2021.12.05.21267310v1) (via hasil pencarian; full text 403)
- YOLOv7 yang ditingkatkan di ACNE04: "mAP 83,7%". Jenis mAP-nya tidak jelas. — [Automatic Acne Detection Model Based on Improved YOLOv7 (DOAJ record)](https://library.kab.ac.ug/Record/doaj-art-20c5d8ac834d45d08943c46661e60e2a/Description)
- CNN interpretabel (THC 2022): mAP tertinggi 0,536 pada dataset open source. — [Technology and Health Care, thc228014](https://content.iospress.com/articles/technology-and-health-care/thc228014)
- Referensi ACNE04-v2 dari penulisnya sendiri: YOLOv8x @ 1280 (skrip repo). Angkanya tidak dipublikasikan di repo. — [raw train.py](https://raw.githubusercontent.com/AIpourlapeau/acne04v2/main/yolo_scripts/train.py)
- YOLOv11 untuk jerawat inflamasi (dataset non-ACNE04, tesis Telkom): mAP@0.5 90%. — [Telkom University openlibrary](https://openlibrary.telkomuniversity.ac.id/pustaka/files/240439/abstract/deteksi-jerawat-inflamasi-pada-citra-wajah-dengan-model-yolov11-berdasarkan-analisis-warna-kulit-dalam-bentuk-buku-karya-ilmiah.pdf)

### Inferences
- Rentang angka v1 yang sangat lebar (10–84) mencerminkan perbedaan protokol, bukan kemajuan metode:
  - AP@[.5:.95] vs AP50
  - crop vs gambar penuh
  - split yang berbeda
  - kemungkinan kebocoran
- Baseline yang paling layak dijalankan ulang sendiri oleh pengguna pada split-nya (rekomendasi):
  1. YOLOv8x @ 1280, mereplikasi skrip penulis v2.
  2. YOLO11l dan YOLOv8l dengan kepala P2.
  3. RT-DETR-l (tersedia di Ultralytics).
  4. Mask R-CNN atau Faster R-CNN + tiling 640 sebagai jembatan ke literatur v1.

### Gaps
- Tidak ada angka terverifikasi untuk YOLOv9/v10/v12, RT-DETR, DINO, P2, maupun SAHI pada ACNE04/v2 dalam sumber yang saya buka.
- Pencarian khusus 2025–2026 hanya dua query. Kemungkinan ada paper jurnal (MDPI/IEEE Access) yang terlewat.

## 6. Rekomendasi protokol agar klaim dapat dibandingkan

### Takeaway
Tidak ada protokol resmi ACNE04-v2, sehingga klaim yang adil harus berbentuk "perbandingan internal pada split yang dirilis". Laporkan AP50 dan AP@[.5:.95] gaya COCO pada test set gambar penuh (bukan per-tile), dan jalankan ulang baseline literatur di split yang sama, alih-alih menyalin angka paper.

### Cited Findings
- v2 tidak memiliki split resmi. — [GitHub AIpourlapeau/acne04v2](https://github.com/AIpourlapeau/acne04v2)
- Protokol penulis v2 adalah mAP50 dan recall dengan conf 0,10–0,50, NMS 0,6, imgsz 1280. — [raw evaluate.py](https://raw.githubusercontent.com/AIpourlapeau/acne04v2/main/yolo_scripts/evaluate.py)
- Literatur v1 memakai split acak 80/20 + crop 640 + AP bergaya PASCAL/COCO (APS/M/L). — [arXiv 2301.12219](https://arxiv.org/html/2301.12219); [ACNet](https://ar5iv.labs.arxiv.org/html/2105.14891)
- Reviewer MICCAI meminta uji statistik dan definisi AP. — [MICCAI 2024 paper page](https://papers.miccai.org/miccai-2024/042-Paper2216.html)

### Inferences
Rekomendasi:
1. Rilis daftar file split 963/120/121 (dan seed) beserta aturan konversi lingkaran→box.
2. Metrik utama: AP50 dan AP@[.5:.95] (Ultralytics default conf=0.001, iou=0.7, max_det=1000) per gambar penuh di test. Tambahkan AP untuk objek kecil (COCO APS < 32² px) karena 73,5% lesi < 16 px. Sertakan juga metrik counting (MAE, korelasi) karena itulah tujuan klinis ACNE04.
3. Sebagai protokol sekunder, laporkan mAP50 dan recall dengan skrip penulis v2 (conf sweep, iou=0.6) agar bisa disejajarkan dengan protokol mereka.
4. Opsional, sebagai jembatan ke literatur v1: 5-fold atau split 1.165/292 pada ACNE04 v1, crop 640, AP gaya COCO. Hanya angka ini yang secara longgar sebanding dengan ACNet (20,5) dan DSDH (23,0).
5. Laporkan CI bootstrap atau beberapa seed, karena reviewer MICCAI menyoroti tidak adanya uji statistik.
6. SAHI harus dilaporkan sebagai protokol inferensi terpisah: gambar penuh, NMS antar-tile, dan imgsz/tile disebutkan. Jangan dicampur dengan angka gambar penuh standar.

### Gaps
- Matching berbasis lingkaran (IoU lingkaran atau jarak pusat) yang mungkin dipakai AcneAI belum terverifikasi karena PDF tidak terbaca.
