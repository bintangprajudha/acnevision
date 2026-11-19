# AcneVision: Deteksi & Penghitungan Lesi Jerawat (YOLOv8)

Model **YOLOv8l** yang mendeteksi setiap lesi jerawat pada foto wajah, menghitung jumlahnya, dan menurunkan tingkat
keparahan (Hayashi) dari jumlah itu. Dilatih dan dievaluasi pada **ACNE04-v2** (1.204 gambar, 32.443 lesi).

> Fokus saat ini: **pengembangan model** dan penulisan paper. Status terkini: [`docs/STATE.md`](docs/STATE.md).

## Model terpilih

**YOLOv8l dilatih pada tile 2×2, inferensi dengan SAHI 1728.** Bobot: `runs/detect/yolov8l_960_tiles2/weights/best.pt`
(tidak ada di git; lihat [Bobot & data](#bobot--data)).

Hasil di **split test** (121 gambar, 3.507 lesi), `imgsz` 960, conf dipilih di val, rata-rata ± simpangan baku dari 3 seed
(42, 7, 123). Sumber: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), `runs/eval/<run>/metrics.json`.

| Mode inferensi | F1 | Recall | Precision | MAE (lesi/gambar) | Waktu per gambar (RTX 4060) |
|---|---|---|---|---|---|
| **SAHI 1728** (akurasi) | 0.580 ± 0.003 | 0.550 ± 0.021 | 0.615 ± 0.027 | 8.02 ± 0.33 | ±622 ms |
| plain (cepat) | 0.524 ± 0.004 | 0.502 ± 0.017 | 0.550 ± 0.020 | 7.87 ± 0.27 | ±70 ms |

Ringkasan metode untuk presentasi: [`docs/PIPELINE_MODEL_TERBAIK.md`](docs/PIPELINE_MODEL_TERBAIK.md).
Peringkat semua model: [`docs/report/ranking_models.csv`](docs/report/ranking_models.csv).

## Cara kerja singkat

```text
ACNE04 (gambar) + ACNE04-v2 (lingkaran)
   -> convert_to_yolo.py : lingkaran -> kotak, split 963/120/121 (seed 42)
   -> preprocess.py      : potong train/val jadi tile 2x2 (overlap 10%)
   -> train.py           : YOLOv8l @ 960 dari bobot COCO, 100 epoch
   -> evaluate.py        : inferensi SAHI 1728 + conf dipilih di val -> deteksi, counting, grading Hayashi
```

## Struktur folder

```text
acnevision/
├── README.md                 # Halaman ini
├── AGENTS.md                 # Aturan kerja agen AI di proyek ini
├── requirements.txt
├── configs/                  # 1 file = 1 eksperimen (indeks: configs/README.md)
├── scripts/                  # Kode: train, evaluate, predict, report (indeks: scripts/README.md)
│   └── data/                 # Unduh data, konversi ke YOLO, tiling
├── splits/                   # Daftar split train/val/test resmi (acne04v2_split.csv) + aturan konversi
├── docs/                     # Dokumentasi (indeks & aturan: docs/README.md)
│   ├── report/               # Tabel & grafik hasil (summary.json, ranking_models.csv, figures/)
│   └── experiments/history/  # results.csv & args.yaml run lama
├── reports/                  # Laporan riset benchmark literatur ACNE04
├── research_notes/           # Catatan mentah di balik laporan riset
├── presentasi/               # Deck PowerPoint + pembuatnya (build_deck.js)
├── notebooks/                # Notebook ringkasan hasil (hanya membaca runs/, jalan di CPU)
├── references/               # Salinan lokal paper acuan (di-ignore git, berhak cipta)
│
│  Tidak masuk git (ada di komputer lokal):
├── data/                     # Dataset YOLO: images/ dan labels/ (data/data.yaml ikut git)
├── data_tiles2/              # Dataset tile 2x2 hasil preprocess.py
├── pretrained/               # Bobot awal COCO (yolov8l.pt, dll.)
├── runs/
│   ├── detect/<run>/weights/ # Bobot hasil training: best.pt, last.pt
│   ├── eval/<run>/           # metrics.json, counts_test.csv, confusion matrix
│   └── predict/<run>/        # Gambar hasil prediksi
└── .venv/                    # Python 3.10 (CUDA)
```

## Memulai

Semua perintah dijalankan dari root proyek dengan Python venv (`.venv\Scripts\python.exe`).

```powershell
# 1. Cek environment (CUDA + ultralytics)
.venv\Scripts\python.exe scripts\verify_env.py

# 2. Data (sekali): unduh -> konversi ke YOLO -> tile 2x2
.venv\Scripts\python.exe scripts\data\download_dataset.py
.venv\Scripts\python.exe scripts\data\download_and_extract_images.py
.venv\Scripts\python.exe scripts\data\convert_to_yolo.py
.venv\Scripts\python.exe scripts\data\preprocess.py --grid 2 --overlap 0.1

# 3. Training model terpilih (uji cepat dulu: tambahkan --epochs 1 --fraction 0.02 --project runs\smoke)
.venv\Scripts\python.exe scripts\train.py --cfg configs\yolov8l_960_tiles2.yaml

# 4. Evaluasi standar (SAHI 1728) -> runs\eval\yolov8l_960_tiles2_sahi1728\metrics.json
.venv\Scripts\python.exe scripts\evaluate.py --weights runs\detect\yolov8l_960_tiles2\weights\best.pt --mode sahi --slice 1728

# 5. Prediksi + visualisasi -> runs\predict\
.venv\Scripts\python.exe scripts\predict.py --source data\images\test --sahi --slice 1728
```

Satu eksperimen baru: salin satu file di `configs/`, ubah **satu** variabel, beri `name` unik, lalu jalankan `train.py`.
Training penuh memakan waktu berjam-jam (model terpilih ±650 menit).

## Aturan penting

- **Test set hanya untuk laporan akhir.** Pemilihan epoch dan conf memakai split val.
- **Evaluasi hanya lewat `scripts/evaluate.py`** (conf 0.001, IoU 0.7, `max_det` 1000, `imgsz` sama dengan training).
  Angka dari protokol lain tidak dibandingkan.
- Setiap run dicatat di [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), termasuk yang gagal atau lebih buruk.
- Bobot, `runs/`, dan dataset tidak masuk git (lihat `.gitignore`).

## Dokumentasi: mulai dari mana?

| Ingin tahu | Baca |
|---|---|
| Status proyek dan langkah berikutnya | [`docs/STATE.md`](docs/STATE.md) |
| Spesifikasi, protokol evaluasi, target dan standar minimum | [`docs/SPEC.md`](docs/SPEC.md) |
| Penjelasan metode model terbaik (bahan slide) | [`docs/PIPELINE_MODEL_TERBAIK.md`](docs/PIPELINE_MODEL_TERBAIK.md) |
| Semua eksperimen dan hasilnya | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| Laporan hasil lengkap | [`docs/REPORT.md`](docs/REPORT.md) |
| Rencana eksperimen dan benchmark paper | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| API lokal untuk aplikasi mobile (`/api/v1/detect`, tunnel, kode error) | [`docs/API.md`](docs/API.md) |
| Daftar lengkap dokumen dan aturan penulisan | [`docs/README.md`](docs/README.md) |

## Bobot & data

- Bobot model tidak disimpan di git (`*.pt` di-ignore). Simpan cadangan `best.pt` secara terpisah.
- Gambar: ACNE04 (Wu dkk., ICCV 2019), via HuggingFace `Layered-Labs/ACNE04`.
- Anotasi lesi: ACNE04-v2 (Gazeau dkk., MICCAI 2024), https://github.com/AIpourlapeau/acne04v2.
- Lisensi: penggunaan akademik/riset.
