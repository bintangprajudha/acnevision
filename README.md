# AcneVision: Deteksi & Penghitungan Lesi Jerawat (YOLOv8)

Model **YOLOv8l** yang mendeteksi setiap lesi jerawat pada foto wajah dan menghitung
jumlahnya, dilatih pada dataset **ACNE04-v2** (1.204 gambar, 32.443 lesi).

> Fokus saat ini: **pengembangan model**. Status: `docs/STATE.md`.

## Model Terpilih: YOLOv8l @ 960px

Split test (121 gambar, 3.507 lesi), protokol standar `scripts/evaluate.py`:

| Precision | Recall | mAP@50 | mAP@50-95 | MAE counting | Inferensi (RTX 4060) |
|---|---|---|---|---|---|
| 56.31% | 48.10% | 49.80% | 19.14% | 8.69 lesi/gambar | 45 ms |

Perbandingan dengan run lain: `docs/EXPERIMENTS.md`.

## Struktur Folder

```text
acnevision/
├── AGENTS.md                 # Aturan kerja agen AI di proyek ini
├── configs/
│   └── yolov8l_960.yaml      # Hyperparameter model terpilih (1 file = 1 eksperimen)
├── data/                     # Dataset YOLO (images/ & labels/ di-ignore git)
│   └── data.yaml
├── docs/                     # Lihat docs/README.md untuk indeks & aturan
│   ├── STATE.md  SPEC.md  PIPELINE.md  EXPERIMENTS.md  ROADMAP.md
│   └── experiments/history/  # results.csv & args.yaml run lama
├── scripts/
│   ├── common.py             # Path, device, konstanta bersama
│   ├── train.py              # Training dari config YAML (auto-resume)
│   ├── evaluate.py           # Metrik deteksi (test) + counting (conf dipilih di val)
│   ├── predict.py            # Visualisasi gambar/folder, opsi --sahi
│   ├── report.py             # Grafik & ringkasan untuk docs/REPORT.md
│   ├── run_experiments.ps1   # Rantai evaluasi semua varian + fine-tune E4
│   ├── verify_env.py         # Cek CUDA & ultralytics
│   └── data/                 # download_dataset, download_and_extract_images, convert_to_yolo
├── pretrained/               # Bobot AWAL (yolov8l.pt COCO) untuk memulai training, di-ignore git
├── runs/                     # Output training/eval/predict, di-ignore git
│   ├── detect/<run>/weights/ # Bobot HASIL training per run: best.pt, last.pt
│   ├── eval/<run>/           # metrics.json, confusion matrix, counts_test.csv
│   └── predict/<run>_<mode>/ # Visualisasi prediksi
├── .claude/skills/           # Skill AI-research yang relevan
└── requirements.txt
```

## Penggunaan

```bash
# Cek environment
.venv\Scripts\python.exe scripts\verify_env.py

# Training (default: configs/yolov8l_960.yaml); smoke test dengan --epochs 1 --fraction 0.02
.venv\Scripts\python.exe scripts\train.py --cfg configs\yolov8l_960.yaml

# Evaluasi standar -> runs/eval/<run>/metrics.json
.venv\Scripts\python.exe scripts\evaluate.py --weights runs\detect\yolov8l_960\weights\best.pt

# Prediksi + visualisasi -> runs/predict/
.venv\Scripts\python.exe scripts\predict.py --source data\images\test\levle2_118.jpg
.venv\Scripts\python.exe scripts\predict.py --source data\images\test --sahi --slice 960
```

## Dataset

- Gambar: ACNE04 (Wu et al., ICCV 2019), via HuggingFace `Layered-Labs/ACNE04`
- Anotasi: ACNE04-v2 (Gazeau et al., MICCAI 2024), https://github.com/AIpourlapeau/acne04v2
- Lisensi: penggunaan akademik/riset
