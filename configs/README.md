# Indeks Config Eksperimen

Satu file = satu eksperimen. Jangan mengubah config run yang sudah tercatat di `docs/EXPERIMENTS.md`; salin lalu ubah satu variabel.
Kode run (B0, E3, dst.) adalah nama resmi di `docs/EXPERIMENTS.md`.

| File | Kode | Model / perubahan | Status |
|---|---|---|---|
| `yolov8l_960_tiles2.yaml` | E9 | **YOLOv8l dilatih pada tile 2×2** dari bobot COCO | **Model terpilih** |
| `yolov8l_960_tiles2_s7.yaml`, `_s123.yaml` | E9 seed 7, 123 | Sama dengan E9, hanya seed berbeda | Dipakai untuk rata-rata ± SD |
| `yolov8l_960_default.yaml` | B0 | YOLOv8l bawaan (tanpa tuning) | Baseline |
| `yolov8l_960.yaml` | E3 | YOLOv8l resep tuning @ 960 | Model terpilih sebelumnya |
| `yolov8l_960_tiles2_ft.yaml` | E4 | Resep tuning, fine-tune pada tile 10 epoch | Pembanding terdekat |
| `yolov8l_960_tiles2_ft60.yaml` | E8 | Resep tuning, fine-tune pada tile 60 epoch | Kandidat plain |
| `yolov8l_960_default_tiles2_ft.yaml` | E5 | Bawaan, fine-tune pada tile | Kontrol |
| `yolov8l_1280.yaml` | E12 | YOLOv8l @ 1280 | Pembanding setup AcneAI |
| `yolov8l_960_p2.yaml` | E10 | Head P2 (screening di val) | Ablasi |
| `yolov8l_960_augnoise.yaml` | E11 | Augmentasi noise/kecerahan (screening di val) | Ablasi |
| `yolo11l_960.yaml`, `yolo11l_960_tiles2_ft.yaml` | E6, E7 | YOLO11l | Tidak masuk tabel utama |
| `yolo11n_640.yaml` | B1 | YOLO11n @ 640 | Dihentikan |
| `rtdetr_l_960.yaml` | B2 | RT-DETR-l | Dibatalkan |

Konfigurasi memakai path relatif terhadap root proyek (`model`, `data`); `scripts/train.py` yang menyelesaikannya.
