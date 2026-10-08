---
phase: 2
plan: 1
wave: 2
depends_on: [1.1, 1.2]
files_modified:
  - scripts/train.py
  - configs/train_config.yaml
autonomous: true
user_setup: []

must_haves:
  truths:
    - "YOLOv8s berhasil training pada dataset ACNE04-v2"
    - "Training berjalan di GPU RTX 4060 tanpa OOM error"
    - "Metrik baseline tercatat: mAP@50, mAP@50-95, Precision, Recall"
  artifacts:
    - "runs/detect/train/ berisi weights (best.pt, last.pt)"
    - "runs/detect/train/ berisi training curves dan confusion matrix"
    - "scripts/train.py — script training reusable"
---

# Plan 2.1: Baseline YOLOv8s Training

<objective>
Train model YOLOv8s pada dataset ACNE04-v2 dengan default hyperparameters
untuk mendapatkan baseline performance.

Purpose: Baseline memberikan referensi awal untuk mengukur improvement dari tuning.
Output: Model weights `best.pt` + metrik baseline.
</objective>

<context>
Load for context:
- docs/SPEC.md
- data/data.yaml
</context>

<tasks>

<task type="auto">
  <name>Buat training script dan jalankan baseline training</name>
  <files>scripts/train.py</files>
  <action>
    1. Buat `scripts/train.py` yang menggunakan Ultralytics API:
       ```python
       from ultralytics import YOLO

       model = YOLO('yolov8s.pt')  # pretrained weights
       results = model.train(
           data='data/data.yaml',
           epochs=50,
           imgsz=640,       # Start dengan 640; jika mAP rendah, coba 1280
           batch=16,        # RTX 4060 8GB — start with 16, reduce if OOM
           device=0,        # GPU 0
           project='runs/detect',
           name='baseline',
           patience=10,     # early stopping
           save=True,
           plots=True,
           verbose=True,
           # Small object optimization
           close_mosaic=5,  # Disable mosaic last 5 epochs untuk stabilitas
       )
       # NOTE: Gambar asli ACNE04 beresolusi ~3456x3112 dengan lesi 10-30px.
       # Jika mAP@50 < 0.2 pada 640, re-run dengan imgsz=1280 dan batch=4.
       ```
    2. Jalankan training: `python scripts/train.py`
    3. Jika batch=16 menyebabkan OOM, turunkan ke batch=8.

    AVOID: Jangan set epochs terlalu tinggi tanpa early stopping — 50 epochs dengan patience=10 sudah cukup untuk baseline.
    AVOID: Jangan lupa set `device=0` untuk memastikan training di GPU, bukan CPU.
    AVOID: Jangan ubah augmentation defaults — ini baseline, kita mau tahu performa "out-of-the-box".
  </action>
  <verify>
    ```
    # Cek model weights ada
    dir runs\detect\baseline\weights\ /b
    # Cek metrik
    type runs\detect\baseline\results.csv | more
    # Cek confusion matrix plot
    dir runs\detect\baseline\confusion_matrix*.png
    ```
  </verify>
  <done>
    - `runs/detect/baseline/weights/best.pt` exists
    - `runs/detect/baseline/results.csv` berisi metrik per epoch
    - Confusion matrix dan training curves ter-generate
    - Baseline mAP@50 tercatat (target awal: >0.3 sebagai sanity check)
  </done>
</task>

<task type="auto">
  <name>Validasi baseline pada test set dan catat metrik</name>
  <files>scripts/validate.py</files>
  <action>
    1. Buat `scripts/validate.py`:
       ```python
       from ultralytics import YOLO

       model = YOLO('runs/detect/baseline/weights/best.pt')
       metrics = model.val(
           data='data/data.yaml',
           split='test',
           imgsz=640,
           batch=16,
           device=0,
           plots=True,
           save_json=True,
       )
       print(f"Precision: {metrics.box.mp:.4f}")
       print(f"Recall:    {metrics.box.mr:.4f}")
       print(f"mAP@50:    {metrics.box.map50:.4f}")
       print(f"mAP@50-95: {metrics.box.map:.4f}")
       ```
    2. Jalankan: `python scripts/validate.py`
    3. Catat semua metrik baseline.

    AVOID: Jangan evaluate pada validation set — gunakan test set (`split='test'`) untuk evaluasi final yang jujur.
  </action>
  <verify>
    Output script menunjukkan 4 metrik: Precision, Recall, mAP@50, mAP@50-95.
  </verify>
  <done>
    - Metrik test set tercatat
    - Confusion matrix pada test set ter-generate
    - Baseline performance documented
  </done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] Training completed tanpa error
- [ ] best.pt dan last.pt ada di runs/detect/baseline/weights/
- [ ] Metrik baseline: Precision, Recall, mAP@50, mAP@50-95 semua tercatat
- [ ] Confusion matrix ter-generate untuk test set
</verification>

<success_criteria>
- [ ] Baseline model ter-training dan ter-evaluasi
- [ ] Semua metrik tercatat sebagai referensi untuk fase tuning
</success_criteria>
