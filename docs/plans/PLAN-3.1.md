---
phase: 3
plan: 1
wave: 3
depends_on: [2.1]
files_modified:
  - scripts/train_tuned.py
  - configs/augmentation.yaml
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Model tuned memiliki mAP@50 lebih tinggi dari baseline"
    - "Augmentasi dan hyperparameter tuning diterapkan"
    - "Model terbaik dipilih berdasarkan mAP@50-95 pada validation set"
  artifacts:
    - "runs/detect/tuned/weights/best.pt — model terbaik setelah tuning"
    - "docs/METRICS.md — perbandingan baseline vs tuned"
---

# Plan 3.1: Hyperparameter Tuning & Augmentation

<objective>
Improve performa model dari baseline melalui hyperparameter tuning dan
strategi augmentasi yang lebih agresif.

Purpose: Baseline jarang optimal — tuning diharapkan meningkatkan mAP@50 dan Recall.
Output: Model tuned `best.pt` + dokumen perbandingan metrik.
</objective>

<context>
Load for context:
- docs/SPEC.md
- runs/detect/baseline/results.csv (baseline metrics)
- data/data.yaml
</context>

<tasks>

<task type="auto">
  <name>Training dengan hyperparameter tuning dan augmentasi lanjutan</name>
  <files>scripts/train_tuned.py</files>
  <action>
    1. Buat `scripts/train_tuned.py` dengan penyesuaian:
       ```python
       from ultralytics import YOLO

       model = YOLO('yolov8s.pt')
       results = model.train(
           data='data/data.yaml',
           epochs=100,
           imgsz=640,
           batch=16,
           device=0,
           project='runs/detect',
           name='tuned',
           patience=20,
           # Hyperparameter tuning
           lr0=0.001,          # lower initial LR for fine-grained learning
           lrf=0.01,           # final LR factor
           warmup_epochs=5,
           # Augmentation agresif
           hsv_h=0.02,         # hue augmentation
           hsv_s=0.7,          # saturation
           hsv_v=0.4,          # value/brightness
           degrees=15,         # rotation
           translate=0.2,      # translation
           scale=0.5,          # scale
           fliplr=0.5,         # horizontal flip
           mosaic=1.0,         # mosaic augmentation
           mixup=0.1,          # mixup
           copy_paste=0.1,     # copy-paste augmentation
           # Optimization
           optimizer='AdamW',
           weight_decay=0.001,
           cos_lr=True,        # cosine LR scheduler
           save=True,
           plots=True,
       )
       ```
    2. Jalankan training.
    3. Jika overfitting terdeteksi (train loss turun tapi val loss naik),
       tingkatkan augmentasi atau kurangi epochs.

    AVOID: Jangan ubah imgsz dari 640 — ini standard YOLOv8 dan sudah optimal.
    AVOID: Jangan disable mosaic — sangat membantu untuk dataset kecil seperti ACNE04-v2.
  </action>
  <verify>
    ```
    dir runs\detect\tuned\weights\ /b
    type runs\detect\tuned\results.csv | more
    ```
  </verify>
  <done>
    - `runs/detect/tuned/weights/best.pt` exists
    - Training curves menunjukkan convergence
    - mAP@50 >= baseline mAP@50
  </done>
</task>

<task type="auto">
  <name>Evaluasi tuned model dan buat laporan perbandingan</name>
  <files>scripts/compare_models.py, docs/METRICS.md</files>
  <action>
    1. Buat `scripts/compare_models.py` yang:
       - Load baseline dan tuned model
       - Evaluate keduanya pada test set
       - Print tabel perbandingan metrik
       - Hitung inference speed kedua model
       - Generate side-by-side confusion matrix
    2. Buat `docs/METRICS.md` berisi:
       ```markdown
       # Model Comparison: Baseline vs Tuned
       | Metric | Baseline | Tuned | Delta |
       |--------|----------|-------|-------|
       | Precision | X.XXX | X.XXX | +X.XXX |
       | Recall | X.XXX | X.XXX | +X.XXX |
       | mAP@50 | X.XXX | X.XXX | +X.XXX |
       | mAP@50-95 | X.XXX | X.XXX | +X.XXX |
       | Inference (ms) | X.X | X.X | ... |
       ```
    3. Pilih model terbaik berdasarkan mAP@50-95.

    AVOID: Jangan hanya lihat accuracy/precision — Recall sangat penting untuk mendeteksi SEMUA jerawat.
  </action>
  <verify>
    `docs/METRICS.md` berisi tabel perbandingan lengkap dengan angka aktual.
  </verify>
  <done>
    - Kedua model ter-evaluasi pada test set
    - Tabel perbandingan lengkap di METRICS.md
    - Model terbaik teridentifikasi
  </done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] Tuned model menghasilkan metrik yang lebih baik atau setara dengan baseline
- [ ] METRICS.md berisi perbandingan lengkap
- [ ] Model terbaik teridentifikasi
</verification>

<success_criteria>
- [ ] mAP@50 tuned >= mAP@50 baseline
- [ ] Perbandingan terdokumentasi
</success_criteria>
