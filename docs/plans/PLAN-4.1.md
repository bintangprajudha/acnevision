---
phase: 4
plan: 1
wave: 4
depends_on: [3.1]
files_modified:
  - scripts/export_onnx.py
  - scripts/inference.py
  - models/acnevision.onnx
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Model terbaik ter-export ke format ONNX"
    - "ONNX model menghasilkan prediksi yang konsisten dengan PyTorch model"
    - "Inference script berfungsi end-to-end pada gambar baru"
    - "Inference speed tercatat"
  artifacts:
    - "models/acnevision.onnx — model ONNX final"
    - "scripts/inference.py — script inferensi end-to-end"
    - "scripts/export_onnx.py — script export reusable"
---

# Plan 4.1: ONNX Export & Inference Pipeline

<objective>
Export model terbaik ke format ONNX untuk deployment mobile, dan buat
script inferensi end-to-end yang mendeteksi + menghitung jerawat.

Purpose: Deliverable akhir — model siap diintegrasikan ke aplikasi mobile.
Output: `acnevision.onnx` + `inference.py` untuk testing.
</objective>

<context>
Load for context:
- docs/SPEC.md
- docs/METRICS.md (model terbaik)
</context>

<tasks>

<task type="auto">
  <name>Export model terbaik ke ONNX</name>
  <files>scripts/export_onnx.py, models/acnevision.onnx</files>
  <action>
    1. Buat `scripts/export_onnx.py`:
       ```python
       from ultralytics import YOLO

       # Load model terbaik (baseline atau tuned — sesuai hasil METRICS.md)
       best_model_path = 'runs/detect/tuned/weights/best.pt'  # atau baseline
       model = YOLO(best_model_path)

       # Export ke ONNX
       model.export(
           format='onnx',
           imgsz=640,
           half=False,       # FP32 untuk kompatibilitas maksimal
           simplify=True,    # ONNX graph simplification
           opset=12,         # opset 12 untuk kompatibilitas luas
       )

       # Copy ke models/
       import shutil
       shutil.copy(best_model_path.replace('.pt', '.onnx'), 'models/acnevision.onnx')
       print("Export successful: models/acnevision.onnx")
       ```
    2. Jalankan script.
    3. Verifikasi ukuran file ONNX (harusnya ~22MB untuk YOLOv8s).

    AVOID: Jangan gunakan half=True (FP16) karena tidak semua mobile device mendukung FP16 inference.
    AVOID: Jangan lupa simplify=True — mengurangi ukuran model dan meningkatkan kompatibilitas.
  </action>
  <verify>
    ```
    dir models\acnevision.onnx
    python -c "import onnx; model = onnx.load('models/acnevision.onnx'); onnx.checker.check_model(model); print('ONNX model valid')"
    ```
  </verify>
  <done>
    - `models/acnevision.onnx` exists dan valid
    - Ukuran file reasonable (~20-25MB)
    - ONNX checker passes tanpa error
  </done>
</task>

<task type="auto">
  <name>Buat inference script end-to-end</name>
  <files>scripts/inference.py</files>
  <action>
    1. Buat `scripts/inference.py` yang:
       - Menerima path gambar sebagai argument
       - Load model ONNX (atau .pt sebagai fallback)
       - Jalankan deteksi pada gambar
       - Gambar bounding box pada setiap jerawat yang terdeteksi
       - Hitung dan tampilkan jumlah total jerawat
       - Simpan gambar hasil deteksi ke `outputs/`
       - Print: jumlah jerawat, confidence scores, dan inference time

       ```python
       from ultralytics import YOLO
       import argparse, time, os

       parser = argparse.ArgumentParser()
       parser.add_argument('--image', required=True, help='Path to input image')
       parser.add_argument('--model', default='models/acnevision.onnx', help='Model path')
       parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
       args = parser.parse_args()

       model = YOLO(args.model)
       start = time.time()
       results = model(args.image, conf=args.conf)
       inference_time = (time.time() - start) * 1000

       # Count and display
       detections = results[0].boxes
       acne_count = len(detections)
       print(f"\n{'='*40}")
       print(f"Acne spots detected: {acne_count}")
       print(f"Inference time: {inference_time:.1f}ms")
       print(f"{'='*40}")

       # Save annotated image
       os.makedirs('outputs', exist_ok=True)
       annotated = results[0].plot()
       output_path = f"outputs/result_{os.path.basename(args.image)}"
       import cv2
       cv2.imwrite(output_path, annotated)
       print(f"Saved: {output_path}")
       ```

    AVOID: Jangan hardcode confidence threshold — gunakan argparse agar adjustable.
  </action>
  <verify>
    ```
    # Test pada gambar dari test set
    python scripts/inference.py --image data/images/test/[any_test_image.jpg]
    dir outputs\
    ```
  </verify>
  <done>
    - Script menerima gambar input dan menghasilkan output dengan bounding box
    - Jumlah jerawat terdeteksi dan ditampilkan
    - Inference time tercatat
    - Gambar hasil tersimpan di outputs/
  </done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] ONNX model valid dan loadable
- [ ] Inference script berjalan end-to-end
- [ ] Bounding box tergambar pada jerawat yang terdeteksi
- [ ] Jumlah jerawat terhitung dengan benar
- [ ] Inference time < 100ms pada GPU
</verification>

<success_criteria>
- [ ] Model ONNX ter-export dan terverifikasi
- [ ] Inference pipeline functional
- [ ] Semua deliverable tercapai sesuai SPEC.md
</success_criteria>
