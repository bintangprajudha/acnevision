---
phase: 1
plan: 1
wave: 1
depends_on: []
files_modified:
  - requirements.txt
  - scripts/setup_env.py
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Python virtual environment aktif dengan semua dependensi terinstall"
    - "ultralytics, torch, torchvision, onnx, onnxruntime semua terinstall"
    - "GPU CUDA terdeteksi oleh PyTorch"
  artifacts:
    - "requirements.txt exists di root project"
    - "Virtual environment .venv/ aktif dan functional"
---

# Plan 1.1: Environment Setup & Dependencies

<objective>
Setup Python virtual environment dan install semua dependensi yang diperlukan
untuk training YOLOv8s object detection.

Purpose: Foundation — tanpa environment yang benar, tidak ada yang bisa dijalankan.
Output: Virtual environment `.venv/` dengan semua packages + `requirements.txt`.
</objective>

<context>
Load for context:
- docs/SPEC.md
</context>

<tasks>

<task type="auto">
  <name>Create requirements.txt dan setup virtual environment</name>
  <files>requirements.txt</files>
  <action>
    1. Buat `requirements.txt` dengan dependensi berikut:
       ```
       ultralytics>=8.2.0
       torch>=2.0.0
       torchvision>=0.15.0
       onnx>=1.14.0
       onnxruntime>=1.15.0
       opencv-python>=4.8.0
       matplotlib>=3.7.0
       seaborn>=0.12.0
       pandas>=2.0.0
       Pillow>=10.0.0
       tqdm>=4.65.0
       PyYAML>=6.0
       ```
    2. Buat virtual environment: `python -m venv .venv`
    3. Activate dan install: `.venv\Scripts\activate && pip install -r requirements.txt`

    AVOID: Jangan install torch CPU-only. Gunakan `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` untuk CUDA support pada RTX 4060.
    AVOID: Jangan gunakan conda — user sudah punya Python 3.10 sistem dan kita butuh venv ringan.
  </action>
  <verify>
    ```
    .venv\Scripts\python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
    .venv\Scripts\python -c "from ultralytics import YOLO; print('Ultralytics OK')"
    ```
  </verify>
  <done>
    - `requirements.txt` ada di root project
    - `.venv/` tercipta dan aktif
    - PyTorch mendeteksi CUDA + RTX 4060
    - `from ultralytics import YOLO` berhasil tanpa error
  </done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] `pip list` menunjukkan ultralytics, torch, torchvision terinstall
- [ ] `torch.cuda.is_available()` returns True
- [ ] `torch.cuda.get_device_name(0)` returns "NVIDIA GeForce RTX 4060"
</verification>

<success_criteria>
- [ ] Virtual environment functional dengan GPU support
- [ ] Semua packages terinstall tanpa conflict
</success_criteria>
