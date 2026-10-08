---
phase: 1
plan: 2
wave: 1
depends_on: []
files_modified:
  - scripts/download_dataset.py
  - scripts/convert_to_yolo.py
  - data/data.yaml
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Gambar ACNE04 asli ter-download dari sumber eksternal (HuggingFace/Kaggle)"
    - "File anotasi Acne04-v2_annotations.json ter-download dari GitHub"
    - "Anotasi lingkaran (cx, cy, r) terkonversi ke format YOLOv8 bbox (class x_center y_center w h)"
    - "Hanya 1.204 gambar yang listed di JSON yang diproses (bukan 1.457)"
    - "Split train/valid/test tersedia dengan fixed seed"
    - "data.yaml valid dan mengarah ke path yang benar"
  artifacts:
    - "data/images/{train,val,test}/ berisi gambar per split"
    - "data/labels/{train,val,test}/ berisi .txt YOLO per split"
    - "data/data.yaml berisi konfigurasi dataset YOLOv8"
---

# Plan 1.2: Dataset Download & YOLO Conversion (REVISED)

> **REVISED** berdasarkan riset mendalam repo ACNE04-v2.
> Temuan kritis: gambar tidak ada di repo, anotasi berformat lingkaran (bukan bbox),
> tidak ada yolo_scripts/, tidak ada split bawaan.

<objective>
Download gambar ACNE04 asli + anotasi ACNE04-v2 dari GitHub, konversi
anotasi lingkaran ke format YOLOv8 bounding box, dan siapkan struktur
direktori siap training.

Purpose: Dataset adalah input utama — training tidak bisa dimulai tanpa data terstruktur.
Output: Folder `data/` dengan struktur YOLOv8 dan `data.yaml`.
</objective>

<context>
Load for context:
- docs/SPEC.md
- docs/DATASET_RESEARCH.md
- https://github.com/AIpourlapeau/acne04v2 (README)
- Format anotasi: {"annotation_id", "image_id", "center": [cx, cy], "radius": r}
- Format gambar: {"image_id", "file_name", "height", "width"}
</context>

<tasks>

<task type="auto">
  <name>Download gambar ACNE04 dan anotasi ACNE04-v2</name>
  <files>scripts/download_dataset.py</files>
  <action>
    1. Buat script `scripts/download_dataset.py` yang:

       **Step A — Download anotasi ACNE04-v2:**
       - Clone atau download `Acne04-v2_annotations.json` dari
         https://github.com/AIpourlapeau/acne04v2
       - Simpan ke `data/raw/Acne04-v2_annotations.json`

       **Step B — Download gambar ACNE04 asli:**
       - Sumber primer: Hugging Face `Layered-Labs/ACNE04` (direct download)
         atau Kaggle mirror dataset ACNE04
       - Alternatif: Google Drive dari repo `xpwu95/LDL` (Classification.tar)
       - Simpan gambar ke `data/raw/images/`
       - Gunakan `huggingface_hub` atau `requests` untuk download

       **Step C — Verifikasi dan filter:**
       - Parse JSON, extract semua `file_name` dari array `images`
       - Verifikasi bahwa gambar yang listed di JSON ada di folder download
       - Print summary: X gambar ditemukan dari Y yang dibutuhkan
       - PENTING: Hanya 1.204 dari 1.457 gambar ACNE04 yang digunakan di v2

    2. Jalankan script.

    AVOID: Jangan download ke root project — simpan di `data/raw/`.
    AVOID: Jangan asumsikan semua 1.457 gambar ACNE04 digunakan — filter hanya
           yang ada di Acne04-v2_annotations.json (1.204 gambar).
    NOTE: File name di JSON mungkin memiliki typo bawaan (misal `levle0` bukan `level0`).
          Jaga nama file persis seperti di JSON.
  </action>
  <verify>
    ```
    python scripts/download_dataset.py
    python -c "import json; d=json.load(open('data/raw/Acne04-v2_annotations.json')); print(f'Images: {len(d[\"images\"])}, Annotations: {len(d[\"annotations\"])}')"
    dir data\raw\images\ /b | find /c /v ""
    ```
  </verify>
  <done>
    - `data/raw/Acne04-v2_annotations.json` exists dengan 1.204 images dan 32.443 annotations
    - `data/raw/images/` berisi gambar-gambar ACNE04
    - Minimal 1.100+ gambar dari 1.204 yang listed berhasil ditemukan
  </done>
</task>

<task type="auto">
  <name>Konversi anotasi lingkaran ke format YOLOv8 dan split dataset</name>
  <files>scripts/convert_to_yolo.py, data/data.yaml</files>
  <action>
    1. Buat script `scripts/convert_to_yolo.py` yang:

       **Step A — Parse JSON:**
       - Load `data/raw/Acne04-v2_annotations.json`
       - Build mapping: image_id → {file_name, width, height}
       - Build mapping: image_id → [list of annotations]

       **Step B — Konversi circle → YOLO bbox:**
       Untuk setiap anotasi dengan center=[cx, cy] dan radius=r,
       pada gambar dengan width=W dan height=H:

       ```python
       # Boundary clipping (jerawat di tepi gambar)
       xmin = max(0, cx - r)
       ymin = max(0, cy - r)
       xmax = min(W, cx + r)
       ymax = min(H, cy + r)

       # YOLO format (normalized 0-1)
       box_w = xmax - xmin
       box_h = ymax - ymin
       x_center_norm = (xmin + box_w / 2) / W
       y_center_norm = (ymin + box_h / 2) / H
       w_norm = box_w / W
       h_norm = box_h / H

       # Label line: "0 x_center y_center width height"
       label_line = f"0 {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
       ```

       PENTING: Gunakan boundary clipping (max/min) untuk mencegah koordinat keluar batas gambar!

       **Step C — Split train/val/test:**
       - Gunakan random split dengan fixed seed (seed=42):
         80% train / 10% val / 10% test
       - Split berdasarkan IMAGE (bukan annotation) agar tidak ada data leakage
       - Shuffle image list sebelum split

       **Step D — Buat struktur folder YOLOv8:**
       ```
       data/
       ├── images/
       │   ├── train/   # copy/move gambar
       │   ├── val/
       │   └── test/
       ├── labels/
       │   ├── train/   # tulis .txt YOLO labels
       │   ├── val/
       │   └── test/
       └── data.yaml
       ```

       **Step E — Print statistik:**
       - Jumlah gambar per split
       - Jumlah total anotasi per split
       - Rata-rata anotasi per gambar
       - Min/max radius (untuk analisis small object)

    2. Buat `data/data.yaml`:
       ```yaml
       path: D:/bintang/acnevision/data
       train: images/train
       val: images/val
       test: images/test
       nc: 1
       names: ['acne']
       ```
       NOTE: Gunakan absolute path di data.yaml agar tidak ambigu saat training.

    3. Jalankan script konversi.

    AVOID: Jangan lupa normalize koordinat ke range 0-1 (relatif terhadap W dan H gambar).
    AVOID: Jangan split per-annotation — split per-IMAGE untuk menghindari data leakage.
    AVOID: Jangan abaikan boundary clipping — jerawat di tepi gambar akan menghasilkan
           koordinat negatif atau >1 tanpa clipping.
    NOTE: Nama file label .txt harus sama dengan nama file gambar (tanpa ekstensi).
          Contoh: gambar `levle0_1.jpg` → label `levle0_1.txt`
  </action>
  <verify>
    ```
    python scripts/convert_to_yolo.py

    # Verifikasi beberapa label file — setiap baris harus format "0 x y w h"
    type data\labels\train\*.txt | findstr "^0 " | more

    # Verifikasi semua koordinat dalam range 0-1
    python -c "
    import glob
    for f in glob.glob('data/labels/**/*.txt', recursive=True)[:10]:
        for line in open(f):
            parts = line.strip().split()
            vals = [float(v) for v in parts[1:]]
            assert all(0 <= v <= 1 for v in vals), f'Out of range in {f}: {vals}'
    print('All checked labels have valid coordinates (0-1)')
    "

    # Verifikasi jumlah file
    python -c "
    import glob
    for split in ['train', 'val', 'test']:
        imgs = len(glob.glob(f'data/images/{split}/*'))
        lbls = len(glob.glob(f'data/labels/{split}/*'))
        print(f'{split}: {imgs} images, {lbls} labels, match={imgs==lbls}')
    "

    # Verifikasi data.yaml
    type data\data.yaml
    ```
  </verify>
  <done>
    - `data/images/{train,val,test}/` berisi gambar
    - `data/labels/{train,val,test}/` berisi .txt YOLO (setiap baris: `0 x y w h`)
    - `data/data.yaml` valid dengan path absolut
    - Jumlah file gambar == jumlah file label per split
    - Semua koordinat dalam range 0.0-1.0
    - Split: ~960 train / ~120 val / ~120 test (berdasarkan 1.204 gambar)
    - Statistik printed: avg ~27 anotasi/gambar
  </done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] Struktur folder `data/` lengkap (images + labels + data.yaml)
- [ ] Setiap .txt label berisi baris format `0 x_center y_center width height`
- [ ] Semua koordinat dalam range 0.0-1.0 (boundary clipping bekerja)
- [ ] Jumlah gambar == jumlah label per split
- [ ] Total ~1.204 gambar dan ~32.443 anotasi terproses
- [ ] Tidak ada gambar ACNE04 yang bukan bagian dari v2 annotations
</verification>

<success_criteria>
- [ ] Dataset siap untuk training YOLOv8
- [ ] data.yaml mengarah ke path yang benar
- [ ] ~960+ gambar training tersedia
- [ ] Konversi circle→bbox terverifikasi benar
</success_criteria>
