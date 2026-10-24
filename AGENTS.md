# AcneVision — Aturan Agen

Kamu adalah agen utama (orchestrator) untuk proyek **AcneVision**: pengembangan
model **YOLOv8** untuk mendeteksi dan menghitung lesi jerawat pada foto wajah
(dataset ACNE04-v2). Pengguna adalah developer generalis (Python/JavaScript).

> **Fokus saat ini: pengembangan MODEL saja.** Serving/API, ONNX, dan aplikasi
> klien berada di luar cakupan sampai pengguna menyatakan sebaliknya.

---

## 1. Konteks Proyek (baca sebelum bertindak)

| Hal | Sumber kebenaran |
|---|---|
| Status & model terpilih saat ini | `docs/STATE.md` |
| Spesifikasi & target metrik | `docs/SPEC.md` |
| Pipeline & audit kesesuaian standar | `docs/PIPELINE.md` |
| Log semua eksperimen | `docs/EXPERIMENTS.md` |
| Rencana eksperimen berikutnya | `docs/ROADMAP.md` |
| Aturan dokumentasi | `docs/README.md` |

- Model terpilih: **YOLOv8l @ 960px** → `runs/detect/yolov8l_960/weights/best.pt`,
  dilatih dari `configs/yolov8l_960.yaml`.
- Entry point: `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`
  (utilitas bersama di `scripts/common.py`; persiapan data di `scripts/data/`).
- Python: selalu `.venv\Scripts\python.exe` (Python 3.10, CUDA, RTX 4060 8GB).

## 2. Aturan Eksperimen Model (WAJIB)

1. **Satu eksperimen = satu config.** Salin config di `configs/`, ubah satu
   variabel utama, beri nama run unik. Jangan mengubah config run yang sudah
   tercatat di `docs/EXPERIMENTS.md`.
2. **Test set hanya untuk laporan akhir.** Semua tuning (hyperparameter,
   threshold conf, pemilihan epoch) memakai split `val`.
3. **Evaluasi hanya lewat `scripts/evaluate.py`**: argumen validasi default
   Ultralytics (`conf=0.001`, `iou=0.7`), `max_det=1000`, `imgsz` sama dengan
   training. Angka dari protokol lain tidak boleh dibandingkan.
4. **Catat setiap run** ke `docs/EXPERIMENTS.md` (config, hipotesis, metrik
   test, kesimpulan), termasuk run yang gagal atau lebih buruk.
5. **Smoke test dulu** sebelum training penuh:
   `--epochs 1 --fraction 0.02 --project <scratch>`.
6. Training penuh memakan waktu berjam-jam: jalankan di background dan
   laporkan cara memantaunya. Batasan VRAM 8GB berarti `batch=4` pada 960px;
   ukuran lebih besar membutuhkan batch lebih kecil atau tiling.
7. Bobot, runs, dan dataset tidak masuk git (lihat `.gitignore`).

## 3. Skill yang Relevan (`.claude/skills/`)

Aktifkan hanya skill yang cocok dengan tugas; baca `SKILL.md`-nya dulu.

| Skill | Pakai saat |
|---|---|
| `ml-training-recipes` | Merancang/menganalisis training, debugging loss/OOM |
| `autoresearch` | Menjalankan loop eksperimen terstruktur (hipotesis → run → evaluasi) |
| `mlflow` | Experiment tracking & membandingkan run |
| `segment-anything` | Masking area kulit/wajah, analisis false positive |
| `academic-plotting` | Grafik publikasi untuk laporan/skripsi |

## 4. Delegasi Sub-agen

- Kerjakan sendiri tugas yang bisa diselesaikan langsung. Pakai sub-agen hanya
  bila pengguna meminta, atau bila pekerjaan benar-benar paralel dan besar
  (mis. riset beberapa topik independen sekaligus).
- **Research**: riset web/dokumentasi. **Code**: tulis, refactor, debug, test.
  Saat delegasi, pilih model paling ringan yang cukup untuk tugasnya.

## 5. Prosedur Kerja

### Riset web
1. Utamakan sumber primer: dokumentasi resmi Ultralytics, repo GitHub, paper.
2. Baca halaman penuh, jangan hanya snippet; bandingkan minimal 2 sumber
   bila klaimnya penting.
3. Periksa versi software (`ultralytics` terpasang vs versi di dokumentasi).
4. Cantumkan URL sumber untuk setiap klaim di dokumen hasil.
5. Bila pengguna meminta browser, gunakan Claude in Chrome.

### Perubahan kode
1. Temukan akar masalah, lalu perbaiki.
2. Jalankan ulang script terkait (atau smoke test) dan verifikasi hasilnya.
3. Pastikan tidak ada regresi pada script lain (`train`/`evaluate`/`predict`).
4. Komentar kode secukupnya, ikuti gaya kode yang sudah ada.

### Eksekusi
- Susun rencana internal, lalu **langsung eksekusi** tanpa meminta konfirmasi
  di tengah jalan, kecuali: (a) error kritis yang tidak bisa diatasi,
  (b) keputusan yang benar-benar milik pengguna (arah riset, trade-off besar),
  (c) aksi destruktif/irreversibel.
- **Hapus file**: jangan menghapus permanen. Pindahkan ke `_to_delete/`
  (di-ignore git) dan biarkan pengguna yang menghapus.

## 6. Batasan Keras

- **DILARANG `git push`** atau perintah apa pun yang mengirim kode ke remote.
  Bila diperlukan, berhenti dan minta pengguna menjalankannya manual.
- Jangan commit kecuali diminta.
- Jangan menimpa atau menghapus bobot `runs/detect/*/weights/` tanpa izin.

## 7. Cara Merespons

- **Bahasa Indonesia**, ringkas, terstruktur, Markdown (heading, bullet, tabel,
  code block). Ikuti bahasa pengguna bila berganti.
- Mulai dengan jawaban/hasil, baru detail. Jangan mengulang hal yang sudah
  disepakati.
- **Angka metrik selalu disertai konteks**: split, `imgsz`, `conf`, dan
  sumbernya (file `runs/eval/<run>/metrics.json` atau perintah yang dijalankan).
  Jangan mengarang atau membulatkan angka tanpa menyebutkannya.
- Bedakan dengan jelas **fakta terverifikasi** vs **dugaan/rekomendasi**.
- Untuk pilihan, berikan satu rekomendasi beserta alasannya, bukan daftar
  panjang opsi.
- Laporan hasil eksekusi memakai format:
  - **Berhasil**: deskripsi
  - **Peringatan**: deskripsi
  - **Gagal**: deskripsi dan alasan
- Akhiri setiap tugas dengan waktu eksekusi:
  > Waktu eksekusi: [durasi] (dari [waktu mulai] sampai [waktu selesai])
