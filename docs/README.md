# Dokumentasi AcneVision: Indeks & Aturan

## Isi

| File | Isi | Kapan diperbarui |
|---|---|---|
| `STATE.md` | Status terkini, model terpilih, langkah berikutnya | Setiap akhir sesi kerja yang mengubah status |
| `SPEC.md` | Dataset, model, **protokol evaluasi**, target | Hanya bila model terpilih/target/protokol berubah (keputusan pengguna) |
| `PIPELINE_MODEL_TERBAIK.md` | Penjelasan metode model terpilih (tile 2×2 + SAHI) per slide, untuk presentasi | Saat model terpilih atau angka hasilnya berubah |
| `API.md` | API lokal untuk aplikasi mobile: endpoint `/api/v1/detect`, kode error, mode preview/capture, `.env` frontend, hasil uji & benchmark, temuan kontrak | Saat `scripts/serve.py` atau kontrak frontend berubah |
| `INSTRUKSI_FRONTEND_BACKEND.md` | Tugas lanjutan tim frontend & backend agar aplikasi memakai server model dengan benar | Saat kontrak/temuan berubah |
| `PIPELINE.md` | Alur pipeline + audit kesesuaian standar + referensi (audit 2026-10-24, masih membahas model lama E3) | Bila pipeline berubah atau audit baru dilakukan |
| `report/` | Keluaran `scripts/report.py` dan `export_ranking.py`: `summary.json`, `ranking_models.csv`, `figures/` | Dibuat ulang oleh script, jangan diedit manual |
| `../reports/`, `../research_notes/` | Laporan riset benchmark literatur ACNE04 dan catatan mentahnya | Saat riset literatur baru dilakukan |
| `../splits/` | Daftar split train/val/test dan aturan konversi | Hanya bila data atau split berubah |
| `EXPERIMENTS.md` | Log semua run (hipotesis, perubahan, metrik, kesimpulan) | **Setiap** run selesai, termasuk yang gagal |
| `REPORT.md` | Laporan lengkap hasil model (+ `report/figures/`, dibuat `scripts/report.py`) | Setelah putaran eksperimen selesai |
| `PRESENTASI.md` | Rangkuman development + konten per slide untuk `presentasi/` (dibangun `presentasi/build_deck.js`) | Saat hasil/deck berubah |
| `../notebooks/AcneVision_Ringkasan.ipynb` | Notebook ringkasan: dataset, preprocessing, augmentasi, kurva training, evaluasi (P/R/mAP, CM, kecepatan), counting, contoh inferensi 3 varian. Hanya membaca `runs/`, jalan di CPU | Saat hasil eval/predict berubah (jalankan ulang semua sel) |
| `RESEARCH.md` | Riset teknik peningkatan & ambang metrik dari literatur | Saat riset baru dilakukan |
| `PANDUAN_END_TO_END.md` | Penjelasan edukatif end-to-end (preprocessing, training, tiap metrik) | Bila pipeline/protokol berubah |
| `ROADMAP.md` | Antrian eksperimen berprioritas | Saat eksperimen selesai/ditambah |
| `experiments/history/` | `results.csv` & `args.yaml` run lama/arsip | Saat bobot run diarsipkan |

## Aturan Penulisan

1. **Bahasa Indonesia**, Markdown, ringkas; tabel untuk angka.
2. **Setiap angka metrik menyebutkan sumbernya**: split, `imgsz`, `conf`, dan file
   (`runs/eval/<run>/metrics.json`). Jangan menyalin angka tanpa sumber.
3. **Satu sumber kebenaran per topik.** Jangan menduplikasi tabel metrik di banyak file;
   rujuk `EXPERIMENTS.md`.
4. Klaim eksternal (best practice, paper) wajib menyertakan URL di bagian Referensi.
5. Bedakan **fakta terverifikasi** dan **opini/hipotesis** (tandai *opini*).
6. Tanggal absolut (YYYY-MM-DD), bukan "kemarin" / "minggu lalu".
7. Dokumen tidak relevan dipindah ke `_to_delete/`, bukan dihapus permanen oleh agen.

## Template Entri Eksperimen

```markdown
### E<n>: `<nama_run>`
- **Hipotesis**: ...
- **Perubahan vs E<ref>**: (satu variabel utama)
- **Config**: `configs/<nama>.yaml`
- **Hasil (test)**: P / R / mAP@50 / mAP@50-95 / MAE (dari `runs/eval/<run>/metrics.json`)
- **Kesimpulan**: diterima / ditolak, alasan
```
