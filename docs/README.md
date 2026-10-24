# Dokumentasi AcneVision: Indeks & Aturan

## Isi

| File | Isi | Kapan diperbarui |
|---|---|---|
| `STATE.md` | Status terkini, model terpilih, langkah berikutnya | Setiap akhir sesi kerja yang mengubah status |
| `SPEC.md` | Dataset, model, **protokol evaluasi**, target | Hanya bila model terpilih/target/protokol berubah (keputusan pengguna) |
| `PIPELINE.md` | Alur pipeline + audit kesesuaian standar + referensi | Bila pipeline berubah atau audit baru dilakukan |
| `EXPERIMENTS.md` | Log semua run (hipotesis, perubahan, metrik, kesimpulan) | **Setiap** run selesai, termasuk yang gagal |
| `REPORT.md` | Laporan lengkap hasil model (+ `report/figures/`, dibuat `scripts/report.py`) | Setelah putaran eksperimen selesai |
| `PRESENTASI.md` | Rangkuman development + konten per slide untuk `presentasi/` (dibangun `presentasi/build_deck.js`) | Saat hasil/deck berubah |
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
