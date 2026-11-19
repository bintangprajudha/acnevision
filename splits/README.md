# Split ACNE04-v2 (AcneVision)

`acne04v2_split.csv` = daftar file per split (kolom `filename,split,n_lesions`), dibuat dari `data/images/{train,val,test}`.

- Sumber gambar: ACNE04 (Wu et al., ICCV 2019), subset 1.204 gambar; anotasi ACNE04-v2 (Gazeau et al., MICCAI 2024, repo `AIpourlapeau/acne04v2`).
- Split acak seed 42 (`scripts/data/convert_to_yolo.py`): **963 train / 120 val / 121 test**.
- Konversi lingkaran → box: `xmin = max(0, cx−r)`, `xmax = min(W, cx+r)`, sama untuk y (dipotong ke batas gambar); box berluas nol dibuang.
- Jumlah box di label YOLO: train 25.521, val 3.366, test 3.507 (total 32.394). JSON v2 memuat 32.443 lesi.
  Selisih 49 belum diverifikasi penyebabnya (dugaan: box berluas nol yang dibuang).
- ACNE04-v2 tidak punya split resmi; angka dari split ini tidak boleh dibandingkan langsung dengan paper lain.
