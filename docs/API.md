# API AcneVision untuk Aplikasi Mobile

> Dibuat 2026-11-19. Kode server: `scripts/serve.py`. Kontrak acuan: repo `keannnw/acnevision-mobile` commit `928f728`
> (`src/shared/types/api.ts`, `src/shared/api/client.ts`). Server ini adalah **server model**, bukan backend aplikasi.

## 1. Ringkasan

| Hal | Isi |
|---|---|
| Endpoint utama | `POST /api/v1/detect` (jalur tamu, **tanpa** API key), respons `DetectResponse` |
| Model | E9: YOLOv8l @ 960 dilatih pada tile 2×2, bobot `runs/detect/yolov8l_960_tiles2/weights/best.pt` (sha256 `305dda97…`), **conf 0.21** (dipilih di val) |
| Mode | `capture` (default) = SAHI 1728, hasil resmi; `preview` = plain 960, cepat, jumlah hanya indikatif |
| Tidak disediakan | `/analyze`, `/analyze/{id}/gradcam`, `/scans`, `/chat/*`, `/me`, `/admin/*` → 404 `NOT_FOUND`. Itu tugas backend tim. |
| Penyimpanan | Gambar **tidak disimpan**; log hanya mencatat mode, ukuran gambar, jumlah lesi, dan waktu |

## 2. Menjalankan server

Prasyarat: `.venv` proyek ini, bobot E9, dan `cloudflared` (`winget install --id Cloudflare.cloudflared`, lalu buka terminal baru).

```powershell
cd D:\bintang\acnevision
powershell -ExecutionPolicy Bypass -File scripts\run_server.ps1            # server + tunnel HTTPS
powershell -ExecutionPolicy Bypass -File scripts\run_server.ps1 -NoTunnel  # server lokal saja (http://127.0.0.1:8000)
```

Skrip menunggu model dimuat (±6 detik termasuk warm-up), membuka quick tunnel, lalu mencetak isi `.env` frontend:

```dotenv
VITE_USE_MOCK=false
VITE_API_BASE_URL=https://<acak>.trycloudflare.com/api/v1
```

- **HTTPS wajib untuk aplikasi HP.** `capacitor.config.ts` memakai `androidScheme: 'https'` (origin `https://localhost`), jadi API
  `http://` (mis. IP LAN) diblokir WebView sebagai *mixed content*. Tunnel memberi URL `https://`.
- Untuk Vite dev di PC yang sama (`http://localhost:5173`), `VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1` juga bisa.
- URL quick tunnel **berubah setiap kali** skrip dijalankan dan tanpa SLA. Untuk URL tetap dipakai *named tunnel* (butuh akun
  Cloudflare dan domain): `cloudflared tunnel login`, `cloudflared tunnel create acnevision`,
  `cloudflared tunnel route dns acnevision api.<domain>`, lalu `cloudflared tunnel run --url http://127.0.0.1:8000 acnevision`.
  *Belum diuji di proyek ini.*
- Log: `runs\serve\server.err.log` (aplikasi + uvicorn), `runs\serve\cloudflared.log`.

## 3. `POST /api/v1/detect`

Request `multipart/form-data` (sama dengan `client.ts`):

| Field | Wajib | Nilai | Catatan |
|---|---|---|---|
| `image` | ya | file JPEG / PNG / WEBP | ≤ 10 MiB (10 × 1024 × 1024 B, = `MAX_UPLOAD_BYTES`), ≤ 50 MP, sisi ≥ 32 px. Tipe dicek dari isi file (magic bytes), bukan dari header MIME |
| `mode` | tidak | `capture` (default) \| `preview` | lihat §4 |
| `source` | tidak | `web` (default) \| `mobile` | hanya dicatat di log |
| `input_method` | tidak | `upload` (default) \| `camera` | hanya dicatat di log |

Header opsional `Authorization: Bearer <token>`: bila ada token, `is_guest=false` dan `upgrade=null`. Token **tidak divalidasi**
(server ini tidak terhubung ke Supabase).

```bash
curl -F "image=@foto.jpg;type=image/jpeg" -F source=mobile -F input_method=camera -F mode=capture \
     https://<acak>.trycloudflare.com/api/v1/detect
```

Contoh respons 200 (gambar val `levle0_105.jpg`, `lesions` dipotong menjadi 2 dari 7):

```json
{
  "detection_id": "b293ec93-48de-4872-826f-c3c267e7fff1",
  "is_guest": true,
  "image": { "width": 3112, "height": 3456 },
  "lesions": [
    { "idx": 0, "bbox": { "x1": 1650.19, "y1": 927.78, "x2": 1708.13, "y2": 987.13 }, "det_confidence": 0.2764, "label": "Jerawat" },
    { "idx": 1, "bbox": { "x1": 2879.78, "y1": 2634.84, "x2": 2933.94, "y2": 2689.16 }, "det_confidence": 0.329, "label": "Jerawat" }
  ],
  "summary": { "total_lesions": 7, "severity": "moderate", "severity_is_estimate": true },
  "timing_ms": { "detection": 1026 },
  "model_info": { "detection": { "name": "AcneVision YOLOv8l tile 2x2 (E9)", "version": "305dda97 · SAHI 1728 · conf 0.21" } },
  "disclaimer": "Hasil ini merupakan analisis AI dan bukan diagnosis medis. Konsultasikan ke dokter kulit untuk penanganan yang tepat.",
  "advice": "Jika ada jerawat yang terasa nyeri, terasa dalam di bawah kulit, atau tidak membaik, konsultasikan ke dokter kulit.",
  "upgrade": {
    "message": "Masuk untuk melihat analisis lengkap.",
    "unlocks": ["Tingkat keparahan dari model AI", "Tingkat keyakinan AI", "Visualisasi Grad-CAM",
                "Rekomendasi perawatan dari AI", "Simpan dan riwayat analisis"]
  }
}
```

Arti field:

| Field | Isi |
|---|---|
| `image.width/height` | Ukuran gambar **setelah orientasi EXIF diterapkan**. Foto kamera depan yang dikirim dalam keadaan *mirrored* tidak di-un-mirror, sehingga kotak cocok dengan file yang dikirim klien |
| `lesions[].bbox` | **Koordinat piksel** `x1,y1,x2,y2` pada gambar di atas (keputusan proyek: mengikuti keluaran model), dibulatkan 2 desimal. Ubah ke 0–1 dengan `x / width`, `y / height` |
| `lesions[].idx` | Mulai **0**, urut posisi pusat kotak: atas→bawah, lalu kiri→kanan. UI menampilkan `idx + 1` |
| `lesions[].det_confidence` | Skor model (4 desimal), selalu ≥ 0.21 |
| `summary.severity` | Estimasi Hayashi dari jumlah lesi: 1–5 `mild`, 6–20 `moderate`, 21–50 `severe`, > 50 `very_severe`, **`null` bila 0 lesi**. Ambangnya identik dengan `SEVERITY_THRESHOLDS` di `classes.ts` (diuji 0..300) |
| `timing_ms.detection` | Waktu inferensi model terukur (ms), tanpa unggah, decode, atau antre. Rincian lengkap ada di header `Server-Timing` (`decode`, `queue`, `infer`, `total`) |
| `model_info.detection` | Nama model + 8 karakter sha256 bobot + mode inferensi + conf |
| `disclaimer`, `advice`, `upgrade` | Teks persis dari `COPY` di `copy.ts`. `upgrade` = `null` bila ada token |

## 4. Mode `preview` vs `capture`

| | `capture` (default) | `preview` |
|---|---|---|
| Inferensi | SAHI: potongan 1728 px (overlap 0.2), tiap potongan di-resize ke 960, NMS IoU 0.5 | Plain: seluruh gambar di-resize ke 960 |
| Status | **Hasil resmi** (F1 test 0.583 pada seed 42; `docs/SPEC.md`) | Hanya untuk overlay cepat; recall di bawah standar minimum SPEC |
| Saat GPU sibuk | Mengantre FIFO (maks 4 menunggu; lebih dari itu 503) | **Langsung ditolak** 503 `SERVER_BUSY` + `Retry-After: 1`, tidak pernah mengantre |

**Prioritas GPU**: hanya satu inferensi berjalan pada satu waktu. Preview ditolak bila GPU sibuk **atau** ada capture menunggu,
jadi capture selalu didahulukan. GPU tidak bisa dipreempsi, sehingga capture paling lama tertunda satu preview yang sedang
berjalan (±40 ms).

Panduan posisi wajah tetap dikerjakan di HP (MediaPipe). Server tidak memeriksa ada atau tidaknya wajah.

## 5. Kode error

Semua error memakai format `ApiError`: `{"error": {"code": "...", "message": "...", "details": {...}?}}`. Pesannya dalam Bahasa Indonesia.

| HTTP | `code` | Kapan |
|---|---|---|
| 400 | `INVALID_IMAGE` | Bukan JPEG/PNG/WEBP, file rusak, resolusi > 50 MP, atau sisi < 32 px. Pesan: "File harus berupa gambar JPEG, PNG, atau WEBP." (sama dengan mock) |
| 400 | `VALIDATION_ERROR` | Field `image` tidak ada, atau nilai `mode`/`source`/`input_method` tidak dikenal (`details.field`) |
| 413 | `FILE_TOO_LARGE` | File > 10 MiB. Pesan: "Ukuran file maksimal 10 MB." Body yang jauh lebih besar ditolak sebelum dibaca |
| 429 | `RATE_LIMITED` | Kuota per IP habis; header `Retry-After` dan `details.retry_after_s` (**kode baru**, belum ditangani khusus frontend) |
| 503 | `SERVER_BUSY` | Preview saat GPU sibuk, antrean capture penuh, atau memori GPU penuh; header `Retry-After` (**kode baru**) |
| 404 | `NOT_FOUND` | Endpoint tidak disediakan (mis. `/api/v1/analyze`) |
| 405 | `METHOD_NOT_ALLOWED` | Metode HTTP salah |
| 401 | `UNAUTHORIZED` | `/predict` internal tanpa API key atau dengan key yang salah |
| 500 | `INTERNAL_ERROR` | Error tak terduga; traceback ada di log server |

## 6. Endpoint lain

- `GET /health` dan `GET /api/v1/health`: status, nama model, sha256 bobot, conf, `gpu.busy`, `gpu.waiting_captures`, dan batas
  upload/rate. Tidak memuat path lokal.
- `POST /predict` dan `POST /predict/image`: untuk keperluan internal, wajib header `X-API-Key` (env `ACNEVISION_API_KEY`; bila
  kosong dibuat acak dan dicetak saat startup). Field: `image`, `mode` (`plain`|`sahi`), `conf`. Ikut antrean capture.
- `/docs` (Swagger) mati secara bawaan; aktifkan dengan `ACNEVISION_DOCS=1`.

## 7. Keamanan dan batasan

| Lapisan | Perilaku | Env |
|---|---|---|
| Rate limit per IP | Sliding window: capture 10 per 60 detik, preview 120 per 60 detik (kuota terpisah) | `ACNEVISION_RATE_CAPTURE`, `ACNEVISION_RATE_PREVIEW` (format `jumlah/detik`) |
| IP klien | Dari `CF-Connecting-IP` (atau `X-Forwarded-For`), **hanya** bila koneksi berasal dari loopback (cloudflared lokal). Request lokal langsung tanpa header itu (tes/benchmark) tidak dibatasi | |
| Antrean capture | Maks 4 menunggu | `ACNEVISION_MAX_QUEUE` |
| CORS | `https://localhost` (Android WebView), `capacitor://localhost` (iOS), `http://localhost:5173` (Vite dev); tanpa credentials | `ACNEVISION_CORS_ORIGINS` (dipisah koma) |
| Ukuran dan tipe | ≤ 10 MiB, magic bytes JPEG/PNG/WEBP, ≤ 50 MP | |
| Bind | `127.0.0.1` saja; dunia luar hanya lewat tunnel | |

Keterbatasan yang perlu diketahui tim:
- **`is_guest` tidak bisa dipercaya**: siapa pun bisa mengirim `Bearer` palsu. Dampaknya hanya `is_guest=false` dan `upgrade=null`,
  tanpa data tambahan. Validasi token Supabase adalah tugas backend tim.
- Siapa pun yang tahu URL tunnel bisa memakai `/detect` (dibatasi rate limit). Tidak ada autentikasi di jalur tamu sesuai kontrak.
- Rate limit dan antrean disimpan di memori: reset bila server dimulai ulang, dan hanya berlaku untuk satu proses.
- Request pertama setelah server lama menganggur bisa lebih lambat: teramati 880 ms dan 1026 ms untuk capture 3112×3456,
  dibandingkan rata-rata 574 ms saat beban berurutan.

## 8. Hasil uji (2026-11-19, RTX 4060, hanya gambar val)

- **`scripts/test_client.py`: 33/33 PASS** (`runs/serve/test_client_20261119_184450.json`). Cakupan:
  - bentuk respons persis `DetectResponse`, Bearer, dan EXIF (orientation 6 → lebar/tinggi tertukar)
  - preview saat GPU sibuk → 503 dalam 3 ms sementara capture tetap 200
  - file bukan gambar, JPEG rusak, 10 MiB + 1 B, 12 MiB, mode salah, `image` hilang, `/api/v1/analyze`, `/predict` tanpa key
  - CORS (3 origin diizinkan, 1 ditolak), ambang Hayashi, dan rate limit (request ke-121 → 429; kuota capture terpisah)
- **Konsistensi** `mode=capture` vs `predict.py --sahi --slice 1728 --conf 0.21` pada 3 gambar val (`levle0_105`, `levle1_107`,
  `levle2_109`): jumlah identik (7, 14, 113). Selisih maksimum 0.005 px dan 5e-5 skor, yaitu pembulatan JSON (2 dan 4 desimal).
  Kedua jalur memakai fungsi yang sama (`predict.sahi_predict`) dan decoder yang sama (OpenCV, EXIF diterapkan).
- **E2E di Chrome** (Vite dev `acnevision-mobile@928f728`, `VITE_USE_MOCK=false`, unggah `levle1_107.jpg` di `/scan`):
  - Request sukses, dan halaman hasil menampilkan "14 jerawat" dan "Sedang" (sama dengan respons server).
  - 14 kotak dibuat di DOM, tetapi dengan `left: 299980%` dan seterusnya, sehingga **tidak terlihat**. Lihat temuan 1.
- PNG dan WEBP 720×800 → 200.

## 9. Benchmark latensi (`scripts/bench_serve.py`, 120 gambar val, berurutan, HTTP lokal tanpa tunnel)

Sumber: `runs/serve/bench_20261119_184632.json`. Latensi klien mencakup unggah lokal, decode, inferensi, dan JSON.
Latensi tunnel/jaringan HP **belum diukur**.

| Skenario | Ukuran gambar | Klien mean / p50 / p95 (ms) | Inferensi server mean / p95 (ms) | Gambar/detik (berurutan, terukur) |
|---|---|---|---|---|
| `capture` foto penuh | 3112×3456 (113), 4912×3264 (6), 510×796 (1) | 608 / 602 / 632 | 574 / 590 | 1.65 |
| `preview` frame 720p (sisi pendek 720, JPEG q92) | 720×800, 1084×720, … | 45 / 45 / 46 | 40 / 41 | 22.3 |
| `capture` frame 720p (jalur kamera `Scan.tsx`) | sama | 48 / 48 / 50 | 42 / 44 | 20.9 |

Diagnostik pada val. **Ini bukan metrik resmi**; metrik resmi hanya dari `scripts/evaluate.py`.

| Skenario | Deteksi rata-rata per gambar (GT 28.1) | Deteksi dengan `det_confidence` < 0.6 |
|---|---|---|
| `capture` foto penuh | 25.0 | **95.6%** (2.871 dari 3.003) |
| `preview` 720p | 17.1 | 96.6% |
| `capture` 720p | 14.6 | 96.0% |

Usulan percepatan, **belum dijalankan** dan perlu persetujuan:
- Ekspor TensorRT FP16 (`model.export(format="engine", half=True)`) dapat mempercepat inferensi plain maupun potongan SAHI.
- FP16 mengubah skor, sehingga jumlah lesi pada conf 0.21 bisa bergeser. Bila dipakai, wajib dievaluasi ulang lewat
  `evaluate.py` dan uji konsistensi diulang.

## 10. Temuan perbedaan kontrak (untuk tim frontend; tidak diubah oleh server)

1. **Format `bbox`**: `DetectedImage.tsx` dan `DetectResult.tsx` memakai `left: x1 * 100%`, yaitu menganggap koordinat 0–1, sedangkan
   server mengirim **piksel**. Akibatnya kotak tergambar jauh di luar foto (terbukti di uji Chrome). Solusi di frontend: bagi dengan
   `image.width` / `image.height`.
2. **`mode=preview` tidak pernah dikirim** frontend. Kotak "live" di `Scan.tsx` adalah simulasi: kotak ditempatkan pada titik
   landmark wajah tetap dengan confidence 82–98% yang dikarang, **bukan** keluaran model.
3. **Jalur kamera mengirim frame video** (`frameToBlob`, ideal 1280×720), bukan foto resolusi penuh.
   - Pada 720p, SAHI 1728 tidak memotong (gambar lebih kecil dari satu potongan), sehingga capture ≈ plain.
   - Lesi juga jauh lebih kecil daripada di foto ACNE04. Di val, rata-rata deteksi turun dari 25.0 (foto penuh) menjadi
     14.6 per gambar (GT 28.1).
   - Disarankan memakai foto resolusi penuh (mis. `ImageCapture.takePhoto()` / plugin Camera Capacitor) dari jarak dekat.
4. **`LOW_CONFIDENCE_THRESHOLD = 0.6`** vs conf model 0.21: **95.6%** deteksi capture pada val akan ditandai "kurang yakin"
   (garis putus-putus), sehingga penanda itu kehilangan makna. Ambang UI perlu ditetapkan ulang oleh tim.
5. **`DetectResult.tsx`** (halaman hasil tamu dan, lewat `Result.tsx`, juga `/analysis/:id`):
   - `very_severe` ditampilkan "Sedang", dan `severity: null` (0 lesi) jatuh ke "Sedang".
   - Bila request gagal, halaman menampilkan nilai contoh: 18 jerawat dan foto contoh. `Scan.tsx` tetap pindah ke
     `/detect-result` saat error.
   - Rincian Komedo/Papula/Pustula **dikarang** dari total (55% / 33% / sisa), padahal model hanya mendeteksi "Jerawat"
     tanpa jenis.
   - Tanggal hard-coded "8 Oktober 2026 • 08.42".
   - Gambar `object-cover object-top` membuat kotak (dalam %) tidak sejajar dengan foto yang terpotong.
6. **Kode `SERVER_BUSY` dan `RATE_LIMITED`** belum ditangani khusus. `Scan.tsx` menelan semua error dan menampilkan pesan umum.
7. **`upgrade`**: mock selalu mengirim objek; server mengirim `null` untuk pengguna ber-token.
8. **Jalur login** di `Scan.tsx` memanggil `/analyze` dulu. Server ini menjawab 404, lalu frontend fallback ke `/detect`
   (satu request tambahan).
9. `.env.example` memakai `http://localhost:8000/api/v1`. Alamat ini tidak bisa dipakai dari aplikasi HP (lihat §2).
