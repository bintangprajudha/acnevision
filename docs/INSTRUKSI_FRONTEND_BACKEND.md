# Instruksi Lanjutan untuk Tim Frontend dan Backend

> Dibuat 2026-11-19 untuk server model AcneVision (`scripts/serve.py`). Kontrak lengkap, contoh respons, dan kode error ada di
> [`API.md`](API.md). Acuan frontend: `keannnw/acnevision-mobile` commit `928f728`.

## 0. Yang sudah tersedia dari tim model

| Hal | Status |
|---|---|
| Model | **E9 = YOLOv8l @ 960 dilatih pada tile 2×2** (`runs/detect/yolov8l_960_tiles2/weights/best.pt`, sha256 `305dda97…`), model terbaik proyek. Conf 0.21 dipilih di val |
| Endpoint | `POST /api/v1/detect` (`mode=capture` = SAHI 1728, hasil resmi; `mode=preview` = plain, cepat), `GET /api/v1/health` |
| Format | Persis `DetectResponse` dan `ApiError` di `src/shared/types/api.ts`. Satu-satunya perbedaan dengan asumsi frontend: **bbox dalam piksel**. `idx` mulai 0, sama dengan frontend |
| Uji | 33/33 uji klien lolos. Capture identik dengan skrip prediksi resmi. Uji E2E di Chrome berhasil (lihat `API.md` §8) |
| Latensi lokal (RTX 4060) | capture foto penuh ±608 ms, preview 720p ±45 ms (tanpa tunnel) |
| Latensi lewat tunnel | 1 pengukuran dari PC server (2026-11-19 19:08): capture foto 0,9 MB **3,0 s** total, 0,9 s di antaranya di server. Dari HP belum diukur |
| Tidak disediakan | `/analyze`, Grad-CAM, `/scans`, `/chat`, `/me`, `/admin` |

### Koneksi ke server model

1. Tim model menjalankan `powershell -ExecutionPolicy Bypass -File scripts\run_server.ps1` dan membagikan URL yang dicetak skrip.
   Contoh sesi 2026-11-19: `https://careful-ruby-tests-gourmet.trycloudflare.com/api/v1`. Sesi ini **sementara**, minta URL terbaru sebelum menguji.
2. URL quick tunnel **berubah setiap kali server dinyalakan ulang**, dan Cloudflare kadang menolak pembuatannya (status 429).
   Selama belum ada named tunnel (B8), perlakukan URL sebagai konfigurasi yang bisa berubah. Jangan di-hard-code; selalu lewat `.env`.
3. Cek server hidup: buka `<URL>/health` di browser HP. Harus muncul `"status":"ok"` dan nama model E9.
4. Server hanya hidup selama PC tim model menyala dan jendela skrip terbuka. Sepakati jadwal uji dengan tim model.
5. **HP, laptop tim, dan PC server model berada di jaringan berbeda.** Selalu pakai URL tunnel `https://…/api/v1`.
   `127.0.0.1`, `localhost`, dan IP LAN PC server **tidak bisa** dijangkau dari perangkat lain. Tunnel Cloudflare bisa
   diakses dari jaringan mana pun (Wi-Fi lain maupun data seluler).
6. API key `/predict` (hanya untuk backend) sekarang **tetap**. Tim model memberikannya lewat jalur privat; lihat `API.md` §6.
7. Templat `.env` siap pakai ada di `handoff/frontend.env` (salin menjadi `.env` di repo mobile) dan `handoff/backend.env`.
   Isi rahasia (Supabase, API key) diisi masing-masing tim dan tidak di-commit.

---

## 1. Frontend (`acnevision-mobile`)

### Wajib (hasil salah atau tidak tampil tanpa ini)

| # | Tugas | File | Detail |
|---|---|---|---|
| F1 | **Normalisasi bbox** | `src/components/DetectedImage.tsx`, `src/pages/DetectResult.tsx` | Server mengirim **piksel** pada gambar setelah EXIF. Bagi koordinat dengan ukuran gambar, mis. `x1 / detection.image.width`, `y1 / detection.image.height`. Tanpa ini kotak tergambar di luar foto (terbukti: `left: 299980%`). Saran: satu helper `toNormalized(lesion, image)` di `src/lib/detection.ts`, dan mock ikut diubah ke piksel agar perilakunya sama |
| F2 | Pakai `.env` asli | `.env` | `VITE_USE_MOCK=false` dan `VITE_API_BASE_URL=https://<url-tunnel>/api/v1`. Jangan `http://` untuk APK (WebView `https://localhost` memblokir mixed content) |
| F3 | Label keparahan lengkap | `src/pages/DetectResult.tsx` | `very_severe` saat ini tampil "Sedang", dan `severity: null` (0 lesi) jatuh ke "Sedang". Pakai `SEVERITY_META[severity].label`; untuk `null` tampilkan "Tidak ada jerawat terdeteksi" |
| F4 | Hapus data palsu | `src/pages/DetectResult.tsx` | Hapus fallback `total ?? 18`, `severity ?? 'moderate'`, `det_confidence \|\| 0.88`, rincian Komedo/Papula/Pustula (55/33/sisa %), dan tanggal hard-coded. **Model hanya mendeteksi "Jerawat", tanpa jenis lesi** |
| F5 | Tangani error dengan benar | `src/pages/Scan.tsx` | Saat ini semua error tetap pindah ke `/detect-result` dan menampilkan hasil contoh. Tampilkan `err.message` di halaman Scan dan jangan navigasi bila gagal |

### Penting (kualitas hasil)

| # | Tugas | Detail |
|---|---|---|
| F6 | **Kirim foto resolusi penuh untuk capture** | `frameToBlob` mengambil frame video (ideal 1280×720). Pada ukuran ini SAHI tidak memotong dan lesi terlalu kecil: di val rata-rata deteksi turun dari 25,0 menjadi 14,6 per gambar (GT 28,1). Pakai `@capacitor/camera` (`getPhoto`, kualitas 90–95) atau `ImageCapture.takePhoto()`, dan minta resolusi maksimum. Arahkan pengguna agar wajah memenuhi bingkai dari jarak dekat. Upload dari galeri sudah benar |
| F7 | Ambang "kurang yakin" | `LOW_CONFIDENCE_THRESHOLD = 0.6` membuat **95,6%** kotak capture di val bergaris putus-putus. Tetapkan ambang baru bersama tim model (mis. dari sebaran skor), atau tampilkan confidence apa adanya tanpa label "kurang yakin" |
| F8 | Kode error baru | Tangani `SERVER_BUSY` (503) dan `RATE_LIMITED` (429). Baca `Retry-After` / `details.retry_after_s`, tampilkan pesan, lalu coba ulang otomatis sekali untuk capture |
| F9 | Batas waktu | Axios 60 detik sudah cukup (capture ±0,6–1 s di server). Tambahkan indikator proses karena latensi tunnel dari HP belum diukur |

### Opsional

| # | Tugas | Detail |
|---|---|---|
| F10 | Overlay live dengan model | Kotak "live" di `Scan.tsx` saat ini **simulasi** (landmark tetap, confidence karangan 82–98%). Bila ingin memakai model: kirim frame ±720p dengan `mode: 'preview'`, maks ±2 per detik dan **hanya satu request berjalan** pada satu waktu. Abaikan 503 (berarti GPU sedang capture). Batas server: 120 preview/menit per IP. Bila tidak dipakai, beri label jelas bahwa kotak live hanya panduan posisi, bukan deteksi |
| F11 | Jalur login | Pengguna login memanggil `/analyze` dulu (server model menjawab 404), lalu fallback ke `/detect`. Setelah backend tim punya `/analyze`, fallback ini tidak diperlukan |
| F12 | `upgrade` | Untuk pengguna ber-token server mengirim `upgrade: null`; pastikan UI tidak bergantung pada objek ini |
| F13 | Perbarui mock | Samakan mock dengan server: bbox piksel, `upgrade` null bila login, `model_info` |

### Cara uji frontend
1. Isi `.env` (F2), lalu `npm run dev`. Unggah foto di `/scan` dan pastikan kotak menempel pada jerawat (F1).
2. Untuk APK: `npm run build`, lalu `npx cap sync android` dan jalankan di HP. Cek `https://<url-tunnel>/api/v1/health` dari browser HP.
3. Uji 0 lesi (foto tanpa wajah atau wajah bersih) → keparahan `null` harus tampil benar (F3).

---

## 2. Backend tim (Supabase / FastAPI aplikasi)

| # | Tugas | Detail |
|---|---|---|
| B1 | **Validasi token** | Server model **tidak** memvalidasi `Authorization: Bearer`; ia hanya menandai `is_guest=false`. Jangan menjadikan `is_guest` dasar hak akses. Validasi JWT Supabase di backend tim |
| B2 | Pilih arsitektur akses model | **Rekomendasi:** aplikasi → backend tim → server model. Backend memvalidasi token, menyimpan scan, lalu meneruskan gambar ke `POST /api/v1/detect` (atau `/predict` dengan `X-API-Key` untuk jalur internal). Server model tetap tersembunyi di balik backend. Alternatif cepat untuk demo: aplikasi langsung ke URL tunnel (sudah berjalan, khusus jalur tamu) |
| B3 | `/analyze` dan klasifikasi | Tidak disediakan tim model (tim tidak mengerjakan klasifikasi 4 tingkat atau Grad-CAM). Bila `/analyze` dibuat, `lesions` dan `total_lesions` **harus** diambil dari `/detect` server model agar konsisten. Jangan membuat probabilitas atau Grad-CAM palsu |
| B4 | Penyimpanan scan | `/scans` disimpan backend tim. Simpan `detection_id`, `image.width/height`, `lesions` (piksel), `summary`, dan `model_info.detection.version` (sha256 bobot + mode) agar hasil bisa ditelusuri ke versi model |
| B5 | Ambang Hayashi | Bila backend punya `severity.py` sendiri (komentar di `classes.ts`), ambangnya wajib sama: ≤5 mild, ≤20 moderate, ≤50 severe, >50 very_severe, 0 lesi → `null` |
| B6 | Endpoint lain | `/chat`, `/me`, `/admin`, `/scans/trend`: sepenuhnya tugas backend tim |
| B7 | Statistik model (admin) | `ModelStats.active_models.detection` dapat diisi dari `GET /api/v1/health` (`model.name`, `model.weights_sha256`). Metrik resmi diambil dari `docs/SPEC.md` (test: F1 0.583, recall 0.557, precision 0.611 untuk seed 42 + SAHI 1728) |
| B8 | URL tetap | Quick tunnel berubah setiap kali dijalankan dan bisa ditolak Cloudflare (status 429). Untuk demo atau produksi, siapkan **named tunnel** dengan domain tim (langkah di `API.md` §2) atau hosting GPU sendiri |

---

## 3. Contoh implementasi

Contoh di bawah adalah **saran** dan belum diuji di repo mobile maupun repo backend. Sesuaikan dengan gaya kode masing-masing.

### 3.1 Frontend: bbox piksel ke 0–1 (F1)

```ts
// src/lib/detection.ts
import type { BBox, DetectedLesion } from '@acnevision/shared';

/** Server model mengirim bbox dalam piksel; komponen menggambar dalam persen (0-1). */
export function toNormalized(b: BBox, image: { width: number; height: number }): BBox {
  return { x1: b.x1 / image.width, y1: b.y1 / image.height, x2: b.x2 / image.width, y2: b.y2 / image.height };
}

export function normalizeLesions<T extends Pick<DetectedLesion, 'bbox'>>(
  lesions: T[], image: { width: number; height: number },
): T[] {
  return lesions.map((l) => ({ ...l, bbox: toNormalized(l.bbox, image) }));
}
```

Panggil satu kali saat hasil diterima, mis. di `setDetection` (`src/store/scan.ts`) sebelum disimpan:
`setDetection({ ...det, lesions: normalizeLesions(det.lesions, det.image) }, imageUrl)`. Dengan begitu `DetectedImage.tsx` dan
`DetectResult.tsx` tidak perlu diubah. Bila frontend memotong foto (`object-cover`), posisi kotak juga harus mengikuti potongannya;
lebih aman tampilkan foto utuh (`object-contain`).

### 3.2 Frontend: error dan coba ulang (F5, F8)

```ts
import { ApiRequestError } from '@acnevision/shared';

async function detectWithRetry(file: File) {
  try {
    return await api.detect(file, { source: 'mobile', input_method: 'camera', mode: 'capture' });
  } catch (e) {
    if (e instanceof ApiRequestError && (e.code === 'SERVER_BUSY' || e.code === 'RATE_LIMITED')) {
      const wait = Number(e.details?.retry_after_s ?? 2);
      await new Promise((r) => setTimeout(r, wait * 1000));
      return api.detect(file, { source: 'mobile', input_method: 'camera', mode: 'capture' });  // satu kali saja
    }
    throw e;   // tampilkan e.message di halaman Scan; JANGAN navigate('/detect-result')
  }
}
```

Pesan dari server sudah dalam Bahasa Indonesia dan aman ditampilkan apa adanya.

| `code` | Tampilan yang disarankan |
|---|---|
| `INVALID_IMAGE`, `FILE_TOO_LARGE`, `VALIDATION_ERROR` | Pesan server + tombol pilih foto lain |
| `SERVER_BUSY`, `RATE_LIMITED` | Coba ulang otomatis sekali, lalu pesan "coba lagi sebentar lagi" |
| `NETWORK_ERROR` (dari `client.ts`) | "Server model tidak dapat dihubungi"; kemungkinan tunnel mati atau URL sudah berganti |
| `INTERNAL_ERROR` | Pesan umum + laporkan ke tim model beserta jam kejadian |

### 3.3 Frontend: foto resolusi penuh (F6)

```ts
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';

const photo = await Camera.getPhoto({
  source: CameraSource.Camera, resultType: CameraResultType.Uri, quality: 92, correctOrientation: true,
});
const blob = await (await fetch(photo.webPath!)).blob();
const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
```

- Batas server **10 MiB** dan **50 MP**. Bila foto lebih besar, kecilkan sampai sisi terpanjang ±4000 px (JPEG q90) sebelum dikirim.
  Jangan sampai < 2000 px, karena lesi menjadi terlalu kecil.
- Wajah harus memenuhi sebagian besar bingkai. Data latih ACNE04 adalah foto wajah dari dekat.
- Kamera depan boleh tetap *mirrored*. Server tidak membalik gambar, jadi kotak cocok dengan file yang dikirim dan ditampilkan.

### 3.4 Frontend: live preview dengan model (F10, opsional)

```ts
let inFlight = false;
async function previewTick(video: HTMLVideoElement) {
  if (inFlight) return;                       // satu request pada satu waktu
  inFlight = true;
  try {
    const blob = await frameToBlob(video, { mirrored: true, quality: 0.8 });   // ±720p
    const res = await api.detect(new File([blob], 'frame.jpg', { type: 'image/jpeg' }), { mode: 'preview' });
    setLiveLesions(normalizeLesions(res.lesions, res.image));
  } catch { /* 503 = GPU sedang capture; abaikan frame ini */ }
  finally { inFlight = false; }
}
// panggil previewTick tiap 500 ms (maks 2/detik; batas server 120 preview/menit per IP)
```

Jumlah dari preview **hanya indikatif**. Angka resmi untuk hasil selalu dari `mode: 'capture'`.

### 3.5 Backend: meneruskan gambar ke server model (B2)

Contoh FastAPI + `httpx`, dengan token sudah divalidasi lebih dulu:

```python
import os, httpx
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

MODEL_URL = os.environ["ACNE_MODEL_URL"]          # mis. https://api.<domain>/api/v1 atau URL tunnel
router = APIRouter()

@router.post("/detect")
async def detect(image: UploadFile = File(...), user=Depends(verify_supabase_jwt)):   # B1
    data = await image.read()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{MODEL_URL}/detect",
                              files={"image": (image.filename, data, image.content_type)},
                              data={"source": "mobile", "input_method": "camera", "mode": "capture"},
                              headers={"Authorization": "Bearer internal"})   # agar is_guest=false dan upgrade=null
    if r.status_code != 200:
        # Teruskan ApiError apa adanya agar frontend tetap menerima {"error": {...}}
        raise HTTPException(status_code=r.status_code, detail=r.json().get("error"))
    result = r.json()
    # B4: simpan result["detection_id"], result["image"], result["lesions"], result["summary"],
    #     result["model_info"]["detection"]["version"] ke tabel scans milik user.id
    return result
```

- Semua request dari backend datang dari satu IP. Bila backend dan server model berjalan di mesin berbeda, beri tahu tim model agar
  batas `ACNEVISION_RATE_CAPTURE` dinaikkan, atau backend memakai `/predict` dengan `X-API-Key` (tidak dibatasi per IP, ikut antrean capture).
- `/predict` mengembalikan format internal (`count`, `detections[].box`), **bukan** `DetectResponse`. Gunakan `/detect` bila backend
  hanya meneruskan respons ke aplikasi.
- Jangan pernah mengirim `X-API-Key` ke aplikasi HP.

### 3.6 Backend: konversi `lesions` untuk `/analyze` (B3)

`AnalyzeResponse.lesions` memakai tipe yang sama (`Lesion = DetectedLesion`). Ambil langsung dari `/detect`, lalu
`summary.total_lesions = len(lesions)`. Isi `classification` hanya bila model klasifikasi tim benar-benar ada. Bila belum ada,
kembalikan `classification: null` (tipe mengizinkan `null`). Jangan mengarang `probabilities`.

---

## 4. Daftar periksa serah terima

| # | Siapa | Cek | Lulus bila |
|---|---|---|---|
| 1 | Frontend | `<URL>/health` dari browser HP | `"status":"ok"`, model E9 |
| 2 | Frontend | Unggah foto val dari galeri di `/scan` | Jumlah lesi sama dengan contoh tim model, dan kotak menempel pada jerawat |
| 3 | Frontend | Foto tanpa wajah atau kulit bersih | Tampil "tidak ada jerawat terdeteksi", bukan "Sedang" atau "18" |
| 4 | Frontend | Kirim file PDF/teks | Pesan "File harus berupa gambar JPEG, PNG, atau WEBP." di halaman Scan |
| 5 | Frontend | Matikan server lalu ambil foto | Pesan koneksi gagal, tidak ada hasil palsu |
| 6 | Frontend | Foto kamera depan | Kotak tidak terbalik kiri-kanan terhadap foto yang ditampilkan |
| 7 | Backend | Token palsu atau kedaluwarsa | Ditolak backend (401), tidak diteruskan sebagai pengguna |
| 8 | Backend | Simpan scan lalu buka riwayat | Kotak dan jumlah sama dengan saat scan; `model_info.version` tersimpan |
| 9 | Keduanya | Catat waktu capture dari HP (5 foto) | Laporkan ke tim model: waktu unggah vs `timing_ms.detection` |

Laporan bug ke tim model sertakan: jam kejadian, `detection_id` (bila ada), `code` error, ukuran file, dan jenis HP.

---

## 5. Yang tetap dikerjakan tim model

- Menjalankan server dan membagikan URL tunnel. Bila status 429, tunggu beberapa menit; skrip sudah mencoba ulang otomatis.
- Mengubah ambang conf atau model **hanya** lewat eksperimen baru dan `scripts/evaluate.py`; kontrak API tidak berubah.
- Usulan berikutnya (perlu persetujuan): TensorRT FP16 untuk mempercepat, dengan evaluasi ulang wajib.

## 6. Urutan kerja yang disarankan

1. Frontend F1–F5 (wajib), lalu uji dengan server lokal atau tunnel.
2. Backend B1–B2: putuskan apakah aplikasi lewat backend atau langsung ke server model.
3. Frontend F6–F8 (kualitas hasil kamera dan error), lalu uji di HP sungguhan.
4. Opsional F10 (live preview) setelah latensi tunnel dari HP diukur.
