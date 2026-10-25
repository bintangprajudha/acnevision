// AcneVision - Deck presentasi (sidang/seminar), 17 slide, 4 model dibandingkan
// ==============================================================================
// Konten mengikuti docs/PRESENTASI.md. Angka dibaca dari artefak evaluasi
// (docs/report/summary.json, bootstrap.json, runs/eval/*/metrics.json);
// gambar dari docs/report/figures (scripts/report.py, scripts/diagrams.py) dan presentasi/assets.
// Model yang dibandingkan: B0 baseline, E3 tuned (plain), E4 + SAHI (terbaik), E5 + SAHI (kontrol 2x2:
// config E4 dengan bobot awal B0). Varian lain (TTA, E3/B0 + SAHI, E4/E5 plain) hanya di docs/REPORT.md.
// Versi 30 slide: presentasi/arsip/.
//
// Usage (dari root proyek):
//   node presentasi/build_deck.js            -> presentasi/AcneVision_Presentasi.pptx
// Prasyarat: npm install (di presentasi/), python scripts/report.py, python scripts/diagrams.py
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "docs", "report", "figures");
const ASSET = path.join(__dirname, "assets");
const OUT = process.env.DECK_OUT || path.join(__dirname, "AcneVision_Presentasi.pptx");   // DECK_OUT: build uji
const APPLY_THEME = process.env.PPTX_APPLY_THEME;   // path apply_theme.js (skill pptx), opsional

// ---------------------------------------------------------------- data
// Python json.dump menulis NaN (mis. Pearson r level 3 dengan 1 gambar); JSON standar memakai null
const readJson = (p) => JSON.parse(fs.readFileSync(p, "utf-8").replace(/:\s*NaN/g, ": null"));
const summary = readJson(path.join(ROOT, "docs", "report", "summary.json"));
const boot = readJson(path.join(ROOT, "docs", "report", "bootstrap.json"));
const evalm = (run) => readJson(path.join(ROOT, "runs", "eval", run, "metrics.json"));
const row = (label) => summary.find((r) => r["Varian"] === label);

const f2 = (x) => Number(x).toFixed(2);
// Pembulatan half-away-from-zero untuk nilai bootstrap (-0.975 -> -0.98, sama dengan REPORT.md)
const r2 = (x) => (Math.sign(x) * Math.floor(Math.abs(x) * 100 + 0.5 + 1e-9) / 100).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);
const id = (x) => Number(x).toLocaleString("id-ID");

// Empat model yang dibandingkan. Urutan = urutan seri di chart (lolos validator palet:
// pink-biru-oranye-ungu; kontras pink < 3:1 sehingga chart selalu berlabel nilai).
// mAP@50 & mAP@50-95 = definisi internal (scripts/analysis.py) agar SAHI sebanding dengan mode plain.
const VARS = [
  { key: "e4s", label: "E4 + SAHI", run: "yolov8l_960_tiles2_ft_sahi1728", summary: "E4 tile-FT + SAHI", color: "E87BA4" },
  { key: "b0", label: "B0 baseline", run: "yolov8l_960_default", summary: "B0 baseline default", color: "2A78D6" },
  { key: "e3", label: "E3 tuned", run: "yolov8l_960", summary: "E3 aug+hyp tuned", color: "EB6834" },
  { key: "e5s", label: "E5 + SAHI", run: "yolov8l_960_default_tiles2_ft_sahi1728", summary: "E5 tile-FT(B0) + SAHI", color: "7D53C8" },
].map((v) => {
  const m = evalm(v.run), t = m.test_conf_val_tuned, d = t.detection, s = row(v.summary);
  return { ...v, m, t, conf: m.conf_selected_on_val, P: d.precision, R: d.recall, F1: d.F1,
    ap50: m.ap50_test_internal, ap5095: m.map50_95_test_internal ?? null, lat: m.latency_ms_median,
    infer: m.detection_test_ultralytics ? m.detection_test_ultralytics.inference_ms : null,
    MAE: s.MAE, RMSE: s.RMSE, bias: s.Bias, r: s["Pearson r"] };
});
const X = Object.fromEntries(VARS.map((v) => [v.key, v]));
const missing95 = VARS.filter((v) => v.ap5095 === null).map((v) => v.label);
if (missing95.length) console.warn(`[peringatan] mAP@50-95 internal belum ada untuk: ${missing95.join(", ")} ` +
  "-> jalankan scripts/rerun_evals.ps1 lalu build ulang deck");
const m95 = (v) => (v.ap5095 === null ? "—*" : f3(v.ap5095));
const ms = (v) => `${v.lat.toFixed(0)} ms`;

// ---------------------------------------------------------------- tema (sama dengan deck v1)
const THEME = {
  name: "AcneVision",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B2B33", lt1: "FFFFFF", dk2: "14505C", lt2: "EEF4F4",
    accent1: "FF6B5E", accent2: "1C8C99", accent3: "F2B84B", accent4: "6C7A89",
    accent5: "2E9E6B", accent6: "8A5FD1", hlink: "1C8C99", folHlink: "14505C",
  },
};
const C = { ink: "0B2B33", ink2: "14505C", bg2: "EEF4F4", coral: "FF6B5E", teal: "1C8C99",
  yellow: "F2B84B", slate: "6C7A89", white: "FFFFFF", grid: "D9E2E2", coralBg: "FFF1EF", red: "C0392B", green: "2E7D4F" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "AcneVision";
pres.title = "AcneVision: Deteksi & Penghitungan Lesi Jerawat";

const W = 10, MX = 0.5, CW = W - 2 * MX;
const TOP = 1.55;                          // awal area konten

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.32, w: CW, h: 0.62, fontFace: "Cambria",
      fontSize: 28, bold: true, color: C.ink, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "sub", type: "body", x: MX, y: 0.94, w: CW, h: 0.55, fontSize: 12.5,
      color: C.slate, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.1, y: 5.22, w: 0.5, h: 0.25, fontSize: 9, color: C.slate, align: "right" },
});
pres.defineSlideMaster({ title: "DARK", background: { color: C.ink } });

// ---------------------------------------------------------------- helper
function imgSize(p) {
  const b = fs.readFileSync(p);
  if (b[0] === 0x89) return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
  let i = 2;
  while (i < b.length) {
    if (b[i] !== 0xff) { i++; continue; }
    const m = b[i + 1];
    if (m >= 0xc0 && m <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(m)) return { h: b.readUInt16BE(i + 5), w: b.readUInt16BE(i + 7) };
    i += 2 + b.readUInt16BE(i + 2);
  }
  throw new Error(`Ukuran gambar tidak terbaca: ${p}`);
}

// Gambar dengan rasio asli, dipusatkan di dalam kotak (x, y, w, h)
function fit(slide, file, x, y, w, h, alt) {
  const { w: iw, h: ih } = imgSize(file);
  const s = Math.min(w / iw, h / ih);
  const dw = iw * s, dh = ih * s;
  slide.addImage({ path: file, x: x + (w - dw) / 2, y: y + (h - dh) / 2, w: dw, h: dh, altText: alt });
}

// Section dideklarasikan tepat sebelum slide pertamanya
const sections = new Set();
function slideIn(masterName, section) {
  if (!sections.has(section)) { pres.addSection({ title: section }); sections.add(section); }
  return pres.addSlide({ masterName, sectionTitle: section });
}

function content(title, sub, section, notes) {
  const s = slideIn("CONTENT", section);
  s.addText(title, { placeholder: "title" });
  if (sub) s.addText(sub, { placeholder: "sub" });
  if (notes) s.addNotes(notes);
  return s;
}

function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 13, color: C.ink, valign: "top", ...o });
}

function card(slide, x, y, w, h, fill = C.bg2) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill } });
}

function numCircle(slide, n, x, y, d = 0.42, fill = C.teal) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  text(slide, String(n), { x, y, w: d, h: d, align: "center", valign: "middle", bold: true, color: C.white, fontSize: 14 });
}

// Kartu bernomor: lingkaran + judul tebal (Cambria) + isi
function numCard(slide, n, title, body, x, y, w, h, opt = {}) {
  card(slide, x, y, w, h, opt.fill || C.bg2);
  numCircle(slide, n, x + 0.18, y + 0.18, 0.42, opt.circle || C.teal);
  text(slide, title, { x: x + 0.75, y: y + 0.14, w: w - 0.9, h: 0.5, fontFace: "Cambria", bold: true, fontSize: 14, valign: "middle" });
  text(slide, body, { x: x + 0.2, y: y + 0.72, w: w - 0.4, h: h - 0.8, fontSize: opt.size || 12.5 });
}

// Stat callout: angka besar coral + label
function stat(slide, value, label, x, y, w, h, opt = {}) {
  card(slide, x, y, w, h, opt.fill || C.bg2);
  text(slide, value, { x: x + 0.2, y: y + 0.1, w: w - 0.4, h: h * 0.55, fontFace: "Cambria", bold: true,
    fontSize: opt.size || 30, color: opt.color || C.coral, valign: "bottom" });
  text(slide, label, { x: x + 0.2, y: y + h * 0.62, w: w - 0.4, h: h * 0.36, fontSize: 11.5, color: C.ink });
}

function caption(slide, t, x, y, w, h = 0.3) {
  text(slide, t, { x, y, w, h, fontSize: 10, color: C.slate, italic: true });
}

// Tabel: header teal, baris selang-seling
function table(slide, header, rows, x, y, w, colW, opt = {}) {
  const hdr = header.map((h) => ({ text: h, options: { bold: true, color: C.white, fill: { color: C.teal } } }));
  const body = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "object" && c !== null && c.text !== undefined ? c : { text: String(c) };
    return { text: cell.text, options: { fill: { color: i % 2 ? C.white : C.bg2 }, bold: j === 0 || cell.bold,
      color: cell.color || C.ink, ...(cell.options || {}) } };
  }));
  slide.addTable([hdr, ...body], { x, y, w, colW, fontSize: opt.fontSize || 11.5, fontFace: "Calibri", color: C.ink,
    border: { type: "solid", pt: 0.5, color: C.grid }, valign: "middle", margin: [3, 6, 3, 6], rowH: opt.rowH || 0.3 });
}

// Opsi chart seragam: label & grid identik di semua chart
function chartOpts(o) {
  return {
    catAxisLabelColor: C.slate, valAxisLabelColor: C.slate, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
    legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", titleColor: C.ink, titleFontSize: 12,
    valGridLine: { color: "E3E9E9", size: 0.75 }, catGridLine: { style: "none" },
    catAxisLineColor: "BFCACA", valAxisLineShow: false, dataLabelColor: C.ink, dataLabelFontSize: 10,
    legendColor: C.ink, legendFontSize: 10,
    ...o,
  };
}

// Selisih bootstrap A - B: "+0.051 [0.037 s.d. 0.066] · signifikan"
const sig = (d) => (d[1] > 0 || d[2] < 0 ? "signifikan" : "tidak signifikan");
const delta = (d, f = f3) => `${d[0] > 0 ? "+" : ""}${f(d[0])} [${f(d[1])} s.d. ${f(d[2])}] · ${sig(d)}`;

// ---------------------------------------------------------------- 1. Judul
{
  const s = slideIn("DARK", "Pembuka");
  const hero = path.join(ASSET, "image-1-1.jpg");
  const { w: iw, h: ih } = imgSize(hero);
  const hh = 5.625, hw = hh * iw / ih;
  s.addImage({ path: hero, x: W - hw, y: 0, w: hw, h: hh, altText: "Foto wajah dengan anotasi lesi jerawat (ground truth)" });
  text(s, "AI MODEL DEVELOPMENT", { x: 0.6, y: 1.5, w: 4.6, h: 0.3, fontSize: 11, bold: true, color: C.yellow, charSpacing: 4 });
  text(s, "AcneVision", { x: 0.6, y: 1.85, w: 4.8, h: 0.9, fontFace: "Cambria", bold: true, fontSize: 44, color: C.white, valign: "middle" });
  text(s, "Deteksi dan penghitungan lesi jerawat otomatis pada foto wajah", { x: 0.6, y: 2.9, w: 4.6, h: 0.8, fontSize: 17, color: C.white });
  text(s, "YOLOv8l  ·  ACNE04-v2  ·  SAHI sliced inference", { x: 0.6, y: 4.55, w: 4.6, h: 0.3, fontSize: 11, color: "9FB3B8" });
  text(s, "Laporan pengembangan model · 25 Oktober 2026", { x: 0.6, y: 4.85, w: 4.6, h: 0.3, fontSize: 11, color: "9FB3B8" });
  s.addNotes("Foto hero memakai anotasi ground truth (kotak merah) dari dataset, bukan prediksi model. " +
    "Presentasi ini merangkum pengembangan model: data, metode, eksperimen, hasil, dan keterbatasan. " +
    "Sumber konten: docs/PRESENTASI.md; detail lengkap semua varian: docs/REPORT.md.");
}

// ---------------------------------------------------------------- 2. Latar belakang
{
  const s = content("Latar belakang & tujuan",
    "Penghitungan lesi manual lambat dan bervariasi antar-penilai. Model deteksi otomatis dapat memberi hitungan yang konsisten dan cepat.",
    "Pendahuluan",
    "Penilaian keparahan jerawat bergantung pada jumlah lesi. Tan et al. (2006): dermatolog cukup andal menghitung lesi, " +
    "tetapi keandalannya meningkat setelah pelatihan; target klinis ICC >= 0.75. Proyek ini fokus pada pengembangan model; " +
    "serving/API dan aplikasi klien di luar cakupan.");
  const cw = (CW - 0.6) / 3, y = TOP + 0.15, h = 3.2;
  numCard(s, 1, "Deteksi lesi", "Menemukan setiap lesi jerawat pada foto wajah dan menandainya dengan kotak (1 kelas: acne).", MX, y, cw, h);
  numCard(s, 2, "Hitung lesi", "Jumlah lesi per gambar adalah metrik utama secara klinis: MAE, bias, dan korelasi terhadap ground truth.", MX + cw + 0.3, y, cw, h);
  numCard(s, 3, "Standar terukur", "Hasil dibandingkan dengan ambang minimum dari literatur, dengan protokol evaluasi tunggal dan uji signifikansi.", MX + 2 * (cw + 0.3), y, cw, h);
}

// ---------------------------------------------------------------- 3. Ringkasan hasil
{
  const k = X.e4s;
  const s = content("Ringkasan hasil",
    "Model terbaik (E4 + SAHI) memenuhi semua ambang minimum layak; kenaikan deteksi terhadap baseline signifikan secara statistik.",
    "Pendahuluan",
    `Semua angka: split test (121 gambar, 3.507 lesi), conf ${k.conf} dipilih di val, matching IoU 0.5. ` +
    "Sumber runs/eval/yolov8l_960_tiles2_ft_sahi1728/metrics.json. mAP@50 dan mAP@50-95 dihitung dengan matcher proyek " +
    "(conf 0.001, IoU 0.50 dan 0.50:0.05:0.95), karena SAHI tidak bisa dievaluasi lewat model.val Ultralytics. " +
    "Inferensi = median waktu end-to-end per gambar (baca gambar + prediksi + NMS) di RTX 4060. Uji signifikansi: paired bootstrap " +
    "10.000 resample per gambar (docs/report/bootstrap.json). Perbedaan MAE terhadap baseline belum signifikan.");
  const items = [[f3(k.P), "Precision"], [f3(k.R), "Recall"], [f3(k.ap50), "mAP@50"], [m95(k), "mAP@50-95"],
    [ms(k), "Inferensi / gambar\n(RTX 4060)"]];
  const cw = (CW - 4 * 0.2) / 5;
  items.forEach(([v, l], i) => stat(s, v, l, MX + i * (cw + 0.2), TOP + 0.1, cw, 1.45, { size: 26 }));
  const d = boot["E4+SAHI - B0"];
  const y2 = TOP + 1.85;
  card(s, MX, y2, CW, 1.55);
  text(s, "Dibanding baseline B0 (YOLOv8l default @960)", { x: MX + 0.25, y: y2 + 0.15, w: CW - 0.5, h: 0.35, fontFace: "Cambria", bold: true, fontSize: 14 });
  text(s, [
    { text: `F1 +${f3(d.dF1[0])}  [CI 95% ${f3(d.dF1[1])} s.d. ${f3(d.dF1[2])}]  ·  signifikan`, options: { bullet: true, breakLine: true } },
    { text: `Recall +${f3(d.dRecall[0])}  [${f3(d.dRecall[1])} s.d. ${f3(d.dRecall[2])}]  ·  signifikan`, options: { bullet: true, breakLine: true } },
    { text: `MAE ${r2(d.dMAE[0])} lesi/gambar  [${r2(d.dMAE[1])} s.d. ${r2(d.dMAE[2])}]  ·  belum signifikan; inferensi ${ms(k)} vs ${ms(X.b0)}`, options: { bullet: true } },
  ], { x: MX + 0.25, y: y2 + 0.55, w: CW - 0.5, h: 0.95, fontSize: 13, paraSpaceAfter: 3 });
}

// ---------------------------------------------------------------- 4. Dataset & tantangan
{
  const s = content("Dataset & tantangan lesi kecil",
    "1.204 foto wajah ACNE04 dengan 32.443 lesi beranotasi (ACNE04-v2). Pada input 960 px, mayoritas lesi lebih kecil dari 16 px.",
    "Data",
    "ACNE04 (Wu et al., ICCV 2019), anotasi ACNE04-v2 (Gazeau et al., MICCAI 2024), lisensi akademik, 1 kelas (acne). " +
    "Split acak per gambar 80/10/10 seed 42: 963/120/121 gambar, 25.521/3.366/3.507 lesi. Level keparahan Hayashi: train 371/468/105/19, " +
    "test 51/59/10/1 gambar (level 3 hanya 1 gambar sehingga metriknya tidak andal). Ukuran lesi = sisi terpanjang bbox, dihitung eksak " +
    "per gambar dengan skala letterbox (scripts/diagrams.py): median 42 px pada foto asli, 11.7 px pada 960. Lesi < 16 px: 91.5% (640), " +
    "73.5% (960), 52.4% (1280). Gambar test terpadat memuat 434 lesi.");
  const y = TOP + 0.1;
  [["1.204", "gambar wajah"], ["32.443", "lesi beranotasi"], ["11,7 px", "median lesi @960"]].forEach(([v, l], i) => {
    card(s, MX, y + i * 0.8, 2.6, 0.7);
    text(s, v, { x: MX + 0.15, y: y + i * 0.8, w: 1.4, h: 0.7, fontFace: "Cambria", bold: true, fontSize: 22, color: C.coral, valign: "middle" });
    text(s, l, { x: MX + 1.55, y: y + i * 0.8, w: 1.0, h: 0.7, fontSize: 11.5, valign: "middle" });
  });
  s.addChart(pres.charts.BAR, [{ name: "% lesi < 16 px", labels: ["640 px", "960 px", "1280 px"], values: [91.5, 73.5, 52.4] }],
    chartOpts({ x: 3.3, y: y - 0.05, w: 2.9, h: 2.45, barDir: "col", chartColors: [C.slate, C.coral, C.slate], showValue: true,
      dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", showTitle: true, title: "Lesi < 16 px setelah resize",
      showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMaxVal: 100, barGapWidthPct: 55 }));
  fit(s, path.join(FIG, "lesion_size_hist.png"), 6.4, y, W - MX - 6.4, 2.35, "Histogram ukuran lesi pada input 960 px");
  const by = 4.5;
  text(s, "Data split (per gambar, seed 42)", { x: MX, y: by - 0.35, w: 4, h: 0.3, bold: true, fontSize: 12 });
  const parts = [[0.8, C.teal, "Train · 80% · 963 gambar · 25.521 lesi", C.white], [0.1, C.yellow, "Val · 120", C.ink], [0.1, C.coral, "Test · 121", C.ink]];
  let bx = MX;
  parts.forEach(([f, col, lab, tc]) => {
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by, w: CW * f - 0.03, h: 0.45, fill: { color: col }, line: { color: col } });
    text(s, lab, { x: bx, y: by, w: CW * f - 0.03, h: 0.45, align: "center", valign: "middle", bold: true, fontSize: 11.5, color: tc });
    bx += CW * f;
  });
}

// ---------------------------------------------------------------- 5. Pipeline & preprocessing
{
  const s = content("Pipeline & preprocessing",
    "Enam tahap dari data mentah sampai prediksi, satu script per tahap. Anotasi lingkaran diubah menjadi bounding box agar dapat dilatih dengan YOLO.",
    "Metode",
    "Tahap coral memuat teknik resolusi: tiling (preprocessing E4) dan SAHI (inferensi). Augmentasi berjalan online di dalam training. " +
    "Konversi: kotak = [cx-r, cy-r, cx+r, cy+r], di-clip ke batas gambar, dinormalisasi ke [0,1]; letterbox ke 960 px dilakukan saat training. " +
    "Training dan evaluasi sama-sama memakai kotak YOLO ini (IoU kotak prediksi vs kotak GT), bukan lingkaran. Namun ukuran kotak GT " +
    "berasal dari jari-jari pilihan anotator pada lesi bertepi samar, sehingga ukurannya berderau; dengan lesi median 11.7 px, " +
    "geser 1 px sudah menurunkan IoU ke 0.72, karena itu mAP@50-95 (IoU ketat) tertekan. Contoh memakai ground truth. Seed 42, deterministic.");
  fit(s, path.join(FIG, "pipeline.png"), MX, TOP, CW, 1.75, "Diagram pipeline enam tahap");
  const y = TOP + 1.95, iw = 1.6, ih = 1.3, gap = 0.2;
  ["Foto asli", "Lingkaran", "Bounding box"].forEach((lab, i) => {
    const x = MX + i * (iw + gap);
    fit(s, path.join(ASSET, `image-5-${i + 1}.jpg`), x, y, iw, ih, lab);
    text(s, `${i + 1}. ${lab}`, { x, y: y + ih + 0.04, w: iw, h: 0.26, align: "center", fontSize: 11, bold: true });
  });
  const x = MX + 3 * (iw + gap) + 0.1, w = W - MX - x;
  card(s, x, y, w, 1.6);
  text(s, [
    { text: "Lingkaran (cx, cy, r) → kotak persegi", options: { bullet: true, breakLine: true } },
    { text: "Clip ke batas gambar, normalisasi [0, 1]", options: { bullet: true, breakLine: true } },
    { text: "25.521 / 3.366 / 3.507 lesi (train / val / test)", options: { bullet: true } },
  ], { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 1.4, fontSize: 12, valign: "middle", paraSpaceAfter: 4 });
}

// ---------------------------------------------------------------- 6. Tiling & SAHI
{
  const s = content("Tiling & SAHI: memperbesar lesi",
    "Tile 2×2 saat fine-tune (E4, E5) dan slice SAHI saat inferensi menaikkan skala input lesi ±2× (0,28 → 0,53–0,56) tanpa menambah kebutuhan VRAM.",
    "Metode",
    "Tiling: scripts/data/preprocess.py --grid 2 --overlap 0.1, train 963 -> 4.815 gambar (asli + 4 tile), label terpotong dipertahankan bila " +
    ">= 50% terlihat; test tidak di-tile. SAHI: slice 1728 px, overlap 20%, tiap slice di-resize ke 960, digabung NMS IoU 0.5. " +
    "Pada foto 3112x3456 SAHI membuat 9 slice, 3 di antaranya nyaris duplikat (geser 1 px). Dasar: Akyon et al., ICIP 2022 " +
    "(sliced inference + slicing-aided fine-tuning).");
  const cw = (CW - 0.3) / 2, ih = 2.75;
  fit(s, path.join(FIG, "tiling_example.png"), MX, TOP - 0.05, cw, ih, "Contoh tiling 2x2 dan label pada satu tile");
  fit(s, path.join(FIG, "sahi_example.png"), MX + cw + 0.3, TOP - 0.05, cw, ih, "Grid slice SAHI pada foto test");
  [["Training · tile 2×2 (E4, E5)","overlap 10% · train 963 → 4.815 gambar · skala 0,53"],
    ["Inferensi · SAHI 1728 px", "9 slice → 960 px · NMS IoU 0.5 · skala 0,56"]].forEach(([t, b], i) => {
    const x = MX + i * (cw + 0.3), y = TOP + ih + 0.05;
    card(s, x, y, cw, 0.8, i ? C.coralBg : C.bg2);
    text(s, t, { x: x + 0.15, y: y + 0.06, w: cw - 0.3, h: 0.32, fontFace: "Cambria", bold: true, fontSize: 13 });
    text(s, b, { x: x + 0.15, y: y + 0.4, w: cw - 0.3, h: 0.32, fontSize: 11.5 });
  });
}

// ---------------------------------------------------------------- 7. Empat model: cara dibuat
{
  const s = content("Empat model yang dibandingkan",
    "Dua jalur training dari bobot COCO yang sama. E5 mengulang resep fine-tune E4 dari B0, sehingga efek bobot awal dan fine-tune tile bisa dipisahkan.",
    "Eksperimen",
    "Keempat model dipilih dari 9 varian yang diuji (lengkap di docs/REPORT.md). Arsitektur sama: YOLOv8l 43,6 jt parameter, 960 px, " +
    "batch 4 (akumulasi 64), seed 42, RTX 4060 8 GB. B0: setelan default Ultralytics (optimizer auto -> AdamW 2e-3, loss 7.5/0.5/1.5), " +
    "100 epoch (best 80). E3: AdamW 1e-3 cosine, loss box/cls/dfl 10/0.3/2.0, augmentasi custom (mosaic 1.0, mixup 0.1, flip h/v 0.5/0.5, " +
    "rotasi ±10°, scale ±50%, HSV 0.015/0.7/0.4), 100 epoch (best 98). E4 dan E5: config identik (data_tiles2 = 963 asli + tile 2×2 overlap 10%, " +
    "4.815 gambar; AdamW 2e-4 cosine, 10 epoch, close_mosaic 3, loss & augmentasi E3); hanya bobot awal yang berbeda (E3 vs B0). " +
    "Training E4 65 menit, E5 66 menit. Inferensi E4/E5: SAHI slice 1728 px, overlap 20%, resize 960, NMS IoU 0.5. " +
    "Fine-tune tile menurunkan performa inferensi plain (E5 - B0 F1 -0.025, signifikan), jadi E4/E5 hanya dipakai dengan SAHI. " +
    "Temuan audit: copy_paste no-op untuk bbox, albumentations tidak terpasang (CLAHE/Blur tidak aktif).");
  const rowH = 1.12, yA = TOP - 0.05, yB = yA + rowH + 0.22;   // jalur B0 (atas), jalur E3 (bawah)
  const cx = [MX, 2.1, 4.6, 7.35], cw = [1.25, 2.1, 2.1, W - MX - 7.35];
  const cy = (y) => y + rowH / 2;
  // Panah lurus/diagonal dari (x1, y1) ke (x2, y2)
  const arrowTo = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: Math.min(y1, y2), w: x2 - x1,
    h: Math.max(Math.abs(y2 - y1), 0.001), flipV: y2 < y1, line: { color: C.slate, width: 1.5, endArrowType: "triangle" } });
  // Node: kartu dengan tepi berwarna varian, judul tebal, isi, dan baris metrik opsional
  const node = (x, y, w, h, title, body, metric, opt = {}) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: opt.fill || C.bg2 },
      line: { color: opt.edge || C.grid, width: opt.edge ? 2 : 0.75 } });
    text(s, title, { x: x + 0.12, y: y + 0.08, w: w - 0.24, h: 0.3, fontFace: "Cambria", bold: true, fontSize: 13 });
    text(s, body, { x: x + 0.12, y: y + 0.4, w: w - 0.24, h: metric ? 0.38 : h - 0.48, fontSize: 10.5, color: C.ink2 });
    if (metric) text(s, metric, { x: x + 0.12, y: y + h - 0.34, w: w - 0.24, h: 0.26, fontSize: 10.5, bold: true });
  };
  const res = (v) => `F1 ${f3(v.F1)} · R ${f3(v.R)} · ${ms(v)}`;

  const yc = (cy(yA) + cy(yB)) / 2 - 0.4;
  node(cx[0], yc, cw[0], 0.8, "COCO", "yolov8l.pt\npretrained");
  arrowTo(cx[0] + cw[0], yc + 0.4, cx[1], cy(yA));
  arrowTo(cx[0] + cw[0], yc + 0.4, cx[1], cy(yB));
  [[yA, X.b0, "setelan default · 100 ep", X.e5s, "init B0"], [yB, X.e3, "AdamW, loss & aug custom · 100 ep", X.e4s, "init E3"]]
    .forEach(([y, base, baseBody, ft, init]) => {
      node(cx[1], y, cw[1], rowH, base.label, baseBody, `plain: F1 ${f3(base.F1)} · ${ms(base)}`, { edge: base.color });
      arrowTo(cx[1] + cw[1], cy(y), cx[2], cy(y));
      node(cx[2], y, cw[2], rowH, `${ft.label.split(" ")[0]} tile-FT`, `${init} · tile 2×2 (4.815 gbr)\n10 ep · lr 2e-4`, null, { edge: ft.color });
      arrowTo(cx[2] + cw[2], cy(y), cx[3], cy(y));
      text(s, "SAHI", { x: cx[2] + cw[2], y: cy(y) - 0.3, w: cx[3] - cx[2] - cw[2], h: 0.24, fontSize: 9.5, color: C.slate, align: "center" });
      const best = ft === X.e4s;
      node(cx[3], y, cw[3], rowH, ft.label + (best ? " · terbaik" : ""), "slice 1728 px → 960", res(ft),
        { edge: ft.color, fill: best ? C.coralBg : C.bg2 });
    });

  // Dua fakta desain 2x2 (paired bootstrap, test)
  const yf = yB + rowH + 0.18, fw = (CW - 0.3) / 2, fh = 5.1 - yf;
  [["Efek fine-tune tile (dari B0)", "E5 + SAHI − B0 + SAHI", boot["E5+SAHI - B0+SAHI"].dF1,
    "Fine-tune tile membuat SAHI efektif juga pada B0."],
  ["Efek bobot awal E3 (setelah tile-FT)", "E4 + SAHI − E5 + SAHI", boot["E4+SAHI - E5+SAHI"].dF1,
    "Resep E3 tetap unggul: E4 dari E3 adalah pilihan tepat."]].forEach(([t, pair, d, msg], i) => {
    const x = MX + i * (fw + 0.3);
    card(s, x, yf, fw, fh);
    text(s, t, { x: x + 0.15, y: yf + 0.06, w: fw - 0.3, h: 0.3, fontFace: "Cambria", bold: true, fontSize: 12.5 });
    text(s, [{ text: `${pair}: ΔF1 ${delta(d)}`, options: { breakLine: true } }, { text: msg, options: { bold: true } }],
      { x: x + 0.15, y: yf + 0.38, w: fw - 0.3, h: fh - 0.44, fontSize: 10.5, paraSpaceAfter: 3 });
  });
}

// ---------------------------------------------------------------- 8. Protokol & standar
{
  const s = content("Protokol evaluasi & standar minimum",
    "Satu script evaluasi untuk semua varian: threshold dipilih di val, test dipakai sekali di akhir, lalu hasil dinilai terhadap ambang literatur.",
    "Eksperimen",
    "Precision/Recall/confusion matrix pada conf yang dipilih di val (MAE minimum), matching IoU 0.5. mAP@50 dan mAP@50-95 dari prediksi conf 0.001 " +
    "dengan matcher proyek (interpolasi 101 titik; IoU 0.50 dan 0.50:0.05:0.95); pada prediksi yang sama selisihnya terhadap Ultralytics " +
    "<= 0.0002 dan <= 0.0024, dan berlaku juga untuk SAHI. Ambang dari literatur (RESEARCH.md B): Sangha & Rizvi 2021 mAP@50 37.97%; " +
    "Li et al. 2023 AP lesi 0.507; AcneAI 2024 (ACNE04-v2) TPR 0.52 @ FPR 0.09; Seité 2019 F1 61-84%. Angka literatur tidak bisa " +
    "dibandingkan langsung (anotasi dan protokol berbeda). Literatur tidak memberi ambang mAP@50-95.");
  fit(s, path.join(FIG, "eval_protocol.png"), MX, TOP - 0.05, CW, 1.45, "Diagram alur protokol evaluasi");
  const y = TOP + 1.55, cw = 4.3;
  [["Confusion matrix tanpa TN", "Sel background–background tidak terdefinisi pada deteksi objek, sehingga ditandai N/A."],
    ["Signifikansi statistik", "Paired bootstrap 10.000× per gambar test; nyata bila CI 95% tidak memuat 0."]].forEach(([t, b], i) => {
    const yy = y + i * 1.0;
    card(s, MX, yy, cw, 0.9);
    text(s, t, { x: MX + 0.15, y: yy + 0.06, w: cw - 0.3, h: 0.3, fontFace: "Cambria", bold: true, fontSize: 13 });
    text(s, b, { x: MX + 0.15, y: yy + 0.38, w: cw - 0.3, h: 0.48, fontSize: 11 });
  });
  const x = MX + cw + 0.3, w = W - MX - x;
  table(s, ["Metrik", "Minimum", "Baik"], [
    ["Precision", "≥ 0.50", "≥ 0.60"], ["Recall", "≥ 0.50", "≥ 0.60"], ["mAP@50", "≥ 0.45", "≥ 0.55"],
    ["MAE counting", "≤ 10", "≤ 7"], ["Pearson r", "≥ 0.85", "≥ 0.90"], ["Inferensi", "< 1 s", "< 100 ms"],
  ], x, y, w, [1.6, 1.3, w - 2.9], { rowH: 0.27, fontSize: 10.5 });
}

// ---------------------------------------------------------------- 9. Metrik utama
{
  const s = content("Metrik utama",
    "E4 + SAHI unggul di Precision, Recall, mAP@50, dan mAP@50-95, disusul E5 + SAHI; mode plain (B0, E3) ±9× lebih cepat.",
    "Hasil",
    "Split test (121 gambar, 3.507 lesi). Precision/Recall pada conf yang dipilih di val per varian (kolom conf), matching IoU 0.5. " +
    "mAP@50 dan mAP@50-95 dari prediksi conf 0.001 dengan matcher proyek (scripts/analysis.py), sehingga SAHI sebanding dengan mode plain. " +
    "mAP@50-95 rendah karena lesi sangat kecil (median 11.7 px): geser 1 px di x dan y menurunkan IoU ke 0.72, 2 px ke 0.52 " +
    "(scripts/analysis.py box_iou). Ukuran kotak GT juga berderau karena berasal dari jari-jari lingkaran anotator, bukan batas lesi yang tegas. " +
    "IoU tetap dihitung antar-kotak YOLO. " +
    "Inferensi = median end-to-end per gambar, RTX 4060, batch 1. Sumber: runs/eval/<run>/metrics.json.");
  const order = [X.b0, X.e3, X.e5s, X.e4s];
  const best = (f, lower = false) => (lower ? Math.min : Math.max)(...order.map((v) => f(v) ?? (lower ? Infinity : -1)));
  const cell = (v, f, fmt, lower = false) => {
    const x = f(v);
    if (x === null) return { text: "—*", color: C.slate };
    const top = x === best(f, lower);
    return { text: fmt(x), bold: top, color: top ? C.coral : C.ink };
  };
  table(s, ["Varian", "conf", "Precision", "Recall", "mAP@50", "mAP@50-95", "Inferensi"], order.map((v) => [
    v.label, v.conf.toFixed(2), cell(v, (k) => k.P, f3), cell(v, (k) => k.R, f3), cell(v, (k) => k.ap50, f3),
    cell(v, (k) => k.ap5095, f3), cell(v, (k) => k.lat, (x) => `${x.toFixed(0)} ms`, true),
  ]), MX, TOP + 0.05, CW, [1.8, 0.8, 1.25, 1.25, 1.25, 1.35, 1.3], { rowH: 0.42, fontSize: 13 });
  const y = TOP + 2.38, cw = (CW - 0.6) / 3;
  [["Precision", "dari semua kotak yang diprediksi, berapa yang benar lesi"],
    ["Recall", "dari semua lesi GT, berapa yang ditemukan"],
    ["mAP@50 / @50-95", "luas kurva P-R pada IoU 0,5 / rata-rata IoU 0,5–0,95"]].forEach(([t, b], i) => {
    const x = MX + i * (cw + 0.3);
    card(s, x, y, cw, 0.95);
    text(s, t, { x: x + 0.15, y: y + 0.08, w: cw - 0.3, h: 0.3, fontFace: "Cambria", bold: true, fontSize: 13 });
    text(s, b, { x: x + 0.15, y: y + 0.4, w: cw - 0.3, h: 0.5, fontSize: 11 });
  });
  caption(s, "Coral = terbaik per kolom. Test, IoU 0.5; mAP dari matcher proyek (setara Ultralytics)." +
    (missing95.length ? "  * menunggu evaluasi ulang (scripts/rerun_evals.ps1)." : ""), MX, 4.95, CW);
}

// ---------------------------------------------------------------- 10. Hasil deteksi
{
  const s = content("Hasil deteksi",
    "E4 + SAHI unggul di keempat metrik; E5 + SAHI di posisi kedua. Kenaikan F1 keduanya terhadap B0 signifikan secara statistik.",
    "Hasil",
    "Split test; Precision/Recall pada conf dipilih di val, mAP dari prediksi conf 0.001, matching IoU 0.5 (mAP@50-95: IoU 0.50:0.05:0.95). " +
    "Uji: paired bootstrap 10.000 resample per gambar test (seed 42), CI 95%. Tuning E3 tidak mengubah F1 secara nyata dibanding B0; " +
    "kenaikan datang dari SAHI (E3 + SAHI vs E3: F1 +0.047, signifikan). Opini: augmentasi skala E3 membuat model tahan terhadap " +
    "perubahan skala saat slicing, sehingga SAHI pada B0 tidak membantu (lihat REPORT.md). E5 (resep E4 dari B0) menunjukkan " +
    "fine-tune tile membuat SAHI efektif juga pada B0, tetapi bobot awal E3 tetap lebih baik.");
  s.addChart(pres.charts.BAR, VARS.map((v) => ({ name: v.label, labels: ["Precision", "Recall", "mAP@50", "mAP@50-95"],
    values: [v.P, v.R, v.ap50, v.ap5095 ?? 0] })),
  chartOpts({ x: MX, y: TOP - 0.05, w: 5.6, h: 3.65, barDir: "col", barGrouping: "clustered", chartColors: VARS.map((v) => v.color),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 9, showLegend: true, legendPos: "b",
    valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", barGapWidthPct: 50 }));
  const x = 6.35, w = W - MX - x;
  table(s, ["Varian", "F1"], [X.e4s, X.e5s, X.b0, X.e3].map((v, i) => [
    { text: v.label, bold: true }, { text: f3(v.F1), bold: i === 0, color: i === 0 ? C.coral : C.ink }]),
  x, TOP - 0.05, w, [2.0, w - 2.0], { rowH: 0.27, fontSize: 11 });
  const y = TOP + 1.45;
  card(s, x, y, w, 2.2);
  text(s, "Selisih F1 (CI 95%)", { x: x + 0.15, y: y + 0.06, w: w - 0.3, h: 0.28, fontFace: "Cambria", bold: true, fontSize: 12.5 });
  const pairs = [["E4 + SAHI − B0", "E4+SAHI - B0"], ["E4 + SAHI − E5 + SAHI", "E4+SAHI - E5+SAHI"],
    ["E5 + SAHI − B0", "E5+SAHI - B0"], ["E3 − B0", "E3 - B0"]];
  text(s, pairs.flatMap(([lab, k], i) => [{ text: lab, options: { bold: true, breakLine: true } },
    { text: delta(boot[k].dF1), options: { breakLine: i < pairs.length - 1 } }]),
  { x: x + 0.15, y: y + 0.36, w: w - 0.3, h: 1.8, fontSize: 9.5 });
}

// ---------------------------------------------------------------- 11. Confusion matrix
{
  const order = [X.b0, X.e3, X.e5s, X.e4s];
  const s = content("Confusion matrix",
    `E4 + SAHI menemukan ${X.e4s.t.detection.TP - X.b0.t.detection.TP} lesi lebih banyak (TP) dengan ${X.b0.t.detection.FP - X.e4s.t.detection.FP} prediksi salah (FP) lebih sedikit dibanding B0; E5 + SAHI berpola serupa.`,
    "Hasil",
    "Test, IoU 0.5, conf dipilih di val per varian. Baris = prediksi (acne / background), kolom = ground truth. Sel background-background " +
    "(TN) tidak terdefinisi pada deteksi objek sehingga N/A. FP lokalisasi = prediksi yang overlap 0.1 <= IoU < 0.5 dengan lesi GT " +
    "(kotak salah skala/posisi); FP latar = tanpa overlap, sebagian kemungkinan lesi yang tidak dianotasi (opini). " +
    "SAHI memangkas FP lokalisasi karena kotak lebih presisi pada resolusi tinggi, tetapi menambah FP latar.");
  const cw = (CW - 0.45) / 4;
  order.forEach((v, i) => {
    const x = MX + i * (cw + 0.15), d = v.t.detection, fp = v.t.fp_breakdown;
    fit(s, path.join(FIG, `cm_${v.run}.png`), x, TOP - 0.1, cw, 2.6, `Confusion matrix ${v.label}`);
    text(s, `${v.label} · conf ${v.conf.toFixed(2)}`, { x, y: TOP + 2.55, w: cw, h: 0.3, align: "center", bold: true, fontSize: 11.5,
      color: v === X.e4s ? C.coral : C.ink });
    text(s, `TP ${id(d.TP)} · FP ${id(d.FP)} · FN ${id(d.FN)}\nFP lokalisasi ${id(fp.localization)} · latar ${id(fp.background)}`,
      { x, y: TOP + 2.87, w: cw, h: 0.5, align: "center", fontSize: 9.5, color: C.ink2 });
  });
}

// ---------------------------------------------------------------- 12. Kecepatan inferensi
{
  const order = [...VARS].sort((a, b) => a.lat - b.lat);
  const s = content("Kecepatan inferensi",
    `Mode plain ±${X.e3.lat.toFixed(0)} ms per gambar (< 100 ms); E4 / E5 + SAHI ±9× lebih lambat (±${X.e4s.lat.toFixed(0)} ms) tetapi masih < 1 s.`,
    "Hasil",
    "Inferensi = median waktu end-to-end per gambar test (baca gambar + prediksi + NMS/penggabungan slice), batch 1, RTX 4060 8GB, " +
    "gambar pertama (warmup) diabaikan; sumber latency_ms_median di runs/eval/<run>/metrics.json. " +
    `Waktu model saja dari model.val Ultralytics (batch 4): B0 ${f2(X.b0.infer)} ms, E3 ${f2(X.e3.infer)} ms per gambar. ` +
    "SAHI menjalankan 9 slice per foto 3112x3456, 3 di antaranya nyaris duplikat (geser 1 px); menghapusnya diperkirakan " +
    "memangkas sekitar sepertiga waktu (opini).");
  s.addChart(pres.charts.BAR, [{ name: "Inferensi (ms)", labels: order.map((v) => v.label), values: order.map((v) => v.lat) }],
    chartOpts({ x: MX, y: TOP - 0.05, w: 5.9, h: 3.0, barDir: "bar", chartColors: order.map((v) => v.color),
      invertedColors: order.map((v) => v.color), showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
      showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 760,
      catAxisOrientation: "maxMin", showTitle: true, title: "Median ms per gambar (lebih kecil lebih cepat)", barGapWidthPct: 45 }));
  caption(s, "Standar: < 1 s minimum, < 100 ms baik. Waktu end-to-end, RTX 4060, batch 1.", MX, TOP + 3.1, 5.9);
  const x = 6.65, w = W - MX - x;
  const fps = (v) => (1000 / v.lat < 10 ? (1000 / v.lat).toFixed(1).replace(".", ",") : (1000 / v.lat).toFixed(0));
  order.forEach((v, i) => stat(s, ms(v), `${v.label}${v.key.endsWith("s") ? "" : " plain"} · ±${fps(v)} gambar/detik`,
    x, TOP - 0.05 + i * 0.9, w, 0.8, { size: 20, color: v.key.endsWith("s") ? C.coral : C.teal }));
}

// ---------------------------------------------------------------- 13. Counting
{
  const s = content("Hasil counting",
    "Keempat model cenderung under-count (bias negatif); E4 + SAHI memiliki MAE terendah dan korelasi tertinggi dengan ground truth.",
    "Hasil",
    "MAE = rata-rata |pred - GT| lesi per gambar; bias = rata-rata (pred - GT). Test 121 gambar, conf dipilih di val. " +
    `RMSE B0 ${f2(X.b0.RMSE)}, E3 ${f2(X.e3.RMSE)}, E5 + SAHI ${f2(X.e5s.RMSE)}, E4 + SAHI ${f2(X.e4s.RMSE)}. Satu gambar level 3 (434 lesi) mendominasi RMSE: ` +
    `B0 menghitung ${X.b0.t.per_level.level3.counting.total_pred}, E4 + SAHI ${X.e4s.t.per_level.level3.counting.total_pred}. ` +
    "Error membesar seiring kepadatan lesi (per level: REPORT.md). Perbedaan MAE terhadap B0 tidak signifikan (bootstrap): " +
    `E4 + SAHI - B0 ${r2(boot["E4+SAHI - B0"].dMAE[0])} [${r2(boot["E4+SAHI - B0"].dMAE[1])}, ${r2(boot["E4+SAHI - B0"].dMAE[2])}], ` +
    `E5 + SAHI - B0 ${r2(boot["E5+SAHI - B0"].dMAE[0])} [${r2(boot["E5+SAHI - B0"].dMAE[1])}, ${r2(boot["E5+SAHI - B0"].dMAE[2])}]. ` +
    "Scatter di kanan: baseline B0 vs model terbaik E4 + SAHI.");
  const order = [X.e4s, X.e5s, X.e3, X.b0];
  s.addChart(pres.charts.BAR, [{ name: "MAE", labels: order.map((v) => v.label), values: order.map((v) => v.MAE) }],
    chartOpts({ x: MX, y: TOP - 0.1, w: 4.0, h: 2.3, barDir: "bar", chartColors: [C.teal], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.00", showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 12,
      catAxisOrientation: "maxMin", showTitle: true, title: "MAE (lesi/gambar, lebih kecil lebih baik)", barGapWidthPct: 45 }));
  table(s, ["Varian", "Bias", "Pearson r"], order.map((v, i) => [v.label, f2(v.bias),
    { text: f3(v.r), bold: i === 0, color: i === 0 ? C.coral : C.ink }]), MX, TOP + 2.25, 4.0, [1.8, 1.0, 1.2], { rowH: 0.26, fontSize: 11 });
  fit(s, path.join(FIG, "count_scatter.png"), 4.7, TOP - 0.05, W - MX - 4.7, 3.35, "Scatter jumlah lesi prediksi vs ground truth (B0 dan E4 + SAHI)");
  caption(s, "Scatter: baseline B0 vs terbaik E4 + SAHI.", 4.7, TOP + 2.9, W - MX - 4.7);
}

// ---------------------------------------------------------------- 14. Contoh prediksi
{
  const s = content("Contoh prediksi (E4 + SAHI)",
    "Seluruh 121 gambar test diprediksi. Model kuat pada lesi yang jelas, tetapi gagal pada lesi tiny yang padat dan rancu pada bekas samar.",
    "Hasil",
    "Output: runs/predict/yolov8l_960_tiles2_ft_sahi/test/ (121 gambar + predictions.csv) dan versi baseline di runs/predict/yolov8l_960_default_plain/test/. " +
    "Kotak tanpa label teks agar wajah tidak tertutup; jumlah ditulis di strip header. levle1_288: banyak FP pada bintik kemerahan " +
    "yang mungkin lesi/bekas tidak dianotasi, dan kotak berskala lain di sekitar lesi GT (FP + FN sekaligus). levle3_89: lesi tiny berdempetan terlewat.");
  const ex = [["levle1_482", "Terbaik · GT 21, pred 17, F1 0.79"], ["levle1_288", "F1 terendah · TP 4 / FP 21 / FN 12"], ["levle3_89", "Terpadat · GT 434, pred 264"]];
  const iw = 2.55, gap = (CW - 3 * iw) / 2;
  ex.forEach(([nm, cap], i) => {
    const x = MX + i * (iw + gap);
    fit(s, path.join(FIG, "examples", `${nm}_e4sahi.jpg`), x, TOP - 0.1, iw, 3.3, `Prediksi ${nm}`);
    text(s, cap, { x, y: TOP + 3.25, w: iw, h: 0.3, fontSize: 11, bold: true, align: "center" });
  });
  text(s, [
    { text: "■ ", options: { color: "008CFF" } }, { text: "TP benar   " },
    { text: "■ ", options: { color: "FFD700" } }, { text: "FP salah   " },
    { text: "■ ", options: { color: "E62828" } }, { text: "FN terlewat" },
  ], { x: MX, y: 5.2, w: 5, h: 0.25, fontSize: 10.5, color: C.slate });
}

// ---------------------------------------------------------------- 15. Hasil vs standar
{
  const ok = (s) => ({ text: s, color: C.green, bold: true });
  // [label, getter, format, minimum, baik, lebih besar lebih baik, teks minimum, teks baik]
  const M = [
    ["Precision", (v) => v.P, f3, 0.50, 0.60, true, "≥ 0.50", "≥ 0.60"],
    ["Recall", (v) => v.R, f3, 0.50, 0.60, true, "≥ 0.50", "≥ 0.60"],
    ["mAP@50", (v) => v.ap50, f3, 0.45, 0.55, true, "≥ 0.45", "≥ 0.55"],
    ["mAP@50-95", (v) => v.ap5095, f3, null, null, true, "—", "—"],
    ["Inferensi", (v) => v.lat, (x) => `${x.toFixed(0)} ms`, 1000, 100, false, "< 1 s", "< 100 ms"],
    ["MAE counting", (v) => v.MAE, f2, 10, 7, false, "≤ 10", "≤ 7"],
    ["Pearson r", (v) => v.r, f3, 0.85, 0.90, true, "≥ 0.85", "≥ 0.90"],
  ];
  const pass = (x, t, hi) => (hi ? x >= t : x <= t);
  const cellOf = (v, [, g, fmt, mn, , hi]) => {
    const x = g(v);
    if (x === null) return { text: "—*", color: C.slate };
    return { text: fmt(x), color: mn !== null && !pass(x, mn, hi) ? C.red : C.ink };
  };
  const status = (m) => {
    const [, g, , mn, gd, hi] = m, x = g(X.e4s);
    if (mn === null) return { text: "tanpa ambang literatur", color: C.slate };
    if (x === null) return "—";
    return pass(x, gd, hi) ? ok("✓ baik") : pass(x, mn, hi) ? ok("✓ minimum") : { text: "✗ di bawah minimum", color: C.red, bold: true };
  };
  const s = content("Hasil vs standar minimum",
    "E4 + SAHI dan E5 + SAHI memenuhi semua ambang minimum; B0 dan E3 gagal di Recall, tetapi hanya mode plain yang memenuhi target < 100 ms.",
    "Hasil",
    "Test, conf dipilih di val per varian (runs/eval/<run>/metrics.json). mAP@50 dan mAP@50-95 dari matcher proyek (setara Ultralytics, " +
    "berlaku juga untuk SAHI). Literatur deteksi jerawat tidak melaporkan mAP@50-95 secara konsisten, sehingga tidak ada ambang. " +
    "mAP@50 E4 + SAHI 0.5499, tepat di bawah ambang baik 0.55 (ditampilkan 0.550). Siap klinis tetap membutuhkan validasi terhadap " +
    "beberapa dermatolog (ICC >= 0.75).");
  table(s, ["Metrik (test)", X.b0.label, X.e3.label, X.e5s.label, X.e4s.label, "Minimum", "Baik", "Status E4 + SAHI"],
    M.map((m) => [m[0], cellOf(X.b0, m), cellOf(X.e3, m), cellOf(X.e5s, m), { ...cellOf(X.e4s, m), bold: true },
      m[6], m[7], status(m)]),
    MX, TOP + 0.05, CW, [1.4, 0.95, 0.9, 1.0, 1.0, 0.85, 0.9, 2.0], { rowH: 0.4, fontSize: 11 });
  caption(s, "Merah: di bawah ambang minimum." + (missing95.length ? "  * menunggu evaluasi ulang (scripts/rerun_evals.ps1)." : ""), MX, 4.9, CW);
}

// ---------------------------------------------------------------- 16. Kesimpulan & langkah lanjut
{
  const s = content("Kesimpulan & langkah lanjut",
    "Resolusi efektif adalah tuas terbesar; pilihan model akhir bergantung pada kebutuhan akurasi vs kecepatan.",
    "Penutup",
    "Kesimpulan didukung uji bootstrap. Penetapan model terpilih di SPEC.md menunggu keputusan: E4 + SAHI bila akurasi diutamakan, " +
    "E3 plain bila inferensi < 100 ms dibutuhkan. Keterbatasan (REPORT.md bagian 9): satu seed per konfigurasi; test 121 gambar tidak " +
    "terstratifikasi (level 3 hanya 1, level 2 hanya 10); split per gambar sehingga kebocoran pasien belum terverifikasi; ukuran kotak GT dari jari-jari lingkaran " +
    "berderau dan lesi kecil, sehingga mAP@50-95 tertekan; belum ada validasi dermatolog (ICC >= 0.75). Roadmap (ROADMAP.md): training tile/1280 px dari nol, head P2, " +
    ">= 3 seed + split terstratifikasi, hapus slice SAHI duplikat, validasi klinis.");
  const cw = (CW - 0.3) / 2, y = TOP + 0.05, h = 1.1;
  [["Memenuhi standar", `E4 + SAHI memenuhi semua ambang minimum; Precision ${f3(X.e4s.P).replace(".", ",")} dan r ${f3(X.e4s.r).replace(".", ",")} mencapai ambang baik.`],
    ["Kontributor utama", `SAHI. Fine-tune tile dari B0: F1 +${f3(boot["E5+SAHI - B0+SAHI"].dF1[0])}; bobot awal E3 tetap lebih baik (+${f3(boot["E4+SAHI - E5+SAHI"].dF1[0])}).`],
    ["Trade-off", `SAHI ±9× lebih lambat (${ms(X.e4s)} vs ${ms(X.e3)}), tetap < 1 s.`],
  ].forEach(([t, b], i) => numCard(s, i + 1, t, b, MX, y + i * (h + 0.12), cw, h, { size: 11.5 }));
  const x = MX + cw + 0.3, rh = 0.95;
  [[C.coralBg, "Akurasi diutamakan", "E4 + SAHI", C.coral, X.e4s], [C.bg2, "Kecepatan diutamakan (< 100 ms)", "E3 plain", C.teal, X.e3]]
    .forEach(([bg, lab, nm, col, v], i) => {
      const yy = y + i * (rh + 0.1);
      card(s, x, yy, cw, rh, bg);
      text(s, lab, { x: x + 0.2, y: yy + 0.08, w: cw - 0.4, h: 0.28, fontSize: 11.5, color: C.slate, bold: true });
      text(s, nm, { x: x + 0.2, y: yy + 0.36, w: 1.9, h: 0.5, fontFace: "Cambria", bold: true, fontSize: 19, color: col, valign: "middle" });
      text(s, `F1 ${f3(v.F1)} · MAE ${f2(v.MAE)}\n${ms(v)} / gambar`, { x: x + 2.1, y: yy + 0.36, w: cw - 2.25, h: 0.52, fontSize: 11, valign: "middle" });
    });
  const yl = y + 2 * (rh + 0.1), hl = 3.54 - 2 * (rh + 0.1);
  card(s, x, yl, cw, hl);
  text(s, [
    { text: "Keterbatasan: ", options: { bold: true } }, { text: "satu seed, test kecil (level 3: 1 gambar), belum divalidasi dermatolog.", options: { breakLine: true } },
    { text: "Berikutnya: ", options: { bold: true } }, { text: "tile / 1280 px dari nol, head P2, multi-seed, validasi klinis (ICC)." },
  ], { x: x + 0.15, y: yl + 0.08, w: cw - 0.3, h: hl - 0.16, fontSize: 11.5, valign: "middle", paraSpaceAfter: 4 });
}

// ---------------------------------------------------------------- 17. Penutup & referensi
{
  const s = slideIn("DARK", "Penutup");
  text(s, "Terima kasih", { x: 0.6, y: 0.6, w: 8.8, h: 0.8, fontFace: "Cambria", bold: true, fontSize: 36, color: C.white });
  text(s, "Dokumen pendukung: docs/REPORT.md · docs/PRESENTASI.md · docs/EXPERIMENTS.md", { x: 0.6, y: 1.4, w: 8.8, h: 0.35, fontSize: 12, color: C.yellow });
  text(s, "REFERENSI", { x: 0.6, y: 2.1, w: 8.8, h: 0.3, fontSize: 11, bold: true, color: C.yellow, charSpacing: 3 });
  const refs = [
    "Wu et al. Joint acne image grading and counting via label distribution learning. ICCV 2019 (ACNE04).",
    "Gazeau et al. AcneAI. MICCAI 2024 (ACNE04-v2).",
    "Akyon et al. Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection. ICIP 2022.",
    "Ultralytics YOLOv8 documentation: configuration & validation (docs.ultralytics.com).",
    "Li et al. Front. Med. 2023 · Seité et al. Exp. Dermatol. 2019 · Tan et al. J. Cutan. Med. Surg. 2006.",
  ];
  text(s, refs.map((r, i) => ({ text: r, options: { bullet: true, breakLine: i < refs.length - 1 } })),
    { x: 0.6, y: 2.45, w: 8.8, h: 2.5, fontSize: 12, color: "D7E3E5", paraSpaceAfter: 5 });
  s.addNotes("URL lengkap setiap referensi tersedia di docs/RESEARCH.md dan docs/REPORT.md.");
}

// ---------------------------------------------------------------- tulis
(async () => {
  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) {
    const { applyTheme } = require(APPLY_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log(`Deck: ${OUT}`);
})();
