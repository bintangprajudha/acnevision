// AcneVision - Deck presentasi (sidang/seminar)
// ==============================================
// Konten mengikuti docs/PRESENTASI.md. Angka dibaca dari artefak evaluasi
// (docs/report/summary.json, bootstrap.json, runs/eval/*/metrics.json, runs/detect/*/results.csv);
// gambar dari docs/report/figures (scripts/report.py, scripts/diagrams.py) dan presentasi/assets.
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
const V = {
  b0: row("B0 baseline default"), e3: row("E3 aug+hyp tuned"), tta: row("E3 + TTA"),
  e3s: row("E3 + SAHI"), e4s: row("E4 tile-FT + SAHI"), b0s: row("B0 + SAHI"),
};
const E4 = evalm("yolov8l_960_tiles2_ft");

// Metrik utama per varian langsung dari metrics.json (test, conf dipilih di val, IoU 0.5).
// mAP@50 & mAP@50-95 = definisi internal (scripts/analysis.py) agar SAHI sebanding dengan mode plain.
const RUNS = [
  ["B0 baseline", "yolov8l_960_default", "b0"], ["E3 tuned", "yolov8l_960", "e3"],
  ["E3 + TTA", "yolov8l_960_tta", "tta"], ["E4 tile-FT", "yolov8l_960_tiles2_ft", "e4"],
  ["B0 + SAHI", "yolov8l_960_default_sahi1728", "b0s"], ["E3 + SAHI", "yolov8l_960_sahi1728", "e3s"],
  ["E4 + SAHI", "yolov8l_960_tiles2_ft_sahi1728", "e4s"],
];
const KEY = Object.fromEntries(RUNS.map(([label, run, color]) => {
  const m = evalm(run), d = m.test_conf_val_tuned.detection;
  return [run, { label, run, color, conf: m.conf_selected_on_val, P: d.precision, R: d.recall, F1: d.F1,
    ap50: m.ap50_test_internal, ap5095: m.map50_95_test_internal ?? null, lat: m.latency_ms_median,
    infer: m.detection_test_ultralytics ? m.detection_test_ultralytics.inference_ms : null }];
}));
const K = (run) => KEY[run];
const missing95 = Object.values(KEY).filter((k) => k.ap5095 === null).map((k) => k.label);
if (missing95.length) console.warn(`[peringatan] mAP@50-95 internal belum ada untuk: ${missing95.join(", ")} ` +
  "-> jalankan scripts/rerun_evals.ps1 lalu build ulang deck");
const m95 = (k) => (k.ap5095 === null ? "—*" : f3(k.ap5095));
const BEST = evalm("yolov8l_960_tiles2_ft_sahi1728").test_conf_val_tuned;
const B0M = evalm("yolov8l_960_default").test_conf_val_tuned;

function curve(run) {
  const lines = fs.readFileSync(path.join(ROOT, "runs", "detect", run, "results.csv"), "utf-8").trim().split(/\r?\n/);
  const head = lines[0].split(",").map((s) => s.trim());
  const ie = head.indexOf("epoch"), im = head.indexOf("metrics/mAP50(B)");
  const byEpoch = new Map();
  for (const l of lines.slice(1)) {
    const c = l.split(",");
    byEpoch.set(+c[ie], +c[im]);          // resume menulis ulang epoch: ambil yang terakhir
  }
  return [...byEpoch.entries()].sort((a, b) => a[0] - b[0]);
}

const f2 = (x) => Number(x).toFixed(2);
// Pembulatan half-away-from-zero untuk nilai bootstrap (-0.975 -> -0.98, sama dengan REPORT.md)
const r2 = (x) => (Math.sign(x) * Math.floor(Math.abs(x) * 100 + 0.5 + 1e-9) / 100).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);
const id = (x) => Number(x).toLocaleString("id-ID");

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
  yellow: "F2B84B", slate: "6C7A89", white: "FFFFFF", grid: "D9E2E2", coralBg: "FFF1EF" };
// Warna varian = palet kategorikal tervalidasi (sama dengan scripts/report.py)
const VC = { b0: "2A78D6", e3: "EB6834", tta: "1BAF7A", e3s: "EDA100", e4s: "E87BA4", b0s: "008300", e4: "6C7A89" };

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
function fit(slide, file, x, y, w, h, alt, align = "center") {
  const { w: iw, h: ih } = imgSize(file);
  const s = Math.min(w / iw, h / ih);
  const dw = iw * s, dh = ih * s;
  const dx = align === "left" ? x : x + (w - dw) / 2;
  slide.addImage({ path: file, x: dx, y: y + (h - dh) / 2, w: dw, h: dh, altText: alt });
  return { x: dx, y: y + (h - dh) / 2, w: dw, h: dh };
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
  const fs_ = opt.fontSize || 11.5;
  const hdr = header.map((h) => ({ text: h, options: { bold: true, color: C.white, fill: { color: C.teal } } }));
  const body = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "object" && c !== null && c.text !== undefined ? c : { text: String(c) };
    return { text: cell.text, options: { fill: { color: i % 2 ? C.white : C.bg2 }, bold: j === 0 || cell.bold,
      color: cell.color || C.ink, ...(cell.options || {}) } };
  }));
  slide.addTable([hdr, ...body], { x, y, w, colW, fontSize: fs_, fontFace: "Calibri", color: C.ink,
    border: { type: "solid", pt: 0.5, color: C.grid }, valign: "middle", margin: [3, 6, 3, 6],
    rowH: opt.rowH || 0.3 });
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
  text(s, "Laporan pengembangan model · 24 Oktober 2026", { x: 0.6, y: 4.85, w: 4.6, h: 0.3, fontSize: 11, color: "9FB3B8" });
  s.addNotes("Foto hero memakai anotasi ground truth (kotak merah) dari dataset, bukan prediksi model. " +
    "Presentasi ini merangkum seluruh pengembangan model: data, metode, eksperimen, hasil, dan keterbatasan. " +
    "Sumber konten: docs/PRESENTASI.md; detail: docs/REPORT.md.");
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
  const t = BEST;
  const s = content("Ringkasan hasil",
    "Model terbaik (E4 + SAHI) memenuhi semua ambang minimum layak; kenaikan deteksi terhadap baseline signifikan secara statistik.",
    "Pendahuluan",
    `Semua angka: split test (121 gambar, 3.507 lesi), conf ${t.conf} dipilih di val, matching IoU 0.5. ` +
    "Sumber runs/eval/yolov8l_960_tiles2_ft_sahi1728/metrics.json. mAP@50 dan mAP@50-95 dihitung dengan matcher proyek " +
    "(conf 0.001, IoU 0.50 dan 0.50:0.05:0.95), karena SAHI tidak bisa dievaluasi lewat model.val Ultralytics. " +
    "Inferensi = median waktu end-to-end per gambar (baca gambar + prediksi + NMS) di RTX 4060. Uji signifikansi: paired bootstrap " +
    "10.000 resample per gambar (docs/report/bootstrap.json). Perbedaan MAE terhadap baseline belum signifikan.");
  const k = K("yolov8l_960_tiles2_ft_sahi1728");
  const items = [
    [f3(k.P), "Precision"], [f3(k.R), "Recall"], [f3(k.ap50), "mAP@50"], [m95(k), "mAP@50-95"],
    [`${k.lat.toFixed(0)} ms`, "Inferensi / gambar\n(RTX 4060)"],
  ];
  const cw = (CW - 4 * 0.2) / 5;
  items.forEach(([v, l], i) => stat(s, v, l, MX + i * (cw + 0.2), TOP + 0.1, cw, 1.45, { size: 26 }));
  const d = boot["E4+SAHI - B0"];
  const y2 = TOP + 1.85;
  card(s, MX, y2, CW, 1.55);
  text(s, "Dibanding baseline B0 (YOLOv8l default @960)", { x: MX + 0.25, y: y2 + 0.15, w: CW - 0.5, h: 0.35, fontFace: "Cambria", bold: true, fontSize: 14 });
  text(s, [
    { text: `F1 +${f3(d.dF1[0])}  [CI 95% ${f3(d.dF1[1])} s.d. ${f3(d.dF1[2])}]  ·  signifikan`, options: { bullet: true, breakLine: true } },
    { text: `Recall +${f3(d.dRecall[0])}  [${f3(d.dRecall[1])} s.d. ${f3(d.dRecall[2])}]  ·  signifikan`, options: { bullet: true, breakLine: true } },
    { text: `MAE ${r2(d.dMAE[0])} lesi/gambar  [${r2(d.dMAE[1])} s.d. ${r2(d.dMAE[2])}]  ·  belum signifikan; inferensi ${k.lat.toFixed(0)} ms vs ${K("yolov8l_960_default").lat.toFixed(0)} ms`, options: { bullet: true } },
  ], { x: MX + 0.25, y: y2 + 0.55, w: CW - 0.5, h: 0.95, fontSize: 13, paraSpaceAfter: 3 });
}

// ---------------------------------------------------------------- 4. Dataset
{
  const s = content("Dataset",
    "Foto wajah ACNE04 dengan anotasi lesi ACNE04-v2. Data dibagi acak per gambar (seed 42): train untuk belajar, val untuk memilih model & threshold, test hanya untuk laporan.",
    "Data",
    "ACNE04 (Wu et al., ICCV 2019), anotasi ACNE04-v2 (Gazeau et al., MICCAI 2024), lisensi akademik. " +
    "Split 80/10/10 seed 42: 963/120/121 gambar, 25.521/3.366/3.507 lesi. Level keparahan kriteria Hayashi (prefix levle0-3). " +
    "Test per level: 51/59/10/1 gambar; level 3 hanya satu gambar sehingga metriknya tidak andal.");
  const y = TOP + 0.1;
  [["1.204", "gambar wajah"], ["32.443", "lesi beranotasi"], ["1", "kelas: acne"]].forEach(([v, l], i) => {
    card(s, MX, y + i * 0.8, 2.6, 0.7);
    text(s, v, { x: MX + 0.2, y: y + i * 0.8, w: 1.4, h: 0.7, fontFace: "Cambria", bold: true, fontSize: 24, color: C.coral, valign: "middle" });
    text(s, l, { x: MX + 1.55, y: y + i * 0.8, w: 1.0, h: 0.7, fontSize: 12, valign: "middle" });
  });
  s.addChart(pres.charts.BAR, [{ name: "Gambar train", labels: ["Level 0", "Level 1", "Level 2", "Level 3"], values: [371, 468, 105, 19] }],
    chartOpts({ x: 3.3, y: y - 0.05, w: 3.4, h: 2.45, barDir: "col", chartColors: [C.teal], showValue: true, dataLabelPosition: "outEnd",
      showTitle: true, title: "Gambar train per level keparahan", showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, barGapWidthPct: 60 }));
  fit(s, path.join(ASSET, "image-2-1.jpg"), 6.9, y, 2.6, 2.3, "Contoh foto wajah dengan anotasi lesi");
  // Bar split
  const by = 4.25, bw = CW;
  text(s, "Data split (per gambar)", { x: MX, y: by - 0.35, w: 4, h: 0.3, bold: true, fontSize: 12 });
  const parts = [[0.8, C.teal, "Train · 80% · 963 gambar · 25.521 lesi", C.white], [0.1, C.yellow, "Val · 120", C.ink], [0.1, C.coral, "Test · 121", C.ink]];
  let bx = MX;
  parts.forEach(([f, col, lab, tc]) => {
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by, w: bw * f - 0.03, h: 0.45, fill: { color: col }, line: { color: col } });
    text(s, lab, { x: bx, y: by, w: bw * f - 0.03, h: 0.45, align: "center", valign: "middle", bold: true, fontSize: 11.5, color: tc });
    bx += bw * f;
  });
}

// ---------------------------------------------------------------- 5. Tantangan lesi kecil
{
  const s = content("Tantangan data: lesi kecil dan padat",
    "Pada input 960 px, mayoritas lesi hanya belasan piksel dan satu wajah bisa memuat ratusan lesi, jauh lebih sulit dari deteksi objek umum.",
    "Data",
    "Ukuran lesi = sisi terpanjang bbox, dihitung eksak per gambar dengan skala letterbox (imgsz / sisi terpanjang foto), " +
    "scripts/diagrams.py. Median 42 px pada foto asli, 11.7 px pada 960. Angka lama (65%, 13 px) adalah aproksimasi yang tidak dapat " +
    "direproduksi dan sudah dikoreksi di docs. Gambar test terpadat memuat 434 lesi.");
  s.addChart(pres.charts.BAR, [{ name: "% lesi < 16 px", labels: ["640 px", "960 px (dipakai)", "1280 px"], values: [91.5, 73.5, 52.4] }],
    chartOpts({ x: MX, y: TOP, w: 3.6, h: 2.7, barDir: "col", chartColors: [C.slate, C.coral, C.slate], showValue: true,
      dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", showTitle: true, title: "Lesi < 16 px setelah resize (%)",
      showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMaxVal: 100, barGapWidthPct: 55 }));
  fit(s, path.join(FIG, "lesion_size_hist.png"), 4.3, TOP, 5.2, 2.55, "Histogram ukuran lesi pada input 960 px");
  const y = 4.35, cw = (CW - 0.6) / 3;
  [["11.7 px", "median ukuran lesi pada input 960"], ["434", "lesi pada gambar test terpadat"], ["Lingkaran", "bentuk anotasi asli (cx, cy, r)"]]
    .forEach(([v, l], i) => {
      card(s, MX + i * (cw + 0.3), y, cw, 0.75);
      text(s, v, { x: MX + i * (cw + 0.3) + 0.15, y, w: 1.35, h: 0.75, fontFace: "Cambria", bold: true, fontSize: 18, color: C.coral, valign: "middle" });
      text(s, l, { x: MX + i * (cw + 0.3) + 1.5, y, w: cw - 1.6, h: 0.75, fontSize: 11, valign: "middle" });
    });
}

// ---------------------------------------------------------------- 6. Pipeline
{
  const s = content("Pipeline end-to-end",
    "Enam tahap dari data mentah sampai prediksi. Setiap tahap punya satu script dan seluruh parameter berada di config YAML sehingga dapat direproduksi.",
    "Metode",
    "Tahap yang disorot coral memuat teknik resolusi: tiling (preprocessing E4) dan SAHI (inferensi). Augmentasi berjalan online " +
    "di dalam training (tidak menambah file). Test set baru disentuh pada evaluasi akhir. Seed 42, deterministic.");
  fit(s, path.join(FIG, "pipeline.png"), MX, TOP + 0.05, CW, 2.1, "Diagram pipeline enam tahap");
  const y = TOP + 2.35, cw = (CW - 0.6) / 3;
  numCard(s, "A", "Data terkontrol", "Split tetap (seed 42); test hanya untuk laporan akhir.", MX, y, cw, 1.45, { size: 12 });
  numCard(s, "B", "Resolusi efektif", "Tile 2×2 saat training (E4) dan SAHI saat inferensi.", MX + cw + 0.3, y, cw, 1.45, { size: 12, circle: C.coral });
  numCard(s, "C", "Evaluasi tunggal", "Semua varian diukur lewat evaluate.py, protokol sama.", MX + 2 * (cw + 0.3), y, cw, 1.45, { size: 12 });
}

// ---------------------------------------------------------------- 7. Preprocessing circle -> bbox
{
  const s = content("Preprocessing 1: lingkaran ke bounding box",
    "Anotasi asli berupa lingkaran (cx, cy, r), sedangkan YOLO membutuhkan bounding box. Setiap lingkaran diubah menjadi kotak persegi lalu dinormalisasi.",
    "Metode",
    "Kotak = [cx-r, cy-r, cx+r, cy+r], di-clip ke batas gambar, dinormalisasi ke [0,1] (format YOLO txt). Gambar disalin tanpa resize; " +
    "letterbox ke 960 px dilakukan Ultralytics saat training. Konversi ini membuat sudut kotak berada di luar lesi bulat, sehingga " +
    "mAP@50-95 (IoU ketat) tertekan; karena itu mAP@50 dan metrik counting lebih representatif. Contoh memakai ground truth.");
  const iw = 2.2, gap = 0.35, y = TOP + 0.05;
  ["1. Foto asli", "2. Anotasi lingkaran", "3. Bounding box"].forEach((lab, i) => {
    const x = MX + i * (iw + gap);
    fit(s, path.join(ASSET, `image-5-${i + 1}.jpg`), x, y, iw, iw, lab);
    text(s, lab, { x, y: y + iw + 0.05, w: iw, h: 0.3, align: "center", fontSize: 11.5, bold: true });
  });
  const x = MX + 3 * (iw + gap) - 0.1, w = W - MX - x;
  card(s, x, y, w, 3.35);
  text(s, [
    { text: "Konversi", options: { bold: true, fontFace: "Cambria", breakLine: true } },
    { text: "Lingkaran → kotak persegi", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Clipping & normalisasi", options: { bold: true, fontFace: "Cambria", breakLine: true } },
    { text: "Koordinat ke [0, 1]", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Hasil", options: { bold: true, fontFace: "Cambria", breakLine: true } },
    { text: "25.521 / 3.366 / 3.507 lesi (train / val / test)" },
  ], { x: x + 0.15, y: y + 0.15, w: w - 0.3, h: 3.05, fontSize: 12 });
}

// ---------------------------------------------------------------- 8. Tiling
{
  const s = content("Preprocessing 2: tiling 2×2 (E4)",
    "Memotong foto menjadi 4 tile beroverlap membuat lesi tampil ±1,9× lebih besar pada input 960 px, tanpa menambah kebutuhan VRAM.",
    "Metode",
    "scripts/data/preprocess.py --grid 2 --overlap 0.1 membuat data_tiles2/: foto asli (hardlink) + 4 tile per foto, train 963 -> 4.815 gambar. " +
    "Label lesi yang terpotong dipertahankan bila >= 50% bagiannya terlihat. Test TIDAK di-tile: evaluasi selalu pada foto asli. " +
    "Dasar: slicing-aided fine-tuning (Akyon et al., ICIP 2022). Contoh: foto train levle1_302.");
  fit(s, path.join(FIG, "tiling_example.png"), MX, TOP, 6.3, 3.55, "Contoh tiling 2x2 dan label pada satu tile");
  const x = 7.05, w = W - MX - x;
  stat(s, "0,53", "skala input tile\n(vs 0,28 foto utuh)", x, TOP, w, 1.15, { size: 26 });
  stat(s, "4.815", "gambar train\n(963 asli + 3.852 tile)", x, TOP + 1.3, w, 1.15, { size: 26 });
  stat(s, "10%", "overlap antar tile;\nlabel ≥ 50% terlihat", x, TOP + 2.6, w, 1.15, { size: 26 });
}

// ---------------------------------------------------------------- 9. Augmentasi
{
  const s = content("Augmentasi data",
    "Augmentasi memberi variasi acak pada gambar training di setiap epoch sehingga model lebih tahan terhadap perbedaan posisi, ukuran, dan pencahayaan.",
    "Metode",
    "Parameter E3/E4 dari configs/yolov8l_960.yaml; B0 memakai default Ultralytics (flipud 0, degrees 0, mixup 0). " +
    "Temuan audit: copy_paste hanya bekerja untuk task segmentasi sehingga no-op pada bbox; paket albumentations tidak terpasang " +
    "sehingga augmentasi bawaan CLAHE/Blur/ToGray tidak pernah aktif di run mana pun.");
  table(s, ["Teknik", "B0 (default)", "E3 / E4", "Fungsi"], [
    ["Mosaic", "1.0", "1.0 (off 10 / 3 epoch akhir)", "Gabung 4 gambar; konteks & skala beragam"],
    ["Mixup", "0", "0.1", "Campur dua gambar; kurangi overfitting"],
    ["Flip horizontal / vertikal", "0.5 / 0", "0.5 / 0.5", "Variasi orientasi"],
    ["Rotasi", "0°", "± 10°", "Variasi sudut pengambilan foto"],
    ["Scale / translate", "± 50% / 10%", "± 50% / 10%", "Variasi ukuran lesi & posisi"],
    ["HSV (h / s / v)", "0.015 / 0.7 / 0.4", "0.015 / 0.7 / 0.4", "Variasi warna kulit & cahaya"],
  ], MX, TOP + 0.05, CW, [2.1, 1.55, 2.15, 3.2], { rowH: 0.36, fontSize: 12 });
  card(s, MX, 4.4, CW, 0.62, C.coralBg);
  text(s, [
    { text: "Temuan: ", options: { bold: true } },
    { text: "copy_paste tidak aktif untuk label bbox (no-op), dan albumentations tidak terpasang, sehingga CLAHE/Blur bawaan tidak pernah dipakai." },
  ], { x: MX + 0.2, y: 4.4, w: CW - 0.4, h: 0.62, fontSize: 12, valign: "middle" });
}

// ---------------------------------------------------------------- 10. Konfigurasi training
{
  const s = content("Konfigurasi model & training",
    "Arsitektur sama (YOLOv8l, transfer learning dari COCO); yang dibedakan adalah setelan optimasi, augmentasi, dan data.",
    "Eksperimen",
    "YOLOv8l 43.6M parameter. Batch 4 pada 960 px karena VRAM 8 GB; Ultralytics mengakumulasi gradien hingga batch nominal 64. " +
    "B0 memakai optimizer=auto yang memilih AdamW lr 0.002 (lr0 diabaikan). E4 tidak dilatih dari nol karena anggaran GPU <= 4 jam. " +
    "Waktu training: E3 ±2,2 jam; E4 65 menit (10 epoch pada 4.815 gambar).");
  table(s, ["", "B0 baseline", "E3 tuned", "E4 tile fine-tune"], [
    ["Inisialisasi", "COCO yolov8l.pt", "COCO yolov8l.pt", "E3 best.pt"],
    ["Data train", "963 foto asli", "963 foto asli", "4.815 (asli + tile 2×2)"],
    ["Optimizer", "auto → AdamW lr 0.002", "AdamW lr0 1e-3, cosine", "AdamW lr0 2e-4, cosine"],
    ["Loss box / cls / dfl", "7.5 / 0.5 / 1.5", "10 / 0.3 / 2.0", "10 / 0.3 / 2.0"],
    ["Augmentasi", "default", "custom (slide 9)", "custom (slide 9)"],
    ["Epoch (best)", "100 (80)", "100 (98)", "10 (10)"],
    ["Umum", { text: "960 px · batch 4 (akumulasi ke 64) · seed 42 · RTX 4060 8 GB · ultralytics 8.4.170", options: { colspan: 3 } }],
  ], MX, TOP + 0.05, CW, [1.9, 2.3, 2.4, 2.4], { rowH: 0.4, fontSize: 12 });
}

// ---------------------------------------------------------------- 11. Riset teknik
{
  const s = content("Riset teknik peningkatan performa",
    "Teknik dipilih berdasarkan masalah inti (lesi tiny dan padat), lalu diuji dengan protokol evaluasi yang sama.",
    "Eksperimen",
    "Sumber: docs/RESEARCH.md bagian A. Paper SAHI (Akyon 2022) melaporkan AP naik 5-7% hanya dari sliced inference, dan lebih besar " +
    "bila dikombinasikan dengan fine-tuning pada slice. Ultralytics: imgsz lebih besar membantu objek kecil. imgsz 1280, head P2, " +
    "dan 300 epoch tidak muat dalam anggaran GPU <= 4 jam dan VRAM 8 GB.");
  const G = { text: "positif, signifikan", color: "2E7D4F" }, N = { text: "negatif", color: "C0392B" };
  table(s, ["Teknik", "Kategori", "Diuji", "Hasil pada test"], [
    ["SAHI sliced inference (1728 px)", "Inferensi / resolusi", "E3, E4, B0", { ...G, text: "F1 +0.047 pada E3 (signifikan)" }],
    ["Slicing-aided fine-tuning (tile 2×2)", "Preprocessing / data", "E4", "+0.010 F1 vs E3+SAHI (tidak signifikan)"],
    ["Augmentasi + bobot loss tuned", "Augmentasi / hyperparameter", "E3 vs B0", "setara default (ΔF1 −0.005)"],
    ["Test-time augmentation", "Inferensi", "E3", { ...N, text: "AP@50 0.491 → 0.467" }],
    ["Threshold conf dipilih di val", "Post-processing", "semua", "MAE E3 9.85 → 8.69 (vs conf 0.25)"],
    ["imgsz 1280, head P2, 300 epoch", "Resolusi / arsitektur", "–", "belum diuji (anggaran)"],
  ], MX, TOP + 0.05, CW, [2.9, 2.05, 1.15, 2.9], { rowH: 0.42, fontSize: 12 });
}

// ---------------------------------------------------------------- 12. Desain eksperimen
{
  const s = content("Desain eksperimen",
    "Tujuh varian dari dua model dasar memisahkan efek tuning training, teknik inferensi, dan data tile.",
    "Eksperimen",
    "B0 vs E3 menguji efek tuning. Cabang E3 menguji teknik inferensi (TTA, SAHI) tanpa training baru. E4 = fine-tune E3 pada data tile, " +
    "lalu dievaluasi plain dan dengan SAHI. B0 + SAHI adalah kontrol: apakah SAHI saja cukup tanpa model E3. " +
    "Angka pada node: F1 dan MAE di test, conf dipilih di val.");
  fit(s, path.join(FIG, "experiment_tree.png"), MX, TOP - 0.05, CW, 3.65, "Silsilah varian eksperimen dengan F1 dan MAE");
}

// ---------------------------------------------------------------- 13. SAHI
{
  const s = content("Inferensi SAHI (Slicing Aided Hyper Inference)",
    "Foto dipotong menjadi slice 1728 px yang masing-masing di-resize ke 960, sehingga skala input naik dari 0,28 ke 0,56.",
    "Eksperimen",
    "Parameter (evaluate.py/predict.py): slice 1728 px, overlap 20%, tiap slice di-resize ke imgsz 960, hasil digabung dengan NMS IoU 0.5. " +
    "Pada foto 3112x3456 SAHI membuat 9 slice, 3 di antaranya nyaris duplikat (bergeser 1 px karena slice terakhir digeser ke tepi). " +
    "Opini: menghapus slice duplikat dapat memangkas latency sekitar sepertiga tanpa mengubah akurasi.");
  fit(s, path.join(FIG, "sahi_example.png"), MX, TOP - 0.05, 3.7, 3.65, "Grid slice SAHI pada foto test");
  const x = 4.55, w = W - MX - x, h = 1.05;
  numCard(s, 1, "Slice + resize", "9 slice 1728 px (overlap 20%) → tiap slice 960 px.", x, TOP, w, h, { size: 12 });
  numCard(s, 2, "Prediksi per slice", "Lesi tampil 2× lebih besar dibanding foto utuh.", x, TOP + h + 0.18, w, h, { size: 12 });
  numCard(s, 3, "Gabung (NMS)", `Biaya: ${V.e4s["Latency ms"].toFixed(0)} ms vs ${V.e3["Latency ms"].toFixed(0)} ms per gambar; 3 slice nyaris duplikat.`, x, TOP + 2 * (h + 0.18), w, h, { size: 12, circle: C.coral });
}

// ---------------------------------------------------------------- 14. Protokol evaluasi
{
  const s = content("Protokol evaluasi",
    "Angka hanya bisa dibandingkan bila diukur dengan cara yang sama: satu script, tuning hanya di val, dan test dipakai sekali di akhir.",
    "Eksperimen",
    "mAP Ultralytics: conf 0.001, iou 0.7, max_det 1000, imgsz = training. AP@50 internal: matching greedy per skor, IoU >= 0.5, " +
    "interpolasi 101 titik; mAP@50-95 internal = rata-rata AP pada IoU 0.50:0.05:0.95. Pada prediksi yang sama, selisih terhadap Ultralytics " +
    "<= 0.0002 (mAP@50) dan <= 0.0024 (mAP@50-95); berlaku juga untuk SAHI. " +
    "Confusion matrix lama dibuat pada conf 0.001 (57.574 FP) sehingga tidak bermakna; versi baru memakai conf operasional dari val. " +
    "Audit pipeline: 17 poin, 5 perbaikan (PIPELINE.md).");
  fit(s, path.join(FIG, "eval_protocol.png"), MX, TOP, CW, 1.85, "Diagram alur protokol evaluasi");
  const y = TOP + 2.1, cw = (CW - 0.3) / 2;
  numCard(s, 1, "Confusion matrix tanpa TN", "Sel background–background tidak terdefinisi pada deteksi objek (jumlah \"bukan lesi\" tak terhingga), sehingga ditandai N/A.", MX, y, cw, 1.45, { size: 12 });
  numCard(s, 2, "Signifikansi statistik", "Paired bootstrap 10.000× per gambar test; selisih dianggap nyata bila CI 95% tidak memuat 0.", MX + cw + 0.3, y, cw, 1.45, { size: 12 });
}

// ---------------------------------------------------------------- 15. Standar metrik
{
  const s = content("Standar metrik minimum dari literatur",
    "Ambang \"layak\" diturunkan dari studi deteksi jerawat, terutama yang memakai dataset ACNE04/ACNE04-v2.",
    "Eksperimen",
    "Sumber lengkap & URL: docs/RESEARCH.md bagian B. Angka literatur tidak bisa dibandingkan langsung: anotasi ACNE04-v2 jauh lebih " +
    "lengkap (lesi kecil) dibanding ACNE04 v1, beberapa studi mengaugmentasi sebelum split (risiko kebocoran), dan protokol berbeda. " +
    "Ambang adalah opini berbasis bukti; siap klinis membutuhkan validasi terhadap beberapa dermatolog (ICC >= 0.75).");
  table(s, ["Studi", "Data / tugas", "Metrik"], [
    ["Sangha & Rizvi 2021", "ACNE04, 403 gambar, YOLOv5", "mAP@50 37.97%"],
    ["Li et al. 2023", "Identifikasi lesi + grading", "AP lesi 0.507, recall 0.775"],
    ["Gazeau et al. 2024 (AcneAI)", "ACNE04-v2 (dataset kita)", "TPR 0.52 @ FPR 0.09"],
    ["Seité et al. 2019", "Smartphone, per tipe lesi", "F1 61% – 84%"],
  ], MX, TOP + 0.05, 5.3, [1.9, 2.0, 1.4], { rowH: 0.42, fontSize: 11.5 });
  table(s, ["Metrik", "Minimum", "Baik"], [
    ["mAP@50", "≥ 0.45", "≥ 0.55"], ["Recall", "≥ 0.50", "≥ 0.60"], ["Precision", "≥ 0.50", "≥ 0.60"],
    ["MAE counting", "≤ 10", "≤ 7"], ["Pearson r", "≥ 0.85", "≥ 0.90"], ["Inferensi", "< 1 s", "< 100 ms"],
  ], 6.1, TOP + 0.05, W - MX - 6.1, [1.3, 1.05, 1.05], { rowH: 0.42, fontSize: 11.5 });
}

// ---------------------------------------------------------------- 16. Kurva training
{
  const b0 = curve("yolov8l_960_default"), e3 = curve("yolov8l_960");
  const s = content("Kurva training",
    "B0 mencapai puncak di epoch 80, sedangkan E3 masih naik di akhir training sehingga belum konvergen penuh.",
    "Hasil",
    "mAP@50 split val per epoch dari runs/detect/<run>/results.csv. B0 terhenti di epoch 45 karena RAM sistem lalu dilanjutkan dari last.pt " +
    "(epoch duplikat dibersihkan). Kurva E4 tidak diplot karena val-nya berisi tile (data_tiles2), tidak sebanding. " +
    "Puncak val B0 lebih tinggi, tetapi di test mAP@50 keduanya setara (0.4997 vs 0.4980).");
  const best = (c) => c.reduce((a, b) => (b[1] > a[1] ? b : a));
  s.addChart(pres.charts.LINE, [
    { name: "B0 baseline", labels: b0.map((r) => String(r[0])), values: b0.map((r) => r[1]) },
    { name: "E3 tuned", labels: e3.map((r) => String(r[0])), values: e3.map((r) => r[1]) },
  ], chartOpts({ x: MX, y: TOP - 0.05, w: 6.2, h: 3.65, chartColors: [VC.b0, VC.e3], lineSize: 2, lineDataSymbol: "none",
    showLegend: true, legendPos: "b", catAxisLabelFrequency: 10, valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
    showTitle: true, title: "mAP@50 split val per epoch", catAxisTitle: "Epoch", showCatAxisTitle: true, catAxisTitleColor: C.slate, catAxisTitleFontSize: 10 }));
  const x = 7.0, w = W - MX - x;
  const [be, bv] = best(b0), [ee, ev] = best(e3);
  stat(s, f3(bv), `B0 puncak val\n(epoch ${be})`, x, TOP, w, 1.1, { size: 24 });
  stat(s, f3(ev), `E3 puncak val\n(epoch ${ee}/100)`, x, TOP + 1.25, w, 1.1, { size: 24 });
  card(s, x, TOP + 2.5, w, 1.1);
  text(s, "E3 belum konvergen: training lebih panjang masuk roadmap.", { x: x + 0.15, y: TOP + 2.5, w: w - 0.3, h: 1.1, fontSize: 12, valign: "middle" });
}

// ---------------------------------------------------------------- 17. Riwayat mAP
{
  const s = content("Riwayat eksperimen: mAP@50 di test",
    "Resolusi 960 dan model besar membawa mAP ke ±0,50; tuning dan fine-tune tile tidak menembusnya. SAHI menaikkan AP@50 ke 0,55.",
    "Hasil",
    "E1 dari dokumentasi lama (protokol tidak lengkap terdokumentasi). E2-E4: mAP@50 Ultralytics (conf 0.001, iou 0.7). " +
    "Varian SAHI tidak punya mAP Ultralytics, jadi yang ditampilkan AP@50 internal (definisi setara, selisih <= 0.009 pada mode plain). " +
    "Sumber: docs/EXPERIMENTS.md dan runs/eval/*/metrics.json.");
  const labs = ["E1 v8s 640", "E2 v8l 640", "E3 v8l 960", "B0 v8l 960", "E4 tile-FT", "E3 + SAHI*", "E4 + SAHI*"];
  const vals = [0.3582, 0.4373, V.e3["mAP50 (Ultralytics)"], V.b0["mAP50 (Ultralytics)"], E4.detection_test_ultralytics.mAP50,
    V.e3s["AP50 (internal)"], V.e4s["AP50 (internal)"]];
  s.addChart(pres.charts.BAR, [{ name: "mAP@50", labels: labs, values: vals }],
    chartOpts({ x: MX, y: TOP - 0.05, w: CW, h: 3.3, barDir: "col", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
      chartColors: [C.slate, C.slate, C.teal, C.teal, C.teal, C.coral, C.coral], showLegend: false, valAxisHidden: true,
      valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 0.62, barGapWidthPct: 45 }));
  caption(s, "* AP@50 internal (matcher proyek, IoU 0.5), karena SAHI tidak memiliki mAP Ultralytics.  Abu-abu: 640 px · teal: 960 px · coral: dengan SAHI.", MX, 4.9, CW);
}

// ---------------------------------------------------------------- 18. Metrik utama per varian
{
  const s = content("Metrik utama per varian",
    "Precision, Recall, mAP@50, mAP@50-95, dan waktu inferensi untuk semua varian, diukur dengan protokol yang sama di split test.",
    "Hasil",
    "Split test (121 gambar, 3.507 lesi). Precision/Recall pada conf yang dipilih di val per varian (kolom conf), matching IoU 0.5. " +
    "mAP@50 dan mAP@50-95 dihitung dari prediksi conf 0.001 dengan matcher proyek (scripts/analysis.py), sehingga varian SAHI sebanding " +
    "dengan mode plain; pada prediksi yang sama selisihnya terhadap Ultralytics <= 0.0002 (mAP@50) dan <= 0.0024 (mAP@50-95). " +
    "mAP@50-95 rendah untuk semua varian karena bbox diturunkan dari anotasi lingkaran dan lesi sangat kecil: pergeseran 1-2 px " +
    "sudah menurunkan IoU di bawah 0.75. Inferensi = median end-to-end per gambar, RTX 4060, batch 1. Sumber: runs/eval/<run>/metrics.json.");
  const best = (f) => Math.max(...RUNS.map(([, r]) => f(K(r)) ?? -1));
  const cell = (k, f, fmt) => {
    const v = f(k);
    if (v === null) return { text: "—*", color: C.slate };
    return { text: fmt(v), bold: v === best(f), color: v === best(f) ? C.coral : C.ink };
  };
  const mode = { "yolov8l_960_tta": "TTA", "yolov8l_960_default_sahi1728": "SAHI", "yolov8l_960_sahi1728": "SAHI", "yolov8l_960_tiles2_ft_sahi1728": "SAHI" };
  const fastest = Math.min(...RUNS.map(([, r]) => K(r).lat));
  table(s, ["Varian", "Mode", "conf", "Precision", "Recall", "mAP@50", "mAP@50-95", "Inferensi"], RUNS.map(([, r]) => {
    const k = K(r);
    return [k.label, mode[r] || "plain", k.conf.toFixed(2), cell(k, (x) => x.P, f3), cell(k, (x) => x.R, f3),
      cell(k, (x) => x.ap50, f3), cell(k, (x) => x.ap5095, f3),
      { text: `${k.lat.toFixed(0)} ms`, bold: k.lat === fastest, color: k.lat === fastest ? C.coral : C.ink }];
  }), MX, TOP, CW, [1.55, 0.8, 0.7, 1.05, 1.0, 1.0, 1.15, 1.75], { rowH: 0.42, fontSize: 12.5 });
  caption(s, "Coral = terbaik per kolom. Test, IoU 0.5; mAP dari matcher proyek (setara Ultralytics)." +
    (missing95.length ? "  * menunggu evaluasi ulang (scripts/rerun_evals.ps1)." : ""), MX, 4.95, CW);
}

// ---------------------------------------------------------------- 19. Hasil deteksi
{
  const s = content("Hasil deteksi",
    "Varian dengan SAHI unggul di Precision, Recall, dan mAP@50; E4 + SAHI tertinggi dengan Precision 0,611 dan Recall 0,536.",
    "Hasil",
    "Split test; Precision/Recall pada conf dipilih di val, mAP dari prediksi conf 0.001, matching IoU 0.5 (mAP@50-95: IoU 0.50:0.05:0.95). " +
    "Urutan seri = urutan F1 menurun. " +
    `E3 + TTA: mAP@50 ${f3(K("yolov8l_960_tta").ap50)}, F1 ${f3(V.tta.F1)} (lebih buruk). B0 + SAHI: mAP@50 ${f3(K("yolov8l_960_default_sahi1728").ap50)}, ` +
    `F1 ${f3(V.b0s.F1)}: SAHI tidak membantu B0. Opini: augmentasi skala E3 membuat model tahan terhadap perubahan skala saat slicing.`);
  const series = [["E4 + SAHI", "yolov8l_960_tiles2_ft_sahi1728", VC.e4s], ["E3 + SAHI", "yolov8l_960_sahi1728", VC.e3s],
    ["B0 baseline", "yolov8l_960_default", VC.b0], ["E3 tuned", "yolov8l_960", VC.e3]];
  s.addChart(pres.charts.BAR, series.map(([n, r]) => {
    const k = K(r);
    return { name: n, labels: ["Precision", "Recall", "mAP@50", "mAP@50-95"], values: [k.P, k.R, k.ap50, k.ap5095 ?? 0] };
  }), chartOpts({ x: MX, y: TOP - 0.05, w: 6.3, h: 3.65, barDir: "col", barGrouping: "clustered", chartColors: series.map((r) => r[2]),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 8, showLegend: true, legendPos: "b",
    valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", barGapWidthPct: 40 }));
  const x = 7.0, w = W - MX - x;
  table(s, ["Varian", "F1"], [
    [{ text: "E4 + SAHI", bold: true }, { text: f3(V.e4s.F1), bold: true }], ["E3 + SAHI", f3(V.e3s.F1)], ["B0 + SAHI", f3(V.b0s.F1)],
    ["B0", f3(V.b0.F1)], ["E3", f3(V.e3.F1)], ["E3 + TTA", f3(V.tta.F1)],
  ], x, TOP, w, [1.5, w - 1.5], { rowH: 0.36, fontSize: 12 });
  caption(s, "F1 = rata-rata harmonik Precision & Recall; test, IoU 0.5.", x, TOP + 2.6, w, 0.5);
}

// ---------------------------------------------------------------- 20. Confusion matrix
{
  const b = B0M.detection, e = BEST.detection;
  const s = content("Confusion matrix & jenis kesalahan",
    `E4 + SAHI menambah ${e.TP - b.TP} lesi yang ditemukan (TP) dan mengurangi ${b.FP - e.FP} prediksi salah (FP) dibanding baseline.`,
    "Hasil",
    `B0 pada conf ${B0M.conf}, E4 + SAHI pada conf ${BEST.conf} (keduanya dipilih di val), test, IoU 0.5. ` +
    "FP lokalisasi = prediksi yang overlap 0.1 <= IoU < 0.5 dengan lesi GT (kotak salah skala/posisi); FP latar = tanpa overlap. " +
    "SAHI memangkas FP lokalisasi karena kotak lebih presisi pada resolusi tinggi, tetapi menambah FP latar. " +
    "Sebagian FP latar kemungkinan lesi yang tidak dianotasi (opini).");
  fit(s, path.join(FIG, "cm_yolov8l_960_default.png"), MX, TOP - 0.05, 3.15, 2.85, "Confusion matrix B0");
  fit(s, path.join(FIG, "cm_yolov8l_960_tiles2_ft_sahi1728.png"), MX + 3.3, TOP - 0.05, 3.15, 2.85, "Confusion matrix E4 + SAHI");
  caption(s, `B0 baseline · conf ${B0M.conf}`, MX, TOP + 2.85, 3.15);
  caption(s, `E4 + SAHI · conf ${BEST.conf}`, MX + 3.3, TOP + 2.85, 3.15);
  const x = 7.15, w = W - MX - x;
  table(s, ["Jenis FP", "B0", "E4+SAHI"], [
    ["Lokalisasi", id(B0M.fp_breakdown.localization), id(BEST.fp_breakdown.localization)],
    ["Latar", id(B0M.fp_breakdown.background), id(BEST.fp_breakdown.background)],
    [{ text: "Total FP" }, id(b.FP), id(e.FP)],
  ], x, TOP, w, [0.95, 0.65, w - 1.6], { rowH: 0.36, fontSize: 11.5 });
  card(s, x, TOP + 1.65, w, 1.6);
  text(s, "Sel background–background = N/A: TN tidak terdefinisi pada deteksi objek.", { x: x + 0.15, y: TOP + 1.7, w: w - 0.3, h: 1.5, fontSize: 12, valign: "middle" });
}

// ---------------------------------------------------------------- 21. Kecepatan inferensi
{
  const order = [...RUNS].map(([, r]) => K(r)).sort((a, b) => a.lat - b.lat);
  const modeColor = (k) => (k.run.includes("sahi") ? C.coral : k.run.endsWith("_tta") ? C.yellow : C.teal);
  const e3 = K("yolov8l_960"), best = K("yolov8l_960_tiles2_ft_sahi1728");
  const s = content("Kecepatan inferensi",
    `Mode plain ±${e3.lat.toFixed(0)} ms per gambar (< 100 ms); SAHI ±9× lebih lambat (±${best.lat.toFixed(0)} ms) tetapi masih < 1 s.`,
    "Hasil",
    "Inferensi = median waktu end-to-end per gambar test (baca gambar + prediksi + NMS/penggabungan slice), batch 1, RTX 4060 8GB, " +
    "gambar pertama (warmup) diabaikan; sumber latency_ms_median di runs/eval/<run>/metrics.json. " +
    `Waktu model saja dari model.val Ultralytics (batch 4): B0 ${f2(K("yolov8l_960_default").infer)} ms, E3 ${f2(e3.infer)} ms, ` +
    `TTA ${f2(K("yolov8l_960_tta").infer)} ms per gambar. SAHI menjalankan 9 slice per foto 3112x3456, 3 di antaranya nyaris duplikat ` +
    "(geser 1 px); menghapusnya diperkirakan memangkas sekitar sepertiga waktu (opini).");
  s.addChart(pres.charts.BAR, [{ name: "Inferensi (ms)", labels: order.map((k) => k.label), values: order.map((k) => k.lat) }],
    chartOpts({ x: MX, y: TOP - 0.05, w: 5.9, h: 3.65, barDir: "bar", chartColors: order.map(modeColor), invertedColors: order.map(modeColor),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", showLegend: false, valAxisHidden: true,
      valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 780, catAxisOrientation: "maxMin",
      showTitle: true, title: "Median ms per gambar (lebih kecil lebih cepat)", barGapWidthPct: 40 }));
  const x = 6.65, w = W - MX - x;
  stat(s, `${e3.lat.toFixed(0)} ms`, `E3 plain · ±${(1000 / e3.lat).toFixed(0)} gambar/detik`, x, TOP, w, 1.05, { size: 24, color: C.teal });
  stat(s, `${best.lat.toFixed(0)} ms`, `E4 + SAHI · ±${(1000 / best.lat).toFixed(1).replace(".", ",")} gambar/detik`, x, TOP + 1.2, w, 1.05, { size: 24 });
  card(s, x, TOP + 2.4, w, 1.2);
  text(s, [
    { text: "■ ", options: { color: C.teal } }, { text: "plain   " },
    { text: "■ ", options: { color: C.yellow } }, { text: "TTA   " },
    { text: "■ ", options: { color: C.coral } }, { text: "SAHI", options: { breakLine: true } },
    { text: "Standar: < 1 s minimum, < 100 ms baik.", options: { color: C.ink2 } },
  ], { x: x + 0.15, y: TOP + 2.5, w: w - 0.3, h: 1.0, fontSize: 11.5, valign: "middle" });
}

// ---------------------------------------------------------------- 22. Counting
{
  const s = content("Hasil counting",
    "Semua varian cenderung under-count (bias negatif); E4 + SAHI memiliki MAE terendah dan korelasi tinggi dengan ground truth.",
    "Hasil",
    "MAE = rata-rata |pred - GT| lesi per gambar; bias = rata-rata (pred - GT). Test 121 gambar, conf dipilih di val. " +
    `RMSE B0 ${f2(V.b0.RMSE)} vs E4 + SAHI ${f2(V.e4s.RMSE)}. Satu gambar level 3 (434 lesi) mendominasi RMSE; tanpa gambar itu MAE B0 7.18 vs E4 + SAHI 6.94. ` +
    "Perbedaan MAE antar varian tidak signifikan (bootstrap).");
  const order = [["E4 + SAHI", V.e4s], ["E3 + SAHI", V.e3s], ["E3 tuned", V.e3], ["E3 + TTA", V.tta], ["B0 baseline", V.b0], ["B0 + SAHI", V.b0s]];
  s.addChart(pres.charts.BAR, [{ name: "MAE", labels: order.map((r) => r[0]), values: order.map((r) => r[1].MAE) }],
    chartOpts({ x: MX, y: TOP - 0.05, w: 4.0, h: 3.65, barDir: "bar", chartColors: [C.teal], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.00", showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 12,
      catAxisOrientation: "maxMin", showTitle: true, title: "MAE (lesi/gambar, lebih kecil lebih baik)", barGapWidthPct: 45 }));
  fit(s, path.join(FIG, "count_scatter.png"), 4.65, TOP - 0.05, W - MX - 4.65, 2.6, "Scatter jumlah lesi prediksi vs ground truth");
  const y = TOP + 2.7, cw = (W - MX - 4.65 - 0.2) / 2;
  stat(s, f3(V.e4s["Pearson r"]), "Pearson r E4 + SAHI\n(B0: " + f3(V.b0["Pearson r"]) + ")", 4.65, y, cw, 0.95, { size: 20 });
  stat(s, f2(V.e4s.Bias), "bias lesi/gambar\n(B0: " + f2(V.b0.Bias) + ")", 4.65 + cw + 0.2, y, cw, 0.95, { size: 20 });
}

// ---------------------------------------------------------------- 23. Per level
{
  const pl = (run) => evalm(run).test_conf_val_tuned.per_level;
  const runs = [["E4 + SAHI", "yolov8l_960_tiles2_ft_sahi1728", VC.e4s], ["E3 + SAHI", "yolov8l_960_sahi1728", VC.e3s],
    ["B0 baseline", "yolov8l_960_default", VC.b0], ["E3 tuned", "yolov8l_960", VC.e3]];
  const lv = ["level0", "level1", "level2"];
  const n = lv.map((l) => pl(runs[0][1])[l].counting.n_images);
  const s = content("Akurasi per level keparahan",
    "Error counting membesar seiring kepadatan lesi; wajah level 3 (n = 1) adalah under-count terbesar.",
    "Hasil",
    "MAE per level di test, conf dipilih di val. Level 2 hanya 10 gambar dan level 3 hanya 1 gambar, sehingga angka keduanya tidak andal; " +
    "split tidak terstratifikasi menjadi item roadmap.");
  s.addChart(pres.charts.BAR, runs.map(([nm, r]) => ({ name: nm, labels: lv.map((l, i) => `Level ${i} (n=${n[i]})`), values: lv.map((l) => pl(r)[l].counting.MAE) })),
    chartOpts({ x: MX, y: TOP - 0.05, w: 6.3, h: 3.65, barDir: "col", barGrouping: "clustered", chartColors: runs.map((r) => r[2]),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 9, showLegend: true, legendPos: "b",
      valAxisMinVal: 0, valAxisMaxVal: 11, valAxisLabelFormatCode: "0", showTitle: true, title: "MAE per level (lesi/gambar)", barGapWidthPct: 40 }));
  const x = 7.0, w = W - MX - x, l3 = (r) => pl(r).level3.counting;
  card(s, x, TOP, w, 3.55);
  text(s, "Level 3 · 1 gambar", { x: x + 0.2, y: TOP + 0.15, w: w - 0.4, h: 0.35, fontFace: "Cambria", bold: true, fontSize: 14 });
  text(s, [
    { text: `Ground truth: ${l3(runs[0][1]).total_gt} lesi`, options: { bullet: true, breakLine: true } },
    { text: `B0: ${l3(runs[2][1]).total_pred} terhitung`, options: { bullet: true, breakLine: true } },
    { text: `E3 + SAHI: ${l3(runs[1][1]).total_pred}`, options: { bullet: true, breakLine: true } },
    { text: `E4 + SAHI: ${l3(runs[0][1]).total_pred}`, options: { bullet: true } },
  ], { x: x + 0.2, y: TOP + 0.6, w: w - 0.4, h: 1.6, fontSize: 12.5, paraSpaceAfter: 4 });
  text(s, "SAHI membantu, tetapi lesi tiny yang berdempetan tetap terlewat.", { x: x + 0.2, y: TOP + 2.35, w: w - 0.4, h: 1.0, fontSize: 12, color: C.ink2 });
}

// ---------------------------------------------------------------- 24. Signifikansi
{
  const s = content("Uji signifikansi",
    "Kenaikan deteksi dari SAHI nyata secara statistik; perbedaan MAE antar varian belum bisa dibedakan dari noise dengan 121 gambar test.",
    "Hasil",
    "Paired bootstrap: 10.000 resample gambar test (seed 42), selisih metrik varian A - B dihitung pada sampel yang sama. " +
    "Teal = CI 95% tidak memuat 0 (signifikan). Signifikan: E4+SAHI - B0 (F1 +0.051, recall +0.059), E4+SAHI - E3 (F1 +0.057), " +
    "E3+SAHI - E3 (F1 +0.047). Tidak signifikan: E4+SAHI - E3+SAHI (+0.010), E3 - B0, E4 - E3, B0+SAHI - B0, dan semua selisih MAE.");
  fit(s, path.join(FIG, "bootstrap_ci.png"), MX, TOP - 0.05, CW, 3.7, "Forest plot selisih F1 dan MAE dengan CI 95%");
}

// ---------------------------------------------------------------- 25. Contoh prediksi
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

// ---------------------------------------------------------------- 26. Hasil vs standar
{
  const t = BEST, v = V.e4s, kb = K("yolov8l_960_default"), ke = K("yolov8l_960_tiles2_ft_sahi1728");
  const ok = (s) => ({ text: s, color: "2E7D4F", bold: true });
  const s = content("Hasil vs standar minimum",
    "E4 + SAHI memenuhi semua ambang minimum layak; Precision dan Pearson r mencapai ambang baik. Baseline B0 gagal di Recall.",
    "Hasil",
    `Test, conf ${t.conf} dipilih di val (runs/eval/yolov8l_960_tiles2_ft_sahi1728/metrics.json). mAP@50 dan mAP@50-95 dari matcher proyek ` +
    "(setara Ultralytics, berlaku juga untuk SAHI). Literatur deteksi jerawat tidak melaporkan mAP@50-95 secara konsisten, sehingga tidak ada ambang; " +
    "metrik ini ditampilkan sebagai ukuran ketepatan lokalisasi. Inferensi = median end-to-end per gambar di RTX 4060. " +
    "Siap klinis tetap membutuhkan validasi terhadap beberapa dermatolog (ICC >= 0.75).");
  table(s, ["Metrik (test)", "B0 baseline", "E4 + SAHI", "Minimum", "Baik", "Status E4 + SAHI"], [
    ["Precision", f3(kb.P), f3(ke.P), "≥ 0.50", "≥ 0.60", ok("✓ baik")],
    ["Recall", { text: f3(kb.R), color: "C0392B" }, f3(ke.R), "≥ 0.50", "≥ 0.60", ok("✓ minimum")],
    ["mAP@50", f3(kb.ap50), f3(ke.ap50), "≥ 0.45", "≥ 0.55", ok("✓ minimum (batas baik)")],
    ["mAP@50-95", m95(kb), m95(ke), "—", "—", { text: "tanpa ambang literatur", color: C.slate }],
    ["Inferensi / gambar", `${kb.lat.toFixed(0)} ms`, `${ke.lat.toFixed(0)} ms`, "< 1 s", "< 100 ms", ok("✓ minimum")],
    ["MAE counting", f2(V.b0.MAE), f2(v.MAE), "≤ 10", "≤ 7", ok("✓ minimum")],
    ["Pearson r", f3(V.b0["Pearson r"]), f3(v["Pearson r"]), "≥ 0.85", "≥ 0.90", ok("✓ baik")],
  ], MX, TOP + 0.05, CW, [1.75, 1.3, 1.3, 1.05, 1.1, 2.5], { rowH: 0.4, fontSize: 12.5 });
  caption(s, "Merah: di bawah ambang minimum. Confusion matrix: slide \"Confusion matrix & jenis kesalahan\"." +
    (missing95.length ? "  * menunggu evaluasi ulang." : ""), MX, 4.9, CW);
}

// ---------------------------------------------------------------- 27. Kesimpulan
{
  const s = content("Kesimpulan & rekomendasi model",
    "Resolusi efektif adalah tuas terbesar; pilihan model akhir bergantung pada kebutuhan akurasi vs kecepatan.",
    "Penutup",
    "Kesimpulan didukung uji bootstrap (slide Uji signifikansi). Penetapan model terpilih di SPEC.md menunggu keputusan: " +
    "E4 + SAHI bila akurasi diutamakan, E3 plain bila latency < 100 ms dibutuhkan.");
  const cw = (CW - 0.3) / 2, y = TOP + 0.05, h = 1.1;
  [["Memenuhi standar", "E4 + SAHI memenuhi semua ambang minimum; Precision 0,611 dan r 0,943 mencapai ambang baik."],
    ["Kontributor utama", "SAHI (signifikan). Tuning E3 dan fine-tune tile E4 memberi efek kecil, tetapi E3 diperlukan agar SAHI efektif."],
    ["Trade-off", "SAHI ±9× lebih lambat (628 vs 69 ms), tetap < 1 s."],
  ].forEach(([t, b], i) => numCard(s, i + 1, t, b, MX, y + i * (h + 0.12), cw, h, { size: 11.5 }));
  const x = MX + cw + 0.3;
  card(s, x, y, cw, 1.6, C.coralBg);
  text(s, "Akurasi diutamakan", { x: x + 0.2, y: y + 0.12, w: cw - 0.4, h: 0.35, fontSize: 12, color: C.slate, bold: true });
  text(s, "E4 + SAHI 1728", { x: x + 0.2, y: y + 0.45, w: cw - 0.4, h: 0.5, fontFace: "Cambria", bold: true, fontSize: 22, color: C.coral });
  text(s, `F1 ${f3(V.e4s.F1)} · MAE ${f2(V.e4s.MAE)} · ${V.e4s["Latency ms"].toFixed(0)} ms`, { x: x + 0.2, y: y + 1.0, w: cw - 0.4, h: 0.4, fontSize: 12.5 });
  card(s, x, y + 1.75, cw, 1.6);
  text(s, "Kecepatan diutamakan (< 100 ms)", { x: x + 0.2, y: y + 1.87, w: cw - 0.4, h: 0.35, fontSize: 12, color: C.slate, bold: true });
  text(s, "E3 plain", { x: x + 0.2, y: y + 2.2, w: cw - 0.4, h: 0.5, fontFace: "Cambria", bold: true, fontSize: 22, color: C.teal });
  text(s, `F1 ${f3(V.e3.F1)} · MAE ${f2(V.e3.MAE)} · ${V.e3["Latency ms"].toFixed(0)} ms`, { x: x + 0.2, y: y + 2.75, w: cw - 0.4, h: 0.4, fontSize: 12.5 });
}

// ---------------------------------------------------------------- 28. Keterbatasan
{
  const s = content("Keterbatasan",
    "Hal-hal yang perlu diketahui sebelum hasil ini dipakai lebih jauh.",
    "Penutup",
    "Sumber: docs/REPORT.md bagian 9 dan docs/PIPELINE.md bagian 2 (butir 11-15). B0 dilatih dalam dua sesi (resume dari last.pt di epoch 45), " +
    "pengaruhnya diperkirakan kecil karena state optimizer ikut tersimpan.");
  const cw = (CW - 0.3) / 2, h = 1.6, y = TOP + 0.05;
  numCard(s, 1, "Satu seed per konfigurasi", "Selisih kecil (mis. E4 vs E3 + SAHI) bisa berasal dari noise; E4 hanya fine-tune 10 epoch.", MX, y, cw, h, { size: 12 });
  numCard(s, 2, "Test kecil, tidak terstratifikasi", "121 gambar; level 3 hanya 1 gambar dan level 2 hanya 10.", MX + cw + 0.3, y, cw, h, { size: 12 });
  numCard(s, 3, "Risiko kebocoran pasien", "Split per gambar; ID pasien tidak tersedia sehingga belum terverifikasi.", MX, y + h + 0.25, cw, h, { size: 12 });
  numCard(s, 4, "Anotasi & validasi klinis", "Lingkaran → bbox menekan mAP@50-95, anotasi diduga tidak lengkap, dan belum ada validasi dermatolog (ICC).", MX + cw + 0.3, y + h + 0.25, cw, h, { size: 12 });
}

// ---------------------------------------------------------------- 29. Roadmap
{
  const s = content("Roadmap",
    "Langkah berikutnya untuk menaikkan recall dan akurasi counting, diakhiri validasi klinis.",
    "Penutup",
    "Urutan mengikuti docs/ROADMAP.md: resolusi efektif terbukti tuas utama, sehingga training tile dari nol / imgsz 1280 diprioritaskan. " +
    "Serving/API dan aplikasi di luar cakupan sampai ada keputusan lanjut.");
  const steps = [
    ["Tile / 1280 px", "Training dari nol, lebih panjang, pada resolusi efektif tinggi"],
    ["Head P2", "Stride 4 untuk lesi < 16 px (73,5% lesi)"],
    ["Multi-seed + split", "≥ 3 seed, stratifikasi level, cek pasien"],
    ["Latency SAHI", "Hapus slice duplikat, batch slice"],
    ["Validasi klinis", "Bandingkan dengan beberapa dermatolog (ICC)"],
  ];
  const cw = (CW - 4 * 0.2) / 5, y = TOP + 0.85;
  s.addShape(pres.shapes.LINE, { x: MX + cw / 2, y: y + 0.25, w: CW - cw, h: 0, line: { color: C.grid, width: 2 } });
  steps.forEach(([t, b], i) => {
    const x = MX + i * (cw + 0.2);
    numCircle(s, i + 1, x + cw / 2 - 0.25, y, 0.5, i === 4 ? C.coral : C.teal);
    text(s, t, { x, y: y + 0.7, w: cw, h: 0.6, align: "center", fontFace: "Cambria", bold: true, fontSize: 13, valign: "top" });
    text(s, b, { x, y: y + 1.35, w: cw, h: 1.2, align: "center", fontSize: 11.5, color: C.ink2 });
  });
}

// ---------------------------------------------------------------- 30. Penutup & referensi
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
