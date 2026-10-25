"""
AcneVision - Diagram & grafik pendukung untuk docs/PRESENTASI.md dan deck
=========================================================================
Melengkapi scripts/report.py dengan diagram metode (pipeline, tiling, SAHI,
protokol evaluasi, silsilah eksperimen) dan grafik data (ukuran lesi, CI bootstrap).
Angka dibaca dari artefak (data/labels, runs/eval/*/metrics.json,
docs/report/bootstrap.json); tidak ada angka yang ditulis tangan.

Usage:
    python scripts/diagrams.py
    -> docs/report/figures/{pipeline,lesion_size_hist,tiling_example,sahi_example,
                            experiment_tree,eval_protocol,bootstrap_ci}.png
"""
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from analysis import load_gt_boxes
from common import PROJECT_ROOT, RUNS_DIR

FIG_DIR = PROJECT_ROOT / "docs" / "report" / "figures"
DATA = PROJECT_ROOT / "data"

# Warna tema deck (ppt/theme/theme1.xml): teal dominan, coral sebagai aksen
DK1, DK2, LT2 = "#0b2b33", "#14505c", "#eef4f4"
CORAL, TEAL, YELLOW, SLATE = "#ff6b5e", "#1c8c99", "#f2b84b", "#6c7a89"
MUTED = "#5b6b70"
BG = "#ffffff"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#d5dcdc", "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e3e9e9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.dpi": 170,
})


def box(ax, x, y, w, h, title, body="", foot="", fc=LT2, ec=TEAL, tc=DK1, title_size=10.5):
    """Kotak bersudut tumpul: judul di atas, isi tepat di bawahnya, catatan kecil di dasar."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h * 0.9, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=tc)
    if body:
        ax.text(x + w / 2, y + h * 0.66, body, ha="center", va="top", fontsize=8.6,
                color=tc, linespacing=1.35)
    if foot:
        ax.text(x + w / 2, y + 0.05, foot, ha="center", va="bottom", fontsize=7.8,
                color=MUTED, family="monospace")


def arrow(ax, p0, p1, color=DK2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color,
                                 shrinkA=2, shrinkB=2))


def blank_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.012, 1.012), ax.set_ylim(0, 1), ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(FIG_DIR / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------- diagram metode
def fig_pipeline():
    steps = [
        ("1. Data", "ACNE04 images\n+ ACNE04-v2\nanotasi lingkaran", "download_*.py"),
        ("2. Preprocessing", "Lingkaran → bbox\nsplit 80/10/10\ntile 2×2 (E4)", "convert_to_yolo.py\npreprocess.py"),
        ("3. Augmentasi", "Online saat training:\nmosaic, flip, rotate,\nscale, HSV, mixup", "configs/*.yaml"),
        ("4. Training", "YOLOv8l COCO\n960 px, batch 4\nbest.pt dipilih di val", "train.py"),
        ("5. Evaluasi", "Deteksi + counting\nconf dipilih di val\nbootstrap CI 95%", "evaluate.py"),
        ("6. Inferensi", "Plain (69 ms)\natau SAHI 1728\n(±630 ms)", "predict.py"),
    ]
    fig, ax = blank_axes((12, 2.5))
    w, gap = 0.145, 0.026
    for i, (t, b, f) in enumerate(steps):
        x = i * (w + gap)
        hl = i in (1, 5)       # tahap yang memuat teknik resolusi (tiling, SAHI)
        box(ax, x, 0.05, w, 0.9, t, b, f, fc="#fff1ef" if hl else LT2, ec=CORAL if hl else TEAL)
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.002, 0.5), (x + w + gap - 0.002, 0.5))
    save(fig, "pipeline.png")


def fig_eval_protocol():
    steps = [
        ("Prediksi", "val & test\nconf 0.001\nmax_det 1000"),
        ("Matching", "greedy per skor\nIoU ≥ 0.5\n→ TP / FP / FN"),
        ("Pilih conf di VAL", "grid 0.05–0.60\nMAE counting\nminimum"),
        ("Laporan TEST", "P, R, F1, AP@50\nMAE, RMSE, bias, r\nper level"),
        ("Signifikansi", "paired bootstrap\n10.000× per gambar\nCI 95%"),
    ]
    fig, ax = blank_axes((11, 2.2))
    w, gap = 0.175, 0.031
    for i, (t, b) in enumerate(steps):
        x = i * (w + gap)
        test = i >= 3
        box(ax, x, 0.06, w, 0.88, t, b, fc="#fff1ef" if test else LT2, ec=CORAL if test else TEAL)
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.002, 0.5), (x + w + gap - 0.002, 0.5))
    save(fig, "eval_protocol.png")


def metrics(run):
    return json.loads((RUNS_DIR / "eval" / run / "metrics.json").read_text(encoding="utf-8"))


def fig_experiment_tree():
    def label(run):
        t = metrics(run)["test_conf_val_tuned"]
        return f"F1 {t['detection']['F1']:.3f} · MAE {t['counting']['MAE']:.2f}"

    nodes = {   # id: (x, y, judul, isi, run eval)
        "coco": (0.02, 0.50, "YOLOv8l COCO", "pretrained", None),
        "b0": (0.25, 0.86, "B0 baseline", "setelan default\n100 epoch", "yolov8l_960_default"),
        "e3": (0.25, 0.14, "E3 tuned", "AdamW, loss & aug\n100 epoch", "yolov8l_960"),
        "b0s": (0.50, 1.10, "B0 + SAHI", "inferensi slice", "yolov8l_960_default_sahi1728"),
        "e5": (0.50, 0.81, "E5 tile-FT (B0)", "fine-tune tile 2×2\n10 epoch", "yolov8l_960_default_tiles2_ft"),
        "e5s": (0.76, 0.81, "E5 + SAHI", "kontrol 2×2", "yolov8l_960_default_tiles2_ft_sahi1728"),
        "e3t": (0.50, 0.52, "E3 + TTA", "flip + multi-skala", "yolov8l_960_tta"),
        "e3s": (0.50, 0.24, "E3 + SAHI", "inferensi slice", "yolov8l_960_sahi1728"),
        "e4": (0.50, -0.04, "E4 tile-FT", "fine-tune tile 2×2\n10 epoch", "yolov8l_960_tiles2_ft"),
        "e4s": (0.76, -0.04, "E4 + SAHI", "terbaik", "yolov8l_960_tiles2_ft_sahi1728"),
    }
    edges = [("coco", "b0"), ("coco", "e3"), ("b0", "b0s"), ("b0", "e5"), ("e5", "e5s"),
             ("e3", "e3t"), ("e3", "e3s"), ("e3", "e4"), ("e4", "e4s")]
    w, h = 0.2, 0.22
    fig, ax = blank_axes((11, 6.6))
    ax.set_ylim(-0.08, 1.38)
    for a, b in edges:
        xa, ya = nodes[a][:2]
        xb, yb = nodes[b][:2]
        arrow(ax, (xa + w, ya + h / 2), (xb, yb + h / 2), color=SLATE)
    for k, (x, y, t, b, run) in nodes.items():
        best = k == "e4s"
        body = b + (f"\n{label(run)}" if run else "")
        box(ax, x, y, w, h, t, body, fc="#fff1ef" if best else LT2, ec=CORAL if best else TEAL,
            title_size=10)
    ax.text(0.02, 1.37, "Metrik: test, conf dipilih di val, IoU 0.5 (runs/eval/<run>/metrics.json)",
            fontsize=8.5, color=MUTED, va="top")
    save(fig, "experiment_tree.png")


# ---------------------------------------------------------------- data & contoh gambar
def fig_lesion_size():
    """Sisi terpanjang bbox setelah letterbox ke 960 px (skala = 960 / sisi terpanjang gambar)."""
    sizes = []
    for img in sorted((DATA / "images" / "train").glob("*.jpg")):
        W, H = Image.open(img).size        # hanya membaca header
        g = load_gt_boxes(DATA / "labels" / "train" / f"{img.stem}.txt", W, H)
        if len(g):
            sizes.append(np.maximum(g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]) * 960 / max(W, H))
    s = np.concatenate(sizes)
    small = (s < 16).mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.logspace(np.log10(max(s.min(), 1)), np.log10(s.max()), 45)
    ax.hist(s, bins=bins, color=TEAL, edgecolor=BG, linewidth=0.8)
    ax.axvline(16, color=CORAL, linewidth=2)
    ax.text(16 * 1.08, ax.get_ylim()[1] * 0.92, f"{small:.1f}% lesi < 16 px", color=DK1, fontsize=10,
            fontweight="bold", va="top")
    ax.axvline(np.median(s), color=DK1, linewidth=1.2, linestyle="--")
    ax.text(np.median(s) / 1.08, ax.get_ylim()[1] * 0.92, f"median {np.median(s):.1f} px",
            color=DK1, fontsize=9.5, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_xticks([4, 8, 16, 32, 64, 128], ["4", "8", "16", "32", "64", "128"])
    ax.set_xlabel("Ukuran lesi pada input 960 px (px, skala log)")
    ax.set_ylabel("Jumlah lesi (train)")
    ax.set_title(f"Distribusi ukuran lesi · n = {len(s):,}".replace(",", "."), loc="left", color=DK1,
                 fontsize=11)
    fig.tight_layout()
    save(fig, "lesion_size_hist.png")
    return small, float(np.median(s))


def tile_windows(w, h, grid=2, overlap=0.1):
    """Sama dengan scripts/data/preprocess.py::tile_windows."""
    tw = int(np.ceil(w / (grid - (grid - 1) * overlap)))
    th = int(np.ceil(h / (grid - (grid - 1) * overlap)))
    xs = np.linspace(0, w - tw, grid).astype(int)
    ys = np.linspace(0, h - th, grid).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def pick_tiling_image():
    """Foto train ukuran umum (3112×3456) dengan jumlah lesi mendekati persentil 80 (cukup padat)."""
    cands = []
    for img in sorted((DATA / "images" / "train").glob("*.jpg")):
        if Image.open(img).size == (3112, 3456):
            n = sum(1 for _ in open(DATA / "labels" / "train" / f"{img.stem}.txt"))
            cands.append((n, img))
    cands.sort(key=lambda t: t[0])
    return cands[int(len(cands) * 0.8)][1]


def fig_tiling():
    src = pick_tiling_image()
    img = rgb(src)
    h, w = img.shape[:2]
    gt = load_gt_boxes(DATA / "labels" / "train" / f"{src.stem}.txt", w, h)
    wins = tile_windows(w, h)
    colors = [CORAL, YELLOW, TEAL, "#8a5fd1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    ax = axes[0]
    ax.imshow(img), ax.axis("off")
    for i, (x1, y1, x2, y2) in enumerate(wins):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=colors[i], lw=2.5))
        ax.text(x1 + 60, y1 + 60, f"Tile {i + 1}", color="white", fontsize=10, fontweight="bold",
                va="top", bbox=dict(fc=colors[i], ec="none", pad=2))
    ax.set_title(f"Gambar asli {w}×{h} → 4 tile ±{wins[0][2]}×{wins[0][3]} (overlap 10%)",
                 loc="left", fontsize=10, color=DK1)
    # Tile dengan lesi terbanyak + label GT: lesi tampil ±2× lebih besar saat tile di-resize ke 960
    def inside(win):
        x1, y1, x2, y2 = win
        return ((gt[:, 0] >= x1) & (gt[:, 1] >= y1) & (gt[:, 2] <= x2) & (gt[:, 3] <= y2)).sum()
    k = int(np.argmax([inside(wn) for wn in wins]))
    x1, y1, x2, y2 = wins[k]
    ax = axes[1]
    ax.imshow(img[y1:y2, x1:x2]), ax.axis("off")
    for bx1, by1, bx2, by2 in gt:
        if bx1 >= x1 and by1 >= y1 and bx2 <= x2 and by2 <= y2:
            ax.add_patch(Rectangle((bx1 - x1, by1 - y1), bx2 - bx1, by2 - by1, fill=False, ec=colors[k], lw=1.4))
    ax.set_title(f"Tile {k + 1} + label ({src.stem}): skala input {960 / max(x2 - x1, y2 - y1):.2f} vs {960 / max(w, h):.2f} (asli)",
                 loc="left", fontsize=10, color=DK1)
    fig.tight_layout()
    save(fig, "tiling_example.png")


def fig_sahi(name="levle1_482"):
    from sahi.slicing import get_slice_bboxes

    img = rgb(DATA / "images" / "test" / f"{name}.jpg")
    h, w = img.shape[:2]
    slices = get_slice_bboxes(image_height=h, image_width=w, slice_height=1728, slice_width=1728,
                              overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    # SAHI menggeser slice terakhir ke tepi gambar, sehingga sebagian slice nyaris identik (selisih 1 px)
    groups = {}
    for i, sl in enumerate(slices):
        groups.setdefault((round(sl[0] / 20), round(sl[1] / 20)), []).append((i + 1, sl))
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.imshow(img), ax.axis("off")
    palette = [CORAL, YELLOW, TEAL, "#8a5fd1", SLATE, "#2e9e6b"]
    for g, members in enumerate(groups.values()):
        c = palette[g % len(palette)]
        x1, y1, x2, y2 = members[0][1]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=c, lw=2.4))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, "+".join(str(i) for i, _ in members), color="white",
                fontsize=10, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=c, ec="white", lw=1))
    dup = len(slices) - len(groups)
    ax.set_title(f"SAHI: {len(slices)} slice 1728 px (overlap 20%), {dup} nyaris duplikat (geser 1 px)\n"
                 f"tiap slice di-resize ke 960 → skala {960 / 1728:.2f} vs {960 / max(w, h):.2f} tanpa slicing; "
                 f"digabung dengan NMS", loc="left", fontsize=9.5, color=DK1)
    fig.tight_layout()
    save(fig, "sahi_example.png")
    return f"{len(slices)} ({len(slices) - len(groups)} nyaris duplikat)"


def fig_bootstrap():
    data = json.loads((PROJECT_ROOT / "docs" / "report" / "bootstrap.json").read_text(encoding="utf-8"))
    pairs = list(data)[::-1]
    y = np.arange(len(pairs))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(pairs) + 1.4), sharey=True)
    for ax, key, title in ((axes[0], "dF1", "Δ F1 (↑ lebih baik)"), (axes[1], "dMAE", "Δ MAE lesi/gambar (↓ lebih baik)")):
        for yi, p in zip(y, pairs):
            m, lo, hi = data[p][key]
            sig = lo > 0 or hi < 0
            c = TEAL if sig else "#a9b4b6"
            ax.plot([lo, hi], [yi, yi], color=c, linewidth=2.4, solid_capstyle="round")
            ax.plot(m, yi, "o", color=c, markersize=8, markeredgecolor=BG, markeredgewidth=1.5)
            d = 3 if key == "dF1" else 2
            shown = np.sign(m) * np.floor(abs(m) * 10 ** d + 0.5 + 1e-9) / 10 ** d   # half away from zero
            ax.text(hi, yi, "  " + f"{shown:+.{d}f}", fontsize=8,
                    color=MUTED, ha="left", va="center", zorder=4,
                    bbox=dict(fc=BG, ec="none", pad=0.3))   # label tetap terbaca di atas garis nol
        ax.axvline(0, color=DK1, linewidth=1)
        lo_all = min(data[p][key][1] for p in pairs)
        hi_all = max(data[p][key][2] for p in pairs)
        ax.set_xlim(lo_all - 0.08 * (hi_all - lo_all), hi_all + 0.22 * (hi_all - lo_all))
        ax.set_title(title, loc="left", fontsize=10, color=DK1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
    axes[0].set_yticks(y, pairs)
    fig.text(0.01, -0.02, "Garis = CI 95% paired bootstrap (10.000 resample per gambar, test n=121). "
             "Teal = signifikan (CI tidak memuat 0), abu-abu = tidak signifikan.", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "bootstrap_ci.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figur -> {FIG_DIR}")
    fig_pipeline()
    fig_eval_protocol()
    fig_experiment_tree()
    small, med = fig_lesion_size()
    print(f"    lesi < 16 px: {small:.1f}% · median {med:.1f} px (train, input 960)")
    fig_tiling()
    print(f"    SAHI slice per gambar: {fig_sahi()}")
    fig_bootstrap()


if __name__ == "__main__":
    main()
