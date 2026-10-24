"""
AcneVision - Grafik & tabel untuk docs/REPORT.md
================================================
Semua angka dibaca dari artefak evaluasi (runs/eval/<run>/metrics.json,
counts_test.csv) dan log training (runs/detect/<run>/results.csv); tidak ada
angka yang ditulis tangan.

Usage:
    python scripts/report.py
    -> docs/report/figures/*.png  +  docs/report/summary.json
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import PROJECT_ROOT, RUNS_DIR

FIG_DIR = PROJECT_ROOT / "docs" / "report" / "figures"
EVAL = RUNS_DIR / "eval"

# Varian yang dibandingkan: (label tampilan, folder runs/eval, folder runs/detect untuk kurva training, warna)
# Palet kategorikal tervalidasi (skill dataviz); warna melekat pada varian, bukan pada urutan
VARIANTS = [
    ("B0 baseline default", "yolov8l_960_default", "yolov8l_960_default", "#2a78d6"),
    ("E3 aug+hyp tuned", "yolov8l_960", "yolov8l_960", "#eb6834"),
    ("E3 + TTA", "yolov8l_960_tta", None, "#1baf7a"),
    ("E3 + SAHI", "yolov8l_960_sahi1728", None, "#eda100"),
    # Kurva training E4 tidak diplot: val-nya berisi tile (data_tiles2), tidak sebanding dengan B0/E3
    ("E4 tile-FT + SAHI", "yolov8l_960_tiles2_ft_sahi1728", None, "#e87ba4"),
    ("B0 + SAHI", "yolov8l_960_default_sahi1728", None, "#008300"),
]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160,
})


def load_variants():
    """Baca metrics.json yang tersedia; varian tanpa hasil dilewati (dengan peringatan)."""
    out = []
    for label, eval_dir, train_dir, color in VARIANTS:
        f = EVAL / eval_dir / "metrics.json"
        if not f.exists():
            print(f"[lewati] {label}: {f} belum ada")
            continue
        out.append({"label": label, "color": color, "eval_dir": eval_dir,
                    "train_dir": train_dir, "m": json.loads(f.read_text(encoding="utf-8"))})
    return out


def summary_rows(variants):
    rows = []
    for v in variants:
        m, t = v["m"], v["m"]["test_conf_val_tuned"]
        det = m.get("detection_test_ultralytics") or {}
        rows.append({
            "Varian": v["label"],
            "mAP50 (Ultralytics)": det.get("mAP50"), "mAP50-95": det.get("mAP50_95"),
            "AP50 (internal)": m["ap50_test_internal"],
            # mAP@50-95 internal (IoU 0.50:0.05:0.95) tersedia untuk semua mode, termasuk SAHI;
            # None bila metrics.json dibuat sebelum metrik ini ditambahkan (jalankan ulang evaluate.py)
            "mAP50-95 (internal)": m.get("map50_95_test_internal"),
            "conf (val)": m["conf_selected_on_val"],
            "Precision": t["detection"]["precision"], "Recall": t["detection"]["recall"],
            "F1": t["detection"]["F1"],
            "TP": t["detection"]["TP"], "FP": t["detection"]["FP"], "FN": t["detection"]["FN"],
            "FP lokalisasi": t.get("fp_breakdown", {}).get("localization"),
            "MAE": t["counting"]["MAE"], "RMSE": t["counting"]["RMSE"], "Bias": t["counting"]["bias"],
            "Pearson r": t["counting"]["pearson_r"],
            "Latency ms": m.get("latency_ms_median"),
            "Inference ms (Ultralytics)": det.get("inference_ms"),
        })
    return pd.DataFrame(rows)


def bar_end(ax):
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def fig_detection(variants, df):
    """Grouped bar: AP50 internal, Precision, Recall, F1 per varian (skala 0-1)."""
    metrics = ["AP50 (internal)", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    w = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for i, v in enumerate(variants):
        vals = df.loc[i, metrics].astype(float).to_numpy()
        bars = ax.bar(x + (i - (len(variants) - 1) / 2) * w, vals, w * 0.9, color=v["color"],
                      label=v["label"], edgecolor="#fcfcfb", linewidth=1)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.008, f"{val:.2f}", ha="center",
                    fontsize=7, color=INK2)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, max(0.8, df[metrics].astype(float).max().max() + 0.08))
    ax.set_ylabel("Skor (test, conf dipilih di val, IoU 0.5)")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax.set_title("Metrik deteksi per varian", loc="left", color=INK, fontsize=11)
    bar_end(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "detection_metrics.png")
    plt.close(fig)


def fig_counting(variants, df):
    """Bar horizontal MAE + bias per varian (dua panel, satu skala per panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.55 * len(variants) + 1.4), sharey=True)
    y = np.arange(len(variants))[::-1]
    for ax, col, title in ((axes[0], "MAE", "MAE (lesi/gambar, ↓)"), (axes[1], "Bias", "Bias pred−GT (0 = ideal)")):
        vals = df[col].astype(float).to_numpy()
        ax.barh(y, vals, 0.6, color=[v["color"] for v in variants])
        for yi, val in zip(y, vals):
            ax.text(val + (0.15 if val >= 0 else -0.15), yi, f"{val:.2f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.tick_params(length=0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [v["label"] for v in variants])
    lo = min(0, df["Bias"].astype(float).min()) - 2
    axes[1].set_xlim(lo, max(2, df["Bias"].astype(float).max() + 2))
    axes[0].set_xlim(0, df["MAE"].astype(float).max() * 1.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "counting_metrics.png")
    plt.close(fig)


def fig_scatter(variants):
    """Small multiples: jumlah prediksi vs GT per gambar (baseline vs terbaik)."""
    pick = [variants[0], min(variants, key=lambda v: v["m"]["test_conf_val_tuned"]["counting"]["MAE"])]
    if pick[0] is pick[1]:
        pick = pick[:1]
    fig, axes = plt.subplots(1, len(pick), figsize=(4.6 * len(pick), 4.4), squeeze=False)
    for ax, v in zip(axes[0], pick):
        d = pd.read_csv(EVAL / v["eval_dir"] / "counts_test.csv")
        lim = max(d["gt_count"].max(), d["pred_count"].max()) * 1.05
        ax.plot([0, lim], [0, lim], color=INK2, linewidth=1, linestyle="--", label="y = x")
        ax.scatter(d["gt_count"], d["pred_count"], s=26, color=v["color"], edgecolor="#fcfcfb",
                   linewidth=1, alpha=0.9)
        c = v["m"]["test_conf_val_tuned"]["counting"]
        ax.set_title(f"{v['label']}\nMAE {c['MAE']:.2f} · r {c['pearson_r']:.3f}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Jumlah lesi GT"), ax.set_ylabel("Jumlah lesi prediksi")
        ax.set_xscale("symlog", linthresh=10), ax.set_yscale("symlog", linthresh=10)
        ax.set_xlim(0, lim), ax.set_ylim(0, lim)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "count_scatter.png")
    plt.close(fig)


def fig_per_level(variants):
    """Grouped bar MAE per level keparahan (level 3 n=1 dikeluarkan dari plot)."""
    levels = ["level0", "level1", "level2"]
    x = np.arange(len(levels))
    w = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, v in enumerate(variants):
        pl = v["m"]["test_conf_val_tuned"]["per_level"]
        vals = [pl[l]["counting"]["MAE"] for l in levels]
        ax.bar(x + (i - (len(variants) - 1) / 2) * w, vals, w * 0.9, color=v["color"], label=v["label"],
               edgecolor="#fcfcfb", linewidth=1)
    n = [variants[0]["m"]["test_conf_val_tuned"]["per_level"][l]["counting"]["n_images"] for l in levels]
    ax.set_xticks(x, [f"Level {i} (n={k})" for i, k in enumerate(n)])
    ax.set_ylabel("MAE (lesi/gambar)")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax.set_title("Akurasi counting per level keparahan (test)", loc="left", color=INK, fontsize=11)
    bar_end(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mae_per_level.png")
    plt.close(fig)


def fig_training(variants):
    """Kurva mAP50 validasi per epoch untuk run yang dilatih."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for v in variants:
        if not v["train_dir"]:
            continue
        f = RUNS_DIR / "detect" / v["train_dir"] / "results.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        r.columns = [c.strip() for c in r.columns]
        r = r.drop_duplicates("epoch", keep="last")      # resume menulis ulang epoch terakhir
        label = v["label"].split(" + ")[0]
        ax.plot(r["epoch"], r["metrics/mAP50(B)"], color=v["color"], linewidth=2, label=label)
    ax.set_xlabel("Epoch"), ax.set_ylabel("mAP@50 (val, selama training)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Kurva training: mAP@50 split val", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    variants = load_variants()
    df = summary_rows(variants)
    print(df.to_string(index=False))
    fig_detection(variants, df)
    fig_counting(variants, df)
    fig_scatter(variants)
    fig_per_level(variants)
    fig_training(variants)
    # Salin confusion matrix (conf dipilih di val) tiap varian
    for v in variants:
        conf = v["m"]["conf_selected_on_val"]
        src = EVAL / v["eval_dir"] / f"confusion_matrix_conf{conf:.2f}.png"
        if src.exists():
            shutil.copy2(src, FIG_DIR / f"cm_{v['eval_dir']}.png")
    (FIG_DIR.parent / "summary.json").write_text(
        # NaN (mAP Ultralytics tidak ada untuk SAHI) ditulis sebagai null agar JSON valid
        json.dumps(df.astype(object).where(df.notna(), None).to_dict(orient="records"), indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print(f"Figur: {FIG_DIR}")


if __name__ == "__main__":
    main()
