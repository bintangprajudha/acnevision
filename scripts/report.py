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

from analysis import grading_metrics
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
    # E5 = config E4 dengan init B0 (desain 2x2 init x tile-FT); dilewati bila belum dievaluasi
    ("E5 tile-FT(B0) + SAHI", "yolov8l_960_default_tiles2_ft_sahi1728", None, "#7d53c8"),
    # E9 = training tile dari COCO (resep E3); slot 8 palet
    ("E9 tile dari COCO + SAHI", "yolov8l_960_tiles2_sahi1728", None, "#e34948"),
]
# Durasi training tile: (label, run, warna). Diplot terpisah (plain & SAHI) + baris summary
# untuk varian yang belum ada di VARIANTS.
TILE = [
    ("E4 fine-tune 10 ep", "yolov8l_960_tiles2_ft", "#e87ba4"),
    ("E8 fine-tune 60 ep", "yolov8l_960_tiles2_ft60", "#6250d6"),
    ("E9 tile dari COCO 100 ep", "yolov8l_960_tiles2", "#e34948"),
]
# YOLOv8l vs YOLO11l pada resep yang sama: (resep, run YOLOv8l, run YOLO11l, label YOLO11l).
# Tidak masuk grafik utama (sudah 7 varian); diplot terpisah dengan 2 seri warna = arsitektur.
ARCH = [
    ("E3", "yolov8l_960", "yolo11l_960", "E6 YOLO11l (resep E3)"),
    ("E3 + SAHI", "yolov8l_960_sahi1728", "yolo11l_960_sahi1728", "E6 + SAHI"),
    ("E4 tile-FT", "yolov8l_960_tiles2_ft", "yolo11l_960_tiles2_ft", "E7 YOLO11l tile-FT (resep E4)"),
    ("E4 tile-FT + SAHI", "yolov8l_960_tiles2_ft_sahi1728", "yolo11l_960_tiles2_ft_sahi1728", "E7 tile-FT + SAHI"),
]
ARCH_COLORS = {"YOLOv8l": "#2a78d6", "YOLO11l": "#eb6834"}
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


def counting_extra(eval_dir: str) -> dict:
    """MAE/RMSE relatif + grading Hayashi (ala Wu 2019) dari counts_test.csv (conf operasional).
    Dihitung ulang dari CSV agar run lama (metrics.json tanpa metrik ini) tetap tercakup."""
    f = EVAL / eval_dir / "counts_test.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    gt, pred = d["gt_count"].to_numpy(), d["pred_count"].to_numpy()
    err = pred - gt
    g = grading_metrics(gt, pred)
    return {"MAE rel %": round(100 * np.abs(err).mean() / gt.mean(), 1),
            "RMSE rel %": round(100 * np.sqrt((err ** 2).mean()) / gt.mean(), 1),
            "Hayashi acc %": g["accuracy"], "Hayashi prec %": g["precision"],
            "Hayashi sens %": g["sensitivity"], "Hayashi spec %": g["specificity"],
            "Youden %": g["youden"]}


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
            **counting_extra(v["eval_dir"]),
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


def fig_arch():
    """YOLOv8l vs YOLO11l per resep: F1, mAP@50-95 internal, MAE (test) + kurva val E3 vs E6.

    Mengembalikan varian YOLO11l yang tersedia (untuk tabel summary)."""
    rows = [r for r in ARCH if (EVAL / r[1] / "metrics.json").exists() and (EVAL / r[2] / "metrics.json").exists()]
    if not rows:
        print("[lewati] perbandingan arsitektur: hasil YOLO11l belum ada")
        return []
    load = lambda d: json.loads((EVAL / d / "metrics.json").read_text(encoding="utf-8"))
    panels = [("F1 (↑)", lambda m: m["test_conf_val_tuned"]["detection"]["F1"], "{:.3f}"),
              ("mAP@50-95 internal (↑)", lambda m: m["map50_95_test_internal"], "{:.3f}"),
              ("MAE lesi/gambar (↓)", lambda m: m["test_conf_val_tuned"]["counting"]["MAE"], "{:.2f}")]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (title, get, fmt) in zip(axes, panels):
        for j, (arch, col) in enumerate(((1, "YOLOv8l"), (2, "YOLO11l"))):
            vals = [get(load(r[arch])) for r in rows]
            bars = ax.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=ARCH_COLORS[col], label=col,
                          edgecolor="#fcfcfb", linewidth=1)
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, val, fmt.format(val), ha="center", va="bottom",
                        fontsize=7, color=INK2)
        ax.set_xticks(x, [r[0] for r in rows], fontsize=8, rotation=15)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        bar_end(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, ncol=2,
               loc="upper right", bbox_to_anchor=(1, 1.06))
    fig.text(0.01, -0.03, "Test (121 gambar), conf dipilih di val per varian, IoU 0.5; mAP@50-95 dari matcher proyek "
             "(conf 0.001). Resep (data, hyperparameter, jadwal) identik; hanya arsitektur yang berbeda.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "arch_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Kurva val E3 vs E6 (keduanya val asli; E4/E7 val-nya berisi tile sehingga tidak diplot)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for run, col in (("yolov8l_960", "YOLOv8l"), ("yolo11l_960", "YOLO11l")):
        f = RUNS_DIR / "detect" / run / "results.csv"
        if f.exists():
            r = pd.read_csv(f)
            r.columns = [c.strip() for c in r.columns]
            r = r.drop_duplicates("epoch", keep="last")
            ax.plot(r["epoch"], r["metrics/mAP50(B)"], color=ARCH_COLORS[col], linewidth=2, label=f"{col} ({run})")
    ax.set_xlabel("Epoch"), ax.set_ylabel("mAP@50 (val, selama training)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Kurva training resep E3: YOLOv8l vs YOLO11l (split val)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "arch_training_curves.png")
    plt.close(fig)
    return [{"label": r[3], "eval_dir": r[2], "m": load(r[2])} for r in rows]


def fig_tile():
    """E4 vs E8 vs E9: F1, mAP@50-95 internal, MAE untuk inferensi plain dan SAHI 1728 (test).

    Mengembalikan varian E8 (plain, SAHI) dan E9 plain untuk tabel summary."""
    rows = [t for t in TILE if (EVAL / t[1] / "metrics.json").exists()
            and (EVAL / f"{t[1]}_sahi1728" / "metrics.json").exists()]
    if len(rows) < 2:
        print("[lewati] perbandingan durasi tile: hasil E8/E9 belum ada")
        return []
    load = lambda d: json.loads((EVAL / d / "metrics.json").read_text(encoding="utf-8"))
    panels = [("F1 (↑)", lambda m: m["test_conf_val_tuned"]["detection"]["F1"], "{:.3f}"),
              ("mAP@50-95 internal (↑)", lambda m: m["map50_95_test_internal"], "{:.3f}"),
              ("MAE lesi/gambar (↓)", lambda m: m["test_conf_val_tuned"]["counting"]["MAE"], "{:.2f}")]
    modes = [("Plain 960", ""), ("SAHI 1728", "_sahi1728")]
    x = np.arange(len(modes))
    w = 0.8 / len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (title, get, fmt) in zip(axes, panels):
        for j, (label, run, color) in enumerate(rows):
            vals = [get(load(run + suf)) for _, suf in modes]
            bars = ax.bar(x + (j - (len(rows) - 1) / 2) * w, vals, w * 0.9, color=color, label=label,
                          edgecolor="#fcfcfb", linewidth=1)
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, val, fmt.format(val), ha="center", va="bottom",
                        fontsize=7, color=INK2)
        ax.set_xticks(x, [m[0] for m in modes], fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        bar_end(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, ncol=len(rows),
               loc="upper right", bbox_to_anchor=(1, 1.06))
    fig.text(0.01, -0.03, "Test (121 gambar), conf dipilih di val per varian, IoU 0.5; mAP@50-95 dari matcher proyek "
             "(conf 0.001). E4/E8: fine-tune tile dari E3 (lr 2e-4); E9: resep E3 pada data tile dari bobot COCO.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tile_duration.png", bbox_inches="tight")
    plt.close(fig)
    have = {v[1] for v in VARIANTS}
    out = []
    for label, run, _ in rows:
        for mname, suf in modes:
            if run + suf not in have and run != "yolov8l_960_tiles2_ft":
                out.append({"label": f"{label.split(' ')[0]} {mname.split(' ')[0]}", "eval_dir": run + suf,
                            "m": load(run + suf)})
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    variants = load_variants()
    df = summary_rows(variants)
    fig_detection(variants, df)
    fig_counting(variants, df)
    fig_scatter(variants)
    fig_per_level(variants)
    fig_training(variants)
    # Varian YOLO11l hanya masuk grafik arsitektur + tabel summary (setelah grafik utama)
    extra = fig_arch() + fig_tile()
    if extra:
        df = pd.concat([df, summary_rows(extra)], ignore_index=True)
    print(df.to_string(index=False))
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
