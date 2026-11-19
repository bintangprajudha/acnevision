"""
AcneVision - Analisis deteksi: matching TP/FP/FN, AP@50, dan confusion matrix.

Dipakai oleh evaluate.py dan predict.py agar definisi TP/FP/FN identik di
seluruh proyek. Matching mengikuti protokol standar deteksi objek: prediksi
diurutkan dari skor tertinggi, masing-masing dipasangkan ke GT yang belum
terpakai dengan IoU tertinggi (>= iou_thr).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_gt_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    """Baca label YOLO (cls cx cy w h, ternormalisasi) -> array xyxy piksel (N, 4)."""
    if not label_path.exists():
        return np.zeros((0, 4))
    rows = [list(map(float, l.split()[1:5])) for l in label_path.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4))
    cx, cy, w, h = np.array(rows).T
    return np.stack([(cx - w / 2) * width, (cy - h / 2) * height,
                     (cx + w / 2) * width, (cy + h / 2) * height], axis=1)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU berpasangan antara a (N,4) dan b (M,4) format xyxy -> (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, iou_thr: float = 0.5):
    """Greedy matching berdasarkan skor.

    Returns:
        pred_tp (bool, N_pred): prediksi yang cocok dengan GT (TP), sisanya FP.
        gt_hit (bool, N_gt): GT yang ditemukan, sisanya FN.
    """
    order = np.argsort(-pred_scores)
    iou = box_iou(pred_boxes[order], gt_boxes)
    pred_tp = np.zeros(len(pred_boxes), bool)
    gt_hit = np.zeros(len(gt_boxes), bool)
    for rank, idx in enumerate(order):
        if iou.shape[1] == 0:
            break
        cand = np.where(~gt_hit, iou[rank], -1.0)
        j = int(cand.argmax())
        if cand[j] >= iou_thr:
            pred_tp[idx] = True
            gt_hit[j] = True
    return pred_tp, gt_hit


def ap50(scores: list, tps: list, n_gt: int) -> float:
    """AP@IoU0.5 dataset-level (interpolasi 101 titik ala COCO) dari skor & flag TP per prediksi."""
    s = np.concatenate(scores) if scores else np.zeros(0)
    t = np.concatenate(tps) if tps else np.zeros(0, bool)
    if n_gt == 0 or len(s) == 0:
        return 0.0
    order = np.argsort(-s, kind="stable")
    t = t[order]
    tp_cum, fp_cum = np.cumsum(t), np.cumsum(~t)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # envelope monoton turun
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


IOU_GRID = np.round(np.arange(0.5, 0.96, 0.05), 2)   # IoU 0.50:0.05:0.95 ala COCO


def map50_95(scores: list, tps_by_iou: list, n_gt: int) -> float:
    """mAP@50-95: rata-rata AP (fungsi ap50, IoU berbeda) atas IOU_GRID.
    tps_by_iou[k] = list flag TP per gambar hasil match(..., iou_thr=IOU_GRID[k])."""
    return float(np.mean([ap50(scores, tps, n_gt) for tps in tps_by_iou]))


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn),
            "precision": round(p, 4), "recall": round(r, 4), "F1": round(f1, 4)}


# Kriteria Hayashi seperti dipakai Wu et al. (ICCV 2019, Tabel 1) pada ACNE04:
# Mild 1-5, Moderate 6-20, Severe 21-50, Very severe > 50 lesi (0 lesi dimasukkan ke Mild).
HAYASHI_UPPER = (5, 20, 50)
HAYASHI_NAMES = ("mild", "moderate", "severe", "very_severe")


def hayashi_grade(counts) -> np.ndarray:
    """Jumlah lesi -> grade Hayashi 0..3."""
    return np.searchsorted(HAYASHI_UPPER, np.asarray(counts), side="left")


def grading_metrics(gt_counts, pred_counts) -> dict:
    """Metrik grading ala Wu 2019, grade diturunkan dari jumlah lesi (GT dan prediksi).

    Precision/sensitivity/specificity = rata-rata makro one-vs-rest atas kelas yang ada;
    Youden Index = sensitivity + specificity - 1 (dalam persen, seperti Wu 2019)."""
    g, p = hayashi_grade(gt_counts), hayashi_grade(pred_counts)
    prec, sens, spec = [], [], []
    for k in range(len(HAYASHI_NAMES)):
        tp = int(((p == k) & (g == k)).sum())
        if (g == k).any():
            sens.append(tp / (g == k).sum())
            spec.append(((p != k) & (g != k)).sum() / max((g != k).sum(), 1))
        if (p == k).any():
            prec.append(tp / (p == k).sum())
    se, sp = float(np.mean(sens)), float(np.mean(spec))
    return {"accuracy": round(100 * float((g == p).mean()), 2),
            "precision": round(100 * float(np.mean(prec)), 2) if prec else 0.0,
            "sensitivity": round(100 * se, 2), "specificity": round(100 * sp, 2),
            "youden": round(100 * (se + sp - 1), 2),
            "n_per_grade_gt": {n: int((g == k).sum()) for k, n in enumerate(HAYASHI_NAMES)}}


def plot_confusion(tp: int, fp: int, fn: int, path: Path, title: str) -> None:
    """Confusion matrix deteksi 2x2. Sel background-background diberi keterangan N/A:
    pada deteksi objek, 'true negative' (area kulit yang benar tidak diprediksi) tidak
    terhitung, sehingga sel itu tidak terdefinisi (bukan nol)."""
    m = prf(tp, fp, fn)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    cells = {(0, 0): (tp, "TP"), (0, 1): (fp, "FP"), (1, 0): (fn, "FN")}
    vmax = max(tp, fp, fn, 1)
    for (r, c), (val, tag) in cells.items():
        shade = plt.cm.Blues(0.15 + 0.75 * val / vmax)
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color=shade))
        txt_color = "white" if val / vmax > 0.55 else "#1a1a1a"
        ax.text(c + 0.5, r + 0.45, f"{val:,}".replace(",", "."), ha="center", va="center",
                fontsize=20, weight="bold", color=txt_color)
        ax.text(c + 0.5, r + 0.68, tag, ha="center", va="center", fontsize=11, color=txt_color)
    ax.add_patch(plt.Rectangle((1, 1), 1, 1, facecolor="#f0f0f0", hatch="//", edgecolor="#c8c8c8"))
    ax.text(1.5, 1.5, "N/A\n(TN tidak terdefinisi\npada deteksi objek)", ha="center", va="center",
            fontsize=9.5, color="#555555")
    ax.set_xlim(0, 2), ax.set_ylim(2, 0)
    ax.set_xticks([0.5, 1.5], ["acne", "background"])
    ax.set_yticks([0.5, 1.5], ["acne", "background"])
    ax.set_xlabel("Ground truth"), ax.set_ylabel("Prediksi")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, pad=10)
    fig.text(0.5, 0.015, f"Precision {m['precision']:.3f} · Recall {m['recall']:.3f} · F1 {m['F1']:.3f}",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)
