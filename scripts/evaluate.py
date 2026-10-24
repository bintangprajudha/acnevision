"""
AcneVision - Evaluasi Model (Deteksi + Counting + Confusion Matrix)
===================================================================
Protokol evaluasi standar proyek (lihat docs/SPEC.md):

1. Metrik deteksi Ultralytics pada split test dengan argumen default
   (conf=0.001, iou=0.7) agar mAP sebanding dengan literatur (mode plain/tta).
2. AP@50 internal + TP/FP/FN dari matcher proyek (scripts/analysis.py), sehingga
   mode SAHI/TTA dapat dibandingkan dengan definisi yang sama.
3. Threshold confidence operasional DIPILIH di split val (MAE counting minimum),
   lalu diterapkan ke test: counting (MAE, RMSE, bias, r) dan confusion matrix.

Prediksi dijalankan sekali per split dengan conf rendah; matching dihitung sekali,
lalu semua threshold diturunkan dari skor yang tersimpan.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --weights runs/detect/exp/weights/best.pt
    python scripts/evaluate.py --mode tta
    python scripts/evaluate.py --mode sahi --slice 960
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

from analysis import IOU_GRID, ap50, box_iou, load_gt_boxes, map50_95, match, plot_confusion, prf
from common import (DATA_YAML, DEFAULT_WEIGHTS, MAX_DET, PROJECT_ROOT, RUNS_DIR,
                    get_device, severity_from_name, train_imgsz)

MIN_CONF = 0.001                       # conf prediksi mentah (untuk AP)
CONF_GRID = np.round(np.arange(0.05, 0.61, 0.01), 2)
IMG_EXT = {".jpg", ".jpeg", ".png"}


def detection_metrics(model, imgsz, device, out_dir: Path, augment: bool) -> dict:
    """mAP/P/R Ultralytics pada split test dengan argumen validasi default."""
    m = model.val(
        data=str(DATA_YAML), split="test", imgsz=imgsz, batch=4, device=device,
        conf=MIN_CONF, iou=0.7, max_det=MAX_DET, augment=augment, save_json=True, plots=True,
        project=str(out_dir.parent), name=out_dir.name, exist_ok=True, verbose=False,
    )
    return {
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "inference_ms": round(float(m.speed["inference"]), 2),
    }


def make_predictor(model, weights: Path, mode: str, imgsz: int, device, slice_size: int):
    """Kembalikan fungsi path -> (boxes xyxy, scores, (h, w))."""
    if mode == "sahi":
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(weights), confidence_threshold=MIN_CONF,
            image_size=imgsz, device="cuda:0" if device != "cpu" else "cpu",   # slice di-resize ke imgsz
        )

        def predict(path: Path):
            res = get_sliced_prediction(
                str(path), sahi_model, slice_height=slice_size, slice_width=slice_size,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                postprocess_type="NMS", postprocess_match_metric="IOU",
                postprocess_match_threshold=0.5, verbose=0,
            )
            preds = res.object_prediction_list
            boxes = np.array([p.bbox.to_xyxy() for p in preds]).reshape(-1, 4)
            scores = np.array([p.score.value for p in preds])
            return boxes, scores, (res.image_height, res.image_width)

        return predict

    def predict(path: Path):
        r = model.predict(str(path), conf=MIN_CONF, iou=0.7, imgsz=imgsz, max_det=MAX_DET,
                          augment=(mode == "tta"), device=device, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.orig_shape

    return predict


def collect(predict, split: str) -> pd.DataFrame:
    """Prediksi semua gambar split, lalu matching sekali terhadap GT (IoU 0.5)."""
    img_dir = PROJECT_ROOT / "data" / "images" / split
    lbl_dir = PROJECT_ROOT / "data" / "labels" / split
    rows = []
    for path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        t0 = time.perf_counter()
        boxes, scores, (h, w) = predict(path)
        latency_ms = (time.perf_counter() - t0) * 1000   # end-to-end termasuk baca gambar
        gt = load_gt_boxes(lbl_dir / f"{path.stem}.txt", w, h)
        tp, _ = match(boxes, scores, gt)
        iou = box_iou(boxes, gt)
        rows.append({"image": path.name, "severity": severity_from_name(path.name),
                     "gt_count": len(gt), "scores": scores, "tp": tp, "latency_ms": latency_ms,
                     "tp_iou": [match(boxes, scores, gt, t)[0] for t in IOU_GRID],   # untuk mAP@50-95
                     "max_iou": iou.max(axis=1) if iou.size else np.zeros(len(boxes))})
    return pd.DataFrame(rows)


def counts_at(df: pd.DataFrame, conf: float) -> np.ndarray:
    return np.array([(s >= conf).sum() for s in df["scores"]])


def tp_fp_fn_at(df: pd.DataFrame, conf: float) -> pd.DataFrame:
    """TP/FP/FN per gambar pada threshold conf (matching greedy tidak berubah oleh filter conf)."""
    tp = np.array([(t & (s >= conf)).sum() for s, t in zip(df["scores"], df["tp"])])
    fp = counts_at(df, conf) - tp
    return pd.DataFrame({"TP": tp, "FP": fp, "FN": df["gt_count"].to_numpy() - tp})


def counting_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - gt
    r = float(np.corrcoef(gt, pred)[0, 1]) if len(gt) > 1 and pred.std() > 0 else float("nan")
    return {
        "n_images": int(len(gt)),
        "MAE": round(float(np.abs(err).mean()), 2),
        "RMSE": round(float(np.sqrt((err ** 2).mean())), 2),
        "bias": round(float(err.mean()), 2),          # negatif = under-count
        "pearson_r": round(r, 4),
        "total_gt": int(gt.sum()),
        "total_pred": int(pred.sum()),
    }


def report_at(df: pd.DataFrame, conf: float) -> dict:
    """Counting + TP/FP/FN keseluruhan dan per level keparahan pada satu threshold."""
    gt = df["gt_count"].to_numpy()
    pred = counts_at(df, conf)
    tff = tp_fp_fn_at(df, conf)
    # Jenis FP: overlap sebagian dengan lesi GT (0.1 <= IoU < 0.5) = kesalahan lokalisasi/skala,
    # tanpa overlap berarti = FP latar (kulit/latar belakang, atau lesi yang tidak dianotasi)
    fp_loc = sum(int((~t & (s >= conf) & (m >= 0.1)).sum())
                 for s, t, m in zip(df["scores"], df["tp"], df["max_iou"]))
    out = {"conf": conf, "counting": counting_metrics(gt, pred), "detection": prf(*tff.sum()),
           "fp_breakdown": {"localization": fp_loc, "background": int(tff["FP"].sum()) - fp_loc},
           "per_level": {}}
    for lvl in sorted(df["severity"].unique()):
        m = (df["severity"] == lvl).to_numpy()
        out["per_level"][f"level{lvl}"] = {"counting": counting_metrics(gt[m], pred[m]),
                                           "detection": prf(*tff[m].sum())}
    return out


def main():
    parser = argparse.ArgumentParser(description="AcneVision - Evaluasi deteksi, counting & confusion matrix")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=None, help="Default: imgsz saat training")
    parser.add_argument("--mode", choices=["plain", "tta", "sahi"], default="plain")
    parser.add_argument("--slice", type=int, default=960, help="Ukuran patch SAHI (px)")
    args = parser.parse_args()

    weights = args.weights.resolve()
    model = YOLO(str(weights))
    imgsz = args.imgsz or train_imgsz(model)
    device = get_device()
    suffix = {"plain": "", "tta": "_tta", "sahi": f"_sahi{args.slice}"}[args.mode]
    run_name = weights.parent.parent.name + suffix
    out_dir = RUNS_DIR / "eval" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {weights}\nmode: {args.mode} | imgsz: {imgsz} | output: {out_dir}")

    # 1) Metrik deteksi Ultralytics (tidak tersedia untuk SAHI)
    det = None
    if args.mode != "sahi":
        det = detection_metrics(model, imgsz, device, out_dir, augment=args.mode == "tta")
        print(f"[Deteksi/test Ultralytics] {det}")

    # 2) Prediksi + matching val & test
    predict = make_predictor(model, weights, args.mode, imgsz, device, args.slice)
    val_df = collect(predict, "val")
    test_df = collect(predict, "test")

    # 3) Threshold operasional dipilih di val (MAE counting minimum)
    val_gt = val_df["gt_count"].to_numpy()
    val_mae = {c: np.abs(counts_at(val_df, c) - val_gt).mean() for c in CONF_GRID}
    best_conf = float(min(val_mae, key=val_mae.get))

    n_gt = int(test_df["gt_count"].sum())
    ap = round(ap50(list(test_df["scores"]), list(test_df["tp"]), n_gt), 4)
    tps_by_iou = [[r[k] for r in test_df["tp_iou"]] for k in range(len(IOU_GRID))]
    ap5095 = round(map50_95(list(test_df["scores"]), tps_by_iou, n_gt), 4)
    at_default, at_tuned = report_at(test_df, 0.25), report_at(test_df, best_conf)
    print(f"[AP50 internal/test] {ap} | [mAP50-95 internal/test] {ap5095}")
    print(f"[Test conf 0.25] {at_default['counting']} {at_default['detection']}")
    print(f"[Test conf {best_conf:.2f} (val)] {at_tuned['counting']} {at_tuned['detection']}")

    # 4) Confusion matrix (conf operasional) + CSV per gambar
    for label, rep in (("0.25", at_default), (f"{best_conf:.2f}", at_tuned)):
        d = rep["detection"]
        plot_confusion(d["TP"], d["FP"], d["FN"], out_dir / f"confusion_matrix_conf{label}.png",
                       f"{run_name} · test · conf={label} · IoU=0.5")
    tff = tp_fp_fn_at(test_df, best_conf)
    csv = test_df[["image", "severity", "gt_count"]].assign(
        pred_count=counts_at(test_df, best_conf), **{c: tff[c] for c in tff})
    csv.to_csv(out_dir / "counts_test.csv", index=False)

    report = {
        "weights": str(weights.relative_to(PROJECT_ROOT)),
        "mode": args.mode, "imgsz": imgsz, "max_det": MAX_DET, "iou_match": 0.5,
        "slice": args.slice if args.mode == "sahi" else None,
        "detection_test_ultralytics": det,
        "ap50_test_internal": ap,
        "map50_95_test_internal": ap5095,
        # Median latency end-to-end per gambar test (gambar pertama = warmup, diabaikan)
        "latency_ms_median": round(float(test_df["latency_ms"].iloc[1:].median()), 1),
        "conf_selected_on_val": best_conf,
        "test_conf_0.25": at_default,
        "test_conf_val_tuned": at_tuned,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Laporan: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
