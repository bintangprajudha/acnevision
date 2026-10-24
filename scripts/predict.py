"""
AcneVision - Prediksi & Visualisasi (gambar tunggal atau folder)
================================================================
Kotak digambar tanpa teks apa pun di atas wajah. Jumlah lesi dan legenda
diletakkan di strip header DI LUAR area gambar, sehingga tidak ada label
yang menutupi lesi.

Mode --compare (butuh label GT di data/labels/<split>/):
    TP = biru (prediksi benar) · FP = kuning (prediksi salah) · FN = merah (lesi terlewat)
    dan menulis predictions.csv (GT, pred, TP, FP, FN per gambar).

Usage:
    python scripts/predict.py --source data/images/test/levle2_118.jpg
    python scripts/predict.py --source data/images/test --compare --conf 0.21
    python scripts/predict.py --source data/images/test --sahi --slice 960
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from analysis import load_gt_boxes, match, prf
from common import DEFAULT_WEIGHTS, MAX_DET, PROJECT_ROOT, RUNS_DIR, get_device, train_imgsz

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# Warna BGR
GREEN, BLUE, YELLOW, RED = (60, 200, 60), (255, 140, 0), (0, 215, 255), (40, 40, 230)


def line_width(img) -> int:
    """Ketebalan garis proporsional resolusi (≈3 px pada foto 3456 px)."""
    return max(2, round(max(img.shape[:2]) / 1200))


def draw_boxes(img, boxes, color, lw):
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, lw, cv2.LINE_AA)


def add_header(img, items: list):
    """Tambahkan strip gelap di ATAS gambar berisi teks/legenda: items = [(teks, warna BGR)]."""
    h, w = img.shape[:2]
    scale = max(0.8, w / 1600)
    thick = max(2, round(scale * 2))
    (_, th), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    bar_h = int(th * 2.4)
    canvas = np.full((h + bar_h, w, 3), 28, np.uint8)
    canvas[bar_h:] = img
    x, y = int(th * 0.8), int(bar_h * 0.68)
    for text, color in items:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] + int(th * 1.6)
    return canvas


def make_predictor(args, model, imgsz, device):
    """Kembalikan fungsi path -> (boxes xyxy, scores)."""
    if args.sahi:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(args.weights), confidence_threshold=args.conf,
            image_size=imgsz, device="cuda:0" if device != "cpu" else "cpu",   # slice di-resize ke imgsz
        )

        def predict(path):
            res = get_sliced_prediction(
                str(path), sahi_model, slice_height=args.slice, slice_width=args.slice,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2, postprocess_type="NMS",
                postprocess_match_metric="IOU", postprocess_match_threshold=0.5, verbose=0,
            )
            preds = res.object_prediction_list
            return (np.array([p.bbox.to_xyxy() for p in preds]).reshape(-1, 4),
                    np.array([p.score.value for p in preds]))

        return predict

    def predict(path):
        r = model.predict(str(path), conf=args.conf, imgsz=imgsz, max_det=MAX_DET,
                          augment=args.tta, device=device, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    return predict


def main():
    parser = argparse.ArgumentParser(description="AcneVision - Prediksi & visualisasi")
    parser.add_argument("--source", type=Path, required=True, help="Gambar atau folder")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=None, help="Default: imgsz saat training")
    parser.add_argument("--compare", action="store_true", help="Bandingkan dengan GT (TP/FP/FN)")
    parser.add_argument("--tta", action="store_true", help="Test-time augmentation")
    parser.add_argument("--sahi", action="store_true", help="Gunakan SAHI sliced inference")
    parser.add_argument("--slice", type=int, default=960, help="Ukuran patch SAHI (px)")
    parser.add_argument("--out", type=Path, default=None, help="Default: runs/predict/<run>/<source>")
    args = parser.parse_args()

    files = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMG_EXT) \
        if args.source.is_dir() else [args.source]
    if not files:
        raise SystemExit(f"Tidak ada gambar di {args.source}")
    run = args.weights.resolve().parent.parent.name
    mode = "sahi" if args.sahi else "tta" if args.tta else "plain"
    out = args.out or RUNS_DIR / "predict" / f"{run}_{mode}" / (args.source.name if args.source.is_dir() else "single")
    out.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = YOLO(str(args.weights))
    imgsz = args.imgsz or train_imgsz(model)
    predict = make_predictor(args, model, imgsz, device)
    print(f"Model: {run} | mode={mode} | imgsz={imgsz} | conf={args.conf} | {len(files)} gambar -> {out}")

    rows = []
    for path in files:
        t0 = time.perf_counter()
        boxes, scores = predict(path)
        ms = (time.perf_counter() - t0) * 1000
        img = cv2.imread(str(path))
        lw = line_width(img)
        row = {"image": path.name, "pred_count": len(boxes), "inference_ms": round(ms, 1)}

        if args.compare:
            # Label GT dicari di data/labels/<nama folder split>/
            lbl = PROJECT_ROOT / "data" / "labels" / path.parent.name / f"{path.stem}.txt"
            gt = load_gt_boxes(lbl, img.shape[1], img.shape[0])
            tp, gt_hit = match(boxes, scores, gt)
            draw_boxes(img, gt[~gt_hit], RED, lw)
            draw_boxes(img, boxes[~tp], YELLOW, lw)
            draw_boxes(img, boxes[tp], BLUE, lw)
            m = prf(tp.sum(), (~tp).sum(), (~gt_hit).sum())
            row.update(gt_count=len(gt), **{k: m[k] for k in ("TP", "FP", "FN")})
            header = [(f"Pred {len(boxes)} | GT {len(gt)}", (255, 255, 255)),
                      (f"TP {m['TP']}", BLUE), (f"FP {m['FP']}", YELLOW), (f"FN {m['FN']}", RED)]
        else:
            draw_boxes(img, boxes, GREEN, lw)
            header = [(f"Acne: {len(boxes)}", (255, 255, 255))]

        cv2.imwrite(str(out / f"{path.stem}.jpg"), add_header(img, header))
        rows.append(row)
        print(f"  {path.name}: {len(boxes)} lesi" + (f" (GT {row['gt_count']})" if args.compare else "")
              + f" | {ms:.0f} ms")

    df = pd.DataFrame(rows)
    df.to_csv(out / "predictions.csv", index=False)
    if args.compare and len(df) > 1:
        tot = prf(df["TP"].sum(), df["FP"].sum(), df["FN"].sum())
        mae = (df["pred_count"] - df["gt_count"]).abs().mean()
        print(f"Total: {tot} | MAE count {mae:.2f}")
    print(f"Selesai: {len(df)} gambar -> {out}")


if __name__ == "__main__":
    main()
