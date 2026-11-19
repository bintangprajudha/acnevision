"""
AcneVision - Ekspor CSV peringkat model (split test, conf dipilih di val).

Membaca docs/report/summary.json (dibuat report.py), menambah YOLOv8l @ 1280 dari
runs/eval/yolov8l_1280, memberi nama metode deskriptif, lalu mengurutkan menurun menurut F1.
Output: docs/report/ranking_models.csv (UTF-8 BOM agar terbuka benar di Excel).
Jalankan: .venv\Scripts\python.exe scripts\export_ranking.py
"""
import json

import pandas as pd

from report import EVAL, summary_rows
from common import PROJECT_ROOT as ROOT

SUMMARY = ROOT / "docs" / "report" / "summary.json"
OUT = ROOT / "docs" / "report" / "ranking_models.csv"

# label di summary.json -> (nama deskriptif, mode inferensi)
NAMES = {
    "B0 baseline default": ("YOLOv8l bawaan (tanpa tuning)", "plain"),
    "B0 + SAHI": ("YOLOv8l bawaan + SAHI", "SAHI 1728"),
    "E3 aug+hyp tuned": ("YOLOv8l resep tuning @ 960", "plain"),
    "E3 + TTA": ("YOLOv8l resep tuning @ 960 + TTA", "TTA"),
    "E3 + SAHI": ("YOLOv8l resep tuning @ 960 + SAHI", "SAHI 1728"),
    "E4 tile-FT + SAHI": ("YOLOv8l resep tuning, fine-tune tile + SAHI", "SAHI 1728"),
    "E5 tile-FT(B0) + SAHI": ("YOLOv8l bawaan, fine-tune tile + SAHI", "SAHI 1728"),
    "E6 YOLO11l (resep E3)": ("YOLO11l resep tuning @ 960", "plain"),
    "E6 + SAHI": ("YOLO11l resep tuning @ 960 + SAHI", "SAHI 1728"),
    "E7 YOLO11l tile-FT (resep E4)": ("YOLO11l fine-tune tile", "plain"),
    "E7 tile-FT + SAHI": ("YOLO11l fine-tune tile + SAHI", "SAHI 1728"),
    "E8 Plain": ("YOLOv8l resep tuning, fine-tune tile 60 epoch", "plain"),
    "E8 SAHI": ("YOLOv8l resep tuning, fine-tune tile 60 epoch + SAHI", "SAHI 1728"),
    "E9 Plain": ("YOLOv8l dilatih pada tile 2x2", "plain"),
    "E9 tile dari COCO + SAHI": ("YOLOv8l dilatih pada tile 2x2 + SAHI", "SAHI 1728"),
    "E12 YOLOv8l @ 1280": ("YOLOv8l @ 1280", "plain"),
}


def main():
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    e12 = EVAL / "yolov8l_1280"
    if (e12 / "metrics.json").exists() and (e12 / "counts_test.csv").exists():
        m = json.loads((e12 / "metrics.json").read_text(encoding="utf-8"))
        extra = summary_rows([{"m": m, "label": "E12 YOLOv8l @ 1280", "eval_dir": "yolov8l_1280"}])
        df = pd.concat([df, extra], ignore_index=True)
    unknown = sorted(set(df["Varian"]) - set(NAMES))
    if unknown:
        raise SystemExit(f"Label belum punya nama deskriptif: {unknown}")
    df.insert(0, "Model", df["Varian"].map(lambda v: NAMES[v][0]))
    df.insert(1, "Mode inferensi", df["Varian"].map(lambda v: NAMES[v][1]))
    df = df.rename(columns={"Varian": "Kode run"})
    df["Peringkat MAE"] = df["MAE"].rank(method="min").astype(int)
    df = df.sort_values("F1", ascending=False, kind="stable").reset_index(drop=True)
    df.insert(0, "Peringkat F1", df["F1"].rank(method="min", ascending=False).astype(int))
    # Kolom ala tabel AcneAI. TPR = Recall. FPR dan AUC butuh definisi 'negatif' (TN tidak terdefinisi pada
    # deteksi objek) dan tidak ditulis di paper yang bisa kami baca -> sengaja tidak dihitung.
    df["Threshold"] = df["conf (val)"]
    df["TPR"] = df["Recall"]
    df["FPR"] = "n/a"
    df["AUC"] = "n/a"
    df["Sumber"] = "model kita; test 121 gambar; conf dari val"
    # Baris acuan AcneAI (Tabel 1 PDF 2216_paper.pdf, dicocokkan ulang 2026-11-13).
    # 'AP' di paper itu = precision-gaya (bukan COCO AP), jangan dibandingkan dengan AP50 internal kita.
    ref = pd.DataFrame([
        {"Model": "ACUAN: AcneAI (Gazeau 2024)", "Mode inferensi": "n/a", "Threshold": 1.81, "TPR": 0.52, "FPR": 0.09,
         "Precision": 0.52, "AP50 (internal)": "0.45 (AP gaya AcneAI)", "AUC": 0.79},
        {"Model": "ACUAN: YOLOv8 @ 1280 AcneAI (Gazeau 2024)", "Mode inferensi": "n/a", "Threshold": 0.18, "TPR": 0.54,
         "FPR": 0.14, "Precision": 0.54, "AP50 (internal)": "0.57 (AP gaya AcneAI)", "AUC": 0.72},
    ])
    ref["Recall"] = ref["TPR"]
    ref["Sumber"] = "AcneAI MICCAI 2024 Tabel 1 (diverifikasi dari PDF); uji pada v2, jumlah gambar tidak tertulis; threshold min |FP-FN|"
    df = pd.concat([df, ref], ignore_index=True)
    first = ["Peringkat F1", "Model", "Mode inferensi", "Threshold", "TPR", "FPR", "Precision", "AP50 (internal)", "AUC", "F1"]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"Tertulis: {OUT} ({len(df)} baris)")
    print(df[["Peringkat F1", "Model", "Threshold", "TPR", "FPR", "Precision", "AP50 (internal)", "AUC", "F1"]].to_string(index=False))


if __name__ == "__main__":
    main()
