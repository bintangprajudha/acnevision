"""
AcneVision - Uji signifikansi paired bootstrap antar varian (split test)
=======================================================================
Resample gambar test dengan pengembalian (10.000x, seed 42); untuk tiap resample dihitung
MAE counting, F1, dan Recall (dari TP/FP/FN per gambar) pada dua varian dengan indeks
gambar yang sama (paired). Output: rata-rata selisih + CI 95% (persentil 2.5-97.5).

Sumber: runs/eval/<run>/counts_test.csv (ditulis evaluate.py pada conf yang dipilih di val).
Pasangan yang hasil evaluasinya belum ada dilewati dengan peringatan.

Usage:
    python scripts/bootstrap.py            -> docs/report/bootstrap.json
"""
import json

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, RUNS_DIR

N_BOOT, SEED = 10_000, 42
OUT = PROJECT_ROOT / "docs" / "report" / "bootstrap.json"

RUNS = {
    "B0": "yolov8l_960_default",
    "B0+SAHI": "yolov8l_960_default_sahi1728",
    "E3": "yolov8l_960",
    "E3+SAHI": "yolov8l_960_sahi1728",
    "E4": "yolov8l_960_tiles2_ft",
    "E4+SAHI": "yolov8l_960_tiles2_ft_sahi1728",
    "E5": "yolov8l_960_default_tiles2_ft",
    "E5+SAHI": "yolov8l_960_default_tiles2_ft_sahi1728",
}
# (a, b) -> selisih a - b; MAE negatif / F1 & Recall positif = a lebih baik
PAIRS = [
    ("E4+SAHI", "B0"), ("E4+SAHI", "E3"), ("E4+SAHI", "E3+SAHI"), ("E3", "B0"),
    ("E3+SAHI", "E3"), ("E4", "E3"), ("B0+SAHI", "B0"),
    # Desain 2x2 (init B0/E3 x tile-FT), semua dengan SAHI 1728
    ("E5+SAHI", "B0+SAHI"),   # efek tile-FT bila init B0
    ("E4+SAHI", "E5+SAHI"),   # efek init E3 vs B0 setelah tile-FT
    ("E5+SAHI", "B0"),
    ("E3+SAHI", "B0+SAHI"),   # efek tuning E3 tanpa tile-FT
    ("E5", "B0"),             # efek tile-FT pada inferensi plain (init B0)
]


def stats(df: pd.DataFrame, idx: np.ndarray):
    """MAE, F1, Recall untuk setiap baris indeks resample (N_BOOT, n)."""
    ae = (df.pred_count - df.gt_count).abs().to_numpy()[idx].mean(1)
    tp, fp, fn = (df[c].to_numpy()[idx].sum(1) for c in ("TP", "FP", "FN"))
    return ae, 2 * tp / (2 * tp + fp + fn), tp / (tp + fn)


def main():
    data = {}
    for key, run in RUNS.items():
        csv = RUNS_DIR / "eval" / run / "counts_test.csv"
        if csv.exists():
            data[key] = pd.read_csv(csv).sort_values("image").reset_index(drop=True)
    n = len(next(iter(data.values())))
    assert all(len(d) == n and (d.image == data["B0"].image).all() for d in data.values()), "daftar gambar test berbeda"
    idx = np.random.default_rng(SEED).integers(0, n, (N_BOOT, n))

    out = {}
    for a, b in PAIRS:
        if a not in data or b not in data:
            print(f"[lewati] {a} - {b}: counts_test.csv belum ada")
            continue
        res = {}
        for name, x, y in zip(["dMAE", "dF1", "dRecall"], stats(data[a], idx), stats(data[b], idx)):
            diff = x - y
            lo, hi = np.percentile(diff, [2.5, 97.5])
            res[name] = [round(float(diff.mean()), 3), round(float(lo), 3), round(float(hi), 3)]
        out[f"{a} - {b}"] = res
        sig = {k: "*" if not (v[1] <= 0 <= v[2]) else "" for k, v in res.items()}
        print(f"{a:>8} - {b:<8} " + "  ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]{sig[k]}" for k, v in res.items()))
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"(* = CI 95% tidak memuat 0)\nDitulis: {OUT}")


if __name__ == "__main__":
    main()
