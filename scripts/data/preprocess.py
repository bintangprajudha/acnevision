"""
AcneVision - Preprocessing: dataset tile untuk slicing-aided fine-tuning
========================================================================
Memotong setiap gambar train/val menjadi grid N x N tile (dengan overlap) pada
resolusi asli, lalu menggabungkannya dengan gambar utuh. Saat training di imgsz
yang sama, lesi pada tile tampil ~N kali lebih besar (mis. N=2: median 13 -> ~26 px
pada 960), sesuai rekomendasi slicing-aided fine-tuning (Akyon et al., 2022, SAHI).

Aturan label: bbox di-clip ke tile; dipertahankan bila >= MIN_VISIBLE luasnya
berada di dalam tile. Split test TIDAK di-tile (evaluasi memakai gambar asli + SAHI).

Usage:
    python scripts/data/preprocess.py --grid 2 --overlap 0.1
    -> data_tiles2/  +  data_tiles2/data.yaml
"""
import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "data"
MIN_VISIBLE = 0.5


def tile_windows(w: int, h: int, grid: int, overlap: float):
    """Koordinat (x1, y1, x2, y2) grid tile dengan overlap relatif terhadap ukuran tile."""
    tw = int(np.ceil(w / (grid - (grid - 1) * overlap)))
    th = int(np.ceil(h / (grid - (grid - 1) * overlap)))
    xs = np.linspace(0, w - tw, grid).astype(int)
    ys = np.linspace(0, h - th, grid).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def read_labels(path: Path, w: int, h: int) -> np.ndarray:
    """Label YOLO -> array (N, 4) xyxy piksel."""
    rows = [list(map(float, l.split()[1:5])) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4))
    cx, cy, bw, bh = np.array(rows).T
    return np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)


def tile_labels(boxes: np.ndarray, win) -> list:
    """Clip bbox ke tile, buang yang terlihat < MIN_VISIBLE, kembalikan baris label YOLO."""
    x1, y1, x2, y2 = win
    tw, th = x2 - x1, y2 - y1
    lines = []
    for bx1, by1, bx2, by2 in boxes:
        cx1, cy1 = max(bx1, x1), max(by1, y1)
        cx2, cy2 = min(bx2, x2), min(by2, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if (cx2 - cx1) * (cy2 - cy1) < MIN_VISIBLE * (bx2 - bx1) * (by2 - by1):
            continue
        cx, cy = ((cx1 + cx2) / 2 - x1) / tw, ((cy1 + cy2) / 2 - y1) / th
        lines.append(f"0 {cx:.6f} {cy:.6f} {(cx2 - cx1) / tw:.6f} {(cy2 - cy1) / th:.6f}")
    return lines


def link_or_copy(src: Path, dst: Path):
    """Hardlink (hemat disk, satu drive) dengan fallback copy."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="AcneVision - buat dataset tile")
    parser.add_argument("--grid", type=int, default=2, help="Jumlah tile per sisi")
    parser.add_argument("--overlap", type=float, default=0.1, help="Overlap antar tile (fraksi tile)")
    args = parser.parse_args()

    dst = PROJECT_ROOT / f"data_tiles{args.grid}"
    for split in ("train", "val"):
        img_out, lbl_out = dst / "images" / split, dst / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for img_path in sorted((SRC / "images" / split).glob("*.jpg")):
            lbl_path = SRC / "labels" / split / f"{img_path.stem}.txt"
            # 1) Gambar utuh (skala asli dataset) tetap disertakan
            link_or_copy(img_path, img_out / img_path.name)
            link_or_copy(lbl_path, lbl_out / lbl_path.name)
            # 2) Tile resolusi penuh
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            boxes = read_labels(lbl_path, w, h)
            for k, win in enumerate(tile_windows(w, h, args.grid, args.overlap)):
                name = f"{img_path.stem}_t{k}"
                x1, y1, x2, y2 = win
                cv2.imwrite(str(img_out / f"{name}.jpg"), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
                lines = tile_labels(boxes, win)
                (lbl_out / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n_img += 1
                n_box += len(lines)
        print(f"{split}: {n_img} tile ({n_box} bbox) + gambar utuh")

    yaml_text = (
        f"# Dataset tile {args.grid}x{args.grid} (overlap {args.overlap}) - dibuat oleh scripts/data/preprocess.py\n"
        f"path: {dst.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"test: {(SRC / 'images' / 'test').as_posix()}   # test tetap gambar asli\n\n"
        "nc: 1\nnames: ['acne']\n"
    )
    (dst / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Selesai: {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
