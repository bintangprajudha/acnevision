"""
AcneVision - Training YOLOv8 dari file konfigurasi
==================================================
Semua hyperparameter berada di configs/*.yaml agar setiap eksperimen reproducible.
Bila checkpoint `last.pt` run dengan nama yang sama sudah ada, training di-resume.

Usage:
    python scripts/train.py                                   # configs/yolov8l_960.yaml
    python scripts/train.py --cfg configs/exp_baru.yaml
    python scripts/train.py --cfg configs/exp_baru.yaml --epochs 1 --fraction 0.05   # smoke test
"""
import argparse
from pathlib import Path

import torch
import yaml

from common import PROJECT_ROOT, RUNS_DIR, get_device, load_model

DEFAULT_CFG = PROJECT_ROOT / "configs" / "yolov8l_960.yaml"


def build_albumentations(specs: list) -> list:
    """List {name, ...param} dari config -> objek albumentations (diteruskan sebagai `augmentations`).

    Menggantikan set default Ultralytics (Blur/MedianBlur/ToGray/CLAHE p=0.01)."""
    import albumentations as A
    return [getattr(A, s["name"])(**{k: v for k, v in s.items() if k != "name"}) for s in specs]


def load_cfg(cfg_path: Path) -> dict:
    """Baca config YAML dan resolve path `model`/`data`/`pretrained` terhadap root project.

    Nama tanpa folder yang tidak ada di project (mis. `yolov8l-p2.yaml` bawaan Ultralytics) diteruskan apa adanya."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("model", "data", "pretrained"):
        if not isinstance(cfg.get(key), str):
            continue
        p = Path(cfg[key])
        if p.is_absolute() or (len(p.parts) == 1 and not (PROJECT_ROOT / p).exists()):
            continue
        cfg[key] = str(PROJECT_ROOT / p)
    if "albumentations" in cfg:
        cfg["augmentations"] = build_albumentations(cfg.pop("albumentations"))
    return cfg


def main():
    parser = argparse.ArgumentParser(description="AcneVision - Training YOLOv8")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="Path config YAML")
    parser.add_argument("--name", type=str, default=None, help="Override nama run")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch")
    parser.add_argument("--fraction", type=float, default=None, help="Subset data train (smoke test)")
    parser.add_argument("--project", type=Path, default=RUNS_DIR / "detect", help="Folder output run")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    model_path = cfg.pop("model")
    for key in ("name", "epochs", "fraction"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    device = get_device()
    gpu = torch.cuda.get_device_name(0) if device != "cpu" else "CPU (sangat lambat)"
    print(f"Config : {args.cfg}\nModel  : {model_path}\nRun    : {cfg['name']}\nDevice : {gpu}")

    last = args.project / cfg["name"] / "weights" / "last.pt"
    if last.exists():
        print(f"Resume dari checkpoint: {last}")
        load_model(last).train(resume=True, device=device, workers=cfg.get("workers", 2))
    else:
        load_model(model_path).train(project=str(args.project), device=device, **cfg)

    print(f"Selesai. Bobot: {args.project / cfg['name'] / 'weights' / 'best.pt'}")
    print("Langkah berikutnya: python scripts/evaluate.py --weights <best.pt>")


if __name__ == "__main__":
    main()
