"""
AcneVision - Benchmark latensi server API (scripts/serve.py) pada gambar VAL
============================================================================
Server harus sudah berjalan. Request dikirim BERURUTAN (satu klien) lewat HTTP lokal, sehingga angka
mencakup unggah + decode + antre + inferensi + JSON, tetapi tidak mencakup jaringan/tunnel.

Skenario (semua gambar val, bukan test set):
    capture_full : file JPEG asli (±3112x3456)                  -> mode=capture (SAHI 1728)
    preview_720  : sisi pendek 720 px, JPEG q92 (= frameToBlob) -> mode=preview (plain 960)
    capture_720  : frame 720 yang sama sebagai capture         -> meniru jalur kamera Scan.tsx (video ±1280x720)

Kolom "diagnostik" (rata-rata deteksi vs GT, persentase det_confidence < 0.6) BUKAN metrik resmi;
metrik resmi hanya dari scripts/evaluate.py.

Usage:
    python scripts/bench_serve.py [--url http://127.0.0.1:8000] [--n 120]
"""
import argparse
import io
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROJECT_ROOT, RUNS_DIR  # noqa: E402

VAL_IMG = PROJECT_ROOT / "data" / "images" / "val"
VAL_LBL = PROJECT_ROOT / "data" / "labels" / "val"
LOW_CONF = 0.6     # LOW_CONFIDENCE_THRESHOLD di frontend (classes.ts)


def to_720(path: Path) -> bytes:
    im = Image.open(path).convert("RGB")
    s = 720 / min(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def run(url: str, items: list, mode: str, warmup: int = 3) -> dict:
    s = requests.Session()

    def post(data):
        return s.post(f"{url}/api/v1/detect", files={"image": ("val.jpg", data, "image/jpeg")},
                      data={"source": "mobile", "input_method": "camera", "mode": mode}, timeout=180)

    for data, _ in items[:warmup]:
        post(data)
    client, server, counts, confs, sizes, errors = [], [], [], [], [], []
    t_all = time.perf_counter()
    for data, gt in items:
        t0 = time.perf_counter()
        r = post(data)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            errors.append(r.status_code)
            continue
        b = r.json()
        client.append(ms)
        server.append(b["timing_ms"]["detection"])
        counts.append(b["summary"]["total_lesions"])
        confs.extend(l["det_confidence"] for l in b["lesions"])
        sizes.append(f"{b['image']['width']}x{b['image']['height']}")
    wall = time.perf_counter() - t_all
    gts = [gt for _, gt in items]
    return {
        "mode": mode, "n": len(client), "errors": errors,
        "image_sizes": sorted(set(sizes)),
        "client_ms": {"mean": statistics.mean(client), "p50": pct(client, 50), "p95": pct(client, 95)},
        "server_detection_ms": {"mean": statistics.mean(server), "p50": pct(server, 50), "p95": pct(server, 95)},
        "sequential_images_per_s": len(client) / wall,
        "diagnostic": {
            "mean_detections": statistics.mean(counts), "mean_gt": statistics.mean(gts),
            "detections_total": len(confs),
            "pct_det_conf_below_0_6": 100 * float(np.mean(np.array(confs) < LOW_CONF)) if confs else 0.0,
            "images_with_0_detections": int(sum(c == 0 for c in counts)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="AcneVision - benchmark latensi server API (val)")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=0, help="Jumlah gambar val (0 = semua)")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    health = requests.get(f"{url}/health", timeout=10).json()
    paths = sorted(VAL_IMG.glob("*.jpg"))
    if args.n:
        paths = paths[:args.n]
    gt = {p: len([l for l in (VAL_LBL / f"{p.stem}.txt").read_text().splitlines() if l.strip()]) for p in paths}
    full = [(p.read_bytes(), gt[p]) for p in paths]
    small = [(to_720(p), gt[p]) for p in paths]
    print(f"{len(paths)} gambar val | server: {health['model']['name']} ({health['model']['weights_sha256']}) "
          f"| GPU {health['gpu']['device']}")

    scen = {}
    for name, items, mode in (("capture_full", full, "capture"), ("preview_720", small, "preview"),
                              ("capture_720", small, "capture")):
        print(f"  {name} ...", flush=True)
        scen[name] = run(url, items, mode)

    print("\n| Skenario | n | Ukuran | Klien mean / p50 / p95 (ms) | Server inferensi mean / p95 (ms) | Gambar/detik "
          "(berurutan) | Deteksi rata-rata (GT) | % conf < 0.6 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, s in scen.items():
        c, sv, d = s["client_ms"], s["server_detection_ms"], s["diagnostic"]
        print(f"| {name} | {s['n']} | {', '.join(s['image_sizes'][:2])} | {c['mean']:.0f} / {c['p50']:.0f} / "
              f"{c['p95']:.0f} | {sv['mean']:.0f} / {sv['p95']:.0f} | {s['sequential_images_per_s']:.2f} | "
              f"{d['mean_detections']:.1f} ({d['mean_gt']:.1f}) | {d['pct_det_conf_below_0_6']:.1f}% |")
        if s["errors"]:
            print(f"  ! {name}: {len(s['errors'])} request gagal: {s['errors'][:5]}")

    out = RUNS_DIR / "serve" / f"bench_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"url": url, "health": health, "n_images": len(paths), "scenarios": scen},
                              indent=1), encoding="utf-8")
    print(f"\nHasil: {out}")


if __name__ == "__main__":
    main()
