# AcneVision — Roadmap (FINALIZED)

> **Status: FINALIZED** — Berdasarkan SPEC.md yang disepakati pada 2026-10-01

## Execution Flow

```
Wave 1 (Paralel)          Wave 2              Wave 3              Wave 4
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Plan 1.1 │  │ Plan 1.2 │  │ Plan 2.1 │  │ Plan 3.1 │  │ Plan 4.1 │
│ Env Setup│  │ Dataset  │→ │ Baseline │→ │ Tuning   │→ │ Export   │
│          │  │ Download │  │ Training │  │          │  │ & Infer  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

## Plans

### Wave 1: Foundation (Paralel)
| Plan | Nama | Deskripsi | File Utama |
|------|------|-----------|------------|
| **1.1** | Environment Setup | Buat venv, install PyTorch + Ultralytics + CUDA | `requirements.txt` |
| **1.2** | Dataset Preparation | Download ACNE04-v2, konversi ke YOLOv8 format | `scripts/download_dataset.py`, `data/data.yaml` |

### Wave 2: Core Training
| Plan | Nama | Deskripsi | Depends On |
|------|------|-----------|------------|
| **2.1** | Baseline Training | Train YOLOv8s default params, evaluate test set | 1.1, 1.2 |

### Wave 3: Optimization
| Plan | Nama | Deskripsi | Depends On |
|------|------|-----------|------------|
| **3.1** | Tuning & Augmentation | HP tuning, augmentasi agresif, perbandingan metrik | 2.1 |

### Wave 4: Delivery
| Plan | Nama | Deskripsi | Depends On |
|------|------|-----------|------------|
| **4.1** | Export & Inference | Export ONNX, buat inference pipeline, hitung jerawat | 3.1 |

## Target Metrik
- **Precision**: > 0.5
- **Recall**: > 0.5
- **mAP@50**: > 0.4
- **mAP@50-95**: > 0.25
- **Inference Speed**: < 100ms per gambar (GPU)

## Deliverables
1. `models/acnevision.onnx` — Model ONNX siap deploy
2. `scripts/inference.py` — Script deteksi + counting end-to-end
3. `docs/METRICS.md` — Dokumentasi performa model
4. `runs/detect/` — Training artifacts (weights, curves, confusion matrix)
