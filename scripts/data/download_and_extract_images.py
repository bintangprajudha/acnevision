"""
Download ACNE04 parquet files from HuggingFace and extract image files.
Uses polars to extract embedded JPEG bytes directly to disk.
"""
import os
import json
import time
from pathlib import Path
from huggingface_hub import hf_hub_download
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_HF = DATA_RAW / "acne04_hf" / "data"
IMAGES_DIR = DATA_RAW / "images"
ANNOTATIONS_FILE = DATA_RAW / "Acne04-v2_annotations.json"

DATA_HF.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_FILES = [
    "data/train-00000-of-00003.parquet",
    "data/train-00001-of-00003.parquet",
    "data/train-00002-of-00003.parquet"
]

def download_parquets():
    print("=== Step 1: Downloading Parquet Files ===")
    for filename in PARQUET_FILES:
        target_file = DATA_RAW / "acne04_hf" / filename
        if target_file.exists() and target_file.stat().st_size > 300 * 1024 * 1024:
            print(f"  [OK] Already downloaded: {target_file.name} ({target_file.stat().st_size / 1024**2:.1f} MB)")
            continue

        print(f"  Downloading {filename}...")
        for attempt in range(1, 4):
            try:
                downloaded_path = hf_hub_download(
                    repo_id="Layered-Labs/ACNE04",
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(DATA_RAW / "acne04_hf"),
                    local_dir_use_symlinks=False
                )
                print(f"  [SUCCESS] Downloaded: {downloaded_path}")
                break
            except Exception as e:
                print(f"  [RETRY {attempt}/3] Error downloading {filename}: {e}")
                time.sleep(5)

def extract_images():
    print("\n=== Step 2: Extracting Images from Parquet ===")
    with open(str(ANNOTATIONS_FILE), "r") as f:
        annot_data = json.load(f)
    
    needed_files = {item["file_name"] for item in annot_data["images"]}
    print(f"  Target files needed from v2 annotation: {len(needed_files)}")

    extracted_count = 0
    total_images_in_parquet = 0

    for filename in PARQUET_FILES:
        target_file = DATA_RAW / "acne04_hf" / filename
        if not target_file.exists():
            print(f"  [SKIP] File not found: {target_file}")
            continue

        print(f"  Reading {target_file.name} with polars...")
        df = pl.read_parquet(str(target_file))
        total_images_in_parquet += len(df)

        for row in df.iter_rows(named=True):
            fname = row["file_name"]
            img_bytes = row["image"]["bytes"]
            
            # Save to images dir
            dst_path = IMAGES_DIR / fname
            if not dst_path.exists() or dst_path.stat().st_size == 0:
                with open(str(dst_path), "wb") as img_out:
                    img_out.write(img_bytes)
            
            if fname in needed_files:
                extracted_count += 1

    print(f"\n=== Extraction Summary ===")
    print(f"  Total images processed in parquets: {total_images_in_parquet}")
    print(f"  Matching v2 images extracted: {extracted_count} / {len(needed_files)}")
    print(f"  Total files in {IMAGES_DIR}: {len(list(IMAGES_DIR.glob('*.jpg')))}")

if __name__ == "__main__":
    download_parquets()
    extract_images()
