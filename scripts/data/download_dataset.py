"""
AcneVision - Download Dataset Script
Downloads ACNE04-v2 annotations (GitHub) and ACNE04 images (HuggingFace).
"""
import os
import json
import urllib.request
import zipfile
import shutil
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = DATA_RAW / "images"

# URLs
ANNOTATIONS_URL = "https://raw.githubusercontent.com/AIpourlapeau/acne04v2/main/Acne04-v2_annotations.json"
ANNOTATIONS_FILE = DATA_RAW / "Acne04-v2_annotations.json"

# HuggingFace ACNE04 dataset
HF_REPO = "Layered-Labs/ACNE04"


def download_file(url, dest, desc=""):
    """Download a file with progress."""
    print(f"  Downloading {desc or url}...")
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Saved: {dest} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        return False


def download_annotations():
    """Download ACNE04-v2 annotations JSON from GitHub."""
    print("\n=== Step 1: Download ACNE04-v2 Annotations ===")
    return download_file(ANNOTATIONS_URL, ANNOTATIONS_FILE, "annotations JSON")


def download_images_huggingface():
    """Download ACNE04 images using huggingface_hub."""
    print("\n=== Step 2: Download ACNE04 Images (HuggingFace) ===")

    try:
        from huggingface_hub import snapshot_download
        print(f"  Using huggingface_hub to download {HF_REPO}...")
        cache_dir = DATA_RAW / "hf_cache"
        local_dir = snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            cache_dir=str(cache_dir),
            local_dir=str(DATA_RAW / "acne04_hf"),
        )
        print(f"  Downloaded to: {local_dir}")
        return Path(local_dir)
    except ImportError:
        print("  huggingface_hub not installed, trying alternative...")
        return download_images_fallback()
    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        return download_images_fallback()


def download_images_fallback():
    """Fallback: download images via direct URL."""
    print("  Attempting direct download fallback...")
    # Try Kaggle-style or direct zip download
    fallback_urls = [
        "https://huggingface.co/datasets/Layered-Labs/ACNE04/resolve/main/Classification.tar",
        "https://huggingface.co/datasets/Layered-Labs/ACNE04/resolve/main/Classification.zip",
    ]
    for url in fallback_urls:
        ext = url.split(".")[-1]
        archive_path = DATA_RAW / f"Classification.{ext}"
        if download_file(url, archive_path, f"ACNE04 archive ({ext})"):
            # Extract
            extract_dir = DATA_RAW / "acne04_extracted"
            extract_dir.mkdir(exist_ok=True)
            print(f"  Extracting {archive_path}...")
            try:
                if ext == "zip":
                    with zipfile.ZipFile(str(archive_path), 'r') as zf:
                        zf.extractall(str(extract_dir))
                elif ext == "tar":
                    import tarfile
                    with tarfile.open(str(archive_path), 'r') as tf:
                        tf.extractall(str(extract_dir))
                print(f"  Extracted to: {extract_dir}")
                return extract_dir
            except Exception as e:
                print(f"  Extraction failed: {e}")
                continue
    return None


def collect_images(source_dir, annotations_data):
    """Find and copy images referenced in annotations to IMAGES_DIR."""
    print("\n=== Step 3: Collect & Filter Images ===")

    # Get list of required file names from annotations
    required_files = {}
    for img_info in annotations_data["images"]:
        fname = img_info["file_name"]
        required_files[fname] = img_info["image_id"]

    print(f"  Required images (from annotations): {len(required_files)}")

    # Build index of all image files in source directory
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    all_source_files = {}
    if source_dir and source_dir.exists():
        for f in source_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                all_source_files[f.name] = f

    print(f"  Available source images: {len(all_source_files)}")

    # Copy matching images
    found = 0
    missing = []
    for fname in required_files:
        dest = IMAGES_DIR / fname
        if dest.exists():
            found += 1
            continue
        if fname in all_source_files:
            shutil.copy2(str(all_source_files[fname]), str(dest))
            found += 1
        else:
            missing.append(fname)

    print(f"  Found & copied: {found}/{len(required_files)}")
    if missing:
        print(f"  Missing: {len(missing)}")
        if len(missing) <= 10:
            for m in missing:
                print(f"    - {m}")
        else:
            for m in missing[:5]:
                print(f"    - {m}")
            print(f"    ... and {len(missing) - 5} more")

    return found, missing


def main():
    print("=" * 60)
    print("AcneVision Dataset Download Script")
    print("=" * 60)

    # Step 1: Download annotations
    if not download_annotations():
        print("\nFATAL: Cannot download annotations. Aborting.")
        sys.exit(1)

    # Load annotations
    with open(str(ANNOTATIONS_FILE), 'r') as f:
        annotations_data = json.load(f)

    n_images = len(annotations_data["images"])
    n_annots = len(annotations_data["annotations"])
    print(f"\n  Annotations loaded:")
    print(f"    Images:      {n_images}")
    print(f"    Annotations: {n_annots}")
    print(f"    Avg per img: {n_annots / n_images:.1f}")

    # Step 2: Download images
    source_dir = download_images_huggingface()

    # Step 3: Collect images
    found, missing = collect_images(source_dir, annotations_data)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Annotations file: {ANNOTATIONS_FILE}")
    print(f"  Images directory:  {IMAGES_DIR}")
    print(f"  Total required:    {n_images}")
    print(f"  Total found:       {found}")
    print(f"  Total missing:     {len(missing)}")
    print(f"  Annotations:       {n_annots}")
    print(f"  Status: {'OK' if found >= n_images * 0.9 else 'INCOMPLETE'}")
    print("=" * 60)

    if found < n_images * 0.5:
        print("\nWARNING: Less than 50% of images found.")
        print("You may need to manually download the ACNE04 dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
