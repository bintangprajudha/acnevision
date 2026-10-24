"""
AcneVision - Convert ACNE04-v2 Annotations to YOLOv8 Format
Converts circle annotations (cx, cy, radius) to YOLO bounding boxes.
Splits dataset into train/val/test with fixed seed.
"""
import json
import os
import shutil
import random
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_FILE = DATA_RAW / "Acne04-v2_annotations.json"
SOURCE_IMAGES = DATA_RAW / "images"

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42


def circle_to_yolo_bbox(cx, cy, radius, img_width, img_height):
    """
    Convert circle annotation to YOLO bounding box format.

    Args:
        cx, cy: Circle center coordinates (pixels)
        radius: Circle radius (pixels)
        img_width, img_height: Image dimensions (pixels)

    Returns:
        tuple: (x_center_norm, y_center_norm, w_norm, h_norm) normalized to [0, 1]
    """
    # Compute bounding box with boundary clipping
    xmin = max(0, cx - radius)
    ymin = max(0, cy - radius)
    xmax = min(img_width, cx + radius)
    ymax = min(img_height, cy + radius)

    # Box dimensions
    box_w = xmax - xmin
    box_h = ymax - ymin

    # Normalize to [0, 1] (YOLO format)
    x_center_norm = (xmin + box_w / 2) / img_width
    y_center_norm = (ymin + box_h / 2) / img_height
    w_norm = box_w / img_width
    h_norm = box_h / img_height

    # Clamp to [0, 1] as safety
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return x_center_norm, y_center_norm, w_norm, h_norm


def load_annotations():
    """Load and parse ACNE04-v2 annotations JSON."""
    print(f"Loading annotations from {ANNOTATIONS_FILE}...")
    with open(str(ANNOTATIONS_FILE), 'r') as f:
        data = json.load(f)

    # Build image info mapping: image_id -> {file_name, width, height}
    image_info = {}
    for img in data["images"]:
        image_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # Build annotations mapping: image_id -> [list of annotations]
    image_annotations = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append({
            "coordinates": ann["coordinates"],
            "radius": ann["radius"],
        })

    print(f"  Images: {len(image_info)}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Images with annotations: {len(image_annotations)}")

    return image_info, image_annotations


def split_dataset(image_ids):
    """Split image IDs into train/val/test with fixed seed."""
    random.seed(RANDOM_SEED)
    ids = list(image_ids)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def convert_and_organize(image_info, image_annotations, splits):
    """Convert annotations and organize files into YOLOv8 structure."""
    stats = {
        "train": {"images": 0, "annotations": 0},
        "val": {"images": 0, "annotations": 0},
        "test": {"images": 0, "annotations": 0},
    }
    skipped_images = []
    radius_values = []

    for split_name, image_ids in splits.items():
        # Create directories
        img_dir = DATA_DIR / "images" / split_name
        lbl_dir = DATA_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_id in image_ids:
            info = image_info[img_id]
            fname = info["file_name"]
            src_path = SOURCE_IMAGES / fname

            # Check if source image exists
            if not src_path.exists():
                skipped_images.append(fname)
                continue

            # Copy image to split directory
            dst_img = img_dir / fname
            if not dst_img.exists():
                shutil.copy2(str(src_path), str(dst_img))

            # Convert annotations to YOLO format
            annotations = image_annotations.get(img_id, [])
            label_name = Path(fname).stem + ".txt"
            dst_label = lbl_dir / label_name

            label_lines = []
            for ann in annotations:
                cx, cy = ann["coordinates"]
                r = ann["radius"]
                radius_values.append(r)

                x_c, y_c, w, h = circle_to_yolo_bbox(
                    cx, cy, r, info["width"], info["height"]
                )

                # Skip invalid boxes (zero area)
                if w <= 0 or h <= 0:
                    continue

                label_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            # Write label file (even if empty — YOLOv8 needs it for negative examples)
            with open(str(dst_label), 'w') as f:
                f.write("\n".join(label_lines))
                if label_lines:
                    f.write("\n")

            stats[split_name]["images"] += 1
            stats[split_name]["annotations"] += len(label_lines)

    return stats, skipped_images, radius_values


def create_data_yaml():
    """Create data.yaml for YOLOv8 training."""
    data_yaml_path = DATA_DIR / "data.yaml"
    # Use absolute path for reliability
    abs_data_dir = str(DATA_DIR).replace("\\", "/")

    content = f"""# AcneVision YOLOv8 Dataset Configuration
# Generated by convert_to_yolo.py

path: {abs_data_dir}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['acne']
"""
    with open(str(data_yaml_path), 'w') as f:
        f.write(content)

    print(f"\n  Created: {data_yaml_path}")
    return data_yaml_path


def main():
    print("=" * 60)
    print("AcneVision - ACNE04-v2 to YOLOv8 Converter")
    print("=" * 60)

    # Step 1: Load annotations
    image_info, image_annotations = load_annotations()

    # Step 2: Split dataset
    print(f"\nSplitting dataset (seed={RANDOM_SEED})...")
    splits = split_dataset(image_info.keys())
    for name, ids in splits.items():
        print(f"  {name}: {len(ids)} images")

    # Step 3: Convert and organize
    print("\nConverting annotations and organizing files...")
    stats, skipped, radius_values = convert_and_organize(
        image_info, image_annotations, splits
    )

    # Step 4: Create data.yaml
    create_data_yaml()

    # Step 5: Print statistics
    print("\n" + "=" * 60)
    print("CONVERSION RESULTS")
    print("=" * 60)

    total_imgs = 0
    total_anns = 0
    for split_name, s in stats.items():
        print(f"  {split_name:5s}: {s['images']:4d} images, {s['annotations']:5d} annotations "
              f"(avg {s['annotations']/max(s['images'],1):.1f}/img)")
        total_imgs += s["images"]
        total_anns += s["annotations"]

    print(f"  {'TOTAL':5s}: {total_imgs:4d} images, {total_anns:5d} annotations")

    if skipped:
        print(f"\n  Skipped (image not found): {len(skipped)}")
        if len(skipped) <= 5:
            for s in skipped:
                print(f"    - {s}")

    if radius_values:
        print(f"\n  Radius statistics:")
        print(f"    Min:    {min(radius_values):.1f} px")
        print(f"    Max:    {max(radius_values):.1f} px")
        print(f"    Mean:   {sum(radius_values)/len(radius_values):.1f} px")
        print(f"    Median: {sorted(radius_values)[len(radius_values)//2]:.1f} px")

    # Verification
    print("\n  Verification:")
    for split_name in ["train", "val", "test"]:
        img_count = len(list((DATA_DIR / "images" / split_name).glob("*")))
        lbl_count = len(list((DATA_DIR / "labels" / split_name).glob("*.txt")))
        match = "[OK]" if img_count == lbl_count else "[MISMATCH]"
        print(f"    {split_name}: {img_count} images, {lbl_count} labels {match}")

    print("=" * 60)
    print(f"  Status: {'OK' if total_imgs > 900 else 'INCOMPLETE - check image download'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
