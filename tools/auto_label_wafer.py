"""
Auto-Label Wafer Images for YOLO Training

Assigns a full-image bounding box (class 0) to every image in the wafer directory.
Non-wafer images get no label file.

Usage:
    python tools/auto_label_wafer.py --wafer-dir dataset/images/train --label-dir dataset/labels/train
    python tools/auto_label_wafer.py --wafer-dir dataset/images/val --label-dir dataset/labels/val

For non-wafer images: simply don't include them in --wafer-dir (no label = negative sample).
"""

import argparse
from pathlib import Path


def auto_label(wafer_dir: str, label_dir: str):
    wafer_path = Path(wafer_dir)
    label_path = Path(label_dir)
    label_path.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images = [f for f in wafer_path.iterdir() if f.suffix.lower() in extensions]

    if not images:
        print(f"No images found in {wafer_path}")
        return

    count = 0
    for img_file in sorted(images):
        label_file = label_path / (img_file.stem + ".txt")

        # Full-image bounding box: class=0, center=(0.5, 0.5), size=(1.0, 1.0)
        with open(label_file, "w") as f:
            f.write("0 0.500000 0.500000 1.000000 1.000000\n")

        count += 1

    print(f"Auto-labeled {count} wafer images in {label_path}")
    print("Label format: 0 0.500000 0.500000 1.000000 1.000000 (full image bbox)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label wafer images with full-image bounding box")
    parser.add_argument("--wafer-dir", type=str, required=True, help="Directory containing wafer images")
    parser.add_argument("--label-dir", type=str, required=True, help="Directory to save YOLO labels")
    args = parser.parse_args()

    auto_label(args.wafer_dir, args.label_dir)
