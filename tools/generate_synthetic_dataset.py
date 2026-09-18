"""
Generate synthetic wafer images for all defect classes.

Creates dataset/train/<class>/ and dataset/val/<class>/ directories
with augmented synthetic wafer maps (noise, rotation, brightness).

Usage:
    python tools/generate_synthetic_dataset.py
    python tools/generate_synthetic_dataset.py --per-class 100
    python tools/generate_synthetic_dataset.py --val-split 0.2
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.synthetic_generator import generate_macro_wafer_map

DEFECT_CLASSES = [
    "scratch",
    "edge_loss",
    "cluster",
    "center",
    "ring",
    "normal",
    "full_fail",
]

DATASET_DIR = PROJECT_ROOT / "dataset"


# ─── Extra augmentations (on top of built-in noise + brightness) ─────
def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply random rotation, flip, brightness jitter, and Gaussian noise."""
    h, w = img.shape[:2]

    # 1. Random rotation (0°, 90°, 180°, 270° or arbitrary small angle)
    if random.random() < 0.5:
        # Exact 90° multiples
        k = random.choice([1, 2, 3])
        img = np.rot90(img, k=k)
    else:
        # Small arbitrary rotation (-30° to +30°)
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    # 2. Random horizontal / vertical flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)  # horizontal
    if random.random() < 0.5:
        img = cv2.flip(img, 0)  # vertical

    # 3. Brightness jitter (±20%)
    factor = random.uniform(0.8, 1.2)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # 4. Additional Gaussian noise
    sigma = random.uniform(2, 8)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 5. Random Gaussian blur
    if random.random() < 0.4:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # 6. Ensure output shape is consistent
    img = cv2.resize(img, (w, h))

    return img


def generate_dataset(
    per_class: int = 50,
    val_split: float = 0.2,
    image_size: int = 224,
) -> None:
    """Generate synthetic images and split into train/val folders."""
    n_val = max(1, int(per_class * val_split))
    n_train = per_class - n_val

    total = per_class * len(DEFECT_CLASSES)
    generated = 0

    for cls in DEFECT_CLASSES:
        # Create directories
        train_dir = DATASET_DIR / "train" / cls
        val_dir = DATASET_DIR / "val" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for i in range(per_class):
            # Generate base synthetic wafer map
            img = generate_macro_wafer_map(cls, size=(image_size, image_size))

            # Apply extra augmentations
            img = augment_image(img)

            # Decide train vs val
            if i < n_train:
                out_path = train_dir / f"synthetic_{cls}_{i:04d}.png"
            else:
                out_path = val_dir / f"synthetic_{cls}_{i:04d}.png"

            cv2.imwrite(str(out_path), img)
            generated += 1

            if generated % 50 == 0 or generated == total:
                print(f"  [{generated}/{total}] Generated {cls} image {i+1}/{per_class}")

    print(f"\n✅ Done! {generated} images saved to {DATASET_DIR}")
    print(f"   Train: {n_train} per class × {len(DEFECT_CLASSES)} classes = {n_train * len(DEFECT_CLASSES)}")
    print(f"   Val:   {n_val} per class × {len(DEFECT_CLASSES)} classes = {n_val * len(DEFECT_CLASSES)}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic wafer defect images.")
    parser.add_argument("--per-class", type=int, default=50, help="Images per class (default: 50)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction for validation (default: 0.2)")
    parser.add_argument("--image-size", type=int, default=224, help="Image dimension (default: 224)")
    args = parser.parse_args()

    print(f"🔬 Generating synthetic wafer dataset...")
    print(f"   Classes: {DEFECT_CLASSES}")
    print(f"   Per class: {args.per_class}")
    print(f"   Val split: {args.val_split}")
    print(f"   Image size: {args.image_size}x{args.image_size}")
    print()

    generate_dataset(
        per_class=args.per_class,
        val_split=args.val_split,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
