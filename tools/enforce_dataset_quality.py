"""
Dataset Quality Enforcement Pipeline.

1. Remove duplicate images (perceptual hash)
2. Remove blurry images (Laplacian variance)
3. Add hard negative samples (confusing class pairs)
4. Rebalance all classes to ~1400 images each

Usage:
    python tools/enforce_dataset_quality.py
    python tools/enforce_dataset_quality.py --target-per-class 1400
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.synthetic_generator import generate_macro_wafer_map

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"

CLASSES = ["normal", "center", "edge_ring", "edge_loss", "scratch", "ring", "cluster", "full_fail"]

# Confusing class pairs for hard negatives
HARD_NEGATIVE_PAIRS = [
    ("ring", "edge_loss"),
    ("scratch", "cluster"),
    ("center", "ring"),
    ("edge_loss", "full_fail"),
    ("cluster", "center"),
]

BLUR_THRESHOLD = 30.0  # Laplacian variance below this = blurry


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: REMOVE DUPLICATES (Perceptual Hash)
# ═══════════════════════════════════════════════════════════════════════
def image_hash(path: Path) -> str:
    """Compute a perceptual hash by resizing to 8x8 grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    resized = cv2.resize(img, (8, 8))
    return hashlib.md5(resized.tobytes()).hexdigest()


def remove_duplicates():
    """Remove duplicate images across each class folder."""
    total_removed = 0
    for cls in CLASSES:
        cls_dir = TRAIN_DIR / cls
        if not cls_dir.exists():
            continue

        seen_hashes = set()
        duplicates = []

        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix not in {".png", ".jpg", ".jpeg"}:
                continue
            h = image_hash(img_path)
            if h in seen_hashes:
                duplicates.append(img_path)
            else:
                seen_hashes.add(h)

        for dup in duplicates:
            dup.unlink()
            total_removed += 1

        if duplicates:
            print(f"  {cls}: removed {len(duplicates)} duplicates")

    print(f"  ✅ Total duplicates removed: {total_removed}")
    return total_removed


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: REMOVE BLURRY IMAGES (Laplacian Variance)
# ═══════════════════════════════════════════════════════════════════════
def remove_blurry():
    """Remove images where Laplacian variance is below threshold."""
    total_removed = 0
    for cls in CLASSES:
        cls_dir = TRAIN_DIR / cls
        if not cls_dir.exists():
            continue

        blurry = []
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix not in {".png", ".jpg", ".jpeg"}:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                blurry.append(img_path)
                continue
            variance = cv2.Laplacian(img, cv2.CV_64F).var()
            if variance < BLUR_THRESHOLD:
                blurry.append(img_path)

        for b in blurry:
            b.unlink()
            total_removed += 1

        if blurry:
            print(f"  {cls}: removed {len(blurry)} blurry images")

    print(f"  ✅ Total blurry removed: {total_removed}")
    return total_removed


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: HARD NEGATIVE GENERATION
# ═══════════════════════════════════════════════════════════════════════
def generate_hard_negative(cls_a: str, cls_b: str, size=(224, 224)) -> np.ndarray:
    """
    Generate a 'confusing' image for cls_a that borrows visual traits from cls_b.
    This forces the model to learn subtle differences between similar classes.
    """
    img_a = generate_macro_wafer_map(cls_a, size=size)
    img_b = generate_macro_wafer_map(cls_b, size=size)

    # Blend: 70% target class + 30% confuser class
    alpha = random.uniform(0.65, 0.80)
    blended = cv2.addWeighted(img_a, alpha, img_b, 1 - alpha, 0)

    # Add slight noise to make it harder
    noise = np.random.normal(0, 5, blended.shape).astype(np.float32)
    blended = np.clip(blended.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Random rotation
    angle = random.uniform(-20, 20)
    h, w = blended.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    blended = cv2.warpAffine(blended, M, (w, h), borderValue=(0, 0, 0))

    return blended


def add_hard_negatives(n_per_pair: int = 50):
    """Generate hard negative samples for each confusing class pair."""
    total = 0
    for cls_a, cls_b in HARD_NEGATIVE_PAIRS:
        cls_dir = TRAIN_DIR / cls_a
        cls_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(cls_dir.glob("*.png")))

        for i in range(n_per_pair):
            img = generate_hard_negative(cls_a, cls_b)
            out_path = cls_dir / f"hard_neg_{cls_b}_{existing + i:04d}.png"
            cv2.imwrite(str(out_path), img)
            total += 1

        print(f"  {cls_a} (vs {cls_b}): +{n_per_pair} hard negatives")

    print(f"  ✅ Total hard negatives added: {total}")
    return total


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: REBALANCE TO TARGET COUNT
# ═══════════════════════════════════════════════════════════════════════
def augment_single(img: np.ndarray) -> np.ndarray:
    """Apply random augmentation to create a new variant."""
    h, w = img.shape[:2]

    # Random rotation
    angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    # Random flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Brightness jitter
    factor = random.uniform(0.8, 1.2)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Noise
    noise = np.random.normal(0, random.uniform(2, 6), img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def rebalance(target: int = 1400):
    """Ensure every class has exactly `target` images via augmentation."""
    for cls in CLASSES:
        cls_dir = TRAIN_DIR / cls
        if not cls_dir.exists():
            continue

        images = sorted([f for f in cls_dir.iterdir() if f.suffix in {".png", ".jpg", ".jpeg"}])
        current = len(images)

        if current == 0:
            print(f"  {cls}: skipping (no seed images found)")
            continue

        if current >= target:
            # Trim excess
            excess = images[target:]
            for f in excess:
                f.unlink()
            print(f"  {cls}: trimmed {len(excess)} → {target}")
        else:
            # Augment to fill
            shortfall = target - current
            for i in range(shortfall):
                src = random.choice(images)
                img = cv2.imread(str(src))
                img = augment_single(img)
                out_path = cls_dir / f"augmented_{cls}_{current + i:04d}.png"
                cv2.imwrite(str(out_path), img)

            print(f"  {cls}: augmented +{shortfall} → {target}")

    print(f"  ✅ All classes rebalanced to {target} images")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Enforce dataset quality.")
    parser.add_argument("--target-per-class", type=int, default=1400)
    parser.add_argument("--hard-neg-per-pair", type=int, default=50)
    args = parser.parse_args()

    print("🔬 Dataset Quality Enforcement Pipeline\n")

    print("📋 Step 1: Removing duplicates...")
    remove_duplicates()

    print("\n📋 Step 2: Removing blurry images...")
    remove_blurry()

    print("\n📋 Step 3: Adding hard negative samples...")
    add_hard_negatives(n_per_pair=args.hard_neg_per_pair)

    print("\n📋 Step 4: Rebalancing to equal class count...")
    rebalance(target=args.target_per_class)

    # Final count
    print("\n📊 Final Dataset Summary:")
    total = 0
    for cls in CLASSES:
        cls_dir = TRAIN_DIR / cls
        count = len(list(cls_dir.glob("*.*"))) if cls_dir.exists() else 0
        total += count
        print(f"  {cls:>12}: {count}")
    print(f"  {'TOTAL':>12}: {total}")
    print("\n🎯 Dataset quality enforcement complete!")


if __name__ == "__main__":
    main()
