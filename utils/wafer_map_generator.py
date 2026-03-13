"""
Synthetic Wafer Map Generator — Macro-Level Defect Patterns
============================================================
Generates full-wafer map images (224×224) for 8 defect classes.
Each sample gets a lot_id for lot-based train/val splitting.

Design decisions:
  - 224×224 matches ImageNet pretraining resolution (no interpolation artifacts).
  - Defects are SPATIAL patterns (rings, clusters), not pixel noise.
  - Augmentation is mild: rotation, flip, brightness jitter.
    NO elastic distortion, NO heavy blur (per constraint).
"""

import numpy as np
import cv2
import random
from typing import Tuple, List, Dict

# ── Class definitions ───────────────────────────────────────────────
DEFECT_CLASSES = [
    "normal",       # 0
    "center",       # 1
    "edge_ring",    # 2
    "edge_loss",    # 3
    "scratch",      # 4
    "ring",         # 5
    "cluster",      # 6
    "full_fail",    # 7
]
NUM_CLASSES = len(DEFECT_CLASSES)
IMG_SIZE = 224


def _draw_wafer_base(size: int = IMG_SIZE) -> np.ndarray:
    """Draw a circular wafer on a black background with subtle grid texture."""
    img = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2 - 12

    # Base wafer fill  (medium gray)
    cv2.circle(img, center, radius, 110, -1)

    # Grid texture for realism
    step = random.randint(10, 18)
    grid_color = random.randint(95, 125)
    for y in range(0, size, step):
        cv2.line(img, (0, y), (size, y), grid_color, 1)
    for x in range(0, size, step):
        cv2.line(img, (x, 0), (x, size), grid_color, 1)

    # Mask to circle
    circle_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)
    img = cv2.bitwise_and(img, img, mask=circle_mask)

    return img, center, radius


# ── Pattern Generators ──────────────────────────────────────────────

def _pattern_normal(img, center, radius):
    """No defect — clean wafer."""
    pass

def _pattern_center(img, center, radius):
    """Bright cluster in the center region (die yield loss at center)."""
    r = random.randint(radius // 6, radius // 3)
    cv2.circle(img, center, r, 255, -1)

def _pattern_edge_ring(img, center, radius):
    """Ring of defects along the wafer edge."""
    thickness = random.randint(8, 18)
    cv2.circle(img, center, radius - 5, 255, thickness)

def _pattern_edge_loss(img, center, radius):
    """Arc of defects at one edge (partial edge failure)."""
    start_angle = random.randint(0, 360)
    end_angle = start_angle + random.randint(60, 140)
    cv2.ellipse(img, center, (radius - 8, radius - 8),
                0, start_angle, end_angle, 255, random.randint(10, 20))

def _pattern_scratch(img, center, radius):
    """One or two bright scratch lines across the wafer."""
    for _ in range(random.randint(1, 2)):
        angle = random.uniform(0, np.pi)
        length = radius * random.uniform(0.6, 1.0)
        x1 = int(center[0] + length * np.cos(angle))
        y1 = int(center[1] + length * np.sin(angle))
        x2 = int(center[0] - length * np.cos(angle))
        y2 = int(center[1] - length * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), 255, random.randint(2, 4))

def _pattern_ring(img, center, radius):
    """Concentric ring (not at the edge — mid-radius)."""
    r = random.randint(radius // 3, int(radius * 0.7))
    cv2.circle(img, center, r, 255, random.randint(6, 14))

def _pattern_cluster(img, center, radius):
    """Localized cluster of defective dies in a random region."""
    cx = center[0] + random.randint(-radius // 3, radius // 3)
    cy = center[1] + random.randint(-radius // 3, radius // 3)
    for _ in range(random.randint(30, 80)):
        dx = random.randint(-25, 25)
        dy = random.randint(-25, 25)
        px, py = cx + dx, cy + dy
        dist = np.sqrt((px - center[0])**2 + (py - center[1])**2)
        if dist < radius:
            cv2.circle(img, (px, py), random.randint(2, 5), 255, -1)

def _pattern_full_fail(img, center, radius):
    """Nearly entire wafer is defective."""
    cv2.circle(img, center, radius - 5, 255, -1)


# Map class name → drawing function
_PATTERN_FN = {
    "normal":    _pattern_normal,
    "center":    _pattern_center,
    "edge_ring": _pattern_edge_ring,
    "edge_loss": _pattern_edge_loss,
    "scratch":   _pattern_scratch,
    "ring":      _pattern_ring,
    "cluster":   _pattern_cluster,
    "full_fail": _pattern_full_fail,
}


# ── Augmentation (mild only) ───────────────────────────────────────

def _augment(img: np.ndarray) -> np.ndarray:
    """
    Mild augmentation: random rotation, flip, brightness jitter.
    NO elastic distortion, NO heavy blur (per spec).
    """
    # Random 90° rotation
    k = random.randint(0, 3)
    img = np.rot90(img, k)

    # Random flip
    if random.random() > 0.5:
        img = np.fliplr(img)
    if random.random() > 0.5:
        img = np.flipud(img)

    # Brightness jitter (±10%)
    scale = random.uniform(0.9, 1.1)
    img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    # Very light Gaussian noise
    noise = np.random.normal(0, 3, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(img)


# ── Public API ──────────────────────────────────────────────────────

def generate_wafer_map(defect_class: str, augment: bool = True) -> np.ndarray:
    """
    Generate a single 224×224 RGB wafer map image.

    Args:
        defect_class: one of DEFECT_CLASSES
        augment: whether to apply mild augmentations

    Returns:
        np.ndarray of shape (224, 224, 3), dtype uint8
    """
    img, center, radius = _draw_wafer_base()
    _PATTERN_FN[defect_class](img, center, radius)

    if augment:
        img = _augment(img)

    # Convert grayscale → RGB (ResNet expects 3 channels)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_rgb


def create_classification_dataset(
    n_per_class: int = 400,
    n_lots: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a balanced dataset with lot IDs for lot-based splitting.

    Returns:
        images:  (N, 224, 224, 3) uint8
        labels:  (N,) int   — class index
        lot_ids: (N,) int   — lot identifier for split
    """
    images, labels, lot_ids = [], [], []

    for class_idx, class_name in enumerate(DEFECT_CLASSES):
        for i in range(n_per_class):
            img = generate_wafer_map(class_name, augment=True)
            lot_id = i % n_lots  # Assign to one of n_lots lots
            images.append(img)
            labels.append(class_idx)
            lot_ids.append(lot_id)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    lot_ids = np.array(lot_ids, dtype=np.int64)

    return images, labels, lot_ids
