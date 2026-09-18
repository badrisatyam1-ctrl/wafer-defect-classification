import random
import numpy as np
import cv2

DEFECT_CLASSES_V2 = [
    "normal",       # 0
    "center",       # 1
    "edge_ring",    # 2
    "edge_loss",    # 3
    "scratch",      # 4
    "ring",         # 5
    "cluster",      # 6
    "full_fail",    # 7
]

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = PROJECT_ROOT / "dataset" / "wm811k_dataset.npz"

import json

EXEMPLARS_PATH = PROJECT_ROOT / "utils" / "exemplars.json"

_DATASET_CACHE = None

def _get_dataset_cache():
    global _DATASET_CACHE
    if _DATASET_CACHE is None and NPZ_PATH.exists():
        try:
            data = np.load(str(NPZ_PATH))
            images = data["images"]
            labels = data["labels"]
            class_map = {}
            if EXEMPLARS_PATH.exists():
                try:
                    with open(EXEMPLARS_PATH, "r") as f:
                        class_map = json.load(f)
                except Exception:
                    pass
            if not class_map:
                for idx, name in enumerate(DEFECT_CLASSES_V2):
                    matches = np.where(labels == idx)[0]
                    if len(matches) > 0:
                        class_map[name] = matches.tolist()
            _DATASET_CACHE = (images, class_map)
        except Exception as e:
            print(f"Warning: failed to load {NPZ_PATH}: {e}")
            _DATASET_CACHE = False
    return _DATASET_CACHE

def generate_macro_wafer_map(defect_type: str, size: tuple = (224, 224)) -> np.ndarray:
    """
    Generate or sample a full-wafer image for macro-level classification.
    Uses authentic WM-811K fab wafer maps when available with rotational/flip
    diversification, providing 100% realistic wafer maps that match the neural model.
    """
    defect_type = defect_type.lower().strip()
    cache = _get_dataset_cache()
    if cache and defect_type in cache[1] and len(cache[1][defect_type]) > 0:
        images, class_map = cache
        chosen_idx = random.choice(class_map[defect_type])
        img = images[chosen_idx].copy()

        # Diversify with random 90-degree rotations and flips
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k)
        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)

        if img.shape[:2] != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w = size
    center = (w // 2, h // 2)
    radius = min(h, w) // 2 - random.randint(5, 12)
    
    # ── BASE BACKGROUND ─────────────────────────────────────────────
    # Randomize the base intensity (dark fab background)
    base_val = random.randint(30, 80)
    img = np.full((h, w), base_val, dtype=np.float32)
    
    # Add a random directional gradient (subtle uneven illumination)
    if random.random() > 0.4:
        grad_dir = random.choice(["horizontal", "vertical", "radial"])
        grad_strength = random.uniform(5, 20)
        if grad_dir == "horizontal":
            gradient = np.linspace(-grad_strength, grad_strength, w).reshape(1, w)
            img += gradient
        elif grad_dir == "vertical":
            gradient = np.linspace(-grad_strength, grad_strength, h).reshape(h, 1)
            img += gradient
        else:
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            img += (dist / (dist.max() + 1e-6)) * grad_strength
            
    # Wafer disc itself
    wafer_brightness = random.randint(70, 120)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Fill wafer area
    img_wafer = np.full((h, w), wafer_brightness, dtype=np.float32)
    img = np.where(mask > 0, img_wafer + (img - base_val), img)
    
    # Subtle per-die "texture" (sparse dots)
    n_dots = random.randint(20, 100)
    for _ in range(n_dots):
        angle = random.uniform(0, 2 * np.pi)
        r = random.uniform(0, radius - 5)
        px = int(center[0] + r * np.cos(angle))
        py = int(center[1] + r * np.sin(angle))
        dot_v = wafer_brightness + random.randint(10, 40)
        cv2.circle(img, (px, py), random.randint(1, 2), dot_v, -1)

    # ── DEFECT PATTERNS ─────────────────────────────────────────────
    defect_bright = random.randint(180, 255)
    defect_dim = random.randint(140, 200)
    safe_min = min(defect_dim, defect_bright)
    safe_max = max(defect_dim, defect_bright)

    if defect_type == "normal":
        pass
    elif defect_type == "center":
        c_rad = random.randint(radius // 8, radius // 2)
        variant = random.choice(["dense", "gradient", "solid"])
        if variant == "dense":
            for _ in range(random.randint(50, 200)):
                r = random.uniform(0, c_rad)
                a = random.uniform(0, 2*np.pi)
                cv2.circle(img, (int(center[0]+r*np.cos(a)), int(center[1]+r*np.sin(a))), random.randint(1, 4), random.randint(safe_min, safe_max), -1)
        elif variant == "gradient":
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            g_mask = np.clip(1.0 - dist / c_rad, 0, 1) * (defect_bright - wafer_brightness)
            img += g_mask
        else:
            cv2.circle(img, center, c_rad, defect_bright, -1)

    elif defect_type == "edge_ring":
        thickness = random.randint(5, 20)
        inner = radius - random.randint(10, 30)
        cv2.circle(img, center, inner, defect_bright, thickness)

    elif defect_type == "edge_loss":
        # Solid arc/sector at edge
        start_a = random.randint(0, 360)
        span = random.randint(30, 120)
        cv2.ellipse(img, center, (radius, radius), 0, start_a, start_a+span, defect_bright, -1)

    elif defect_type == "scratch":
        # Jagged or curved line
        pts = []
        n_pts = random.randint(3, 8)
        curr_x, curr_y = random.randint(center[0]-radius, center[0]+radius), random.randint(center[1]-radius, center[1]+radius)
        for _ in range(n_pts):
            pts.append([curr_x, curr_y])
            curr_x += random.randint(-40, 40)
            curr_y += random.randint(-40, 40)
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], False, defect_bright, random.randint(2, 5))

    elif defect_type == "ring":
        # Concentric rings
        r1 = random.randint(radius // 4, radius // 2)
        cv2.circle(img, center, r1, defect_bright, random.randint(3, 10))
        if random.random() > 0.5:
            cv2.circle(img, center, r1 + random.randint(20, 50), defect_dim, random.randint(2, 8))

    elif defect_type == "cluster":
        # 1-3 clusters of dots
        for _ in range(random.randint(1, 3)):
            cx, cy = int(center[0] + random.uniform(-0.6, 0.6)*radius), int(center[1] + random.uniform(-0.6, 0.6)*radius)
            for _ in range(random.randint(30, 100)):
                dx, dy = int(random.gauss(0, 15)), int(random.gauss(0, 15))
                cv2.circle(img, (cx+dx, cy+dy), random.randint(1, 3), random.randint(safe_min, safe_max), -1)

    elif defect_type == "full_fail":
        # Heavy coverage
        n_dots = random.randint(400, 1000)
        for _ in range(n_dots):
            r = random.uniform(0, radius)
            a = random.uniform(0, 2*np.pi)
            cv2.circle(img, (int(center[0]+r*np.cos(a)), int(center[1]+r*np.sin(a))), random.randint(1, 5), random.randint(safe_min, safe_max), -1)

    # ── POST-PROCESSING ─────────────────────────────────────────────
    # Gaussian noise (subtle)
    noise = np.random.normal(0, random.uniform(2, 6), (h, w)).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Final Wafer Mask (clean edges)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(final_mask, center, radius, 255, -1)
    img = cv2.bitwise_and(img, img, mask=final_mask)
    
    # Random Rotation
    if random.random() > 0.2:
        angle = random.uniform(0, 360)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=0)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def create_classification_dataset_v2(n_samples=3000, size=(224, 224), imbalance=True):
    images, labels, lot_ids = [], [], []
    
    # Balanced distribution
    samples_per_class = max(1, n_samples // len(DEFECT_CLASSES_V2))
    
    lot_counter = 0
    samples_in_current_lot = 0
    current_lot_id = f"LOT_{lot_counter:04d}"
    
    for class_idx, defect_type in enumerate(DEFECT_CLASSES_V2):
        for _ in range(samples_per_class):
            img = generate_macro_wafer_map(defect_type, size)
            
            # Use only one channel since it's a grayscale-like mask
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            images.append(img)
            labels.append(class_idx)
            lot_ids.append(current_lot_id)
            
            samples_in_current_lot += 1
            if samples_in_current_lot >= 25:  # Standard wafer lot size
                lot_counter += 1
                current_lot_id = f"LOT_{lot_counter:04d}"
                samples_in_current_lot = 0
                
    return np.array(images), np.array(labels), np.array(lot_ids)
