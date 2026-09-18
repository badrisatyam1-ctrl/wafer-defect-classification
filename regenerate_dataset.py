"""
Regenerate 10,000 diverse synthetic wafer images using the enhanced V3 generator
and pack them into wafer_dataset_10k.npz for training.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from utils.synthetic_generator import generate_macro_wafer_map, DEFECT_CLASSES_V2

SIZE = (224, 224)
SAMPLES_PER_CLASS = 625
TOTAL = SAMPLES_PER_CLASS * len(DEFECT_CLASSES_V2)  # 5,000

print(f"Generating {TOTAL} diverse synthetic wafer images at {SIZE}...")

images = []
labels = []
lot_ids = []

for cls_idx, cls_name in enumerate(DEFECT_CLASSES_V2):
    print(f"  {cls_name}: generating {SAMPLES_PER_CLASS} images...", end="", flush=True)
    lot_counter = 0
    generated = 0
    while generated < SAMPLES_PER_CLASS:
        lot_size = min(np.random.randint(5, 15), SAMPLES_PER_CLASS - generated)
        lot_id = f"LOT_{cls_name}_{lot_counter:04d}"
        for _ in range(lot_size):
            img = generate_macro_wafer_map(cls_name, size=SIZE)
            # Convert RGB to grayscale for consistent storage
            gray = img[:, :, 0]  # All channels are same since we cvtColor GRAY2RGB
            images.append(gray)
            labels.append(cls_idx)
            lot_ids.append(lot_id)
            generated += 1
        lot_counter += 1
    print(f" done ({generated} images)")

images = np.array(images, dtype=np.uint8)
labels = np.array(labels, dtype=np.int64)
lot_ids = np.array(lot_ids)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wafer_dataset_10k.npz")
print(f"Saving to {out_path}...")
np.savez_compressed(out_path, images=images, labels=labels, lot_ids=lot_ids)
print(f"Done. Shape: {images.shape}, Labels: {labels.shape}")
