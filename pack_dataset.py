import os
import numpy as np
import cv2
from pathlib import Path

ROOT = Path('.').resolve()
DATA_DIR = ROOT / 'dataset' / 'train'
images = []
labels = []
lot_ids = []
classes = ["normal", "center", "edge_ring", "edge_loss", "scratch", "ring", "cluster", "full_fail"]

print(f"Packing images from {DATA_DIR}...")
for i, cls in enumerate(classes):
    cls_dir = DATA_DIR / cls
    if not cls_dir.exists():
        print(f"Warning: {cls_dir} does not exist.")
        continue
    
    count = 0
    for img_p in cls_dir.glob('*.png'):
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Enforce consistent resolution
        img = cv2.resize(img, (512, 512))
        
        # Use filename to guess lot (e.g. 'center_lot2_0001.png')
        parts = img_p.stem.split('_')
        lot = parts[-2] if len(parts) > 1 else 'lot1'
        
        images.append(img)
        labels.append(i)
        lot_ids.append(lot)
        count += 1
    print(f"  {cls}: {count} images")

print(f"Saving to {ROOT / 'wafer_dataset_10k.npz'}...")
np.savez_compressed(
    'wafer_dataset_10k.npz',
    images=np.array(images, dtype=np.uint8),
    labels=np.array(labels, dtype=np.int64),
    lot_ids=np.array(lot_ids)
)
print("Done.")
