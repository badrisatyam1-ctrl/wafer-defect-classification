"""
Extracts the WM-811K Pandas dataframe into a balanced .npz dataset
compatible with our training pipeline. Maps the raw 2D grid matrices
into 224x224 grayscale-equivalent images.
"""
import sys, os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.resnet18_classifier import DEFECT_CLASSES_V2

WMK_TO_V2_MAP = {
    'none': 'normal',
    'Center': 'center',
    'Edge-Ring': 'edge_ring',
    'Edge-Loc': 'edge_loss',
    'Scratch': 'scratch',
    'Donut': 'ring',
    'Loc': 'cluster',
    'Near-full': 'full_fail'
}

# The target classes in exactly the order of training
CLASSES = DEFECT_CLASSES_V2
SAMPLES_PER_CLASS = 1200 # Target ~9600 total samples

def extract_wm811k(pkl_path="dataset/wm811k/LSWMD.pkl", out_path="dataset/wm811k_dataset.npz"):
    print(f"Loading {pkl_path}...")
    df = pd.read_pickle(pkl_path)
    
    # Extract clean string labels
    df['label'] = df['failureType'].apply(
        lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 and isinstance(x[0][0], str) else 'unknown'
    )
    
    # Map to our V2 classes
    df['v2_class'] = df['label'].map(WMK_TO_V2_MAP)
    
    # Filter out unknowns and Randoms
    df = df.dropna(subset=['v2_class'])
    
    print("\nAvailable valid samples per mapped class:")
    print(df['v2_class'].value_counts())
    
    images = []
    labels = []
    lot_ids = []
    timestamps = []
    
    # Stratified sampling
    for cls_name in CLASSES:
        subset = df[df['v2_class'] == cls_name]
        n_samples = min(len(subset), SAMPLES_PER_CLASS)
        
        if n_samples == 0:
            print(f"WARNING: No samples found for {cls_name}")
            continue
            
        sampled = subset.sample(n=n_samples, random_state=42)
        
        print(f"Extracting {n_samples} samples for {cls_name}...")
        for _, row in sampled.iterrows():
            wmap = row['waferMap'] # 2D array: 0=bg, 1=normal, 2=defect
            
            # Convert to pseudo-realistic grayscale visual format
            # Background = 0 (Black)
            # Normal Die = 180 (Light Grey)
            # Defect Die = 50 (Dark Grey/Blackish)
            vis = np.zeros_like(wmap, dtype=np.uint8)
            vis[wmap == 1] = 180
            vis[wmap == 2] = 50
            
            # Resize to 224x224 (Nearest neighbor preserves sharp die edges)
            vis_resized = cv2.resize(vis, (224, 224), interpolation=cv2.INTER_NEAREST)
            
            images.append(vis_resized)
            labels.append(CLASSES.index(cls_name))
            
            # Use lotName or fake a unique ID
            lot_id = row['lotName'] if 'lotName' in row else f"LOT_{len(images)}"
            lot_ids.append(lot_id)
            
            # Fake a timestamp incrementing
            timestamps.append(len(images))

    # Convert to expected np.ndarray
    images_arr = np.array(images, dtype=np.uint8)
    # Expand dims to (N, H, W, 1) or keep (N,H,W) depending on training expectations
    # train_resnet.py expects raw dataset. Usually handles grayscale correctly if WaferDatasetBundle supports it.
    
    labels_arr = np.array(labels, dtype=np.int64)
    lot_ids_arr = np.array(lot_ids, dtype=str)
    timestamps_arr = np.array(timestamps, dtype=np.int64)
    
    print(f"\nExtracted shape: {images_arr.shape}")
    print(f"Label range: {labels_arr.min()} to {labels_arr.max()}")
    
    print(f"Saving to {out_path}...")
    np.savez_compressed(
        out_path,
        images=images_arr,
        labels=labels_arr,
        lot_ids=lot_ids_arr,
        timestamps=timestamps_arr
    )
    print("Done! Dataset ready for train_resnet.py")

if __name__ == "__main__":
    extract_wm811k()
