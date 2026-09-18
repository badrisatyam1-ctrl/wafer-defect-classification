import numpy as np
import cv2
import os

def save_samples(npz_path, out_dir, n_each=3):
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found")
        return
        
    data = np.load(npz_path)
    images = data['images']
    labels = data['labels']
    
    classes = [
        "normal", "center", "edge_ring", "edge_loss", 
        "scratch", "ring", "cluster", "full_fail"
    ]
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    for i in range(len(classes)):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            print(f"No samples for class {classes[i]}")
            continue
            
        selected = np.random.choice(idx, min(n_each, len(idx)), replace=False)
        for j, s_idx in enumerate(selected):
            img = images[s_idx]
            fname = f"{i}_{classes[i]}_{j}.png"
            cv2.imwrite(os.path.join(out_dir, fname), img)
            print(f"Saved {fname}")

if __name__ == "__main__":
    save_samples("wafer_dataset_10k.npz", "tmp/samples")
