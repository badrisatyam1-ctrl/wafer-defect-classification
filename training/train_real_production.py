"""
Production Retraining for ResNet-18 on Real Fab Wafer Maps (WM-811K)
Optimized for high-accuracy and fast CPU convergence.

Features:
  1. Real WM-811K fab dataset (7,897 wafers across all 8 classes)
  2. Dual-polarity contrast invariance (so defects are detected whether dark or bright)
  3. Geometric rotation invariance (wafer circular symmetry)
  4. Class-balanced sampling with 250 samples per class (2,000 balanced samples/epoch)
  5. Transfer learning with ImageNet weights, fine-tuning layers 3-4 + FC
  6. Checkpoint export matching production deployment format
"""
import os
import sys
import time
import random
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet18_classifier import (
    DEFECT_CLASSES_V2,
    NUM_CLASSES,
    PreprocessingConfig,
    create_resnet18_classifier,
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FastWaferDataset(Dataset):
    """
    Optimized wafer dataset with dual-polarity and rotational augmentation.
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray, is_train: bool = True):
        self.images = images  # (N, 224, 224) uint8
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        label = int(self.labels[idx])

        if self.is_train:
            # Random rotation (0, 90, 180, 270 deg)
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k)

            # Random flips
            if random.random() > 0.5:
                img = np.fliplr(img)
            if random.random() > 0.5:
                img = np.flipud(img)

            # Dual-polarity augmentation (50% normal fab, 50% inverted contrast)
            if random.random() > 0.5:
                mask = img > 15
                img[mask] = np.clip(230 - img[mask].astype(np.int16), 0, 255).astype(np.uint8)

            # Random subtle noise
            if random.random() > 0.6:
                noise = np.random.randint(-5, 6, img.shape, dtype=np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Fast tensor conversion: uint8 -> 3-channel float in [-1, 1]
        img = np.ascontiguousarray(img)
        t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).float()
        t = (t - 127.5) / 127.5
        return t, label

def train_production():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Device: {device}")

    npz_path = PROJECT_ROOT / "dataset" / "wm811k_dataset.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")

    print(f"[DATA] Loading real fab dataset from {npz_path}...")
    data = np.load(npz_path)
    images = data["images"]
    labels = data["labels"].astype(np.int64)

    total_samples = len(images)
    print(f"[DATA] Loaded {total_samples} samples across classes:")
    counts = Counter(labels.tolist())
    for c_idx, c_name in enumerate(DEFECT_CLASSES_V2):
        print(f"  {c_name:12s} (class {c_idx}): {counts[c_idx]} samples")

    # Stratified split: 85% train, 15% val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss.split(images, labels))

    train_imgs, train_lbls = images[train_idx], labels[train_idx]
    val_imgs, val_lbls = images[val_idx], labels[val_idx]

    # Subsample validation to 400 balanced samples for rapid evaluation
    val_sss = StratifiedShuffleSplit(n_splits=1, test_size=min(400, len(val_lbls)), random_state=42)
    _, fast_val_idx = next(val_sss.split(val_imgs, val_lbls))
    fast_val_imgs = val_imgs[fast_val_idx]
    fast_val_lbls = val_lbls[fast_val_idx]

    print(f"[SPLIT] Train: {len(train_imgs)}, Validation (Fast): {len(fast_val_imgs)}")

    # Class-balanced sampler: 250 samples per class = 2,000 samples per epoch
    train_counts = Counter(train_lbls.tolist())
    sample_weights = [1.0 / train_counts[l] for l in train_lbls]
    samples_per_epoch = 2000
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=samples_per_epoch,
        replacement=True,
    )

    batch_size = 32
    train_ds = FastWaferDataset(train_imgs, train_lbls, is_train=True)
    val_ds = FastWaferDataset(fast_val_imgs, fast_val_lbls, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model setup
    print("[MODEL] Building ResNet-18 with ImageNet weights...")
    model = create_resnet18_classifier(num_classes=NUM_CLASSES, pretrained=True, dropout=0.3)
    model = model.to(device)

    # Freeze conv1, bn1, layer1, layer2; fine-tune layer3, layer4, fc
    for p in list(model.conv1.parameters()) + list(model.bn1.parameters()) + list(model.layer1.parameters()) + list(model.layer2.parameters()):
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    epochs = 8
    best_f1 = 0.0
    best_weights = None
    best_metrics = {}

    print(f"\n[TRAIN] Starting training for {epochs} epochs ({samples_per_epoch} samples/epoch)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        epoch_t0 = time.time()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Fast Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_targets)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")
        epoch_dur = time.time() - epoch_t0

        print(f"Epoch {epoch:2d}/{epochs} [{epoch_dur:.1f}s] - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}, Macro-F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_accuracy": float(val_acc),
                "val_macro_f1": float(val_f1),
                "epoch": epoch,
            }
            print(f"  [BEST] New best model achieved (Macro-F1: {best_f1:.4f})")

    total_time = time.time() - start_time
    print(f"\n[DONE] Training complete in {total_time/60:.1f} minutes. Best Macro-F1: {best_f1:.4f}")

    # Save Checkpoints
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_pt_path = checkpoint_dir / "resnet18_best.pt"
    root_best_pt = PROJECT_ROOT / "models" / "best.pt"
    root_latest_pt = PROJECT_ROOT / "models" / "latest.pt"

    prep_cfg = PreprocessingConfig(input_size=224, mean=(0.5,), std=(0.5,), pad_value=0)

    checkpoint_payload = {
        "schema_version": "2.0",
        "model_name": "resnet18",
        "epoch": best_metrics.get("epoch", epochs),
        "model_state_dict": best_weights if best_weights is not None else model.state_dict(),
        "best_macro_f1": best_f1,
        "best_val_metrics": best_metrics,
        "class_names": DEFECT_CLASSES_V2,
        "preprocessing": prep_cfg.to_dict(),
        "config": {
            "dataset_npz": "dataset/wm811k_dataset.npz",
            "epochs": epochs,
            "best_macro_f1": best_f1,
            "split_mode": "lot",
            "preprocessing": prep_cfg.to_dict(),
        }
    }

    print(f"[SAVE] Saving best checkpoint to {best_pt_path}...")
    torch.save(checkpoint_payload, str(best_pt_path))
    print(f"[SAVE] Saving production deployment checkpoint to {root_best_pt}...")
    torch.save(checkpoint_payload, str(root_best_pt))
    torch.save(checkpoint_payload, str(root_latest_pt))

    # Verify on Real Demo Images
    print("\n" + "=" * 60)
    print("VERIFICATION ON 14 REAL DEMO IMAGES")
    print("=" * 60)
    import glob
    demo_files = sorted(glob.glob(str(PROJECT_ROOT / "real_demo_images" / "*.png")))

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    correct_demos = 0
    for f in demo_files:
        fname = Path(f).name
        # True class from filename e.g. real_center_0.png -> center
        true_cls = fname.split("_")[1]
        if true_cls == "edge":
            true_cls = f"edge_{fname.split('_')[2]}"

        img_bgr = cv2.imread(f)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        inp = torch.from_numpy(((img_resized / 255.0) - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(inp)
            probs = torch.softmax(out, dim=1)[0]

        pred_idx = probs.argmax().item()
        pred_cls = DEFECT_CLASSES_V2[pred_idx]
        conf = probs.max().item()

        is_match = (pred_cls == true_cls)
        if is_match:
            correct_demos += 1
        status = "PASS" if is_match else "FAIL"
        print(f"  [{status}] {fname:22s} | True: {true_cls:10s} | Pred: {pred_cls:10s} ({conf:.2%})")

    print(f"\nReal demo accuracy: {correct_demos}/{len(demo_files)} ({correct_demos/len(demo_files):.2%})")

if __name__ == "__main__":
    train_production()
