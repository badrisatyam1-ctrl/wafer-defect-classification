"""
Production Retraining V2: Combined NPZ + Augmented Folder Dataset
Trains ResNet-18 on BOTH data distributions to handle all user inputs.

Data Sources:
  1. wm811k_dataset.npz — 7,897 real fab wafer maps (discrete {0,50,180} pixels)
  2. dataset/train/*/*.png — 10,000 augmented wafer maps (continuous 0-255 pixels)

This ensures the model generalizes to ALL wafer image types users may upload.
"""
import os
import sys
import time
import random
import glob
from pathlib import Path
from collections import Counter

# Ensure UTF-8 output so Windows charmap never crashes on prints
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
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


class CombinedWaferDataset(Dataset):
    """
    Combined dataset loading from both NPZ fab data and augmented folder images.
    Both are resized to 224x224 grayscale and augmented uniformly.
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray, is_train: bool = True):
        self.images = images  # (N, 224, 224) uint8
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy().astype(np.float32)
        label = int(self.labels[idx])

        if self.is_train:
            # Random rotation (0, 90, 180, 270 deg)
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy()

            # Random flips
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
            if random.random() > 0.5:
                img = np.flipud(img).copy()

            # Dual-polarity augmentation (contrast inversion)
            if random.random() > 0.5:
                mask = img > 15
                img[mask] = np.clip(230 - img[mask], 0, 255)

            # Random brightness/contrast jitter
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)  # contrast
                beta = random.uniform(-10, 10)       # brightness
                img = np.clip(img * alpha + beta, 0, 255)

            # Random Gaussian noise
            if random.random() > 0.6:
                noise = np.random.normal(0, random.uniform(2, 8), img.shape).astype(np.float32)
                img = np.clip(img + noise, 0, 255)

        # Normalize to 3-channel tensor in [-1, 1]
        img = np.ascontiguousarray(img).astype(np.float32)
        t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        t = (t - 127.5) / 127.5
        return t, label


def load_combined_dataset():
    """Load and combine both NPZ and augmented folder data."""
    all_images = []
    all_labels = []

    # Source 1: NPZ fab data
    npz_path = PROJECT_ROOT / "dataset" / "wm811k_dataset.npz"
    if npz_path.exists():
        print(f"[DATA] Loading NPZ fab dataset from {npz_path}...", flush=True)
        data = np.load(str(npz_path))
        npz_images = data["images"]  # (N, 224, 224) uint8
        npz_labels = data["labels"].astype(np.int64)
        all_images.append(npz_images)
        all_labels.append(npz_labels)
        print(f"[DATA] NPZ: {len(npz_images)} samples loaded", flush=True)

    # Source 2: Augmented folder images
    train_dir = PROJECT_ROOT / "dataset" / "train"
    if train_dir.exists():
        print(f"[DATA] Loading augmented folder dataset from {train_dir}...", flush=True)
        aug_images = []
        aug_labels = []
        for cls_idx, cls_name in enumerate(DEFECT_CLASSES_V2):
            cls_dir = train_dir / cls_name
            if not cls_dir.exists():
                continue
            files = sorted(glob.glob(str(cls_dir / "*.png")))
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Resize to 224x224 if needed
                if img.shape != (224, 224):
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                aug_images.append(img)
                aug_labels.append(cls_idx)
        
        if aug_images:
            aug_images = np.array(aug_images, dtype=np.uint8)
            aug_labels = np.array(aug_labels, dtype=np.int64)
            all_images.append(aug_images)
            all_labels.append(aug_labels)
            print(f"[DATA] Augmented folder: {len(aug_images)} samples loaded", flush=True)

    # Combine
    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"[DATA] Combined total: {len(images)} samples", flush=True)

    counts = Counter(labels.tolist())
    for idx, name in enumerate(DEFECT_CLASSES_V2):
        print(f"  {name:12s}: {counts[idx]} samples", flush=True)

    return images, labels


def train_combined():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Device: {device}", flush=True)

    images, labels = load_combined_dataset()

    # Stratified 85/15 train/val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss.split(images, labels))

    train_imgs, train_lbls = images[train_idx], labels[train_idx]
    val_imgs, val_lbls = images[val_idx], labels[val_idx]

    print(f"[SPLIT] Train: {len(train_imgs)}, Val: {len(val_imgs)}", flush=True)

    # Class-balanced sampler: 2000 samples per epoch for fast, responsive CPU training
    train_counts = Counter(train_lbls.tolist())
    class_weights = {c: 1.0 / count for c, count in train_counts.items()}
    sample_weights = np.array([class_weights[int(l)] for l in train_lbls], dtype=np.float64)
    n_samples_per_epoch = min(2000, len(train_imgs))
    sampler = WeightedRandomSampler(sample_weights, num_samples=n_samples_per_epoch, replacement=True)

    train_ds = CombinedWaferDataset(train_imgs, train_lbls, is_train=True)
    val_ds = CombinedWaferDataset(val_imgs, val_lbls, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    
    # Subsampled validation (600 samples) for fast epoch validation, full on final epoch
    val_subset_idx = np.random.RandomState(42).choice(len(val_imgs), min(600, len(val_imgs)), replace=False)
    val_quick_ds = Subset(val_ds, val_subset_idx)
    val_quick_loader = DataLoader(val_quick_ds, batch_size=64, shuffle=False, num_workers=0)
    val_full_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # Model: ResNet-18
    # Start from existing checkpoint if available (transfer learning from previous stage)
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_pt_path = ckpt_dir / "resnet18_best.pt"

    model = create_resnet18_classifier(num_classes=NUM_CLASSES, pretrained=True, dropout=0.3)
    
    if best_pt_path.exists():
        try:
            prev_ckpt = torch.load(str(best_pt_path), map_location=device, weights_only=False)
            if isinstance(prev_ckpt, dict) and "model_state_dict" in prev_ckpt:
                model.load_state_dict(prev_ckpt["model_state_dict"], strict=False)
                print(f"[MODEL] Warm-started from existing checkpoint at {best_pt_path}", flush=True)
        except Exception as e:
            print(f"[MODEL] Note: couldn't warm start ({e}), using standard pretrained weights", flush=True)

    # Fine-tune layer3, layer4 and fc
    for name, param in model.named_parameters():
        if "layer1" in name or "layer2" in name or "conv1" in name:
            param.requires_grad = False

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] ResNet-18, trainable params: {trainable:,}", flush=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4, weight_decay=1e-4
    )
    n_epochs = 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1 = 0.0
    best_weights = None

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_imgs, batch_lbls in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_lbls = batch_lbls.to(device)

            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = criterion(logits, batch_lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate (quick loader for epochs 1..n_epochs-1, full for final)
        active_val_loader = val_full_loader if epoch == n_epochs else val_quick_loader
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch_imgs, batch_lbls in active_val_loader:
                batch_imgs = batch_imgs.to(device)
                logits = model(batch_imgs)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(batch_lbls.numpy())

        val_acc = accuracy_score(all_true, all_preds)
        val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        improved = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " [BEST]"

        print(
            f"[Epoch {epoch:2d}/{n_epochs}] "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f} | "
            f"LR: {lr_now:.6f} | "
            f"Time: {elapsed:.1f}s{improved}",
            flush=True
        )

    # Save best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Exact matching preprocessing configuration (224x224, 3 channels [-1, 1])
    prep_cfg = PreprocessingConfig(
        input_size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        pad_value=0,
    )
    checkpoint_payload = {
        "schema_version": 2,
        "model_name": "resnet18_combined_v2",
        "epoch": n_epochs,
        "model_state_dict": model.state_dict(),
        "best_macro_f1": best_f1,
        "best_val_metrics": {"macro_f1": best_f1},
        "class_names": DEFECT_CLASSES_V2,
        "preprocessing": {
            "input_size": 224,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "pad_value": 0,
        },
        "config": {
            "num_classes": NUM_CLASSES,
            "dropout": 0.3,
            "pretrained": True,
            "data_sources": ["wm811k_dataset.npz", "dataset/train/"],
        },
    }
    
    print(f"\n[SAVE] Saving best checkpoint (Macro-F1: {best_f1:.4f}) to {best_pt_path}...", flush=True)
    torch.save(checkpoint_payload, str(best_pt_path))
    models_best = PROJECT_ROOT / "models" / "best.pt"
    torch.save(checkpoint_payload, str(models_best))
    print(f"[SAVE] Also updated {models_best}", flush=True)

    # Verification
    print("\n" + "=" * 60, flush=True)
    print("VERIFICATION ON BOTH DATA SOURCES", flush=True)
    print("=" * 60, flush=True)

    model.eval()
    preprocessor_tensor = lambda img_gray: (
        torch.from_numpy(img_gray.astype(np.float32))
        .unsqueeze(0).repeat(3, 1, 1)
        .sub_(127.5).div_(127.5)
        .unsqueeze(0).to(device)
    )

    # Test on augmented folder images (5 random per class)
    print("\n--- Augmented Folder Images ---", flush=True)
    aug_correct = 0
    aug_total = 0
    train_dir = PROJECT_ROOT / "dataset" / "train"
    for cls_idx, cls_name in enumerate(DEFECT_CLASSES_V2):
        cls_dir = train_dir / cls_name
        if not cls_dir.exists():
            continue
        files = sorted(glob.glob(str(cls_dir / "*.png")))
        random.seed(42)
        test_files = random.sample(files, min(5, len(files)))
        for f in test_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img.shape != (224, 224):
                img = cv2.resize(img, (224, 224))
            inp = preprocessor_tensor(img)
            with torch.no_grad():
                probs = torch.softmax(model(inp), dim=1)[0]
                pred_idx = probs.argmax().item()
                pred_cls = DEFECT_CLASSES_V2[pred_idx]
                conf = probs[pred_idx].item()
            is_match = pred_cls == cls_name
            if is_match:
                aug_correct += 1
            aug_total += 1
            status = "PASS" if is_match else "FAIL"
            print(f"  [{status}] {os.path.basename(f):30s} | True: {cls_name:10s} | Pred: {pred_cls:10s} ({conf:.2%})", flush=True)
    if aug_total > 0:
        print(f"\nAugmented accuracy: {aug_correct}/{aug_total} ({aug_correct/aug_total:.1%})", flush=True)

    # Test on real demo images
    print("\n--- Real Demo Images ---", flush=True)
    demo_correct = 0
    demo_total = 0
    demo_files = sorted(glob.glob(str(PROJECT_ROOT / "real_demo_images" / "*.png")))
    for f in demo_files:
        fname = os.path.basename(f)
        parts = fname.replace("real_", "").rsplit("_", 1)[0]
        true_cls = parts
        
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img.shape != (224, 224):
            img = cv2.resize(img, (224, 224))
        inp = preprocessor_tensor(img)
        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1)[0]
            pred_idx = probs.argmax().item()
            pred_cls = DEFECT_CLASSES_V2[pred_idx]
            conf = probs[pred_idx].item()
        is_match = pred_cls == true_cls
        if is_match:
            demo_correct += 1
        demo_total += 1
        status = "PASS" if is_match else "FAIL"
        print(f"  [{status}] {fname:22s} | True: {true_cls:10s} | Pred: {pred_cls:10s} ({conf:.2%})", flush=True)
    
    if demo_total > 0:
        print(f"\nReal demo accuracy: {demo_correct}/{demo_total} ({demo_correct/demo_total:.1%})", flush=True)

    print(f"\n[DONE] Best Macro-F1: {best_f1:.4f}", flush=True)
    print("[DONE] Checkpoint saved successfully.", flush=True)


if __name__ == "__main__":
    train_combined()
