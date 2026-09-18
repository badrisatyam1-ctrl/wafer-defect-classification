"""
Fast SIH training script — trains ResNet18 directly from wafer_dataset_10k.npz
- Stratified 80/20 split (no leakage)
- WeightedRandomSampler (handles imbalance properly)
- Saves best.pt in the format the Streamlit app expects
- Expected time: ~5-10 min on CPU, ~2 min on GPU
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ─────────────────────────── CONFIG ─────────────────────────────────────────
NPZ_PATH = "wafer_dataset_10k.npz"
SAVE_PATH = "models/best.pt"
BATCH_SIZE = 64
EPOCHS = 15
LR = 3e-4
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["center", "cluster", "edge_loss", "edge_ring",
               "full_fail", "normal", "ring", "scratch"]
NUM_CLASSES = len(CLASS_NAMES)

print(f"Device: {DEVICE}  |  Classes: {NUM_CLASSES}  |  Epochs: {EPOCHS}")

# ─────────────────────────── LOAD DATA ──────────────────────────────────────
print("\nLoading NPZ dataset...")
data = np.load(NPZ_PATH)
images = data["images"]  # (N, 224, 224) uint8 grayscale
labels = data["labels"]  # (N,) int

print(f"Total samples: {len(images)} | Label dist: {Counter(labels.tolist())}")

# ─────────────────────────── STRATIFIED SPLIT ───────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx = next(sss.split(images, labels))
print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

# ─────────────────────────── DATASET ────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class WaferNPZDataset(Dataset):
    def __init__(self, imgs, lbls, transform):
        self.imgs = imgs
        self.lbls = lbls
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        # Grayscale → RGB by stacking
        img_rgb = np.stack([img, img, img], axis=-1)  # (H, W, 3)
        img_tensor = self.transform(img_rgb)
        return img_tensor, int(self.lbls[idx])

train_ds = WaferNPZDataset(images[train_idx], labels[train_idx], train_tf)
val_ds   = WaferNPZDataset(images[val_idx],   labels[val_idx],   val_tf)

# Weighted sampler for balanced training
train_labels = labels[train_idx]
class_counts = Counter(train_labels.tolist())
sample_weights = [1.0 / class_counts[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,     num_workers=0)

# ─────────────────────────── MODEL ──────────────────────────────────────────
print("\nBuilding ResNet18...")
model = models.resnet18(weights="IMAGENET1K_V1")  # Transfer learning
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Phase 1: Only train the final classifier (fast convergence)
for name, param in model.named_parameters():
    param.requires_grad = ("fc" in name or "layer4" in name)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# ─────────────────────────── TRAINING ────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

os.makedirs("models", exist_ok=True)
best_f1 = 0.0
patience_counter = 0
best_epoch = 0

print(f"\n{'Ep':>3} | {'Loss':>8} | {'ValAcc':>7} | {'MacroF1':>8} | {'LR':>8} | {'Status':>12}")
print("-" * 65)

def safe_save(state, path):
    tmp = path + "_tmp"
    torch.save(state, tmp)
    if os.path.exists(path):
        try: os.remove(path)
        except: pass
    os.rename(tmp, path)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # After epoch 5, unfreeze all layers for fine-tuning
    if epoch == 5:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch)
        print("   [Unfroze all layers for fine-tuning]")

    # Validation
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            pred = out.argmax(1).cpu().numpy()
            all_p.extend(pred)
            all_l.extend(lbls.numpy())

    val_acc = accuracy_score(all_l, all_p)
    macro_f1 = f1_score(all_l, all_p, average="macro")
    current_lr = optimizer.param_groups[0]["lr"]
    elapsed = time.time() - t0

    # Save best
    checkpoint = {
        "schema_version": 2,
        "model_name": "resnet18",
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "split_mode": "stratified",
        "preprocessing": {"input_size": 224},
        "best_epoch": epoch,
        "val_accuracy": val_acc,
        "macro_f1": macro_f1,
    }
    safe_save(checkpoint, "models/latest.pt")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_epoch = epoch
        patience_counter = 0
        safe_save(checkpoint, SAVE_PATH)
        status = f"BEST F1={best_f1:.4f}"
    else:
        patience_counter += 1
        status = f"no imp {patience_counter}/{PATIENCE}"

    print(f"{epoch:3d} | {running_loss:8.4f} | {val_acc:7.4f} | {macro_f1:8.4f} | {current_lr:8.6f} | {status}")

    scheduler.step()

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# ─────────────────────────── FINAL RESULTS ───────────────────────────────────
print(f"\n{'='*65}")
print(f"TRAINING COMPLETE")
print(f"  Best Epoch  : {best_epoch}")
print(f"  Best Macro F1: {best_f1:.4f}  ({best_f1*100:.2f}%)")
print(f"  Model saved to: {SAVE_PATH}")
print(f"{'='*65}")

# Per-class F1 on best model
ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
all_p, all_l = [], []
with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs = imgs.to(DEVICE)
        all_p.extend(model(imgs).argmax(1).cpu().numpy())
        all_l.extend(lbls.numpy())

f1_per_class = f1_score(all_l, all_p, average=None)
print("\nPer-class F1 (best model):")
for cls, f in zip(CLASS_NAMES, f1_per_class):
    print(f"  {cls:12s}: {f:.4f}")

final_acc = accuracy_score(all_l, all_p)
final_f1  = f1_score(all_l, all_p, average="macro")
print(f"\nFinal Val Accuracy : {final_acc:.4f} ({final_acc*100:.2f}%)")
print(f"Final Macro F1     : {final_f1:.4f} ({final_f1*100:.2f}%)")
