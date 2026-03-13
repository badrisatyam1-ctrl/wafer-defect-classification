"""
Training Script — ResNet18 Wafer Defect Classifier
====================================================
Key design decisions:
  1. Lot-based train/val split (NOT random) — prevents data leakage
     because wafers from the same lot share process conditions.
  2. Focal Loss with per-class alpha weights computed from training
     distribution — handles class imbalance without oversampling.
  3. Cosine annealing LR — smooth convergence, avoids LR cliffs.
  4. Preprocessing identical to inference: resize 224, ImageNet normalize.
"""

import sys
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.resnet18_model import (
    WaferResNet18, FocalLoss,
    IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
)
from utils.wafer_map_generator import (
    create_classification_dataset, DEFECT_CLASSES, NUM_CLASSES
)

# ── Configuration ───────────────────────────────────────────────────

class Config:
    BATCH_SIZE   = 32
    EPOCHS       = 20
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    N_PER_CLASS  = 500       # samples per class
    N_LOTS       = 20        # number of distinct lots
    VAL_LOTS     = {16, 17, 18, 19}  # last 4 lots held out for validation
    SAVE_DIR     = Path("models/checkpoints")
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ── Preprocessing (MUST match inference_api.py) ─────────────────────

# This is the SINGLE SOURCE OF TRUTH for preprocessing.
# Both training and inference use uint8 images → float32 → normalize.
def preprocess_batch(images_uint8: np.ndarray) -> torch.Tensor:
    """
    Convert a batch of uint8 images (N, 224, 224, 3) → normalized tensor.
    This function is used IDENTICALLY in inference_api.py.
    """
    # uint8 [0,255] → float32 [0,1]
    x = images_uint8.astype(np.float32) / 255.0
    # HWC → CHW
    x = np.transpose(x, (0, 3, 1, 2))
    # ImageNet normalize
    mean = np.array(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(1, 3, 1, 1)
    x = (x - mean) / std
    return torch.from_numpy(x).float()


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 RESNET18 WAFER DEFECT CLASSIFIER — TRAINING")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")

    # ── 1. Generate Dataset ─────────────────────────────────────────
    print(f"\n📊 Generating {Config.N_PER_CLASS * NUM_CLASSES} samples...")
    images, labels, lot_ids = create_classification_dataset(
        n_per_class=Config.N_PER_CLASS,
        n_lots=Config.N_LOTS
    )
    print(f"   Images: {images.shape}  Labels: {labels.shape}  Lots: {lot_ids.shape}")

    # ── 2. Lot-Based Split (NO random shuffle) ─────────────────────
    # Wafers in the same lot share fab conditions → must stay together
    val_mask   = np.isin(lot_ids, list(Config.VAL_LOTS))
    train_mask = ~val_mask

    x_train, y_train = images[train_mask], labels[train_mask]
    x_val,   y_val   = images[val_mask],   labels[val_mask]
    print(f"   Train: {len(x_train)}  Val: {len(x_val)}  (split by lot)")

    # ── 3. Preprocess ───────────────────────────────────────────────
    x_train_t = preprocess_batch(x_train)
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t   = preprocess_batch(x_val)
    y_val_t   = torch.from_numpy(y_val).long()

    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds   = TensorDataset(x_val_t, y_val_t)
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False)

    # ── 4. Compute Class Weights ────────────────────────────────────
    # Alpha for Focal Loss: inverse frequency → rare classes get higher weight
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # normalize
    alpha = torch.from_numpy(class_weights).float().to(Config.DEVICE)
    print(f"\n⚖️  Class weights (Focal Loss alpha):")
    for i, name in enumerate(DEFECT_CLASSES):
        print(f"   {name:12s}: {alpha[i]:.3f}  (n={int(class_counts[i])})")

    # ── 5. Model, Loss, Optimizer ───────────────────────────────────
    model = WaferResNet18(num_classes=NUM_CLASSES, pretrained=True).to(Config.DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=Config.LR,
                                   weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    print(f"\n🏗️  Model: ResNet18 ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"   Loss:  Focal Loss (γ=2.0)")
    print(f"   Optimizer: AdamW (lr={Config.LR})")

    # ── 6. Training Loop ────────────────────────────────────────────
    Config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    print(f"\n🏋️  Training for {Config.EPOCHS} epochs...")
    for epoch in range(1, Config.EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        for xb, yb in train_dl:
            xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        train_acc = train_correct / train_total
        val_acc   = val_correct / val_total
        elapsed   = time.time() - t0

        print(f"  Epoch {epoch:2d}/{Config.EPOCHS}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"train_loss={train_loss/train_total:.4f}  "
              f"val_loss={val_loss/val_total:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}  ({elapsed:.1f}s)")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Config.SAVE_DIR / "resnet18_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': DEFECT_CLASSES,
            }, str(save_path))
            print(f"   ✅ Saved best model (val_acc={val_acc:.4f})")

    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE — Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Model saved to: {Config.SAVE_DIR / 'resnet18_best.pth'}")
    print(f"{'='*60}")
