"""
Split dataset 80/20 with class balance, then fast-train on a random subset.

Step 1: Moves 20% of dataset/train/<class> → dataset/val/<class> (balanced)
Step 2: Trains pretrained ResNet18 (frozen backbone) on ~1500 random images for 5 epochs

Usage:
    python training/split_and_train.py
    python training/split_and_train.py --subset 1000 --epochs 5
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / "dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

CLASSES = ["scratch", "edge_loss", "cluster", "center", "ring", "normal", "full_fail"]
NUM_CLASSES = len(CLASSES)


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: BALANCED 80/20 SPLIT
# ═══════════════════════════════════════════════════════════════════════
def split_dataset(val_ratio: float = 0.2):
    """Move val_ratio of images from train/ to val/ per class."""
    train_root = DATASET_DIR / "train"
    val_root = DATASET_DIR / "val"

    if not train_root.exists():
        print("❌ dataset/train/ not found. Generate data first.")
        return

    total_moved = 0

    for cls in sorted(os.listdir(train_root)):
        cls_train_dir = train_root / cls
        cls_val_dir = val_root / cls

        if not cls_train_dir.is_dir():
            continue

        cls_val_dir.mkdir(parents=True, exist_ok=True)

        # Get all images in train
        images = sorted([f for f in cls_train_dir.iterdir() if f.suffix in {".png", ".jpg", ".jpeg"}])

        # Already split? Skip if val has images
        existing_val = list(cls_val_dir.iterdir())
        if len(existing_val) > 10:
            print(f"  {cls}: val/ already has {len(existing_val)} images, skipping")
            continue

        n_val = max(1, int(len(images) * val_ratio))

        # Shuffle and pick val images
        random.shuffle(images)
        val_images = images[:n_val]

        for img_path in val_images:
            dest = cls_val_dir / img_path.name
            shutil.move(str(img_path), str(dest))

        total_moved += n_val
        remaining = len(images) - n_val
        print(f"  {cls}: {remaining} train / {n_val} val")

    print(f"\n✅ Split complete! Moved {total_moved} images to val/")


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: FAST SUBSET TRAINING
# ═══════════════════════════════════════════════════════════════════════
class AddGaussianNoise:
    """Custom transform: inject Gaussian noise after ToTensor."""
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_dataloaders(batch_size: int, subset_size: int, image_size: int = 224):
    """Create balanced subset dataloaders with strong augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.05),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(str(DATASET_DIR / "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(DATASET_DIR / "val"), transform=val_transform)

    # Random subset from train (balanced across classes)
    if subset_size < len(train_dataset):
        per_class = subset_size // NUM_CLASSES
        subset_indices = []
        for cls_idx in range(NUM_CLASSES):
            cls_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if label == cls_idx]
            random.shuffle(cls_indices)
            subset_indices.extend(cls_indices[:per_class])
        random.shuffle(subset_indices)
        train_dataset = Subset(train_dataset, subset_indices)

    print(f"  Train subset: {len(train_dataset)} images")
    print(f"  Val set:      {len(val_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


def create_model(device: str):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train(epochs: int, batch_size: int, subset_size: int, lr: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Fast Subset Training")
    print(f"   Device: {device} | Epochs: {epochs} | Subset: {subset_size} | Batch: {batch_size}")

    train_loader, val_loader = get_dataloaders(batch_size, subset_size)
    model = create_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_pt_path = PROJECT_ROOT / "models" / "best.pt"
    checkpoint_path = CHECKPOINT_DIR / "resnet18_fast.pt"
    best_val_acc = 0.0

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save full checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": CLASSES,
                "config": {
                    "split_mode": "lot",
                    "loss_name": "CrossEntropyLoss",
                    "num_classes": NUM_CLASSES,
                    "backbone": "resnet18",
                    "frozen_backbone": True,
                },
                "preprocessing": {
                    "input_size": 224,
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                },
            }, checkpoint_path)
            # Also save lightweight state_dict to models/best.pt
            torch.save(model.state_dict(), best_pt_path)
            marker = " ✅ best"

        print(f"  {epoch:4d}  | {train_loss:10.4f} | {train_acc:8.2%} | {val_loss:8.4f} | {val_acc:6.2%} | {elapsed:5.1f}s{marker}")

    print(f"\n🏁 Done! Best val accuracy: {best_val_acc:.2%}")
    print(f"   Checkpoint: {checkpoint_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Split dataset + fast subset training.")
    parser.add_argument("--skip-split", action="store_true", help="Skip the 80/20 split step")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default: 0.2)")
    parser.add_argument("--subset", type=int, default=1500, help="Training subset size (default: 1500)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    # Step 1: Split
    if not args.skip_split:
        print("📂 Step 1: Splitting dataset (80/20 balanced)...")
        split_dataset(val_ratio=args.val_ratio)

    # Step 2: Train
    print("\n📊 Step 2: Fast subset training...")
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset_size=args.subset,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
