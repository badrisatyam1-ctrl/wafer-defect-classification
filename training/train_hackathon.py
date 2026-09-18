import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np

# Prevent OpenMP runtime crash on Windows/Anaconda when calculating Confusion Matrix
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ========================
# CONFIG
# ========================
DATA_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 5
SAVE_PATH = "models/best.pt"

# ========================
# TRANSFORMS
# ========================
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std + self.mean, 0.0, 1.0)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.3, 0.3),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.05),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========================
# DATA LOADERS & BALANCING
# ========================
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

from collections import Counter
import random

original_labels = [label for _, label in train_dataset.samples]
class_counts = Counter(original_labels)
total = sum(class_counts.values())
weights = [total / class_counts[i] for i in range(len(class_counts))]
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

max_samples = max(class_counts.values())
balanced_samples = []
for cls_idx in range(NUM_CLASSES):
    cls_samples = [s for s in train_dataset.samples if s[1] == cls_idx]
    balanced_samples.extend(cls_samples)
    shortfall = max_samples - len(cls_samples)
    if shortfall > 0:
        balanced_samples.extend(random.choices(cls_samples, k=shortfall))

train_dataset.samples = balanced_samples
if hasattr(train_dataset, "targets"):
    train_dataset.targets = [s[1] for s in train_dataset.samples]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

CLASS_NAMES = train_dataset.classes
print(f"Classes: {CLASS_NAMES}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ========================
# MODEL — ResNet18 (fast + accurate)
# ========================
model = models.resnet18(pretrained=True)

# Unfreeze the full backbone for maximum performance
for name, param in model.named_parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")

# ========================
# ========================
# We already balance the dataset via oversampling above, so we DO NOT pass class weights here.
# Double-compensating causes the model to overfit on the minority class ("Full" fail).
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ========================
# TRAINING LOOP — saves after EVERY epoch so the app can use it live
# ========================
best_acc = 0
patience_counter = 0
start_epoch = 0
os.makedirs("models", exist_ok=True)

if os.path.exists(SAVE_PATH):
    print(f"\n⚠️  Removing existing checkpoint at {SAVE_PATH} to train from scratch (ImageNet pretrained).")
    os.remove(SAVE_PATH)

print(f"\n{'Epoch':>6} | {'Loss':>10} | {'Val Acc':>9} | {'LR':>10} | {'Status':>12}")
print("-" * 60)

def safe_save(state_dict, path):
    tmp_path = path + "_tmp"
    torch.save(state_dict, tmp_path)
    try:
        if os.path.exists(path):
            try: os.remove(path)
            except: pass
        os.rename(tmp_path, path)
    except Exception:
        pass

for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0

    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Batch-wise live streaming to Streamlit!
        if (step + 1) % 10 == 0:
            checkpoint = {
                "schema_version": 2,
                "model_name": "resnet18",
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "split_mode": "lot",
                "preprocessing": {"input_size": 224},
            }
            safe_save(checkpoint, "models/latest.pt")

    # ========================
    # VALIDATION (softmax probabilities)
    # ========================
    model.eval()
    correct = 0
    total_val = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            total_val += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total_val
    current_lr = optimizer.param_groups[0]['lr']
    # ========================================================
    # SAVE AFTER EVERY EPOCH so the Streamlit app can use it
    # ========================================================
    checkpoint = {
        "schema_version": 2,
        "model_name": "resnet18",
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "split_mode": "lot",
        "preprocessing": {"input_size": 224},
    }
    safe_save(checkpoint, SAVE_PATH)

    status = ""
    if acc > best_acc:
        best_acc = acc
        patience_counter = 0
        status = "✅ best"
    else:
        patience_counter += 1
        status = f"⏳ {patience_counter}/{PATIENCE}"

    print(f"  {epoch+1:4d}  | {running_loss:10.4f} | {acc:8.4f} | {current_lr:10.6f} | {status:>12}")
    print(f"         📡 Model saved → {SAVE_PATH} (app can use it now)")

    scheduler.step()

    if patience_counter >= PATIENCE:
        print(f"\n⚠️ Early stopping at epoch {epoch+1}")
        break

print(f"\n🏁 Best Val Accuracy: {best_acc:.4f}")

# ========================
# CONFUSION MATRIX
# ========================
print("\n📊 Confusion Matrix:")
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for true, pred in zip(all_labels, all_preds):
    cm[true][pred] += 1

header = f"{'':>12}" + "".join(f"{name[:7]:>8}" for name in CLASS_NAMES)
print(header)
print("-" * len(header))
for i, name in enumerate(CLASS_NAMES):
    row = f"{name:>12}" + "".join(f"{cm[i][j]:>8}" for j in range(NUM_CLASSES))
    print(row)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=16)
    fig.colorbar(im)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    print("\n✅ Confusion matrix → models/confusion_matrix.png")
except ImportError:
    pass

print("\n🎯 Training Complete — App is ready!")
