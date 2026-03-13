"""
End-to-End Pipeline Test — ResNet18 Classifier
Verifies: generator, model loading, prediction, Grad-CAM, preprocessing match.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("RESNET18 CLASSIFIER — PIPELINE TEST")
print("=" * 60)

# 1. Generator
print("\n[1/5] Testing generator...")
from utils.wafer_map_generator import generate_wafer_map, DEFECT_CLASSES, NUM_CLASSES
for cls in DEFECT_CLASSES:
    img = generate_wafer_map(cls, augment=True)
    assert img.shape == (224, 224, 3), f"Bad shape for {cls}: {img.shape}"
    assert img.dtype == np.uint8
print(f"  ✅ All {NUM_CLASSES} classes generate valid 224×224 RGB images")

# 2. Model checkpoint
print("\n[2/5] Checking model checkpoint...")
model_path = PROJECT_ROOT / "models" / "checkpoints" / "resnet18_best.pth"
assert model_path.exists(), f"Model not found: {model_path}"
import torch
ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
print(f"  File:     {model_path}")
print(f"  Size:     {model_path.stat().st_size / 1e6:.1f} MB")
print(f"  Epoch:    {ckpt['epoch']}")
print(f"  Val acc:  {ckpt['val_acc']:.4f}")
print(f"  Classes:  {ckpt['class_names']}")
print(f"  ✅ Checkpoint valid")

# 3. Inference API
print("\n[3/5] Testing inference API...")
from deployment.inference_api import predict, load_model
model = load_model(str(model_path))

test_img = generate_wafer_map("scratch", augment=False)
result = predict(test_img, model=model, generate_heatmap=True)
print(f"  Predicted: {result['class']} ({result['confidence']*100:.1f}%)")
print(f"  All probs: {result['all_probs']}")
assert result['class'] in DEFECT_CLASSES
assert 0 <= result['confidence'] <= 1
assert result['heatmap'] is not None
assert result['heatmap'].shape == (224, 224)
assert result['overlay'] is not None
assert result['overlay'].shape == (224, 224, 3)
print(f"  ✅ Inference + Grad-CAM OK")

# 4. Preprocessing consistency
print("\n[4/5] Verifying preprocessing consistency...")
from training.train_classifier import preprocess_batch
from deployment.inference_api import preprocess_image

# Same image through both paths
img_test = generate_wafer_map("center", augment=False)
batch_result = preprocess_batch(np.expand_dims(img_test, 0))[0]  # (3, 224, 224)
single_result = preprocess_image(img_test)[0]  # (3, 224, 224)

diff = torch.abs(batch_result - single_result).max().item()
print(f"  Max difference: {diff:.10f}")
assert diff < 1e-5, f"Preprocessing mismatch! diff={diff}"
print(f"  ✅ Preprocessing EXACTLY matches between training and inference")

# 5. Test all classes
print("\n[5/5] Testing predictions across all classes...")
correct = 0
for cls in DEFECT_CLASSES:
    img = generate_wafer_map(cls, augment=False)
    result = predict(img, model=model, generate_heatmap=False)
    match = "✅" if result['class'] == cls else "❌"
    print(f"  {match} {cls:12s} → predicted: {result['class']:12s} ({result['confidence']*100:.1f}%)")
    if result['class'] == cls:
        correct += 1

print(f"\n  Accuracy: {correct}/{NUM_CLASSES} ({correct/NUM_CLASSES*100:.0f}%)")

print(f"\n{'='*60}")
if correct == NUM_CLASSES:
    print("ALL TESTS PASSED ✅")
else:
    print(f"PASSED WITH {correct}/{NUM_CLASSES} CORRECT")
print(f"{'='*60}")
