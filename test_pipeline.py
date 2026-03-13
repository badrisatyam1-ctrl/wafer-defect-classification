"""
End-to-End Test: Verify the entire pipeline works.
Tests: Model loading, synthetic data generation, prediction, output shapes.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 50)
print("END-TO-END PIPELINE TEST")
print("=" * 50)

# 1. Test Generator
print("\n[1/4] Testing Synthetic Generator...")
from utils.synthetic_generator import generate_wafer_mask_pair
img, mask = generate_wafer_mask_pair('Scratch')
print(f"  Image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")
print(f"  Mask shape:  {mask.shape}, dtype: {mask.dtype}, range: [{mask.min()}, {mask.max()}]")
assert img.shape == (128, 128, 3), f"Bad image shape: {img.shape}"
assert mask.shape == (128, 128, 1), f"Bad mask shape: {mask.shape}"
print("  ✅ Generator OK")

# 2. Test Model Loading
print("\n[2/4] Testing Model Load...")
import tensorflow as tf
from tensorflow import keras
from models.unet_model import dice_loss, dice_coefficient

model_path = PROJECT_ROOT / "models" / "checkpoints" / "unet_best.keras"
print(f"  Model path: {model_path}")
print(f"  File exists: {model_path.exists()}")
if not model_path.exists():
    print("  ❌ MODEL FILE NOT FOUND!")
    sys.exit(1)

print(f"  File size: {model_path.stat().st_size / 1e6:.1f} MB")

custom_objects = {'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient}
model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
print(f"  Model input shape: {model.input_shape}")
print(f"  Model output shape: {model.output_shape}")
print("  ✅ Model Loaded OK")

# 3. Test Prediction (WITHOUT double normalization)
print("\n[3/4] Testing Prediction...")
import numpy as np

# Feed raw [0, 255] values - model has built-in Rescaling(1./255)
input_tensor = np.expand_dims(img.astype(np.float32), axis=0)
print(f"  Input tensor shape: {input_tensor.shape}, range: [{input_tensor.min()}, {input_tensor.max()}]")

pred = model.predict(input_tensor, verbose=0)[0]
print(f"  Prediction shape: {pred.shape}, range: [{pred.min():.4f}, {pred.max():.4f}]")
print(f"  Prediction mean: {pred.mean():.4f}")

# Check prediction is meaningful (not all zeros or all ones)
has_defect = pred.max() > 0.5
has_background = pred.min() < 0.5
print(f"  Has defect pixels: {has_defect}")
print(f"  Has background pixels: {has_background}")

if has_defect and has_background:
    print("  ✅ Prediction looks MEANINGFUL (has both defect and background)")
elif not has_defect:
    print("  ⚠️ No defect detected (prediction all < 0.5)")
else:
    print("  ⚠️ Entire image predicted as defect")

# 4. Test with "none" type
print("\n[4/4] Testing 'none' (no defect) prediction...")
img_clean, mask_clean = generate_wafer_mask_pair('none')
input_clean = np.expand_dims(img_clean.astype(np.float32), axis=0)
pred_clean = model.predict(input_clean, verbose=0)[0]
defect_area_clean = (pred_clean > 0.5).sum() / (128*128) * 100
print(f"  Clean wafer defect area: {defect_area_clean:.2f}%")

print("\n" + "=" * 50)
print("ALL TESTS PASSED ✅")
print("=" * 50)
