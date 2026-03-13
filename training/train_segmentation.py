

"""
Training Script for U-Net Segmentation
Optimized for Dice Score (Research Standard).
"""
import tensorflow as tf
import numpy as np
import sysYou are a senior semiconductor yield analytics ML engineer.

Goal:
Upgrade an existing wafer defect detection model so it detects
MACRO-LEVEL wafer defects at industry standard accuracy.

Constraints:
- Input is a full-wafer image or wafer map (not cropped dies)
- Defects are global spatial patterns, not pixel noise
- Model must be robust to class imbalance and process drift
- Training and inference preprocessing MUST be identical

Defect classes:
[
 "normal",
 "center",
 "edge_ring",
 "edge_loss",
 "scratch",
 "ring",
 "cluster",
 "full_fail"
]

Tasks:
1. Replace any patch-based or random-crop logic with full-wafer input
2. Implement a ResNet18-based classifier with global context
3. Use class-weighted CrossEntropy or Focal Loss
4. Split train/val by lot or time (NOT random)
5. Add Grad-CAM explainability
6. Ensure inference pipeline exactly matches training preprocessing
7. Output:
   - predicted class
   - confidence score
   - heatmap overlay

Do NOT:
- Use elastic distortion
- Use heavy blur
- Use pixel-level segmentation

Deliverables:
- PyTorch training script
- Inference script for website API
- Clear comments explaining each design decision

from pathlib import Path
import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet_model import create_unet_model, dice_loss, dice_coefficient
from utils.synthetic_generator import create_segmentation_dataset

# Configuration
class SegmentationConfig:
    INPUT_SHAPE = (128, 128, 3)
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    MODEL_DIR = Path("models/checkpoints")
    LOG_DIR = Path("logs/segmentation")

if __name__ == "__main__":
    print("="*60)
    print("🚀 STARTING SEGMENTATION TRAINING (U-NET)")
    print("="*60)

    # 1. Generate Data (Image + Mask)
    print("\n🎨 Generating 3,000 Image-Mask pairs...")
    x, y = create_segmentation_dataset(n_samples=3000)
    
    print(f"  Images: {x.shape} (RGB)")
    print(f"  Masks:  {y.shape} (Binary)")
    
    # Split
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    
    # 2. Create U-Net
    print("\n🏗️ Building U-Net Architecture...")
    model = create_unet_model(SegmentationConfig.INPUT_SHAPE)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=SegmentationConfig.LEARNING_RATE),
        loss=dice_loss,
        metrics=[dice_coefficient, 'binary_accuracy']
    )
    
    # Callbacks
    SegmentationConfig.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(SegmentationConfig.MODEL_DIR / "unet_best.keras"),
        monitor='val_dice_coefficient',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(SegmentationConfig.LOG_DIR / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    
    # 3. Train
    print("\n🏋️ Training Loop...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=SegmentationConfig.EPOCHS,
        batch_size=SegmentationConfig.BATCH_SIZE,
        callbacks=[checkpoint, tensorboard]
    )
    
    print("\n✅ TRAINING COMPLETE")
    print(f"Best Dice Score: {max(history.history['val_dice_coefficient']):.4f}")
            