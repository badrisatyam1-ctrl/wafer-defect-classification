"""
Training Script for U-Net Segmentation
Optimized for Dice Score (Research Standard).
"""
import tensorflow as tf
import numpy as np
import sys
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
