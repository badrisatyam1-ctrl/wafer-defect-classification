"""
Training Script for U-Net Segmentation.
Optimized for Dice score.
"""
import datetime
import os
import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))

from models.unet_model import create_unet_model, dice_coefficient, dice_loss
from utils.synthetic_generator import create_segmentation_dataset


class SegmentationConfig:
    INPUT_SHAPE = (128, 128, 3)
    BATCH_SIZE = int(os.getenv("SEG_BATCH_SIZE", "32"))
    EPOCHS = int(os.getenv("SEG_EPOCHS", "15"))
    N_SAMPLES = int(os.getenv("SEG_SAMPLES", "3000"))
    LEARNING_RATE = float(os.getenv("SEG_LR", "1e-3"))
    MODEL_DIR = PROJECT_ROOT / "models" / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "logs" / "segmentation"


if __name__ == "__main__":
    print("=" * 60)
    print("STARTING SEGMENTATION TRAINING (U-NET)")
    print("=" * 60)

    print(f"\nGenerating {SegmentationConfig.N_SAMPLES:,} image-mask pairs...")
    x, y = create_segmentation_dataset(n_samples=SegmentationConfig.N_SAMPLES)

    print(f"  Images: {x.shape} (RGB)")
    print(f"  Masks:  {y.shape} (Binary)")

    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    print("\nBuilding U-Net architecture...")
    model = create_unet_model(SegmentationConfig.INPUT_SHAPE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=SegmentationConfig.LEARNING_RATE),
        loss=dice_loss,
        metrics=[dice_coefficient, "binary_accuracy"],
    )

    SegmentationConfig.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SegmentationConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(SegmentationConfig.MODEL_DIR / "unet_best.weights.h5"),
        monitor="val_dice_coefficient",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1,
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(
            SegmentationConfig.LOG_DIR / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )

    print("\nTraining loop...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=SegmentationConfig.EPOCHS,
        batch_size=SegmentationConfig.BATCH_SIZE,
        callbacks=[checkpoint, tensorboard],
    )

    print("\nTRAINING COMPLETE")
    print(f"Best Dice Score: {max(history.history['val_dice_coefficient']):.4f}")
            