"""
Retraining Script V4 (WITH CLASS WEIGHTS)
Adds class weighting to prioritize defect learning.
"""
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transfer_model import create_mobilenet_model
from training.train_cnn import train_model, evaluate_model, TrainingConfig
from utils.synthetic_generator import create_dataset

if __name__ == "__main__":
    print("=" * 70)
    print("RETRAINING V4 (WITH CLASS WEIGHTS)")
    print("=" * 70)
    
    # 1. Generate Dataset
    print("\n🎨 Generating 4,000 wafer maps...")
    x, y = create_dataset(n_samples=4000)
    
    # Split
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    
    # 2. Compute Class Weights (prioritize rare classes)
    print("\n⚖️ Computing class weights...")
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes.astype(int), weights))
    
    # Boost defect classes (index 1-8) even more
    for i in range(1, 9):
        if i in class_weights:
            class_weights[i] *= 2.0  # Double weight for defects
    
    print(f"  Class weights: {class_weights}")
    
    # 3. Create Model
    print("\n🏗️ Creating MobileNetV2 Model...")
    model = create_mobilenet_model(num_classes=9)
    
    # 4. Train WITH class weights
    print("\n🚀 Training with class weights...")
    config = TrainingConfig()
    config.EPOCHS = 12
    config.BATCH_SIZE = 32
    
    train_model(
        model,
        (x_train, y_train),
        (x_val, y_val),
        class_weights=class_weights,
        config=config
    )
    
    # 5. Save
    save_path = "models/checkpoints/best_model.keras"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"\n💾 Saved: {save_path}")
    
    # 6. Evaluate
    print("\n📊 Evaluation:")
    evaluate_model(model, (x_val, y_val))
    
    print("\n✅ DONE!")
