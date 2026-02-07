"""
CNN Training Script for Wafer Defect Classification

This script trains the CNN model with:
- Appropriate loss function (Categorical Cross-Entropy with class weights)
- Adam optimizer with learning rate scheduling
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score

============================================================================
WHY RECALL IS CRITICAL IN WAFER DEFECT DETECTION
============================================================================

In semiconductor manufacturing, MISSING A DEFECT IS FAR WORSE THAN FALSE ALARMS.

Consider the consequences:

┌─────────────────────────────────────────────────────────────────────────┐
│ SCENARIO           │ PREDICTION │ ACTUAL │ CONSEQUENCE                 │
├─────────────────────────────────────────────────────────────────────────┤
│ FALSE POSITIVE     │ Defect     │ Good   │ Extra inspection cost       │
│ (Type I Error)     │            │        │ (~$0.50 per wafer)          │
├─────────────────────────────────────────────────────────────────────────┤
│ FALSE NEGATIVE     │ Good       │ Defect │ Defective chips shipped!    │
│ (Type II Error)    │            │        │ (~$10,000+ recalls/lawsuits)│
└─────────────────────────────────────────────────────────────────────────┘

THEREFORE: HIGH RECALL IS ESSENTIAL!

METRIC DEFINITIONS:
─────────────────────
• ACCURACY  = (TP + TN) / Total
  → Overall correctness (misleading with imbalanced data!)

• PRECISION = TP / (TP + FP)
  → "Of predicted defects, how many are real?"
  → High precision = few false alarms

• RECALL    = TP / (TP + FN)  ← MOST IMPORTANT FOR DEFECTS
  → "Of actual defects, how many did we catch?"
  → High recall = few missed defects

• F1-SCORE  = 2 × (Precision × Recall) / (Precision + Recall)
  → Harmonic mean of precision and recall
  → Balanced metric when both matter

FOR WAFER DEFECT DETECTION:
─────────────────────────────
• Target Recall > 0.95 (catch 95%+ of defects)
• Accept lower Precision (some false alarms are OK)
• Use F1-Score to balance the trade-off
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from datetime import datetime
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.wafer_cnn import create_simple_cnn, DEFECT_CLASSES


# ==============================================================================
# CUSTOM METRICS
# ==============================================================================

class MetricsCallback(keras.callbacks.Callback):
    """
    Custom callback to compute Precision, Recall, F1 at end of each epoch.
    """
    
    def __init__(self, validation_data, class_names):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.history = {
            'precision': [], 'recall': [], 'f1': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        x_val, y_val = self.validation_data
        y_pred = np.argmax(self.model.predict(x_val, verbose=0), axis=1)
        
        # Calculate metrics (weighted for imbalanced classes)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        # Store in history
        self.history['precision'].append(precision)
        self.history['recall'].append(recall)
        self.history['f1'].append(f1)
        
        # Log
        logs['precision'] = precision
        logs['recall'] = recall
        logs['f1'] = f1
        
        print(f" - precision: {precision:.4f} - recall: {recall:.4f} - f1: {f1:.4f}")


# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

class TrainingConfig:
    """Training configuration parameters."""
    
    # Model
    INPUT_SHAPE = (128, 128, 3)
    NUM_CLASSES = 9
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 10
    LR_REDUCE_PATIENCE = 5
    LR_REDUCE_FACTOR = 0.5
    
    # Paths
    MODEL_DIR = Path("models/checkpoints")
    LOG_DIR = Path("logs/tensorboard")


# ==============================================================================
# LOSS FUNCTION AND OPTIMIZER
# ==============================================================================

def get_loss_function(class_weights=None):
    """
    Get appropriate loss function.
    
    WHY CATEGORICAL CROSS-ENTROPY?
    ─────────────────────────────
    - Standard for multi-class classification
    - Penalizes confident wrong predictions more
    - Works well with softmax output
    
    WITH CLASS WEIGHTS:
    ─────────────────────
    - Increases loss for minority class errors
    - Forces model to pay attention to rare defects
    - Essential for imbalanced datasets like WM-811K
    """
    if class_weights is not None:
        # Weighted loss for imbalanced data
        return keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        return 'sparse_categorical_crossentropy'


def get_optimizer(learning_rate=0.001):
    """
    Get optimizer.
    
    WHY ADAM OPTIMIZER?
    ─────────────────────
    - Adaptive learning rate per parameter
    - Combines momentum + RMSprop benefits
    - Works well out-of-the-box
    - Standard choice for CNNs
    
    LEARNING RATE:
    ─────────────────
    - 0.001 is a good starting point
    - Will be reduced if loss plateaus
    """
    return keras.optimizers.Adam(learning_rate=learning_rate)


# ==============================================================================
# COMPILE AND TRAIN
# ==============================================================================

def compile_model(model, class_weights=None, learning_rate=0.001):
    """
    Compile the model with loss, optimizer, and metrics.
    """
    model.compile(
        optimizer=get_optimizer(learning_rate),
        loss=get_loss_function(class_weights),
        metrics=['accuracy']
    )
    return model


def get_callbacks(config: TrainingConfig, validation_data=None):
    """
    Get training callbacks.
    
    CALLBACKS EXPLAINED:
    ─────────────────────
    1. ModelCheckpoint: Save best model based on val_accuracy
    2. EarlyStopping: Stop if no improvement for N epochs
    3. ReduceLROnPlateau: Reduce learning rate when stuck
    4. TensorBoard: Visualize training in real-time
    """
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=str(config.MODEL_DIR / "best_model.keras"),
            monitor='val_recall',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=str(config.LOG_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    # Add custom metrics callback if validation data provided
    if validation_data is not None:
        callbacks.append(MetricsCallback(validation_data, DEFECT_CLASSES))
    
    return callbacks


def train_model(
    model,
    train_data,
    val_data,
    class_weights=None,
    config=None
):
    """
    Train the model.
    
    Args:
        model: Compiled Keras model
        train_data: Tuple of (x_train, y_train) or tf.data.Dataset
        val_data: Tuple of (x_val, y_val) or tf.data.Dataset
        class_weights: Dict mapping class indices to weights
        config: TrainingConfig object
    
    Returns:
        Training history
    """
    if config is None:
        config = TrainingConfig()
    
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  Input shape:     {config.INPUT_SHAPE}")
    print(f"  Number of classes: {config.NUM_CLASSES}")
    print(f"  Batch size:      {config.BATCH_SIZE}")
    print(f"  Max epochs:      {config.EPOCHS}")
    print(f"  Learning rate:   {config.LEARNING_RATE}")
    print(f"  Class weights:   {'Enabled' if class_weights else 'Disabled'}")
    
    # Prepare data
    if isinstance(train_data, tuple):
        x_train, y_train = train_data
        x_val, y_val = val_data
        
        print(f"\n  Training samples:   {len(x_train)}")
        print(f"  Validation samples: {len(x_val)}")
        
        validation_data = (x_val, y_val)
    else:
        # tf.data.Dataset
        validation_data = val_data
    
    # Get callbacks
    callbacks = get_callbacks(config, val_data if isinstance(val_data, tuple) else None)
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    # Train
    if isinstance(train_data, tuple):
        history = model.fit(
            x_train, y_train,
            validation_data=validation_data,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    
    return history


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_model(model, test_data, class_names=DEFECT_CLASSES):
    """
    Comprehensive model evaluation with all metrics.
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    x_test, y_test = test_data
    
    # Get predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n  OVERALL METRICS:")
    print(f"  ─────────────────────────────────────")
    print(f"  Accuracy:  {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f}  ({precision*100:.1f}%)")
    print(f"  Recall:    {recall:.4f}  ({recall*100:.1f}%)  ← CRITICAL FOR DEFECTS")
    print(f"  F1-Score:  {f1:.4f}  ({f1*100:.1f}%)")
    
    # Per-class report
    print(f"\n  PER-CLASS METRICS:")
    print("  " + "-" * 60)
    # Get labels that exist in test set
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    used_names = [class_names[i] for i in unique_labels if i < len(class_names)]
    report = classification_report(y_test, y_pred, labels=unique_labels, 
                                   target_names=used_names, zero_division=0)
    for line in report.split('\n'):
        print(f"  {line}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_probs
    }


# ==============================================================================
# DEMO WITH SYNTHETIC DATA
# ==============================================================================

def create_synthetic_dataset(n_samples=1000, n_classes=9, img_size=(224, 224, 3)):
    """Create synthetic dataset for testing the training pipeline."""
    np.random.seed(42)
    
    # Create imbalanced class distribution (like WM-811K)
    class_probs = [0.6, 0.08, 0.02, 0.08, 0.1, 0.05, 0.03, 0.03, 0.01]
    
    x = np.random.rand(n_samples, *img_size).astype(np.float32)
    y = np.random.choice(n_classes, size=n_samples, p=class_probs)
    
    # Add class-specific patterns for demo
    for i in range(n_samples):
        cls = y[i]
        if cls == 1:  # Center - add pattern in center
            x[i, 80:144, 80:144, :] += 0.3
        elif cls == 7:  # Scratch - add line
            x[i, 100:120, :, :] += 0.3
    
    x = np.clip(x, 0, 1)
    
    return x, y


def calculate_class_weights(y):
    """Calculate class weights for imbalanced data."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    
    weights = {}
    for cls, count in zip(classes, counts):
        weights[int(cls)] = total / (n_classes * count)
    
    return weights


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WAFER DEFECT CNN - TRAINING SCRIPT")
    print("=" * 70)
    
    # Create synthetic data for demo
    print("\n📊 Creating synthetic dataset...")
    x, y = create_synthetic_dataset(n_samples=500)
    
    # Split into train/val/test
    train_split = int(0.7 * len(x))
    val_split = int(0.85 * len(x))
    
    x_train, y_train = x[:train_split], y[:train_split]
    x_val, y_val = x[train_split:val_split], y[train_split:val_split]
    x_test, y_test = x[val_split:], y[val_split:]
    
    print(f"  Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    print(f"\n⚖️ Class weights calculated for {len(class_weights)} classes")
    
    # Create model
    print("\n🏗️ Creating CNN model...")
    model = create_simple_cnn(
        input_shape=TrainingConfig.INPUT_SHAPE,
        num_classes=TrainingConfig.NUM_CLASSES
    )
    
    # Compile
    model = compile_model(model, class_weights, TrainingConfig.LEARNING_RATE)
    print("✓ Model compiled with Adam optimizer and CrossEntropy loss")
    
    # Train (reduced epochs for demo)
    config = TrainingConfig()
    config.EPOCHS = 5  # Reduced for demo
    
    print("\n🚀 Starting training...")
    history = train_model(
        model,
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        class_weights=class_weights,
        config=config
    )
    
    # Evaluate
    results = evaluate_model(model, (x_test, y_test))
    
    print("\n" + "=" * 70)
    print("WHY RECALL MATTERS - SUMMARY")
    print("=" * 70)
    print("""
    In wafer defect detection:
    
    • Recall measures: "What % of actual defects did we catch?"
    • A missed defect (low recall) → defective chips shipped → costly recalls
    • A false alarm (low precision) → extra inspection → minor cost
    
    THEREFORE: Prioritize HIGH RECALL (>95%) even at cost of some precision!
    
    Your model's recall: {:.1f}%
    """.format(results['recall'] * 100))
    
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
