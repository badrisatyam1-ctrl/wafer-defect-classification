"""
Training History Visualization Script

This script provides functions to visualize training and validation metrics 
(Loss and Accuracy) over epochs.

============================================================================
DIAGNOSING OVERFITTING AND UNDERFITTING
============================================================================

1. UNDERFITTING (High Bias)
   - Symptoms: 
     • Training Loss is HIGH
     • Validation Loss is HIGH
   - Meaning: Model is too simple to learn the patterns.
   - Fix: Increase model complex (layers/filters), train longer, reduce regularization.

2. OVERFITTING (High Variance)
   - Symptoms:
     • Training Loss decreases continuously (LOW)
     • Validation Loss decreases then starts RISING (U-shape)
     • Large gap between Training and Validation Accuracy
   - Meaning: Model is memorizing noise in training data.
   - Fix: Add Dropout, Data Augmentation, Early Stopping, simplify model.

3. GOOD FIT
   - Symptoms:
     • Training and Validation Loss decrease and stabilize close to each other.
     • Gap between accuracies is small.
   - Meaning: Model generalizes well to unseen data.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union

def plot_training_history(
    history: Union[Dict, object], 
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
):
    """
    Plot training and validation loss and accuracy from Keras history.
    
    Args:
        history: Keras History object or dictionary containing 'loss', 'val_loss', etc.
        save_path: Path to save the figure (optional)
        figsize: Figure size
    """
    # Handle Keras History object or raw dict
    if hasattr(history, 'history'):
        metrics = history.history
    else:
        metrics = history
        
    # Extract metrics
    loss = metrics.get('loss', [])
    val_loss = metrics.get('val_loss', [])
    acc = metrics.get('accuracy', [])
    val_acc = metrics.get('val_accuracy', [])
    
    epochs = range(1, len(loss) + 1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # --------------------------------------------------------------------------
    # PLOT 1: LOSS CURVES (The most important check)
    # --------------------------------------------------------------------------
    ax1.plot(epochs, loss, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Diagnosis annotation for Loss
    if len(val_loss) > 0:
        min_val_loss_epoch = np.argmin(val_loss) + 1
        ax1.axvline(x=min_val_loss_epoch, color='g', linestyle='--', alpha=0.5)
        ax1.text(min_val_loss_epoch, max(max(loss), max(val_loss))*0.9, 
                 f' Best Model\n (Epoch {min_val_loss_epoch})', color='g')
        
        # Check for Overfitting
        if len(val_loss) > min_val_loss_epoch + 2 and val_loss[-1] > val_loss[min_val_loss_epoch]:
            ax1.text(epochs[-1], val_loss[-1], ' ← OVERFITTING?', 
                     color='red', fontweight='bold', ha='left')

    # --------------------------------------------------------------------------
    # PLOT 2: ACCURACY CURVES
    # --------------------------------------------------------------------------
    ax2.plot(epochs, acc, 'bo-', label='Training Acc')
    ax2.plot(epochs, val_acc, 'ro-', label='Validation Acc')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Diagnosis annotation for Accuracy
    if len(acc) > 0 and len(val_acc) > 0:
        gap = acc[-1] - val_acc[-1]
        ax2.text(epochs[-len(epochs)//2], (acc[-1]+val_acc[-1])/2, 
                 f'Gap: {gap:.1%}', ha='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training plot saved to: {save_path}")
        
    plt.show()

# ==============================================================================
# DEMO: SIMULATING SCENARIOS
# ==============================================================================

def demo_scenario(scenario: str):
    """Generate synthetic history for demonstration."""
    epochs = 20
    x = np.linspace(0, 10, epochs)
    
    history = {}
    
    if scenario == "overfitting":
        # Train loss goes down, Val loss goes down then UP
        history['loss'] = 1.0 * np.exp(-0.5 * x) + 0.05
        history['val_loss'] = 1.0 * np.exp(-0.5 * x) + 0.05 + 0.02 * (x - 4)**2 * (x > 4)
        
        # Train acc goes high, Val acc plateaus or drops
        history['accuracy'] = 0.5 + 0.45 * (1 - np.exp(-0.5 * x))
        history['val_accuracy'] = 0.5 + 0.35 * (1 - np.exp(-0.5 * x)) - 0.05 * (x > 5) / 10
        
        title = "SCENARIO: OVERFITTING (High Variance)"
        desc = "Note Validation Loss rising after Epoch 5 while Training Loss continues to drop."

    elif scenario == "underfitting":
        # Both losses stay high and flat
        history['loss'] = 0.8 - 0.1 * (1 - np.exp(-0.1 * x))
        history['val_loss'] = 0.85 - 0.1 * (1 - np.exp(-0.1 * x))
        
        # Low accuracy
        history['accuracy'] = 0.4 + 0.1 * np.random.rand(epochs)
        history['val_accuracy'] = 0.38 + 0.1 * np.random.rand(epochs)
        
        title = "SCENARIO: UNDERFITTING (High Bias)"
        desc = "Note both Training and Validation Performance are poor and not improving much."
        
    elif scenario == "good_fit":
        # Both losses decrease together
        history['loss'] = 1.0 * np.exp(-0.5 * x) + 0.1
        history['val_loss'] = 1.1 * np.exp(-0.5 * x) + 0.15 + 0.05 * np.random.rand(epochs)
        
        # Both accuracies increase together
        history['accuracy'] = 0.4 + 0.5 * (1 - np.exp(-0.4 * x))
        history['val_accuracy'] = 0.4 + 0.48 * (1 - np.exp(-0.4 * x))
        
        title = "SCENARIO: GOOD FIT"
        desc = "Training and Validation curves track closely together."

    print("\n" + "="*70)
    print(title)
    print("="*70)
    print(desc)
    
    plot_training_history(history, save_path=f"evaluation/plot_{scenario}.png")

if __name__ == "__main__":
    print("Generating demo plots for diagnosis explanations...")
    import os
    os.makedirs("evaluation", exist_ok=True)
    
    demo_scenario("overfitting")
    demo_scenario("good_fit")
    demo_scenario("underfitting")
