"""
Model Evaluation Script for Wafer Defect Classification

This script evaluates the trained CNN model and generates:
1. Confusion Matrix - Visual representation of predictions vs actual
2. Classification Report - Precision, Recall, F1 per class
3. Interpretation guide for semiconductor manufacturing

============================================================================
HOW TO INTERPRET RESULTS FOR SEMICONDUCTOR MANUFACTURING
============================================================================

UNDERSTANDING THE CONFUSION MATRIX:
───────────────────────────────────
The confusion matrix shows how often each class is predicted correctly
or confused with another class.

                        PREDICTED CLASS
                    ┌─────┬─────┬─────┬─────┐
                    │none │Cent │Donut│Scrat│
           ┌────────┼─────┼─────┼─────┼─────┤
      A    │ none   │ 95  │  2  │  1  │  2  │  ← Row shows actual "none" 
      C    ├────────┼─────┼─────┼─────┼─────┤     predictions
      T    │ Center │  5  │ 85  │  3  │  7  │
      U    ├────────┼─────┼─────┼─────┼─────┤
      A    │ Donut  │  2  │  5  │ 88  │  5  │
      L    ├────────┼─────┼─────┼─────┼─────┤
           │ Scratch│  3  │  4  │  2  │ 91  │
           └────────┴─────┴─────┴─────┴─────┘
                            ↑
                    Column shows predictions

DIAGONAL = Correct predictions (want HIGH values)
OFF-DIAGONAL = Errors (want LOW values)

CRITICAL MANUFACTURING INSIGHTS:
─────────────────────────────────
1. DEFECT MISSED AS GOOD (Row=Defect, Col=none)
   Example: Actual "Scratch", Predicted "none"
   IMPACT: 🔴 CRITICAL! Defective chip shipped to customer!
   ACTION: Improve recall for that defect class

2. GOOD WAFER FALSE ALARM (Row=none, Col=Defect)
   Example: Actual "none", Predicted "Scratch"  
   IMPACT: 🟡 MODERATE - Extra inspection cost
   ACTION: Improve precision if too many false alarms

3. DEFECT MISCLASSIFIED (Row=DefectA, Col=DefectB)
   Example: Actual "Scratch", Predicted "Edge-Loc"
   IMPACT: 🟠 MODERATE - Wrong root cause analysis
   ACTION: May lead to wrong process adjustment
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc
)
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.wafer_cnn import DEFECT_CLASSES


# ==============================================================================
# CONFUSION MATRIX GENERATION
# ==============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = DEFECT_CLASSES,
    normalize: bool = True,
    save_path: str = None,
    figsize: tuple = (12, 10)
):
    """
    Generate and plot confusion matrix with annotations.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Normalize by row (show percentages)
        save_path: Path to save figure
        figsize: Figure size
    """
    # Get unique labels in the data
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    used_names = [class_names[i] for i in unique_labels if i < len(class_names)]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    if normalize:
        # Normalize by row (actual class count)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.1%'
        title = 'Normalized Confusion Matrix (% of Actual Class)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix (Sample Counts)'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color scheme: Green diagonal (correct), Red off-diagonal (errors)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt if normalize else 'd',
        cmap='RdYlGn_r',  # Red=bad, Green=good
        xticklabels=used_names,
        yticklabels=used_names,
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None,
        linewidths=0.5,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Class', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm


# ==============================================================================
# CLASSIFICATION REPORT
# ==============================================================================

def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = DEFECT_CLASSES,
    save_path: str = None
):
    """
    Generate detailed classification report with manufacturing interpretation.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save report
    """
    # Get unique labels
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    used_names = [class_names[i] for i in unique_labels if i < len(class_names)]
    
    # Generate report
    report = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=used_names,
        output_dict=True,
        zero_division=0
    )
    
    # Print formatted report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    
    print("\n┌" + "─" * 68 + "┐")
    print(f"│ {'Class':<12} │ {'Precision':>10} │ {'Recall':>10} │ {'F1-Score':>10} │ {'Support':>10} │")
    print("├" + "─" * 68 + "┤")
    
    for cls in used_names:
        metrics = report[cls]
        print(f"│ {cls:<12} │ {metrics['precision']:>10.3f} │ {metrics['recall']:>10.3f} │ {metrics['f1-score']:>10.3f} │ {int(metrics['support']):>10} │")
    
    print("├" + "─" * 68 + "┤")
    print(f"│ {'MACRO AVG':<12} │ {report['macro avg']['precision']:>10.3f} │ {report['macro avg']['recall']:>10.3f} │ {report['macro avg']['f1-score']:>10.3f} │ {int(report['macro avg']['support']):>10} │")
    print(f"│ {'WEIGHTED AVG':<12} │ {report['weighted avg']['precision']:>10.3f} │ {report['weighted avg']['recall']:>10.3f} │ {report['weighted avg']['f1-score']:>10.3f} │ {int(report['weighted avg']['support']):>10} │")
    print("└" + "─" * 68 + "┘")
    
    # Manufacturing interpretation
    print("\n" + "=" * 70)
    print("MANUFACTURING INTERPRETATION")
    print("=" * 70)
    
    interpret_for_manufacturing(report, used_names)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(classification_report(y_true, y_pred, labels=unique_labels,
                                          target_names=used_names, zero_division=0))
        print(f"\n✓ Report saved to: {save_path}")
    
    return report


def interpret_for_manufacturing(report: dict, class_names: list):
    """
    Provide manufacturing-specific interpretation of metrics.
    """
    print("\n📊 METRIC INTERPRETATION FOR SEMICONDUCTOR FAB:")
    print("-" * 60)
    
    # Find critical issues
    low_recall_classes = []
    low_precision_classes = []
    
    for cls in class_names:
        if cls in report:
            recall = report[cls]['recall']
            precision = report[cls]['precision']
            
            if recall < 0.90 and cls != 'none':
                low_recall_classes.append((cls, recall))
            if precision < 0.80:
                low_precision_classes.append((cls, precision))
    
    # Report low recall (missing defects)
    if low_recall_classes:
        print("\n🔴 CRITICAL - LOW RECALL (Missing Defects):")
        for cls, recall in low_recall_classes:
            print(f"   • {cls}: {recall*100:.1f}% recall")
            print(f"     → {(1-recall)*100:.1f}% of '{cls}' defects are being MISSED!")
            print(f"     → These defective wafers may ship to customers!")
            print(f"     → ACTION: Collect more '{cls}' training samples")
    else:
        print("\n✅ All defect classes have acceptable recall (>90%)")
    
    # Report low precision (false alarms)
    if low_precision_classes:
        print("\n🟡 WARNING - LOW PRECISION (False Alarms):")
        for cls, precision in low_precision_classes:
            print(f"   • {cls}: {precision*100:.1f}% precision")
            print(f"     → {(1-precision)*100:.1f}% of '{cls}' predictions are FALSE ALARMS")
            print(f"     → Causes unnecessary wafer re-inspection")
    else:
        print("\n✅ All classes have acceptable precision (>80%)")
    
    # Overall assessment
    weighted_recall = report['weighted avg']['recall']
    weighted_precision = report['weighted avg']['precision']
    
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT:")
    print("=" * 60)
    
    if weighted_recall >= 0.95:
        print("🟢 EXCELLENT: Catching 95%+ of defects - ready for production!")
    elif weighted_recall >= 0.90:
        print("🟡 GOOD: Catching 90%+ of defects - acceptable but can improve")
    elif weighted_recall >= 0.80:
        print("🟠 FAIR: Missing 20%+ of defects - needs improvement before production")
    else:
        print("🔴 POOR: Missing too many defects - NOT suitable for production!")
    
    print(f"\n   Weighted Recall:    {weighted_recall*100:.1f}%")
    print(f"   Weighted Precision: {weighted_precision*100:.1f}%")
    print(f"   Weighted F1-Score:  {report['weighted avg']['f1-score']*100:.1f}%")


# ==============================================================================
# PER-CLASS ANALYSIS
# ==============================================================================

def analyze_per_class_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = DEFECT_CLASSES
):
    """
    Detailed per-class analysis with recommendations.
    """
    print("\n" + "=" * 70)
    print("PER-CLASS ANALYSIS")
    print("=" * 70)
    
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    for label in unique_labels:
        if label >= len(class_names):
            continue
            
        cls = class_names[label]
        
        # Calculate per-class metrics
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n📌 {cls.upper()}")
        print("-" * 40)
        print(f"   True Positives:  {tp:>5}  (correctly detected)")
        print(f"   False Positives: {fp:>5}  (false alarms)")
        print(f"   False Negatives: {fn:>5}  (MISSED defects)")
        print(f"   True Negatives:  {tn:>5}  (correctly ignored)")
        print(f"   Precision: {precision:.1%} | Recall: {recall:.1%}")
        
        if cls != 'none' and fn > 0:
            print(f"   ⚠️  {fn} '{cls}' defects were MISSED!")


# ==============================================================================
# VISUALIZATION: METRICS BAR CHART
# ==============================================================================

def plot_metrics_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = DEFECT_CLASSES,
    save_path: str = None
):
    """
    Plot precision, recall, F1 for each class as grouped bar chart.
    """
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    used_names = [class_names[i] for i in unique_labels if i < len(class_names)]
    
    report = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=used_names,
        output_dict=True,
        zero_division=0
    )
    
    # Extract metrics
    precision_scores = [report[c]['precision'] for c in used_names]
    recall_scores = [report[c]['recall'] for c in used_names]
    f1_scores = [report[c]['f1-score'] for c in used_names]
    
    # Plot
    x = np.arange(len(used_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision_scores, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall_scores, width, label='Recall', color='#e74c3c')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#2ecc71')
    
    ax.set_xlabel('Defect Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall, F1-Score by Defect Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(used_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Metrics comparison saved to: {save_path}")
    
    plt.show()


# ==============================================================================
# COMPLETE EVALUATION PIPELINE
# ==============================================================================

def evaluate_model_complete(
    model,
    test_data: tuple,
    output_dir: str = "evaluation/results",
    class_names: list = DEFECT_CLASSES
):
    """
    Run complete evaluation pipeline.
    
    Args:
        model: Trained Keras model
        test_data: Tuple of (x_test, y_test)
        output_dir: Directory to save results
        class_names: Class names
    
    Returns:
        Dictionary with all evaluation results
    """
    x_test, y_test = test_data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("WAFER DEFECT MODEL EVALUATION")
    print("=" * 70)
    
    # Get predictions
    print("\n🔄 Generating predictions...")
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 1. Confusion Matrix
    print("\n📊 Generating Confusion Matrix...")
    cm = plot_confusion_matrix(
        y_test, y_pred, class_names,
        normalize=True,
        save_path=str(output_path / "confusion_matrix.png")
    )
    
    # 2. Classification Report
    print("\n📋 Generating Classification Report...")
    report = generate_classification_report(
        y_test, y_pred, class_names,
        save_path=str(output_path / "classification_report.txt")
    )
    
    # 3. Per-class Analysis
    analyze_per_class_performance(y_test, y_pred, class_names)
    
    # 4. Metrics Comparison Chart
    print("\n📈 Generating Metrics Comparison...")
    plot_metrics_comparison(
        y_test, y_pred, class_names,
        save_path=str(output_path / "metrics_comparison.png")
    )
    
    # Calculate overall metrics
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_pred_probs
    }
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {output_path.absolute()}")
    
    return results


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WAFER DEFECT MODEL EVALUATION DEMO")
    print("=" * 70)
    
    # Create synthetic predictions for demo
    np.random.seed(42)
    n_samples = 200
    
    # Simulated ground truth (imbalanced)
    class_probs = [0.5, 0.1, 0.05, 0.1, 0.1, 0.05, 0.03, 0.05, 0.02]
    y_true = np.random.choice(9, size=n_samples, p=class_probs)
    
    # Simulated predictions (imperfect model)
    y_pred = y_true.copy()
    # Add some random errors
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    y_pred[error_indices] = np.random.choice(9, size=len(error_indices))
    
    print(f"\n📊 Demo with {n_samples} samples, ~15% error rate")
    
    # Generate confusion matrix
    print("\n1️⃣ CONFUSION MATRIX")
    cm = plot_confusion_matrix(
        y_true, y_pred, DEFECT_CLASSES,
        normalize=True,
        save_path="evaluation/confusion_matrix_demo.png"
    )
    
    # Generate classification report
    print("\n2️⃣ CLASSIFICATION REPORT")
    report = generate_classification_report(
        y_true, y_pred, DEFECT_CLASSES
    )
    
    # Per-class analysis
    print("\n3️⃣ PER-CLASS ANALYSIS")
    analyze_per_class_performance(y_true, y_pred, DEFECT_CLASSES)
    
    # Metrics comparison
    print("\n4️⃣ METRICS COMPARISON")
    plot_metrics_comparison(
        y_true, y_pred, DEFECT_CLASSES,
        save_path="evaluation/metrics_comparison_demo.png"
    )
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
