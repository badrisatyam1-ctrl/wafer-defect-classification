"""
Inference API — ResNet18 Wafer Defect Classifier
==================================================
Loads the trained model and provides a single predict() function
that returns: predicted_class, confidence, grad_cam_heatmap.

CRITICAL: Preprocessing here is IDENTICAL to train_classifier.py.
          Both use the same preprocess_image() pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet18_model import (
    WaferResNet18, GradCAM, overlay_heatmap,
    IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
)
from utils.wafer_map_generator import DEFECT_CLASSES, NUM_CLASSES

# ── Device ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Preprocessing (MUST match training) ─────────────────────────────

def preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    """
    Preprocess a SINGLE image for inference.
    Exactly matches the preprocessing in train_classifier.py.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image (any resolution)

    Returns:
        tensor: (1, 3, 224, 224) normalized float32
    """
    # Resize to model input size
    img = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE))

    # uint8 [0,255] → float32 [0,1]
    img = img.astype(np.float32) / 255.0

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    # ImageNet normalize (same constants as training)
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = (img - mean) / std

    # Add batch dimension
    return torch.from_numpy(img).float().unsqueeze(0)


# ── Model Loader ────────────────────────────────────────────────────

_model_cache = {}

def load_model(checkpoint_path: str = None) -> WaferResNet18:
    """Load the trained ResNet18 model (cached)."""
    if checkpoint_path is None:
        checkpoint_path = str(PROJECT_ROOT / "models" / "checkpoints" / "resnet18_best.pth")

    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]

    model = WaferResNet18(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    _model_cache[checkpoint_path] = model
    return model


# ── Prediction ──────────────────────────────────────────────────────

def predict(
    image_rgb: np.ndarray,
    model: WaferResNet18 = None,
    generate_heatmap: bool = True,
) -> dict:
    """
    Full prediction pipeline.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image
        model: optional pre-loaded model
        generate_heatmap: whether to compute Grad-CAM

    Returns:
        dict with keys:
          - 'class':      str   — predicted defect class name
          - 'class_idx':  int   — class index
          - 'confidence': float — softmax probability
          - 'all_probs':  dict  — {class_name: probability}
          - 'heatmap':    np.ndarray or None — (224, 224) float32 [0,1]
          - 'overlay':    np.ndarray or None — (224, 224, 3) uint8
    """
    if model is None:
        model = load_model()

    # Preprocess (IDENTICAL to training)
    input_tensor = preprocess_image(image_rgb).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_class = DEFECT_CLASSES[pred_idx]
    confidence = probs[pred_idx].item()

    all_probs = {
        DEFECT_CLASSES[i]: round(probs[i].item(), 4)
        for i in range(NUM_CLASSES)
    }

    # Grad-CAM
    heatmap = None
    overlay = None
    if generate_heatmap:
        cam = GradCAM(model)
        heatmap = cam.generate(input_tensor, target_class=pred_idx)

        # Create overlay on resized input image
        img_resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE))
        overlay = overlay_heatmap(img_resized, heatmap, alpha=0.5)

    return {
        'class': pred_class,
        'class_idx': pred_idx,
        'confidence': confidence,
        'all_probs': all_probs,
        'heatmap': heatmap,
        'overlay': overlay,
    }
