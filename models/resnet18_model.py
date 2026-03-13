"""
ResNet18 Wafer Defect Classifier + Focal Loss + Grad-CAM
=========================================================
Design decisions:
  - ResNet18 pretrained on ImageNet: good feature extractor, fast to fine-tune.
  - Global Average Pooling captures full-wafer spatial context (no patch cropping).
  - Focal Loss (Lin et al., 2017): down-weights easy examples, focuses on
    hard/rare classes — ideal for semiconductor class imbalance.
  - Grad-CAM (Selvaraju et al., 2017): produces heatmap overlays without
    modifying the model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2


# ── Preprocessing constants (MUST match in training AND inference) ──
# ImageNet normalization — standard for pretrained ResNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224


# ── Focal Loss ──────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Reduces the loss contribution from easy examples and focuses training
    on hard, misclassified examples. Essential for wafer defect detection
    where 'normal' may dominate and rare defects like 'scratch' are few.

    Args:
        alpha:  Per-class weight tensor of shape (C,). Set higher for rare classes.
        gamma:  Focusing parameter. γ=0 → standard CE.  γ=2 works well in practice.
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        # Register alpha as a buffer so it moves to the correct device
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            at = self.alpha[targets]
            focal_weight = focal_weight * at

        return (focal_weight * ce_loss).mean()


# ── ResNet18 Classifier ────────────────────────────────────────────

class WaferResNet18(nn.Module):
    """
    ResNet18 fine-tuned for wafer defect classification.

    Architecture:
      ImageNet-pretrained ResNet18 backbone
      → Global Average Pooling (built-in)
      → Dropout(0.3) for regularization
      → FC(512 → num_classes)

    The backbone captures hierarchical spatial features.
    GAP ensures the model sees the ENTIRE wafer, not local patches.
    """
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Keep everything except the final FC layer
        # ResNet18 layers: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.avgpool = backbone.avgpool       # Global Average Pooling
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512, num_classes)  # ResNet18 outputs 512-d features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)           # (B, 512, 7, 7)
        pooled = self.avgpool(features)       # (B, 512, 1, 1)
        flat = torch.flatten(pooled, 1)       # (B, 512)
        flat = self.dropout(flat)
        logits = self.fc(flat)                # (B, num_classes)
        return logits

    def extract_features(self, x: torch.Tensor):
        """Return the last conv feature map (for Grad-CAM)."""
        return self.features(x)


# ── Grad-CAM ───────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Produces a heatmap showing which spatial regions of the wafer
    most influenced the model's prediction. Critical for:
      - Engineer trust: "why did the model flag this wafer?"
      - Debugging: identifying if the model learns spurious features.

    Usage:
        cam = GradCAM(model)
        heatmap = cam.generate(input_tensor, target_class=None)
    """
    def __init__(self, model: WaferResNet18):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook into the last conv layer (layer4 of ResNet)
        target_layer = model.features[-1]  # layer4
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.no_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, 224, 224) normalized tensor
            target_class: class index to explain. If None, uses predicted class.

        Returns:
            heatmap: (224, 224) float32 in [0, 1]
        """
        self.model.eval()

        # We need gradients for this forward pass
        with torch.enable_grad():
            input_tensor = input_tensor.requires_grad_(True)
            logits = self.model(input_tensor)

            if target_class is None:
                target_class = logits.argmax(dim=1).item()

            # Backprop from the target class score
            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

        # Weighted combination of activation maps
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, 512, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, 7, 7)
        cam = F.relu(cam)  # Only positive contributions

        # Resize to input resolution
        cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        image:   (H, W, 3) uint8  — original wafer image
        heatmap: (H, W) float32   — Grad-CAM output in [0, 1]
        alpha:   blending factor

    Returns:
        (H, W, 3) uint8 — blended visualization
    """
    # Resize heatmap to match image if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap (Jet: blue=cold, red=hot)
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return blended
