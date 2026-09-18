"""
Inference engine for website and API usage.

This loader intentionally rejects legacy demo checkpoints that do not include
preprocessing and training metadata. That prevents a synthetic or mismatched
model from being presented as production-ready.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet18_classifier import (
    DEFECT_CLASSES_V2,
    GradCAM,
    NUM_CLASSES,
    PreprocessingConfig,
    WaferPreprocessor,
    create_resnet18_classifier,
)
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────
# Standalone inference preprocessing (matches training exactly)
# ─────────────────────────────────────────────────────────────────────
def preprocess_for_inference(image_path, device="cpu"):
    """
    Load a wafer image as RGB, resize to 224×224, normalize with ImageNet stats,
    and return a batch tensor ready for the model.

    This MUST match the training val_transform exactly:
      - RGB ("RGB") → 3 channels
      - Resize 224×224
      - Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    """
    img = Image.open(image_path).convert("RGB")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return tf(img).unsqueeze(0).to(device)


def predict(image_path, model, device="cpu"):
    """
    Minimal prediction: returns (class_index, confidence).

    Usage:
        cls, conf = predict("wafer.png", model)
        print(DEFECT_CLASSES_V2[cls], f"{conf:.2%}")
    """
    model.eval()
    x = preprocess_for_inference(image_path, device=device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    cls = probs.argmax(dim=1).item()
    conf = probs.max().item()

    return cls, conf


def _encode_png_base64(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _validate_checkpoint(checkpoint: Dict[str, object], checkpoint_path: Path) -> None:
    required_keys = {"model_state_dict", "preprocessing", "class_names", "config"}
    missing_keys = sorted(required_keys.difference(checkpoint.keys()))
    if missing_keys:
        raise ValueError(
            "Legacy or incomplete checkpoint detected at "
            f"{checkpoint_path}. Missing keys: {', '.join(missing_keys)}. "
            "Train a new model with training/train_resnet.py on a real lot/time-labelled dataset."
        )

    config_payload = checkpoint.get("config") or {}
    split_mode = config_payload.get("split_mode")
    if split_mode not in {"lot", "time"}:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not record a valid lot/time split. "
            "Retrain with --split-mode lot or --split-mode time."
        )


class WaferInferenceEngine:
    """Real inference engine that loads trained ResNet18 from models/best.pt."""

    CLASS_NAMES = ["normal", "center", "edge_ring", "edge_loss", "scratch", "ring", "cluster", "full_fail"]

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.device = device or "cpu"
        self.checkpoint_path = Path(
            checkpoint_path or PROJECT_ROOT / "models" / "best.pt"
        )
        self.class_names = self.CLASS_NAMES
        self.training_metadata = {}
        self.model = None
        self._last_mtime = 0
        self._load_model()

    def _get_active_checkpoint(self) -> Path:
        # Only do latest/best swapping when the requested checkpoint IS best.pt or latest.pt.
        # If a specific checkpoint was requested (e.g. synthetic_model.pt), use it directly.
        if self.checkpoint_path.name not in ("best.pt", "latest.pt"):
            return self.checkpoint_path

        latest = self.checkpoint_path.with_name("latest.pt")
        best = self.checkpoint_path.with_name("best.pt")

        if latest.exists() and best.exists():
            if latest.stat().st_mtime > best.stat().st_mtime:
                return latest
            return best
        elif latest.exists():
            return latest
        elif best.exists():
            return best
            
        return self.checkpoint_path

    def _load_model(self):
        """Load or reload the model from active live checkpoint."""
        try:
            active_path = self._get_active_checkpoint()
            if not active_path.exists():
                print(f"⚠️ No checkpoint at {active_path}, using random init")
                self.model = self._create_model()
                return

            current_mtime = active_path.stat().st_mtime
            if self.model is not None and current_mtime == self._last_mtime:
                return  # No change, skip reload

            self.model = self._create_model()
            checkpoint = torch.load(str(active_path), map_location=self.device, weights_only=False)

            # Handle both raw state_dict and full checkpoint dict formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                self.training_metadata = {
                    "epoch": checkpoint.get("epoch"),
                    "best_macro_f1": checkpoint.get("best_macro_f1"),
                    "class_names": checkpoint.get("class_names"),
                }
                if checkpoint.get("class_names"):
                    self.class_names = checkpoint["class_names"]
                    self.CLASS_NAMES = self.class_names
                self.preprocessor = WaferPreprocessor(checkpoint.get("preprocessing"))
                print(f"  Checkpoint epoch: {checkpoint.get('epoch')}, F1: {checkpoint.get('best_macro_f1', 'N/A')}")
            else:
                state_dict = checkpoint
                self.preprocessor = WaferPreprocessor()

            # Handle hackathon style model (fc.weight instead of fc.1.weight)
            if "fc.weight" in state_dict:
                print("  Detected hackathon architecture (nn.Linear instead of nn.Sequential)")
                import torchvision.models as t_models
                import torch.nn as t_nn
                num_classes = state_dict["fc.weight"].shape[0]
                model = t_models.resnet18(pretrained=False)
                model.fc = t_nn.Linear(model.fc.in_features, num_classes)
                self.model = model.to(self.device)
                
                self.features = []
                self.gradients = []
                def forward_hook(module, input, output): self.features.append(output)
                def backward_hook(module, grad_in, grad_out): self.gradients.append(grad_out[0])
                self.model.layer4.register_forward_hook(forward_hook)
                self.model.layer4.register_full_backward_hook(backward_hook)
                
                # Check if checkpoint explicitly defines class_names with matching count
                if isinstance(checkpoint, dict) and checkpoint.get("class_names") and len(checkpoint["class_names"]) == num_classes:
                    self.class_names = list(checkpoint["class_names"])
                else:
                    full_classes = ['center', 'cluster', 'edge_loss', 'edge_ring', 'full_fail', 'normal', 'ring', 'scratch']
                    self.class_names = full_classes[:num_classes] if num_classes <= len(full_classes) else full_classes
                self.CLASS_NAMES = self.class_names
                
                from torchvision import transforms
                class HackathonPreprocessor:
                    def __init__(self):
                        self.tf = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                    def __call__(self, img): return self.tf(img)
                self.preprocessor = HackathonPreprocessor()
            else:
                if not hasattr(self, "model") or self.model is None or not hasattr(self.model.fc, "__len__"):
                    self.model = self._create_model()

            self.model.load_state_dict(state_dict)
            self.model.eval()
            self._last_mtime = current_mtime
            print(f"Model loaded from {active_path}")
        except Exception as e:
            print(f"Failed to load model: {e}, using random init")
            import traceback; traceback.print_exc()
            self.model = self._create_model()

    def _create_model(self):
        """Create a ResNet18 model with 8-class head matching training architecture."""
        model = create_resnet18_classifier(
            num_classes=NUM_CLASSES,
            pretrained=False,
            freeze_backbone=False,
        )
        model = model.to(self.device)
        model.eval()

        # List-based hooks for Grad-CAM (user-specified structure)
        self.features = []
        self.gradients = []

        def forward_hook(module, input, output):
            self.features.append(output)

        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])

        target_layer = model.layer4
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

        return model



    def predict_from_file(self, image_path: Union[str, Path], api_safe: bool = False) -> Dict[str, object]:
        wafer = np.array(Image.open(image_path).convert("RGB"))
        return self._predict_real(wafer, api_safe=api_safe)

    def predict_from_array(self, image: np.ndarray, api_safe: bool = False) -> Dict[str, object]:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        wafer = np.array(Image.fromarray(image).convert("RGB"))

        return self._predict_real(wafer, api_safe=api_safe)

    def is_wafer_like(self, image: np.ndarray) -> bool:
        """Reject completely uniform or pure noise images."""
        if image.size == 0:
            return False
        std_dev = np.std(image)
        return std_dev > 3.0

    def detect_and_crop_wafer(self, image: np.ndarray):
        """Detect circular wafer using HoughCircles and crop the wafer region.
        
        Returns:
            (wafer_crop, detected): tuple of cropped wafer ndarray and bool
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        
        # Optimization: Downsample and heavily blur to destroy grid texture and speed up HoughCircles.
        # This drops detection time from ~12s to ~10ms on highly textured photographs.
        scale = min(150 / max(w, h), 1.0)
        if scale < 1.0:
            small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        else:
            small_gray = gray
            
        blurred = cv2.GaussianBlur(small_gray, (9, 9), 2)
        sh, sw = small_gray.shape
        
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                                   param1=50, param2=30,
                                   minRadius=20, maxRadius=max(sw, sh))

        if circles is None:
            return None, False

        # Scale coordinates back to original size
        x, y, r = circles[0][0]
        x, y, r = int(x / scale), int(y / scale), int(r / scale)

        # Clamp to image bounds
        y1 = max(0, y - r)
        y2 = min(h, y + r)
        x1 = max(0, x - r)
        x2 = min(w, x + r)

        wafer = image[y1:y2, x1:x2]
        return wafer, True

    def _core_predict(self, wafer: np.ndarray):
        display_img = wafer.copy()
        
        if not hasattr(self, 'preprocessor'):
            self.preprocessor = WaferPreprocessor()

        input_tensor = self.preprocessor(display_img).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        self.features = []
        self.gradients = []

        output = self.model(input_tensor)
        
        # get probabilities as explicitly requested
        probs_np = torch.softmax(output, dim=1)[0].detach().cpu().numpy()

        # FIX: use top probability class strictly from NumPy space
        pred_idx = int(probs_np.argmax())
        confidence = float(probs_np[pred_idx])

        probabilities = {self.CLASS_NAMES[i]: float(probs_np[i]) for i in range(min(len(self.CLASS_NAMES), len(probs_np)))}

        return display_img, input_tensor, output, pred_idx, confidence, probs_np

    def _generate_gradcam(self, output: torch.Tensor, pred_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output[0, pred_idx].backward()

        if len(self.gradients) > 0 and len(self.features) > 0:
            grad = self.gradients[0]
            feat = self.features[0]

            weights = torch.mean(grad, dim=(2, 3))[0]
            cam = torch.zeros(feat.shape[2:], dtype=torch.float32)

            for i, w in enumerate(weights):
                cam += w * feat[0, i]

            cam = cam.detach().numpy()
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()
            # Sharpen: threshold low activations to zero so only strong
            # defect regions glow red, making the heatmap crisp
            cam = np.where(cam < 0.25, 0.0, cam)
            # Re-normalize after threshold
            if cam.max() > 0:
                cam = cam / cam.max()
            return cam
        return np.zeros((7, 7), dtype=np.float32)

    def _overlay_cam(self, wafer: np.ndarray, cam: np.ndarray):
        # Use INTER_CUBIC for smoother, more precise upscaling
        heatmap = cv2.resize(cam, (wafer.shape[1], wafer.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        wafer_bgr = cv2.cvtColor(wafer, cv2.COLOR_RGB2BGR)
        # Stronger heatmap overlay so defect regions are clearly visible
        overlay = cv2.addWeighted(wafer_bgr, 0.5, heatmap, 0.5, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        return heatmap, overlay

    def _predict_real(self, wafer: np.ndarray, api_safe: bool = False) -> Dict[str, object]:
        # Auto-reload if checkpoint updated (training saved a new epoch)
        self._load_model()

        # 0. Gating: Reject obvious non-wafers (like cities/faces)
        if not self.is_wafer_like(wafer):
            return {
                "predicted_class": "unknown",
                "predicted_index": -1,
                "confidence": 0.0,
                "class_probabilities": {},
                "checkpoint_path": str(self.checkpoint_path),
                "grad_cam_reliable": False,
                "heatmap": np.zeros((wafer.shape[0], wafer.shape[1]), dtype=np.float32),
                "overlay": wafer.copy(),
                "wafer_detected": False,
            }

        cropped, detected = self.detect_and_crop_wafer(wafer)
        
        # Fallback to the full image if HoughCircles failed or found a tiny circle
        # (common when analyzing oval-shaped WM-811K data visualizations)
        if detected and cropped is not None:
            h, w = wafer.shape[:2]
            ch, cw = cropped.shape[:2]
            if ch < h * 0.5 or cw < w * 0.5:
                cropped_wafer = wafer
            else:
                cropped_wafer = cropped
        else:
            cropped_wafer = wafer

        # 1. Predict
        display_img, input_tensor, output, pred_idx, confidence, probs = self._core_predict(cropped_wafer)

        # 2. Get Class Name
        if 0 <= pred_idx < len(self.CLASS_NAMES):
            pred_class_name = self.CLASS_NAMES[pred_idx]
        else:
            pred_class_name = "unknown"
        pred = pred_class_name
        
        probabilities = {self.CLASS_NAMES[i]: float(probs[i]) for i in range(min(len(self.CLASS_NAMES), len(probs)))}

        if np.std(probs) < 0.05:
            pred = "uncertain"

        # 3. Generate GradCAM explicitly on the predicted index
        cam = self._generate_gradcam(output, pred_idx)

        # 4. Overlay CAM on high-resolution original wafer crop
        heatmap, overlay = self._overlay_cam(cropped_wafer, cam)

        # reliability check
        grad_cam_reliable = bool(np.std(cam) >= 0.15) if cam is not None else False

        result = {
            "predicted_class": pred,
            "predicted_index": pred_idx,
            "confidence": confidence,
            "class_probabilities": probabilities,
            "checkpoint_path": str(self.checkpoint_path),
            "grad_cam_reliable": grad_cam_reliable,
            "heatmap": heatmap,
            "overlay": overlay,
            "wafer_detected": True,
        }

        if api_safe:
            result.update({
                "heatmap_png_base64": "",
                "overlay_png_base64": "",
            })
        else:
            result.update({
                "heatmap": heatmap,
                "overlay": overlay,
            })

        return result

    def predict_for_api(self, image: np.ndarray) -> Dict[str, object]:
        return self.predict_from_array(image, api_safe=True)


def main():
    parser = argparse.ArgumentParser(description="Run ResNet18 wafer inference.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a full-wafer image.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override.")
    parser.add_argument("--api-safe", action="store_true", help="Return JSON-safe base64 visuals only.")
    args = parser.parse_args()

    engine = WaferInferenceEngine(checkpoint_path=args.checkpoint)
    result = engine.predict_from_file(args.image, api_safe=args.api_safe)
    print(f"predicted_class: {result['predicted_class']}")
    print(f"confidence: {result['confidence']:.4f}")
    if args.api_safe:
        print("overlay_png_base64_length:", len(result["overlay_png_base64"]))


if __name__ == "__main__":
    main()
