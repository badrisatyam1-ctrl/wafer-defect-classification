import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from deployment.server import get_model_and_preprocessor
from models.resnet18_classifier import GradCAM

model, prep = get_model_and_preprocessor()
img = cv2.imread('dataset/train/scratch/augmented_scratch_0795.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tensor = prep(img_rgb).unsqueeze(0)

# 1. Old Layer4 Grad-CAM
old_cam_engine = GradCAM(model, 'layer4')
old_hm = old_cam_engine(tensor, 4)
display_img = prep.display_image(img_rgb)
old_overlay = GradCAM.overlay_heatmap(display_img, old_hm)

# 2. Enhanced High-Res Grad-CAM++ with multi-layer & wafer disc constraint
feats = {}
grads = {}

def get_hooks(name):
    def fh(m, i, o): feats[name] = o
    def bh(m, gi, go): grads[name] = go[0]
    return fh, bh

h4_f, h4_b = get_hooks('l4')
h3_f, h3_b = get_hooks('l3')
h4_handle = model.layer4[-1].register_forward_hook(h4_f)
h4_bhandle = model.layer4[-1].register_full_backward_hook(h4_b)
h3_handle = model.layer3[-1].register_forward_hook(h3_f)
h3_bhandle = model.layer3[-1].register_full_backward_hook(h3_b)

t = tensor.clone().requires_grad_(True)
model.zero_grad()
out = model(t)
out[0, 4].backward()

h4_handle.remove(); h4_bhandle.remove()
h3_handle.remove(); h3_bhandle.remove()

def gradcam_pp(feat, grad):
    g = grad[0]
    f = feat[0]
    g2 = g.pow(2)
    g3 = g.pow(3)
    eps = 1e-7
    sum_f = f.sum(dim=(1, 2), keepdim=True)
    alpha = g2 / (2 * g2 + sum_f * g3 + eps)
    weights = (alpha * F.relu(g)).sum(dim=(1, 2), keepdim=True)
    cam = (weights * f).sum(dim=0)
    return F.relu(cam)

cam4 = gradcam_pp(feats['l4'], grads['l4'])
cam3 = gradcam_pp(feats['l3'], grads['l3'])

cam4_up = F.interpolate(cam4.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0, 0]
cam3_up = F.interpolate(cam3.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0, 0]

c4 = (cam4_up - cam4_up.min()) / (cam4_up.max() - cam4_up.min() + 1e-7)
c3 = (cam3_up - cam3_up.min()) / (cam3_up.max() - cam3_up.min() + 1e-7)

# Layer3 has high spatial resolution; Layer4 has semantic class gating
new_hm = 0.4 * c4 + 0.6 * c3
new_hm = (new_hm - new_hm.min()) / (new_hm.max() - new_hm.min() + 1e-7)
new_hm_np = new_hm.detach().cpu().numpy()

# Mask heatmap to wafer disk boundary so it doesn't bleed onto the black background!
gray_display = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
_, wafer_mask = cv2.threshold(gray_display, 20, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
wafer_mask = cv2.morphologyEx(wafer_mask, cv2.MORPH_CLOSE, kernel)
wafer_mask_float = cv2.resize(wafer_mask, (224, 224)).astype(np.float32) / 255.0

new_hm_masked = new_hm_np * wafer_mask_float
new_hm_masked = (new_hm_masked - new_hm_masked.min()) / (new_hm_masked.max() - new_hm_masked.min() + 1e-7)

new_overlay = GradCAM.overlay_heatmap(display_img, new_hm_masked)

# Save comparison side by side
vis = np.hstack([
    cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR),
    cv2.cvtColor(old_overlay, cv2.COLOR_RGB2BGR),
    cv2.cvtColor(new_overlay, cv2.COLOR_RGB2BGR)
])
cv2.imwrite('scratch_gradcam_comparison.png', vis)
print('Comparison saved to scratch_gradcam_comparison.png')
