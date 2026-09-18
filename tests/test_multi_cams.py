import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from deployment.server import get_model_and_preprocessor, DEFECT_CLASSES_V2

model, prep = get_model_and_preprocessor()

def compute_highres_cam(model, tensor, target_class, display_img):
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
    out[0, target_class].backward()

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

    new_hm = 0.4 * c4 + 0.6 * c3
    new_hm = (new_hm - new_hm.min()) / (new_hm.max() - new_hm.min() + 1e-7)
    new_hm_np = new_hm.detach().cpu().numpy()

    # Mask to circular wafer disc
    gray_display = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
    _, wafer_mask = cv2.threshold(gray_display, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    wafer_mask = cv2.morphologyEx(wafer_mask, cv2.MORPH_CLOSE, kernel)
    wafer_mask_float = cv2.resize(wafer_mask, (224, 224)).astype(np.float32) / 255.0

    new_hm_masked = new_hm_np * wafer_mask_float
    if new_hm_masked.max() > new_hm_masked.min():
        new_hm_masked = (new_hm_masked - new_hm_masked.min()) / (new_hm_masked.max() - new_hm_masked.min())

    heatmap_resized = cv2.resize(new_hm_masked, (display_img.shape[1], display_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.45 * heatmap_rgb + 0.55 * display_img)
    return overlay

sample_files = [
    'real_demo_images/real_center_1.png',
    'real_demo_images/real_cluster_1.png',
    'dataset/train/scratch/augmented_scratch_0795.png'
]

results = []
for f in sample_files:
    img = cv2.imread(f)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = prep(img_rgb).unsqueeze(0)
    with torch.no_grad():
        pred_idx = model(tensor).argmax().item()
    display_img = prep.display_image(img_rgb)
    ov = compute_highres_cam(model, tensor, pred_idx, display_img)
    results.append(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))

vis = np.hstack([cv2.cvtColor(r, cv2.COLOR_RGB2BGR) for r in results])
cv2.imwrite('multi_class_cams.png', vis)
print('Multi-class CAMs generated successfully.')
