import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from deployment.server import get_model_and_preprocessor

def compute_cam(model, input_tensor, target_class):
    # Hook layer3 and layer4
    feats = {}
    grads = {}

    def get_hooks(name):
        def fh(m, i, o): feats[name] = o
        def bh(m, gi, go): grads[name] = go[0]
        return fh, bh

    h4_f, h4_b = get_hooks('l4')
    h3_f, h3_b = get_hooks('l3')

    handle_4f = model.layer4[-1].register_forward_hook(h4_f)
    handle_4b = model.layer4[-1].register_full_backward_hook(h4_b)
    handle_3f = model.layer3[-1].register_forward_hook(h3_f)
    handle_3b = model.layer3[-1].register_full_backward_hook(h3_b)

    tensor = input_tensor.clone().requires_grad_(True)
    model.zero_grad()
    logits = model(tensor)
    score = logits[0, target_class]
    score.backward()

    # Clean up hooks
    handle_4f.remove()
    handle_4b.remove()
    handle_3f.remove()
    handle_3b.remove()

    def gradcam_pp(feat, grad):
        # Grad-CAM++ weighting
        # grad: (1, C, H, W), feat: (1, C, H, W)
        g = grad[0]
        f = feat[0]
        
        g2 = g.pow(2)
        g3 = g.pow(3)
        eps = 1e-7
        sum_f = f.sum(dim=(1, 2), keepdim=True)
        alpha = g2 / (2 * g2 + sum_f * g3 + eps)
        
        # weights = sum(alpha * relu(grad))
        weights = (alpha * F.relu(g)).sum(dim=(1, 2), keepdim=True)
        cam = (weights * f).sum(dim=0)
        cam = F.relu(cam)
        return cam

    cam4 = gradcam_pp(feats['l4'], grads['l4']) # (7, 7)
    cam3 = gradcam_pp(feats['l3'], grads['l3']) # (14, 14)

    # Upsample both to 224x224
    cam4_up = F.interpolate(cam4.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0, 0]
    cam3_up = F.interpolate(cam3.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0, 0]

    # Normalize each
    c4 = (cam4_up - cam4_up.min()) / (cam4_up.max() - cam4_up.min() + 1e-7)
    c3 = (cam3_up - cam3_up.min()) / (cam3_up.max() - cam3_up.min() + 1e-7)

    # Fused CAM: 50% Layer4 (semantic class guidance) + 50% Layer3 (high resolution features)
    # Multiplying or combining:
    fused = 0.5 * c4 + 0.5 * c3
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-7)

    return fused.detach().cpu().numpy()

if __name__ == '__main__':
    model, prep = get_model_and_preprocessor()
    img = cv2.imread('dataset/train/scratch/augmented_scratch_0795.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = prep(img_rgb).unsqueeze(0)
    
    heatmap = compute_cam(model, tensor, 4)
    print('Fused Grad-CAM++ heatmap shape:', heatmap.shape)
    print('Max:', heatmap.max(), 'Min:', heatmap.min(), 'Mean:', heatmap.mean())
