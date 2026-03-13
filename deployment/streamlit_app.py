# -*- coding: utf-8 -*-
# Wafer Defect Classification Dashboard (ResNet18)
# Displays: predicted class, confidence, Grad-CAM heatmap overlay.

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
from pathlib import Path
import random

# -- Project root --
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.wafer_map_generator import generate_wafer_map, DEFECT_CLASSES

# -- Page config --
st.set_page_config(
    page_title="Wafer Defect Classifier",
    page_icon="🔬",
    layout="wide"
)

# -- CSS Styling (Using joined lines to avoid triple-quote tokenizer issues) --
css_style = (
    "<style>\n"
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');\n"
    "html, body, [class*='css'] { font-family: 'Inter', sans-serif; }\n"
    ".main-title { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
    "-webkit-background-clip: text; -webkit-text-fill-color: transparent; "
    "font-size: 2.2rem; font-weight: 700; }\n"
    ".result-box { padding: 20px; border-radius: 14px; text-align: center; "
    "font-size: 20px; font-weight: 700; margin: 8px 0; }\n"
    ".defect-found { background: linear-gradient(135deg, #ff5252, #d32f2f); color: white; }\n"
    ".wafer-clear { background: linear-gradient(135deg, #69f0ae, #2e7d32); color: white; }\n"
    ".confidence-high { color:#2e7d32; font-weight:700; }\n"
    ".confidence-mid  { color:#f57c00; font-weight:700; }\n"
    ".confidence-low  { color:#d32f2f; font-weight:700; }\n"
    "</style>"
)
st.markdown(css_style, unsafe_allow_html=True)

# -- Model Loading --
@st.cache_resource
def load_classifier():
    """Load trained ResNet18 model"""
    model_path = PROJECT_ROOT / "models" / "checkpoints" / "resnet18_best.pth"
    if not model_path.exists():
        return None, False
    try:
        from deployment.inference_api import load_model
        model = load_model(str(model_path))
        return model, True
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, False

model, model_loaded = load_classifier()

# -- Sidebar --
st.sidebar.title("🔬 Wafer Analytics")
if model_loaded:
    st.sidebar.success("✅ ResNet18 Loaded")
else:
    st.sidebar.error("❌ Model Not Found")
    st.sidebar.caption("Run: python training/train_classifier.py")

st.sidebar.markdown("---")
st.sidebar.markdown("**Defect Classes:**")
for i, cls in enumerate(DEFECT_CLASSES):
    st.sidebar.markdown(f"`{i}` - {cls}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack:**")
st.sidebar.markdown("- ResNet18 (ImageNet pretrained)")
st.sidebar.markdown("- Focal Loss (gamma=2.0)")
st.sidebar.markdown("- Grad-CAM Explainability")

# -- Main Header --
st.markdown('<p class="main-title">🧬 Wafer Defect Classification</p>', unsafe_allow_html=True)
st.markdown("Industry-grade macro-level defect detection with **Grad-CAM explainability**.")

# -- Mode selection --
mode = st.radio(
    "Select detection mode",
    ["Upload Image / Synthetic", "Real-time Camera"],
    horizontal=True
)

st.markdown("---")

# Layout columns
col_input, col_result, col_heatmap = st.columns([1, 1, 1])

# Image variable initialized
use_image = None

# -- Camera Mode --
if mode == "Real-time Camera":
    with col_input:
        st.subheader("📷 Camera Input")
        camera_image = st.camera_input("Take a picture of wafer")
        if camera_image:
            use_image = np.array(Image.open(camera_image).convert("RGB"))

# -- Upload / Synthetic Mode --
elif mode == "Upload Image / Synthetic":
    with col_input:
        st.subheader("📥 Input")
        uploaded_file = st.file_uploader("Upload wafer image", type=["jpg", "png", "jpeg", "bmp"])
        synth_class = st.selectbox("Or generate synthetic:", ["(random)"] + DEFECT_CLASSES)
        if st.button("🎲 Generate Sample"):
            cls = random.choice(DEFECT_CLASSES) if synth_class == "(random)" else synth_class
            img = generate_wafer_map(cls, augment=True)
            st.session_state["cls_img"] = img
            st.session_state["cls_gt"] = cls

    # Resolve use_image
    if uploaded_file:
        use_image = np.array(Image.open(uploaded_file).convert("RGB"))
        with col_input:
            st.image(use_image, caption="Uploaded", use_container_width=True)
    elif "cls_img" in st.session_state:
        use_image = st.session_state["cls_img"]
        gt = st.session_state.get("cls_gt", "?")
        with col_input:
            st.image(use_image, caption=f"Synthetic ({gt})", use_container_width=True)

# -- Prediction --
if use_image is not None:
    if not model_loaded:
        st.warning("⚠️ No trained model found. Train it first!")
    else:
        from deployment.inference_api import predict
        with st.spinner("Analyzing wafer..."):
            result = predict(use_image, model=model, generate_heatmap=True)
        
        pred_class = result["class"]
        confidence = result["confidence"]
        all_probs = result["all_probs"]
        overlay = result["overlay"]

        # -- Result Column --
        with col_result:
            st.subheader("📊 Prediction")
            if pred_class == "normal":
                st.markdown('<div class="result-box wafer-clear">✅ NORMAL</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box defect-found">⚠️ {pred_class.upper()}</div>', unsafe_allow_html=True)

            css = "confidence-high" if confidence > 0.8 else ("confidence-mid" if confidence > 0.5 else "confidence-low")
            st.markdown(f'<p class="{css}">Confidence: {confidence * 100:.1f}%</p>', unsafe_allow_html=True)

            st.markdown("**Class Probabilities:**")
            prob_data = dict(sorted(all_probs.items(), key=lambda x: -x[1]))
            st.bar_chart(prob_data)

        # -- Heatmap Column --
        with col_heatmap:
            st.subheader("🔥 Grad-CAM")
            if overlay is not None:
                st.image(overlay, caption="Attention Heatmap", use_container_width=True)
                st.caption("Red = high attention.")
            else:
                st.info("Heatmap not available.")