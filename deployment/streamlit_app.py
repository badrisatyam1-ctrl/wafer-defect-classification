"""
Wafer Defect Segmentation Dashboard (U-Net)
Visualizes defect localization using Research-Standard U-Net masks.
"""
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet_model import dice_loss, dice_coefficient
from utils.synthetic_generator import generate_wafer_mask_pair
import random

st.set_page_config(page_title="Wafer Segmentation Research", page_icon="🧬", layout="wide")

# CSS
st.markdown("""
<style>
    .metric-box { padding: 15px; border-radius: 10px; background: #e3f2fd; color: #1565c0; text-align: center; }
    .metric-val { font-size: 24px; font-weight: bold; }
    .def-found { color: #d32f2f; font-weight: bold; }
    .def-clear { color: #388e3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_unet():
    model_path = Path("models/checkpoints/unet_best.keras")
    if model_path.exists():
        try:
            # Must Load Custom Objects for Dice Loss
            custom_objects = {'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient}
            model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
            return model, True
        except Exception as e:
            st.error(f"Load Error: {e}")
    return None, False

model, model_loaded = load_unet()

# SIDEBAR
st.sidebar.title("🔬 Research Lab")
st.sidebar.info(f"Model: {'🟢 U-Net Loaded' if model_loaded else '🔴 Training Required'}")
threshold = st.sidebar.slider("Mask Threshold", 0.1, 0.9, 0.5)

st.title("🧬 Wafer Defect Segmentation (U-Net)")
st.markdown("Locates defect pixels using **semantic segmentation** instead of simple classification.")

col1, col2, col3 = st.columns(3)

# INPUT
with col1:
    st.subheader("1. Input Wafer")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    
    if st.button("Generate Synthetic Sample"):
        cls = random.choice(['Scratch', 'Donut', 'Edge-Ring', 'Loc', 'Random'])
        img, true_mask = generate_wafer_mask_pair(cls)
        st.session_state['seg_img'] = img
        st.session_state['seg_true_mask'] = true_mask
        st.toast(f"Generated: {cls}")

    use_image = None
    if 'seg_img' in st.session_state and not uploaded_file:
        use_image = st.session_state['seg_img']
        st.image(use_image, caption="Synthetic Input", use_column_width=True)
    elif uploaded_file:
        use_image = np.array(Image.open(uploaded_file).convert('RGB'))
        st.image(use_image, caption="Uploaded Input", use_column_width=True)

# PREDICTION
if use_image is not None and model_loaded:
    # Preprocess
    img_resized = cv2.resize(use_image, (128, 128))
    # Normalize [0,1] - U-Net expects this if trained on it
    img_norm = img_resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_norm, axis=0)
    
    # Predict Mask
    pred_mask = model.predict(input_tensor, verbose=0)[0] # (128, 128, 1)
    
    # Threshold
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    
    # Color Overlay
    overlay = img_resized.copy()
    # Red overlay on defect pixels
    overlay[binary_mask[:,:,0] == 255] = [255, 0, 0] 
    
    with col2:
        st.subheader("2. Predicted Mask")
        st.image(binary_mask, caption=f"Segmentation Mask (>{threshold})", use_column_width=True, clamp=True)
        
    with col3:
        st.subheader("3. Defect Overlay")
        st.image(overlay, caption="Localization", use_column_width=True)
        
    # METRICS
    st.markdown("---")
    defect_pixels = np.sum(binary_mask > 0)
    total_pixels = 128*128
    defect_area = (defect_pixels / total_pixels) * 100
    
    m1, m2 = st.columns(2)
    m1.metric("Defect Area", f"{defect_area:.2f}%")
    
    if defect_area > 0.5:
        m2.markdown(f'<div class="metric-box def-found">DEFECT DETECTED</div>', unsafe_allow_html=True)
    else:
        m2.markdown(f'<div class="metric-box def-clear">WAFER CLEAR</div>', unsafe_allow_html=True)

elif not model_loaded:
    st.warning("⚠️ Model not found. Runs `python training/train_segmentation.py` first.")
