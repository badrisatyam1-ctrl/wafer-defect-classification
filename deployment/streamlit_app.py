# -*- coding: utf-8 -*-
# Wafer Defect Classification Dashboard (ResNet18) - Stable Version v2.2
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

# Imports
from utils.wafer_map_generator import generate_wafer_map, DEFECT_CLASSES
from deployment.inference_api import load_model, predict

# -- Page config --
st.set_page_config(
    page_title='Wafer Defect Classifier',
    page_icon='🔬',
    layout='wide'
)

# -- CSS Styling --
css_style = (
    '<style>\n'
    '@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap");\n'
    'html, body, [class*="css"] { font-family: "Inter", sans-serif; }\n'
    '.main-title { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
    '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
    'font-size: 2.2rem; font-weight: 700; }\n'
    '.result-box { padding: 20px; border-radius: 14px; text-align: center; '
    'font-size: 20px; font-weight: 700; margin: 8px 0; }\n'
    '.defect-found { background: linear-gradient(135deg, #ff5252, #d32f2f); color: white; }\n'
    '.wafer-clear { background: linear-gradient(135deg, #69f0ae, #2e7d32); color: white; }\n'
    '.confidence-high { color:#2e7d32; font-weight:700; }\n'
    '.confidence-mid  { color:#f57c00; font-weight:700; }\n'
    '.confidence-low  { color:#d32f2f; font-weight:700; }\n'
    '</style>'
)
st.markdown(css_style, unsafe_allow_html=True)

# -- Top Level Initialization --
use_image = None
model = None
model_loaded = False

# -- Load Model (No caching to avoid RuntimeError with Torch on Windows) --
model_path = PROJECT_ROOT / 'models' / 'checkpoints' / 'resnet18_best.pth'
if model_path.exists():
    try:
        model = load_model(str(model_path))
        model_loaded = True
    except Exception as e:
        st.sidebar.error(f'Model Error: {e}')

# -- Sidebar --
st.sidebar.title('🔬 Wafer Analytics')
st.sidebar.caption('v2.2 Stable Branch')

if model_loaded:
    st.sidebar.success('✅ ResNet18 Loaded')
else:
    st.sidebar.error('❌ Model Not Found')
    st.sidebar.caption('Run: python training/train_classifier.py')

st.sidebar.markdown('---')
st.sidebar.markdown('**Defect Classes:**')
for i, cls in enumerate(DEFECT_CLASSES):
    st.sidebar.markdown(f'`{i}` - {cls}')

# -- Main Header --
st.markdown('<p class="main-title">🧬 Wafer Defect Classification</p>', unsafe_allow_html=True)
st.markdown('Industry-grade macro-level defect detection.')

mode = st.radio(
    'Select detection mode',
    ['Upload Image / Synthetic', 'Real-time Camera'],
    horizontal=True,
    key='dashboard_mode'
)

st.markdown('---')

# Setup layout globally
col_input, col_result, col_heatmap = st.columns([1, 1, 1])

# -- MODE LOGIC --
if mode == 'Real-time Camera':
    with col_input:
        st.subheader('📷 Camera Input')
        camera_image = st.camera_input('Take a picture of wafer', key='cam_input')
        if camera_image:
            use_image = np.array(Image.open(camera_image).convert('RGB'))

elif mode == 'Upload Image / Synthetic':
    with col_input:
        st.subheader('📥 Input')
        uploaded_file = st.file_uploader('Upload wafer image', type=['jpg', 'png', 'jpeg', 'bmp'], key='file_input')
        
        synth_class = st.selectbox('Or generate synthetic:', ['(random)'] + DEFECT_CLASSES, key='synth_sel')
        if st.button('🎲 Generate Sample', key='synth_btn'):
            sel_cls = random.choice(DEFECT_CLASSES) if synth_class == '(random)' else synth_class
            st.session_state['cls_img'] = generate_wafer_map(sel_cls, augment=True)
            st.session_state['cls_gt'] = sel_cls

    if uploaded_file:
        use_image = np.array(Image.open(uploaded_file).convert('RGB'))
        with col_input:
            st.image(use_image, caption='Uploaded', use_container_width=True)
    elif 'cls_img' in st.session_state:
        use_image = st.session_state['cls_img']
        gt_cls = st.session_state.get('cls_gt', '?')
        with col_input:
            st.image(use_image, caption=f'Synthetic ({gt_cls})', use_container_width=True)

def wafer_present(image):
    '''Simple heuristic to detect if a wafer is in the frame using area coverage'''
    if image is None: return False
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return False
        max_area = max(cv2.contourArea(c) for c in contours)
        total_area = image.shape[0] * image.shape[1]
        return (max_area / total_area) > 0.1
    except:
        return True # Fallback to True on error to avoid blocking

# -- PREDICTION & RESULTS --
if use_image is not None:
    if not model_loaded:
        st.warning('⚠️ Model not found. Train it first!')
    elif not wafer_present(use_image):
        st.warning('⚠️ No wafer detected in the frame. Please point the camera at a wafer or upload a valid wafer image!')
    else:
        with st.spinner('Analyzing...'):
            result = predict(use_image, model=model, generate_heatmap=True)
        
        # Display Results
        with col_result:
            st.subheader('📊 Prediction')
            p_class = result['class']
            conf = result['confidence']
            
            # Badge
            if p_class == 'normal':
                st.markdown('<div class="result-box wafer-clear">✅ NORMAL</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box defect-found">⚠️ {p_class.upper()}</div>', unsafe_allow_html=True)

            # Confidence
            clr = 'confidence-high' if conf > 0.8 else ('confidence-mid' if conf > 0.5 else 'confidence-low')
            st.markdown(f'<p class="{clr}">Confidence: {conf * 100:.1f}%</p>', unsafe_allow_html=True)

            # Probabilities
            probs = dict(sorted(result['all_probs'].items(), key=lambda x: -x[1]))
            st.bar_chart(probs)

        with col_heatmap:
            st.subheader('🔥 Grad-CAM')
            if result.get('overlay') is not None:
                st.image(result['overlay'], caption='Attention Heatmap', use_container_width=True)
            else:
                st.info('Heatmap not generated.')