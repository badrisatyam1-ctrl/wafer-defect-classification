"""
Full Streamlit website for macro-level wafer defect classification.

The site is built around the production inference engine so the UI, local
testing flow, and future website API all use the same prediction contract.
"""

from __future__ import annotations

import json
import sys
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deployment.inference import WaferInferenceEngine
from models.resnet18_classifier import DEFECT_CLASSES_V2
from utils.synthetic_generator import generate_macro_wafer_map
from deployment.wafer_detector import detect_wafer

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "resnet18_best.pt"


def _image_to_png_bytes(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


def _sorted_probabilities(class_probabilities: Dict[str, float]):
    return sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)


def _checkpoint_signature() -> int:
    return CHECKPOINT_PATH.stat().st_mtime_ns if CHECKPOINT_PATH.exists() else 0


@st.cache_data(show_spinner=False)
def load_checkpoint_summary(signature: int) -> Optional[Dict[str, Any]]:
    if not signature:
        return None
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config_payload = checkpoint.get("config") or {}
    return {
        "epoch": checkpoint.get("epoch"),
        "best_macro_f1": checkpoint.get("best_macro_f1"),
        "split_mode": config_payload.get("split_mode", checkpoint.get("split_mode")),
        "loss_name": config_payload.get("loss_name", checkpoint.get("loss_name")),
        "preprocessing": checkpoint.get("preprocessing", {}),
        "class_names": checkpoint.get("class_names", DEFECT_CLASSES_V2),
        "is_valid": all(
            key in checkpoint for key in ("model_state_dict", "preprocessing", "class_names", "config")
        ) and config_payload.get("split_mode") in {"lot", "time"},
    }


@st.cache_resource(show_spinner=False)
def load_engine(signature: int) -> WaferInferenceEngine:
    if not signature:
        raise FileNotFoundError(
            f"Missing checkpoint at {CHECKPOINT_PATH}. Train the classifier first."
        )
    return WaferInferenceEngine(checkpoint_path=CHECKPOINT_PATH)


def get_engine_and_error():
    signature = _checkpoint_signature()
    if not signature:
        return None, "Model checkpoint not found. Run `python training/train_resnet.py` first."
    try:
        return load_engine(signature), None
    except Exception as exc:
        return None, f"Checkpoint found but could not be loaded: {exc}"


def is_full_fail(pred, confidence):
    return pred == "full_fail" and confidence > 0.7


def run_prediction(engine: WaferInferenceEngine, frame: np.ndarray):
    detected, box = detect_wafer(frame)

    if not detected:
        st.error("❌ No wafer detected. Please upload a valid wafer image.")
        st.stop()

    # crop wafer
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]

    # Pass cropped wafer region to the classifier
    result = engine.predict_from_array(crop)
    
    # Catastrophic Fail Detection Override
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = list(result["class_probabilities"].values())
    
    if is_full_fail(pred_class, confidence):
        result["predicted_class"] = "full_fail"
        st.error("🚨 FULL WAFER FAILURE DETECTED")
    
    st.session_state["analysis_result"] = result
    
    # Store explicit values for OpenAI Chatbot
    st.session_state["prediction"] = result["predicted_class"]
    st.session_state["confidence"] = result["confidence"]
    
    # Clear history on new wafer analysis
    if "chat_history" in st.session_state:
        st.session_state["chat_history"] = []


def render_sidebar(checkpoint_summary: Optional[Dict[str, Any]], engine_error: Optional[str]):
    st.sidebar.markdown("## ⚙️ Model Status")
    if engine_error:
        st.sidebar.error("❌ " + engine_error)
        st.sidebar.code(
            "python training/train_resnet.py --dataset-npz path\\to\\wafer_dataset.npz --split-mode lot",
            language="powershell",
        )
    else:
        st.sidebar.success("✅ Engine Active")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏷️ Defect Classes")
    
    # Render classes as styled pills
    pills_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
    for class_name in DEFECT_CLASSES_V2:
        color = "#22C55E" if class_name == "normal" else "#EF4444"
        bg_color = "rgba(34, 197, 94, 0.15)" if class_name == "normal" else "rgba(239, 68, 68, 0.15)"
        pills_html += f'<span style="background: {bg_color}; color: {color}; padding: 4px 10px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; border: 1px solid {color}30;">{class_name.title()}</span>'
    pills_html += '</div>'
    st.sidebar.markdown(pills_html, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Deployment Notes")
    st.sidebar.markdown(
        "- Full-wafer input only\n"
        "- Lot/time validation split\n"
        "- Weighted loss for imbalance\n"
        "- Grad-CAM explainability"
    )

    if checkpoint_summary:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💾 Checkpoint")
        epoch = checkpoint_summary.get("epoch")
        best_macro_f1 = checkpoint_summary.get("best_macro_f1")
        split_mode = checkpoint_summary.get("split_mode", "unknown")
        loss_name = checkpoint_summary.get("loss_name", "unknown")
        preprocessing = checkpoint_summary.get("preprocessing", {})
        is_valid = checkpoint_summary.get("is_valid", False)
        
        st.sidebar.markdown(f"**Validated:** `{'Yes' if is_valid else 'No'}`")
        st.sidebar.markdown(f"**Epoch:** `{epoch if epoch is not None else 'n/a'}`")
        if best_macro_f1 is not None:
            st.sidebar.metric("🏆 Best macro F1", f"{best_macro_f1:.4f}")
        st.sidebar.markdown(f"**Split mode:** `{split_mode}`")
        st.sidebar.markdown(f"**Loss:** `{loss_name}`")
        st.sidebar.markdown(
            f"**Input size:** `{preprocessing.get('input_size', 'n/a')}`"
        )


def render_header():
    # Dark modern theme custom CSS
    st.markdown(
        """
<style>
    /* Main Dark Theme background */
    .stApp, .stApp > header {
        background-color: #0E1117;
        color: #F8FAFC;
        font-family: "Inter", "Segoe UI", sans-serif;
    }
    
    /* Hide top header line */
    header {visibility: hidden;}
    
    /* Sidebar dark background */
    [data-testid="stSidebar"] {
        background-color: #1A1F2B !important;
        border-right: 1px solid #2D3748 !important;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    
    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Hero section */
    .hero {
        padding: 2rem;
        background: #1A1F2B;
        border: 1px solid #2D3748;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
        border-top: 4px solid #3B82F6;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        max-width: 800px;
        line-height: 1.6;
    }
    .tag-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .hero-tag {
        background: #2D3748;
        color: #E2E8F0;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #4A5568;
    }
    
    /* Make the 3 columns look like cards */
    [data-testid="column"] {
        background: #1A1F2B;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .panel-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #FFFFFF;
        border-bottom: 1px solid #2D3748;
        padding-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Prediction Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        width: 100%;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .badge-normal {
        background: rgba(34, 197, 94, 0.1);
        color: #4ADE80;
        border: 2px solid #22C55E;
    }
    .badge-defect {
        background: rgba(239, 68, 68, 0.1);
        color: #F87171;
        border: 2px solid #EF4444;
    }
    
    /* Metric boxes */
    .metric-box {
        background: #252D3D;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #2D3748;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .metric-value.warning {
        color: #FBBF24;
    }
    .metric-value.good {
        color: #4ADE80;
    }
    
    /* Overrides for Streamlit elements to fit dark theme better */
    .stRadio label { font-weight: 600 !important; color: #E2E8F0 !important;}
    .stSelectbox label { font-weight: 600 !important; color: #E2E8F0 !important;}
    .stFileUploader label { font-weight: 600 !important; color: #E2E8F0 !important;}
    
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="hero">
  <div class="hero-title">Wafer Yield Analytics Console</div>
  <div class="hero-subtitle">
    Industrial-grade full-wafer macro defect classification. Powered by a ResNet18 backbone, 
    imbalance-aware training, and Grad-CAM explainability overlay.
  </div>
  <div class="tag-container">
    <span class="hero-tag">🔬 Full-wafer only</span>
    <span class="hero-tag">🧠 ResNet18 backbone</span>
    <span class="hero-tag">📡 Grad-CAM viz</span>
    <span class="hero-tag">⚡ Edge Inference</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_prediction_badge(class_name: str):
    is_normal = (class_name.lower() == "normal")
    badge_class = "badge-normal" if is_normal else "badge-defect"
    icon = "✅" if is_normal else "⚠️"
    
    st.markdown(
        f'<div class="badge {badge_class}">{icon} {class_name.replace("_", " ")}</div>',
        unsafe_allow_html=True
    )


def render_metric_box(title: str, value: str, state_class: str = ""):
    st.markdown(
        f'''
        <div class="metric-box">
            <div class="metric-title">{title}</div>
            <div class="metric-value {state_class}">{value}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        page_title="Wafer Yield Analytics",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    render_header()

    checkpoint_summary = load_checkpoint_summary(_checkpoint_signature())
    engine, engine_error = get_engine_and_error()
    render_sidebar(checkpoint_summary, engine_error)
    
    if engine is None:
        st.error(engine_error or "Model is not ready.")
        st.info("Wait for a valid model checkpoint to be generated.")
        return

    # 3-column Layout Setup
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    
    # ─── COLUMN 1: INPUT AREA ─────────────────────────────────────────────────────────
    with col1:
        st.markdown('<div class="panel-title">📷 Input Source</div>', unsafe_allow_html=True)
        
        mode = st.radio(
            "Detection Mode", 
            ["Upload Image", "Synthetic Generation", "Real-time Camera"], 
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
        if mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload full map", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = np.array(Image.open(uploaded_file).convert("RGB"))
                st.session_state["input_image"] = image
                st.session_state["input_source"] = "Uploaded wafer"
                st.image(image, use_container_width=True, caption="Source Image")
                run_prediction(engine, image)
                
        elif mode == "Synthetic Generation":
            selected_class = st.selectbox("Select macro defect type", DEFECT_CLASSES_V2)
            if st.button("Generate & Test", use_container_width=True, type="primary"):
                synth_image = generate_macro_wafer_map(selected_class, size=(256, 256))
                if len(synth_image.shape) == 2:
                    synth_image = cv2.cvtColor(synth_image, cv2.COLOR_GRAY2RGB)
                st.session_state["input_image"] = synth_image
                st.session_state["input_source"] = f"Synthetic: {selected_class}"
            
            if "input_image" in st.session_state and "Synthetic" in st.session_state.get("input_source", ""):
                st.image(st.session_state["input_image"], use_container_width=True, caption=st.session_state["input_source"])
                run_prediction(engine, st.session_state["input_image"])
                
        elif mode == "Real-time Camera":
            st.info("Ensure good lighting over the wafer sample before capturing.")
            camera_image = st.camera_input("Capture live wafer")
            if camera_image:
                image = np.array(Image.open(camera_image).convert("RGB"))
                st.session_state["input_image"] = image
                st.session_state["input_source"] = "Live Camera"
                
                # YOLOv8 Wafer Gatekeeper for Real-time Camera
                detected, box = detect_wafer(image)
                
                if not detected:
                    cv2.putText(image, "No Wafer Detected", (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    st.image(image)
                    st.stop()
                else:
                    # Only run the heavy defect model if a valid wafer is found
                    run_prediction(engine, image)


    # Check if we have results to display in col2 and col3
    has_results = "analysis_result" in st.session_state

    # ─── COLUMN 2: PREDICTION RESULTS ────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="panel-title">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        if not has_results:
            st.markdown(
                '<div style="text-align: center; padding: 3rem 1rem; color: #64748B;">'
                '<i>Waiting for wafer input...</i></div>',
                unsafe_allow_html=True
            )
        else:
            result = st.session_state["analysis_result"]
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            class_probs = result["class_probabilities"]
            sorted_probs = _sorted_probabilities(class_probs)
            top_two_gap = sorted_probs[0][1] - sorted_probs[1][1] if len(sorted_probs) > 1 else sorted_probs[0][1]
            
            # Big Badge
            render_prediction_badge(predicted_class)
            
            # Metrics
            mcol1, mcol2, mcol3 = st.columns(3)
            conf_state = "good" if confidence > 0.80 else ("warning" if confidence >= 0.50 else "")
            
            # Severity Scoring Logic
            severity = "None"
            sev_state = "good"
            if predicted_class != "normal":
                if confidence > 0.85:
                    severity = "High"
                    sev_state = "bad"
                elif confidence >= 0.60:
                    severity = "Medium"
                    sev_state = "warning"
                else:
                    severity = "Low Confidence"
                    sev_state = "warning"
            
            with mcol1:
                render_metric_box("Confidence", f"{confidence:.1%}", conf_state)
            with mcol2:
                render_metric_box("Top-2 Margin", f"{top_two_gap:.1%}")
            with mcol3:
                render_metric_box("Severity", severity, sev_state)
                
            if confidence > 0.80:
                st.success("✅ Strong prediction message. High model confidence.")
            elif confidence >= 0.50:
                st.info("ℹ️ Moderate certainty. Consider manual review.")
            else:
                st.warning("⚠️ Model is uncertain.")
                
            st.markdown("---")
            st.markdown("#### Class Probabilities")
            
            # Improved chart
            for c_name, prob in sorted_probs:
                if prob < 0.01 and c_name != predicted_class:
                    continue # Skip very low probs to reduce clutter
                
                bar_color = "#22C55E" if c_name == "normal" else "#3B82F6"
                if c_name == predicted_class and c_name != "normal":
                    bar_color = "#EF4444"
                
                st.markdown(
                    f'''
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.9rem;">
                        <span>{c_name.replace("_", " ").title()}</span>
                        <span style="font-weight: 600;">{prob:.1%}</span>
                    </div>
                    <div style="width: 100%; background-color: #2D3748; border-radius: 4px; margin-bottom: 12px; height: 10px;">
                        <div style="width: {prob*100}%; background-color: {bar_color}; height: 100%; border-radius: 4px;"></div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )


    # ─── COLUMN 3: GRAD-CAM VIZ ──────────────────────────────────────────────────────
    with col3:
        st.markdown('<div class="panel-title">🎯 Class Activation Map</div>', unsafe_allow_html=True)
        
        if not has_results:
            st.markdown(
                '<div style="text-align: center; padding: 3rem 1rem; color: #64748B;">'
                '<i>Waiting for wafer input...</i></div>',
                unsafe_allow_html=True
            )
        else:
            result = st.session_state["analysis_result"]
            overlay = result["overlay"]
            heatmap_rgb = _heatmap_to_rgb(result["heatmap"])
            
            tab1, tab2 = st.tabs(["Overlay", "Raw Heatmap"])
            
            with tab1:
                st.image(overlay, use_container_width=True)
                st.markdown(
                    "<p style='font-size: 0.85rem; color: #94A3B8; margin-top: 0.5rem;'>"
                    "Hot areas (red/yellow) indicate where the model focuses to determine the defect class.</p>",
                    unsafe_allow_html=True
                )
            with tab2:
                st.image(heatmap_rgb, use_container_width=True)

    # ─── WAFER INTELLIGENCE CHATBOT ───────────────────────────────────────────────────
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("## 🤖 AI Wafer Assistant")

    if "latest_q" in st.session_state:
        with st.chat_message("user"):
            st.markdown(st.session_state["latest_q"])
        with st.chat_message("assistant"):
            st.markdown(st.session_state["latest_a"])

    if user_input := st.chat_input("Ask about this wafer..."):
        prediction = st.session_state.get("prediction")

        if not prediction:
            st.warning("Please upload or capture a wafer image first.")
        else:
            st.session_state["latest_q"] = user_input
            with st.chat_message("user"):
                st.markdown(user_input)

            try:
                from deployment.chatbot import generate_wafer_response
                answer = generate_wafer_response(user_input)
                st.session_state["latest_a"] = answer
                with st.chat_message("assistant"):
                    st.markdown(answer)
            except Exception as e:
                st.error(f"Error generating AI response: {e}")

if __name__ == "__main__":
    main()
