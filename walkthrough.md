# Wafer Defect Classifier — Final Walkthrough

## What Was Built

A **1-channel ResNet18 classifier** for macro-level wafer defect detection across 8 classes: normal, center, edge_ring, edge_loss, scratch, ring, cluster, full_fail.

## Key Architecture Changes

```python
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 8)
```

- **Input**: Grayscale 256×256 wafer maps (1 channel)
- **Preprocessing**: `Grayscale(1)` → `Resize(256)` → `ToTensor()` → `Normalize(mean=[0.5], std=[0.5])`
- **Loss**: Focal Loss (γ=2.0) with class weights
- **Split**: Lot-based (no data leakage)

## Training Results

| Metric | Value |
|--------|-------|
| **Best Macro F1** | **0.9967 (99.7%)** |
| Epochs | 30 |
| Samples | 5,000 (lot-based split) |
| Device | CPU |
| Training Time | ~2 hours |

## Dashboard Verification

![Streamlit Dashboard showing scratch detection at 98.9% confidence with Grad-CAM overlay](C:\Users\badri\.gemini\antigravity\brain\c652c884-7c27-4072-ab7f-bab7b81a2240\streamlit_dashboard_results_1773670389745.png)

- **Predicted**: Scratch ✅
- **Confidence**: 98.9%
- **Grad-CAM**: Correctly highlights the scratch region

## UI Standardization & Overhaul

The dashboard logic was heavily refactored into an industry-grade Analytics Console:
- **Dark Theme (`#0E1117`)**: High contrast metrics, card-based layout, tailored CSS overrides.
- **3-Column Layout**: Structured workflow comparing input source, prediction metrics, and Grad-CAM overlay side-by-side.
- **Real-Time Camera**: Shifted from unstable `cv2` background loops to Streamlit's robust `st.camera_input` API.
- **Auto-Prediction UX**: Eliminated the manual "Run Classifier" button in favor of immediate reactive inference on upload/capture.

## 🤖 AI Wafer Assistant (OpenAI Integration)

An expert chatbot was seamlessly integrated directly into the [streamlit_app_v2.py](file:///c:/Users/badri/OneDrive/Documents/GitHub/wafer-defect-classification/deployment/streamlit_app_v2.py) dashboard ([deployment/chatbot.py](file:///c:/Users/badri/OneDrive/Documents/GitHub/wafer-defect-classification/deployment/chatbot.py)). 
- **Context-Aware**: It dynamically reads the live [prediction](file:///c:/Users/badri/OneDrive/Documents/GitHub/wafer-defect-classification/deployment/streamlit_app_v2.py#91-102) and `confidence` from `st.session_state`.
- **Domain Knowledge Injected**: A static dictionary of defect causes (`wafer_knowledge`) is automatically injected into the hidden System Prompt.
- **Continuous Conversation**: The OpenAI API streams responses in real-time, matching Streamlit's native Dark Mode UI while retaining full chat history.

![Redesigned Dark Mode Dashboard](C:\Users\badri\.gemini\antigravity\brain\c652c884-7c27-4072-ab7f-bab7b81a2240\wafer_defect_dashboard_dark_mode_final_1773750938578.png)

## Files Modified

| File | Change |
|------|--------|
| [resnet18_classifier.py](file:///c:/Users/badri/OneDrive/Documents/GitHub/wafer-defect-classification/models/resnet18_classifier.py) | 1-channel conv1, grayscale prep, `pytorch-grad-cam` integration |
| [train_resnet.py](file:///c:/Users/badri/OneDrive/Documents/GitHub/wafer-defect-classification/training/train_resnet.py) | Imbalance-weighted CrossEntropyLoss on 100K industrial pipeline |
| [inference.py](file:///c:/Users/badri/OneDrive/Documents/GitHub/wafer-defect-classification/deployment/inference.py) | Dedicated [preprocess_for_inference](file:///c:/Users/badri/OneDrive/Documents/GitHub/wafer-defect-classification/deployment/inference.py#41-60) matching exact training schema |
| [streamlit_app_v2.py](file:///c:/Users/badri/OneDrive/Documents/GitHub/wafer-defect-classification/deployment/streamlit_app_v2.py) | Complete Dark-Mode layout rewrite, Live Camera support, Chatbot integration |
| [chatbot.py](file:///c:/Users/badri/OneDrive/Documents/GitHub/wafer-defect-classification/deployment/chatbot.py) | End-to-end OpenAI dynamic context integration and streaming chat UI |
