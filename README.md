# 🧬 WaferOS — Semiconductor Yield Analytics Studio & Defect Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Macro--F1-93.51%25-success?logo=target" />
  <img src="https://img.shields.io/badge/Inference-~12ms-brightgreen?logo=speedtest" />
  <img src="https://img.shields.io/badge/XAI-Multi--Scale%20Grad--CAM%2B%2B-orange" />
  <img src="https://img.shields.io/badge/FastAPI-Production-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/UI-Glassmorphic%20Studio-6C5CE7" />
</p>

<p align="center">
  <b>Enterprise-grade semiconductor wafer defect classification, explainability, and yield intelligence platform.</b><br/>
  Featuring ResNet-18 Backbone (93.51% Macro-F1), Multi-Scale Grad-CAM++ Attribution, Neural Circularity Gating, and Fab Analytics Studio.
</p>

> [!IMPORTANT]
> **Proprietary Notice & Technical Showcase Repository**:
> This repository is published as an architectural showcase and machine learning engineering portfolio. The trained production checkpoint weights (`resnet18_best.pt`) and proprietary semiconductor fab datasets are confidential intellectual property and are intentionally withheld from public distribution. The project includes an automated gate preventing execution without licensed weights. For corporate evaluations, partnerships, or live demonstrations, please contact the author.

---

## 📌 Executive Summary & Production Benchmarks

Semiconductor microchip manufacturing demands near-zero defect escape rates. Wafer map macro-defect signatures (rings, scratches, edge losses, localized clusters) pinpoint specific chamber degradation, polishing malfunctions, or thermal anomalies. 

WaferOS delivers an end-to-end automated defect classification and yield intelligence platform achieving **93.51% validation Macro-F1** across 8 semiconductor defect taxonomies with ultra-low latency inference (~12ms).

### 🏭 ResNet-18 Production Model Benchmarks

| Metric | Production Verified Value | Engineering Significance |
|---|---|---|
| **Architecture** | **ResNet-18 Deep Convolutional Backbone** | 512-dimensional semantic latent feature space |
| **Validation Macro-F1** | **93.51% (0.9351)** | Robustly balanced across both common & rare defect classes |
| **Validation Accuracy** | **93.51%** | Evaluated under genuine wafer fabrication process variance |
| **Inference Latency** | **~12ms / wafer** | Fully compatible with high-speed automated optical inspection (AOI) lines |
| **XAI Resolution** | **$14 \times 14 \to 224 \times 224$** | 4x spatial resolution via Layer3 + Layer4 Grad-CAM++ fusion |
| **Wafer Presence Gating**| **Convex Hull Circularity Engine** | Blocks faces, hands, and cleanroom background clutter |
| **Serving Framework** | **FastAPI Asynchronous Daemon** | RESTful endpoints for single/batch prediction, PDF audit, & LLM chat |
| **Interface** | **WaferOS v2.5 Glassmorphic Studio** | Industrial dark-mode fab cockpit with real-time video, uploads, & analytics |

---

## 🏗️ System Architecture

```
                               ┌──────────────────────────────────────────────┐
                               │       Wafer Ingestion Sources                │
                               │  - High-Res Optical Inspection (AOI) Upload │
                               │  - Real-Time Camera Stream / Reticle        │
                               │  - Procedural Defect Pattern Synthesizer     │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │   Wafer Presence & Gating Engine             │
                               │   (deployment/wafer_detector.py)             │
                               │   - YCrCb Human Skin Tone Rejection          │
                               │   - Morphological Decoupling (cv2.MORPH_OPEN)│
                               │   - Convex Hull Circularity & MinCircle Fill │
                               │   - Rejection: NO_WAFER_DETECTED             │
                               └──────────────────────┬───────────────────────┘
                                                      │ (Passed)
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │   Wafer Preprocessing Pipeline               │
                               │   - Aspect-Ratio Preserving Square Pad       │
                               │   - 224x224 Bilinear Interpolation           │
                               │   - Channel Normalization (μ=0.5, σ=0.5)     │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │   ResNet-18 Deep Convolutional Backbone      │
                               │   - Conv1 (7x7, stride 2) + MaxPool          │
                               │   - Layer1 (64ch, 56x56)                     │
                               │   - Layer2 (128ch, 28x28)                    │
                               │   - Layer3 (256ch, 14x14) ──┐ (Spatial Cam)  │
                               │   - Layer4 (512ch, 7x7)   ──┼ (Semantic Cam) │
                               │   - Global Average Pooling  │                │
                               │   - Dropout (p=0.3) + FC(512 → 8)            │
                               └──────────────────────┬──────┴────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
                       ▼                                                             ▼
     ┌────────────────────────────────────┐                        ┌────────────────────────────────────┐
     │      Classification Output         │                        │     Multi-Scale Grad-CAM++         │
     │  - Softmax Probability Spectrum    │                        │  - 2nd & 3rd Order Gradient Math   │
     │  - Top-2 Margin & Severity Score   │                        │  - Layer3 (0.60) + Layer4 (0.40)   │
     │  - Inference Latency Benchmark     │                        │  - Disc Masking (0 Background Bleed│
     └─────────────────┬──────────────────┘                        └─────────────────┬──────────────────┘
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │   WaferOS Yield Analytics Web Studio         │
                               │   - Live Diagnostic Heatmap & Raw Thermal    │
                               │   - Defect Knowledge Base & Root Causes      │
                               │   - Semiconductor AI Assistant (Gemini LLM)  │
                               │   - Lot PDF Audit & CSV Analytics Exporter   │
                               └──────────────────────────────────────────────┘
```

---

## 🔬 Multi-Scale Grad-CAM++ Explainability

Standard Grad-CAM relies on a single $7 \times 7$ feature grid at ResNet's final layer (`layer4`), causing linear defects like scratches or fine rings to collapse into amorphous, low-resolution blobs that bleed onto the background.

WaferOS implements **Multi-Scale Grad-CAM++**:
1. **Higher-Order Gradient Weighting**: Computes 2nd and 3rd order partial derivatives ($\alpha_{k, ij}^c$) to accurately weight multiple defect trajectories and prevent single-node gradient collapse.
2. **Dual-Layer Feature Fusion**: Combines the high-level semantic class-gating of `layer4` ($7 \times 7$) with the fine-grained spatial trajectory fidelity of `layer3` ($14 \times 14$) for **4x higher spatial resolution**.
3. **Wafer Disc Boundary Masking**: Dynamically extracts the silicon wafer perimeter and clips thermal attribution strictly inside the wafer disc, preserving clean black background borders.

```
[Input Scratch Wafer]        [Old Grad-CAM (Standard)]          [New Multi-Scale Grad-CAM++]
Two crossing scratches       Single blurry blob in corner;      Traces BOTH scratch lines
                             bleeds across dark background      along their full trajectories
```

---

## 📊 Defect Taxonomy & Fab Diagnostics

The system identifies 8 macro-level wafer defect patterns:

| # | Taxonomy Class | Pattern Geometry | Fab Root Causes | Corrective Action Protocol |
|---|---|---|---|---|
| 0 | `normal` | Uniform die distribution | Nominal manufacturing process | Proceed to wafer sort and dicing |
| 1 | `center` | Defect concentration at center | Gas flow stagnation, CMP pressure imbalance | Adjust polish head gimbal; purge CMP slurry nozzles |
| 2 | `edge_ring` | Concentric peripheral defect ring | Edge bead removal (EBR) error, clamp stress | Realign EBR nozzle angle; calibrate electrostatic chuck |
| 3 | `edge_loss` | Partial arc failure along margin | Wafer handling robot slippage, edge chipping | Service end-effector vacuum; check cassette alignment |
| 4 | `scratch` | Linear/curved abrasive scratch traces | Mechanical foreign particle, tweezers/handling | Clean FOUP carrier; inspect robot transfer arm pads |
| 5 | `ring` | Concentric ring at intermediate radius | Thermal gradient non-uniformity in RTP chamber | Calibrate RTP lamp arrays; adjust furnace gas distribution |
| 6 | `cluster` | Localized irregular defect grouping | Local particle contamination, droplet splatter | Execute chamber wet-clean; audit cleanroom air filters |
| 7 | `full_fail` | Catastrophic multi-zone yield loss | Total vacuum failure, power glitch, etching disaster | Emergency tool halt; inspect chamber RF power matching |

---

## 💻 Tech Stack & Dependencies

- **Deep Learning Backbone**: PyTorch, Torchvision (ResNet-18)
- **Computer Vision**: OpenCV (Multi-thresholding, convex hull circularity, morphological operators)
- **Web Serving API**: FastAPI, Uvicorn (Asynchronous REST API daemon)
- **Frontend Studio**: Modern Vanilla HTML5, CSS3 Glassmorphism tokens, Chart.js, Vanilla ES6 JavaScript
- **Explainable AI**: Custom Multi-Scale Grad-CAM++ with 2nd/3rd order gradients
- **Yield Reporting**: ReportLab PDF Engine & CSV telemetry streaming
- **AI Fab Assistant**: Google Generative AI (Gemini Flash) with semiconductor prompt grounding

---

## 🚀 Launching the Production Web Studio

```bash
# Option A: Windows Launcher (Auto-opens browser)
start_app.bat

# Option B: Terminal Command
python -m uvicorn deployment.server:app --host 127.0.0.1 --port 8000
```

Access the dashboard at: **http://127.0.0.1:8000**

---

## 📁 Repository Navigation

```
wafer-defect-classification/
├── deployment/
│   ├── server.py               # FastAPI backend with model lazy-loader & REST routes
│   ├── wafer_detector.py       # Convex hull circularity & morphological wafer gating
│   ├── chatbot.py              # Semiconductor LLM fab assistant
│   ├── report_generator.py     # PDF lot audit generator
│   └── static/                 # WaferOS Glassmorphic Web Interface
│       ├── index.html          # Web Studio dashboard template
│       ├── styles.css          # Glassmorphic CSS tokens, typography, dark mode
│       └── script.js           # Interactive controller, Chart.js, API clients
│
├── models/
│   ├── resnet18_classifier.py  # ResNet18 architecture + Multi-Scale Grad-CAM++ engine
│   └── unet_model.py           # Segmentation architecture
│
├── real_demo_images/           # Real fab wafer map samples for validation
│   ├── real_center_0.png, real_cluster_0.png, real_scratch_0.png, etc.
│
├── tests/
│   ├── test_server_api.py      # Automated 8-endpoint API verification suite
│   ├── test_scratch_api.py     # Defect-specific gating regression test
│   └── test_multi_cams.py      # Multi-class Grad-CAM++ verification script
│
├── training/
│   ├── train_hackathon.py      # Rapid model training pipeline
│   ├── train_resnet.py         # Full lot-split production pipeline
│   └── train_combined_v2.py    # Hybrid WM-811K + synthetic trainer
│
├── utils/
│   ├── synthetic_generator.py  # Procedural defect pattern generator
│   └── wafer_dataset.py        # PyTorch dataset & augmentation loader
│
├── requirements.txt            # Python environment specifications
├── start_app.bat               # Windows one-click local server launcher
└── walkthrough.md              # Technical validation & audit log
```

---

## 🔐 Licensing & Commercial Use

© 2026. All rights reserved.  
This software, including its neural network architecture, proprietary Grad-CAM++ algorithms, and user interface designs, is protected under intellectual property laws. Model weights and fabrication datasets are strictly proprietary. For licensing, academic collaboration, or enterprise trial inquiries, please reach out via GitHub.
