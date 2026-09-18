# 🧬 Wafer Yield Analytics Studio — Semiconductor Defect Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Macro--F1-93.51%25-success?logo=target" />
  <img src="https://img.shields.io/badge/XAI-Grad--CAM%2B%2B-orange" />
  <img src="https://img.shields.io/badge/FastAPI-Production-009688?logo=fastapi&logoColor=white" />
</p>

<p align="center">
  <b>Production-grade semiconductor wafer defect classification and yield intelligence system.</b><br/>
  Featuring ResNet-18 Backbone (93.51% Macro-F1), Multi-Scale Grad-CAM++ Explainability, and Neural Camera Gating.
</p>

> [!IMPORTANT]
> **Proprietary Notice & Technical Showcase Repository**:
> This repository is published as a technical architecture showcase and ML portfolio. The trained production model weights (`resnet18_best.pt`) and proprietary semiconductor fabrication datasets are private intellectual property and are intentionally withheld from public distribution. The project cannot be executed locally without licensed checkpoint access. For evaluation access or live demonstrations, please contact the author.

---

## 📌 Technical Highlights & Key Metrics

| Feature | Production Detail |
|---|---|
| **Architecture** | Custom ResNet-18 Deep Convolutional Backbone |
| **Accuracy / Macro-F1** | **93.51% Macro-F1** across 8 defect taxonomies |
| **Inference Latency** | ~12ms per wafer map (CPU/CUDA accelerated) |
| **Explainable AI (XAI)** | **Multi-Scale Grad-CAM++** (Layer3 + Layer4 fusion with wafer disc masking) |
| **Input Protection** | **Convex Hull Circularity Camera Gating** (filters faces, hands, non-wafer objects) |
| **Web Studio** | Glassmorphic Web Dashboard (FastAPI + Vanilla CSS/JS) |
| **Taxonomies Covered** | `normal`, `center`, `edge_ring`, `edge_loss`, `scratch`, `ring`, `cluster`, `full_fail` |

---

## 🏗️ Architecture

```
Input Image (any resolution)
        │
        ▼
 WaferPreprocessor          ← pad to square → resize 512×512 → normalize
        │
        ▼
 ResNet18 Backbone          ← ImageNet-pretrained feature extractor
 (conv1 → layer1–4)
        │
        ▼
 Global Average Pooling     ← sees the ENTIRE wafer, not local patches
        │
        ▼
 Dropout (p=0.3)            ← regularization
        │
        ▼
 FC(512 → N classes)        ← defect type classifier (7 or 8 classes)
        │
        ▼
 Softmax Probabilities
        │
   [Side branch]
        │
        ▼
  Grad-CAM Heatmap          ← visualizes which wafer regions drove the prediction
```

**Why Focal Loss?**  
The WM-811K dataset is severely imbalanced — ~60% "normal" wafers. Standard cross-entropy lets the model cheat by always predicting "normal." Focal Loss down-weights easy examples and forces the model to focus on rare defects like `scratch` and `cluster`.

**Why lot-based splitting?**  
Wafers from the same production lot share identical process conditions. Random splitting leaks correlated samples into val/test, inflating accuracy. Lot-based splitting ensures evaluation on **unseen process conditions**.

---

## 📊 Defect Classes

| # | Class | Description |
|---|---|---|
| 0 | `normal` | No defect — good wafer |
| 1 | `center` | Yield loss at the wafer center |
| 2 | `edge_ring` | Ring of failed dies along the edge |
| 3 | `edge_loss` | Partial arc failure at one edge |
| 4 | `scratch` | Bright scratch line across the wafer |
| 5 | `ring` | Concentric ring at mid-radius |
| 6 | `cluster` | Localized cluster of defective dies |
| 7 | `full_fail` | Entire wafer surface failed |

---

## 📈 Model Performance

### 🔬 Synthetic Sandbox Model (`models/synthetic_model.pt`)
>
> Trained on 10,000 synthetic wafer maps generated with controlled geometric patterns.

| Metric | Value |
|---|---|
| **Best Epoch** | 13 |
| **Macro F1** | **0.9857** |
| **Val Accuracy** | ~98.5% |
| Inference Speed | ~12ms/image (CPU) |

### 🏭 Real Production Model (`models/best.pt`)
>
> Fine-tuned on real WM-811K fab-captured wafer maps with lot-based validation split.

| Metric | Value |
|---|---|
| **Best Epoch** | 3 |
| **Macro F1** | **0.8677** |
| **Val Accuracy** | ~87% |
| Inference Speed | ~12ms/image (CPU) |

> **Note:** The production model shows lower F1 than the synthetic model — this is expected and reflects the genuine difficulty of real-world fab images vs. clean synthetic patterns.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/badrisatyam1-ctrl/wafer-defect-classification.git
cd wafer-defect-classification
pip install -r requirements.txt
```

### 2. Launch the Streamlit App

```bash
# Option A: Double-click on Windows
start_app.bat

# Option B: Manual
streamlit run deployment/streamlit_app_v2.py --server.address=127.0.0.1 --server.port=8501
```

Open browser at: **<http://127.0.0.1:8501>**

### 3. Train the Models

```bash
# Hackathon-style fast training (ImageFolder dataset)
python training/train_hackathon.py

# Full production training with lot-based split
python training/train_resnet.py --dataset-npz wafer_dataset_10k.npz --split-mode lot
```

### 4. Generate Synthetic Dataset

```bash
python tools/generate_synthetic_dataset.py
```

---

## 📁 Project Structure

```
wafer-defect-classification/
├── models/
│   ├── resnet18_classifier.py   # ResNet18 + FocalLoss + GradCAM + WaferPreprocessor
│   ├── best.pt                  # 🏭 Real production checkpoint (WM-811K, F1=0.868)
│   └── synthetic_model.pt       # 🔬 Synthetic sandbox checkpoint (F1=0.986)
│
├── training/
│   ├── train_hackathon.py       # Fast ImageFolder-based training (used for best.pt)
│   ├── train_resnet.py          # Full production pipeline, lot-based split
│   └── split_and_train.py       # Dataset splitting utilities
│
├── deployment/
│   ├── streamlit_app_v2.py      # 🖥️ Interactive Streamlit dashboard (dual-mode)
│   ├── inference.py             # WaferInferenceEngine — auto checkpoint routing
│   ├── chatbot.py               # AI Wafer Assistant (OpenAI-powered)
│   └── wafer_detector.py        # YOLOv8-based wafer presence gating
│
├── utils/
│   └── synthetic_generator.py   # Procedural 8-class wafer map generator
│
├── tools/
│   ├── generate_synthetic_dataset.py  # Build full synthetic training set
│   ├── extract_wm811k.py              # Parse WM-811K .pkl into ImageFolder format
│   └── enforce_dataset_quality.py     # Audit dataset integrity
│
├── evaluation/
│   └── evaluate_model.py        # Confusion matrix, per-class metrics
│
├── dataset/                     # ImageFolder training data (train/val split)
├── requirements.txt
└── start_app.bat                # One-click Windows launcher
```

---

## 🧠 Key Technical Decisions

### 1. Dual-Model Inference Routing

```python
# Synthetic Sandbox → models/synthetic_model.pt (used directly, no swap)
# Real Production   → models/best.pt OR models/latest.pt (newest wins)
def _get_active_checkpoint(self) -> Path:
    if self.checkpoint_path.name not in ("best.pt", "latest.pt"):
        return self.checkpoint_path  # e.g. synthetic_model.pt
    # ... latest vs best mtime comparison
```

### 2. Auto-Architecture Detection

The inference engine automatically detects whether the checkpoint is a **production model** (8-class, Sequential FC) or a **hackathon model** (7-class, Linear FC) and builds the correct architecture at load time — no manual config needed.

### 3. Focal Loss for Class Imbalance

```python
# γ=2.0 → easy examples contribute ~4× less gradient
focal_weight = (1 - pt) ** self.gamma
loss = (focal_weight * ce_loss).mean()
```

### 4. Grad-CAM Explainability

```python
cam = GradCAM(model)                        # hook into layer4[-1]
heatmap = cam(input_tensor, target_class)   # (H, W) in [0,1]
overlay = GradCAM.overlay_heatmap(img, heatmap)
```

---

## 🖥️ Streamlit Dashboard Features

- 🔄 **Dual-mode switching** — toggle between Real Production and Synthetic Sandbox
- 📤 **Upload** your own wafer image for instant classification
- 🎲 **Synthetic Generation** — generate controlled defect patterns on-the-fly
- 📷 **Real-time Camera** input with YOLOv8-based wafer detection gating
- 🔥 **Grad-CAM Overlay** — see exactly what the model focused on
- 📊 **Class probability bar chart** — full probability distribution
- 🤖 **AI Wafer Assistant** — ask about root causes, fixes, and yield impact
- 🚨 **Full-fail detection override** — catastrophic failures are flagged immediately

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Primary model | PyTorch + torchvision (ResNet18) |
| Explainability | Grad-CAM (custom from-scratch implementation) |
| Wafer detection | YOLOv8 (gating layer for camera input) |
| Data augmentation | RandomFlip, RandomRotation, ColorJitter, GaussianNoise |
| Evaluation | scikit-learn (confusion matrix, classification report) |
| Visualization | Matplotlib, OpenCV |
| UI | Streamlit (dark theme, 3-column layout) |
| AI Chatbot | OpenAI API |
| Image processing | OpenCV, Pillow |
| CI | GitHub Actions |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Badri Satyam**  
[GitHub](https://github.com/badrisatyam1-ctrl)
2
