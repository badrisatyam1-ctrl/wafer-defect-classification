# 🧬 Wafer Defect Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/github/actions/workflow/status/badrisatyam1-ctrl/wafer-defect-classification/ci.yml?label=CI&logo=github" />
</p>

<p align="center">
  <b>Industry-grade macro-level wafer defect detection using ResNet18 + Focal Loss + Grad-CAM.</b><br/>
  Built to mirror real-world semiconductor fab quality control pipelines.
</p>

---

## 📌 Project Overview

Semiconductor wafers are circular silicon disks on which hundreds of chips are etched. Defects — rings, scratches, clusters — reduce yield and cost millions in recalls. This project automates **macro-level defect classification** using deep learning.

| Feature | Detail |
|---|---|
| **Model** | ResNet18 (ImageNet pretrained, fine-tuned) |
| **Loss** | Focal Loss (γ=2.0) — handles severe class imbalance |
| **Explainability** | Grad-CAM heatmaps — highlights defect regions |
| **Architecture extras** | Squeeze-and-Excitation (SE) attention blocks in CNN variant |
| **Data split** | Lot-based (prevents leakage from fab process sharing) |
| **UI** | Streamlit dashboard with real-time inference + heatmap |
| **Dataset** | WM-811K compatible + custom synthetic wafer map generator |

---

## 🏗️ Architecture

```
Input Image (224×224×3)
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
 FC(512 → 8 classes)        ← defect type classifier
        │
        ▼
 Softmax Probabilities
        │
   [Side branch]
        │
        ▼
  Grad-CAM Heatmap          ← visualizes where the model looked
```

**Why Focal Loss?**  
The WM-811K dataset is severely imbalanced — ~60% "normal" wafers. Standard cross-entropy lets the model cheat by always predicting "normal." Focal Loss down-weights easy examples, forcing the model to focus on rare defects like `scratch` and `donut`.

**Why lot-based splitting?**  
Wafers from the same production lot share identical process conditions. Random splitting would leak correlated samples into val/test, inflating accuracy. Lot-based splitting ensures evaluation on **unseen process conditions**.

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

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/badrisatyam1-ctrl/wafer-defect-classification.git
cd wafer-defect-classification
pip install -r requirements.txt
```

### 2. Train the ResNet18 Classifier

```bash
python training/train_classifier.py
```

Trains for 20 epochs with Focal Loss + AdamW + Cosine LR annealing.  
Best model saved to `models/checkpoints/resnet18_best.pth`.

### 3. Launch the Streamlit App

```bash
streamlit run deployment/streamlit_app.py
```

Or double-click `start_app.bat` on Windows.

### 4. Evaluate the Model

```bash
python evaluation/evaluate_model.py
```

Generates confusion matrix, classification report, and per-class metrics comparison chart.

---

## 📁 Project Structure

```
wafer-defect-classification/
├── models/
│   ├── resnet18_model.py       # ResNet18 + FocalLoss + GradCAM (PyTorch)
│   ├── wafer_cnn.py            # Custom CNN + SE-Attention + FocalLoss (Keras)
│   ├── unet_model.py           # U-Net segmentation (legacy)
│   └── checkpoints/
│       ├── resnet18_best.pth   # Trained ResNet18 (~44 MB)
│       └── best_model.keras    # Trained CNN (~9.8 MB)
│
├── training/
│   ├── train_classifier.py     # ResNet18 training (lot-based split, Focal Loss)
│   ├── train_cnn.py            # CNN training (Keras, class-weighted CE)
│   └── retrain_realistic.py    # Realistic retraining pipeline
│
├── deployment/
│   ├── streamlit_app.py        # 🖥️ Interactive Streamlit dashboard
│   └── inference_api.py        # Preprocessing + predict() + GradCAM pipeline
│
├── evaluation/
│   ├── evaluate_model.py       # Confusion matrix, classification report, per-class charts
│   ├── plot_history.py         # Training curve visualizer
│   ├── confusion_matrix_demo.png
│   └── plot_overfitting.png
│
├── utils/
│   ├── wafer_map_generator.py  # Synthetic 224×224 wafer map generator (8 classes)
│   └── synthetic_generator.py  # Additional synthetic data utilities
│
├── preprocessing/
│   └── opencv_preprocess.py    # OpenCV-based image preprocessing
│
├── data/
│   ├── raw/                    # Place WM-811K data here
│   └── processed/              # Preprocessed arrays
│
├── requirements.txt
└── start_app.bat               # Windows launcher
```

---

## 🧠 Key Technical Decisions

### 1. Focal Loss over Weighted Cross-Entropy

```python
# Standard CE treats all mistakes equally — bad for imbal. data
loss = cross_entropy(logits, targets)

# Focal Loss: (1-p_t)^γ  downweights easy examples
# γ=2.0 → easy examples contribute ~4× less loss
focal_weight = (1 - pt) ** self.gamma
loss = (focal_weight * ce_loss).mean()
```

### 2. Grad-CAM for Explainability (Zero Code Change to Model)

```python
cam = GradCAM(model)                           # hook into layer4
heatmap = cam.generate(input_tensor)           # (224, 224) in [0,1]
overlay = overlay_heatmap(image, heatmap)      # jet colormap blend
```

No model architecture changes needed. Critical for engineer trust in a fab environment.

### 3. Squeeze-and-Excitation (SE) Attention

```python
# Learns per-channel importance weights
se = GlobalAveragePooling2D()(x)            # Squeeze
se = Dense(channels // 16, activation='relu')(se)
se = Dense(channels, activation='sigmoid')(se) # Excite
x  = Multiply()([x, se])                   # Scale
```

Helps the CNN focus on faint defect patterns even in minority classes.

### 4. Why Recall > Precision in Semiconductor QC

| Error Type | Consequence | Cost |
|---|---|---|
| False Negative (Miss defect) | Defective chip ships to customer | 💸 $10,000+ recalls/lawsuits |
| False Positive (False alarm) | Extra inspection stop | 💰 ~$0.50/wafer |

**Target: Recall > 95%** even at the cost of some precision.

---

## 📈 Model Performance

> Trained on synthetic wafer map dataset (500 samples/class, 8 classes).

| Metric | ResNet18 |
|---|---|
| Validation Accuracy | ~92% |
| Weighted Recall | >90% |
| Weighted F1 | >88% |
| Inference Speed | ~15ms/image (CPU) |

---

## 🖥️ Streamlit Dashboard

The interactive dashboard (`deployment/streamlit_app.py`) provides:

- **Upload** your own wafer image OR **generate** a synthetic one
- **Real-time ResNet18 inference** with confidence score
- **Grad-CAM heatmap overlay** — see exactly where the model detected the defect
- **Class probability bar chart** — full distribution over all 8 classes
- Color-coded result: 🟢 Normal / 🔴 Defect

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Primary model | PyTorch + torchvision (ResNet18) |
| CNN variant | TensorFlow / Keras |
| Explainability | Grad-CAM (custom implementation) |
| Attention | Squeeze-and-Excitation blocks |
| Data augmentation | OpenCV (rotation, flip, brightness, noise) |
| Evaluation | scikit-learn (confusion matrix, classification report) |
| Visualization | Matplotlib, Seaborn |
| UI | Streamlit |
| Image processing | OpenCV, Pillow |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Badri Satyam**  
[GitHub](https://github.com/badrisatyam1-ctrl)
