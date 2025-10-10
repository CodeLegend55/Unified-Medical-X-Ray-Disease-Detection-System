# 🏥 Unified Medical X-Ray Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

A **3-model ensemble AI system** that detects **8 disease classes** from X-ray images using ResNet50, DenseNet121, and EfficientNetB0 with **95-98% accuracy**.

### Detected Diseases
- **Chest:** COVID-19, Pneumonia, Tuberculosis, Normal
- **Bone:** Osteoporosis, Normal, Fractured, Non-Fractured

---

## ⚡ Quick Start

```bash
# Clone and setup
git clone https://github.com/CodeLegend55/Unified-Medical-X-Ray-Disease-Detection-System.git
cd "Unified Training"
pip install -r requirements.txt

# Prepare dataset
python prepare_unified_dataset.py

# Run web application
python app.py
```

Open browser to **http://localhost:5000**

---

## 📥 Dataset Structure

```
datasets/
├── chest_xray_merged/
│   ├── train/, val/, test/ (covid, normal, pneumonia, tb)
├── osteoporosis/
│   ├── normal/, osteoporosis/
└── Bone_Fracture_Binary_Classification/
    └── train/, val/, test/ (fractured, not fractured)
```

---

## 🎓 Training

Run `unified_model_training.ipynb` to train all 3 models:

| Model | Size | Training Time | Accuracy |
|-------|------|---------------|----------|
| ResNet50 | 94 MB | ~2-3 hours | 92-95% |
| DenseNet121 | 29 MB | ~2-3 hours | 93-96% |
| EfficientNetB0 | ~20 MB | ~1.5-2 hours | 91-94% |
| **Ensemble** | **~143 MB** | **~6-8 hours** | **95-98%** |

**Requirements:** 16GB RAM, GPU recommended

---

## 🚀 Web Application

```bash
python app.py
```

**Features:**
- 🖼️ Drag-and-drop image upload
- 🤖 3-Model ensemble prediction
- 📊 Individual model confidence scores
-  AI-powered medical reports (optional Hugging Face API)
- ⚡ Fast inference (~125ms GPU, ~210ms CPU)

**AI Reports Setup (Optional):**
1. Get free API key: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Update `config.py`: `HUGGINGFACE_API_KEY = "hf_your_key"`
3. See [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) for details

---

## 💻 Python Usage

```python
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['COVID19', 'PNEUMONIA', 'TUBERCULOSIS', 'NORMAL_CHEST',
           'OSTEOPOROSIS', 'NORMAL_BONE', 'FRACTURED', 'NON_FRACTURED']

# Load models (see full code in repository)
# ... load ResNet50, DenseNet121, EfficientNetB0 ...

# Predict
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('xray.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    # Get predictions from all 3 models
    ensemble_probs = (resnet_probs + densenet_probs + efficientnet_probs) / 3
    pred_idx = torch.argmax(ensemble_probs).item()
    print(f"Diagnosis: {CLASSES[pred_idx]} ({ensemble_probs[pred_idx]*100:.1f}%)")
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA Out of Memory | Reduce batch size to 16 or 8 |
| Dataset Not Found | Run `prepare_unified_dataset.py` first |
| Low Accuracy | Increase epochs to 100 |
| Training Too Slow | Use GPU; reduce image size |
| Import Errors | Run `pip install -r requirements.txt` |

---

## � Performance

| Model | Accuracy | Inference Speed |
|-------|----------|-----------------|
| ResNet50 | 92-95% | ~50ms (GPU) |
| DenseNet121 | 93-96% | ~40ms (GPU) |
| EfficientNetB0 | 91-94% | ~35ms (GPU) |
| **Ensemble** | **95-98%** | **~125ms (GPU)** |

**Dataset:** 51,632 images (39,818 train, 6,228 val, 5,586 test)

---

## 📝 Project Structure

```
Unified Training/
├── datasets/              # Source datasets
├── unified_dataset/       # Prepared unified dataset
├── models/                # Trained model weights
│   ├── unified_ResNet50.pth
│   ├── unified_DenseNet121.pth
│   └── unified_EfficientNetB0.pth
├── templates/             # Web UI
├── app.py                 # Flask application
├── config.py              # Configuration
├── prepare_unified_dataset.py
├── unified_model_training.ipynb
└── requirements.txt
```

---

##  Contact

**Repository:** [GitHub](https://github.com/CodeLegend55/Unified-Medical-X-Ray-Disease-Detection-System)  
**License:** MIT

---

*Made with ❤️ for better medical diagnostics*


