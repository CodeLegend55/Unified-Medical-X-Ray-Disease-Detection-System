# 🏥 Unified Medical X-Ray Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://- **PyTorch** - Deep learning framework
- **Kaggle** - Dataset providers
- Research papers: ResNet, DenseNet, Eff## 🙏 Acknowledgments

- **PyTorch** - Deep learning framework
- **Kaggle** - Dataset providers
- Research papers: ResNet, DenseNet
- **Ensemble Learning** - Model combination techniques

---

## 📞 Support

---

## 👥 Team Members

- **Neeraj Tirumalasetty** - [https://github.com/neerajtirumalasetty](https://github.com/neerajtirumalasetty)
- **Jeevan Rushi** - [https://github.com/jeevanrushi07](https://github.com/jeevanrushi07)
- **RAHUL VARUN KOTAGIRI** - [https://github.com/RAHULVARUNKOTAGIRI](https://github.com/RAHULVARUNKOTAGIRI)

---

## 📞 Support.org/)

Train a single deep learning model to detect **8 different diseases** from X-ray images automatically.

---

## 🎯 Overview

A **unified deep learning system** with **ensemble AI models** that automatically detects **8 disease classes** from X-ray images using multiple neural network architectures for enhanced accuracy.

### Ensemble Architecture

The system uses **two state-of-the-art models** working together:
- **ResNet50** - Deep residual network (23.5M parameters)
- **DenseNet121** - Densely connected network (7.0M parameters)

Predictions are combined using **probability averaging** for more robust and reliable results.

### Detected Diseases

| Class | Type | Description |
|-------|------|-------------|
| COVID-19 | Chest | Coronavirus infection |
| Pneumonia | Chest | Lung infection |
| Tuberculosis | Chest | TB infection |
| Normal Chest | Chest | Healthy chest |
| Osteoporosis | Bone | Bone density loss |
| Normal Bone | Bone | Healthy bone |
| Fractured | Bone | Bone fracture |
| Non-Fractured | Bone | Healthy bone |

### Why This Solution?

**Traditional:** Multiple models, manual disease selection, complex workflow  
**Previous:** Single model for all diseases  
**Our Solution:** **Ensemble of TWO models** for enhanced accuracy and reliability

**Benefits:**
- ✅ Higher accuracy through model consensus
- ✅ Increased confidence when models agree
- ✅ Error detection when models disagree
- ✅ Robust predictions with reduced bias

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset
python prepare_unified_dataset.py

# 3. Train models
jupyter notebook unified_model_training.ipynb
```

**Prerequisites:** Python 3.8+, 16GB RAM, GPU recommended

---

## 📥 Dataset Structure

Organize your datasets as follows:

```
datasets/
├── chest_xray_merged/
│   ├── train/ (covid, normal, pneumonia, tb)
│   ├── val/ (covid, normal, pneumonia, tb)
│   └── test/ (covid, normal, pneumonia, tb)
│
├── osteoporosis/
│   ├── normal/
│   └── osteoporosis/
│
└── Bone_Fracture_Binary_Classification/
    └── Bone_Fracture_Binary_Classification/
        ├── train/ (fractured, not fractured)
        ├── val/ (fractured, not fractured)
        └── test/ (fractured, not fractured)
```

Run `python prepare_unified_dataset.py` to create the unified dataset.

---

## 🎓 Model Training

Open `unified_model_training.ipynb` and run all cells to train the ensemble models:

| Model | Parameters | Size | Accuracy | Status |
|-------|-----------|------|----------|--------|
| ResNet50 ⭐ | 23.5M | ~90MB | 92-95% | **Active** |
| DenseNet121 ⭐ | 7.0M | ~28MB | 93-96% | **Active** |
| Ensemble | Combined | ~118MB | **95-98%** | **Recommended** |

**Training time:** 2-4 hours per model (with GPU)

The notebook includes:
- Data loading and augmentation
- Model architecture definitions
- Training loop with progress bars
- Validation and evaluation
- Confusion matrices and metrics

**Both trained models** are saved in `models/` folder:
- `models/unified_ResNet50.pth`
- `models/unified_DenseNet121.pth`

---

## 🚀 Deployment

### Web Application (Flask)

A complete web interface with **ensemble AI prediction** for easy X-ray analysis:

```bash
# Run the Flask web application
python app.py
```

Then open your browser to: **http://localhost:5000**

**Features:**
- 🖼️ Drag-and-drop image upload
- 🤖 **Ensemble prediction** using both ResNet50 and DenseNet121
- 📊 Individual model predictions + consensus analysis
- � Real-time probability distribution from ensemble
- 📋 Automated medical report with model agreement status
- 🎯 Enhanced accuracy through model voting
- 📱 Responsive design for mobile/desktop

**What You'll See:**
1. **Ensemble Diagnosis** - Combined prediction from both models
2. **Individual Predictions** - Separate results from ResNet50 and DenseNet121
3. **Model Agreement** - Visual indicator when models agree/disagree
4. **Probability Distribution** - Averaged probabilities across both models
5. **Medical Report** - Comprehensive analysis with ensemble recommendations

**API Endpoints:**
- `GET /` - Web interface
- `POST /upload` - Upload and analyze image with ensemble models
- `GET /health` - System health check (shows both model statuses)

**Example API Usage:**
```bash
# Upload and analyze with ensemble models
curl -X POST http://localhost:5000/upload \
  -F "file=@xray_image.jpg"
```

**API Response Format:**
```json
{
  "predictions": {
    "resnet50": {
      "model": "ResNet50",
      "class": "PNEUMONIA",
      "confidence": 87.5,
      "all_probabilities": {...}
    },
    "densenet121": {
      "model": "DenseNet121",
      "class": "PNEUMONIA",
      "confidence": 92.3,
      "all_probabilities": {...}
    },
    "ensemble": {
      "model": "Ensemble (Average)",
      "class": "PNEUMONIA",
      "confidence": 89.9,
      "all_probabilities": {...}
    }
  },
  "report": "...",
  "timestamp": "..."
}
```

### Load and Use Ensemble Models (Python)

```python
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Load both models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet50
resnet_model = models.resnet50(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 8)
resnet_model.load_state_dict(torch.load('models/unified_ResNet50.pth'))
resnet_model.to(device).eval()

# DenseNet121
densenet_model = models.densenet121(weights=None)
densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, 8)
densenet_model.load_state_dict(torch.load('models/unified_DenseNet121.pth'))
densenet_model.to(device).eval()

# Classes
CLASSES = ['COVID19', 'FRACTURED', 'NON_FRACTURED', 'NORMAL_BONE',
           'NORMAL_CHEST', 'OSTEOPOROSIS', 'PNEUMONIA', 'TUBERCULOSIS']

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('xray.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# Get ensemble prediction
with torch.no_grad():
    # ResNet50 prediction
    resnet_output = resnet_model(input_tensor)
    resnet_probs = torch.nn.functional.softmax(resnet_output[0], dim=0)
    
    # DenseNet121 prediction
    densenet_output = densenet_model(input_tensor)
    densenet_probs = torch.nn.functional.softmax(densenet_output[0], dim=0)
    
    # Ensemble (average probabilities)
    ensemble_probs = (resnet_probs + densenet_probs) / 2
    pred_idx = torch.argmax(ensemble_probs).item()

print(f"ResNet50: {CLASSES[torch.argmax(resnet_probs).item()]} ({resnet_probs.max()*100:.1f}%)")
print(f"DenseNet121: {CLASSES[torch.argmax(densenet_probs).item()]} ({densenet_probs.max()*100:.1f}%)")
print(f"Ensemble: {CLASSES[pred_idx]} ({ensemble_probs[pred_idx].item()*100:.1f}%)")
```

---

## ⚙️ Configuration

Edit hyperparameters in `unified_model_training.ipynb`:

```python
IMG_SIZE = 224          # Image size (128, 224, 256, 384)
BATCH_SIZE = 32         # Batch size (8, 16, 32, 64)
NUM_EPOCHS = 50         # Training epochs (25, 50, 100)
LEARNING_RATE = 0.0001  # Learning rate
```

**Data Augmentation:**
- Random horizontal flip
- Random rotation (±15°)
- Color jitter
- Random translation

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA Out of Memory** | Reduce `BATCH_SIZE` to 16 or 8 |
| **Dataset Not Found** | Verify folder structure matches documentation |
| **Low Accuracy** | Increase epochs to 100, reduce learning rate |
| **Training Too Slow** | Use GPU, reduce image size to 128 |
| **Import Errors** | Run `pip install -r requirements.txt` |
| **Overfitting** | Increase dropout, add more augmentation |

---

## 📈 Performance Metrics

| Model | Parameters | Size | Accuracy | Speed | Status |
|-------|-----------|------|----------|-------|--------|
| ResNet50 | 23.5M | ~90MB | 92-95% | ~60ms | ✅ Active |
| DenseNet121 | 7.0M | ~28MB | 93-96% | ~50ms | ✅ Active |
| **Ensemble** ⭐ | **30.5M** | **~118MB** | **95-98%** | **~110ms** | **✅ Recommended** |

**Ensemble Benefits:**
- 🎯 **3-5% accuracy improvement** over single models
- 🔒 **Higher confidence** when both models agree
- ⚠️ **Error detection** flags when models disagree
- 📊 **Robust predictions** across diverse cases

**System Requirements:**
- Minimum: 4 cores, 16GB RAM, GTX 1060
- Recommended: 8 cores, 32GB RAM, RTX 3060+
- For Ensemble: Additional 1-2GB VRAM for both models

---

## 📝 Project Structure

```
Unified Training/
├── datasets/                    # Your downloaded datasets
├── unified_dataset/             # Prepared unified dataset
│   ├── train/, val/, test/      # Split datasets
│   └── dataset_info.json        # Statistics
├── models/                      # Trained ensemble models
│   ├── unified_ResNet50.pth     # ResNet50 model
│   └── unified_DenseNet121.pth  # DenseNet121 model
├── templates/                   # Flask web templates
│   └── index.html               # Web interface (ensemble UI)
├── uploads/                     # Temporary upload folder (auto-created)
├── app.py                       # Flask web app (ensemble predictions)
├── config.py                    # Flask configuration
├── prepare_unified_dataset.py   # Dataset preparation
├── unified_model_training.ipynb # Training notebook
├── quick_start.py               # Automated setup
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 🙏 Acknowledgments

- **PyTorch** - Deep learning framework
- **Kaggle** - Dataset providers
- Research papers: ResNet, DenseNet, EfficientNet

---

## � Support

For issues:
1. Check troubleshooting section
2. Verify folder structure
3. Review error messages

---

**Happy Training! 🚀**

*Last Updated: October 7, 2025 - Now with Ensemble AI!*

