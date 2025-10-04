# 🏥 Unified Medical X-Ray Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://- **PyTorch** - Deep learning framework
- **Kaggle** - Dataset providers
- Research papers: ResNet, DenseNet, EfficientNet

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

A **unified deep learning model** that automatically detects **8 disease classes** from X-ray images using a single model.

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
**Our Solution:** ONE model detects ALL diseases automatically

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

Open `unified_model_training.ipynb` and run all cells to train three models:

| Model | Parameters | Size | Accuracy |
|-------|-----------|------|----------|
| ResNet50 | 23.5M | ~90MB | 92-95% |
| DenseNet121 | 7.0M | ~28MB | 93-96% |
| EfficientNetB0 ⭐ | 4.0M | ~16MB | 94-97% |

**Training time:** 2-4 hours per model (with GPU)

The notebook includes:
- Data loading and augmentation
- Model architecture definitions
- Training loop with progress bars
- Validation and evaluation
- Confusion matrices and metrics

Trained models are saved in `models/` folder.

---

## 🚀 Deployment

### Load and Use Trained Model

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = UnifiedEfficientNetB0(num_classes=8)
model.load_state_dict(torch.load('models/unified_EfficientNetB0.pth'))
model.eval()

# Classes
CLASSES = ['COVID19', 'PNEUMONIA', 'TUBERCULOSIS', 'NORMAL_CHEST',
           'OSTEOPOROSIS', 'NORMAL_BONE', 'FRACTURED', 'NON_FRACTURED']

# Preprocess and predict
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('xray.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    pred_idx = torch.argmax(probs).item()

print(f"Prediction: {CLASSES[pred_idx]} ({probs[pred_idx].item()*100:.1f}%)")
```

### Flask Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    # ... preprocess and predict ...
    return jsonify({'prediction': predicted_class, 'confidence': confidence})

if __name__ == '__main__':
    app.run(port=5000)
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

| Model | Parameters | Size | Accuracy | Speed |
|-------|-----------|------|----------|-------|
| ResNet50 | 23.5M | ~90MB | 92-95% | ~60ms |
| DenseNet121 | 7.0M | ~28MB | 93-96% | ~50ms |
| EfficientNetB0 ⭐ | 4.0M | ~16MB | 94-97% | ~40ms |

**System Requirements:**
- Minimum: 4 cores, 16GB RAM, GTX 1060
- Recommended: 8 cores, 32GB RAM, RTX 3060+

---

## 📝 Project Structure

```
Unified Training/
├── datasets/                    # Your downloaded datasets
├── unified_dataset/             # Prepared unified dataset
│   ├── train/, val/, test/      # Split datasets
│   └── dataset_info.json        # Statistics
├── models/                      # Trained models (.pth files)
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

*Last Updated: October 4, 2025*

