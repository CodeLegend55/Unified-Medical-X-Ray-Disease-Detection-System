# 🏥 Unified Medical X-Ray Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

A **comprehensive medical imaging analysis system** featuring:
- **3-model ensemble AI** (ResNet50 + DenseNet121 + EfficientNetB0)
- **8 disease classes** detection with **95-98% accuracy**
- **Custom fine-tuned medical LLM** for automated radiology report generation
- **No external API dependencies** - fully self-contained system

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

### System Requirements
- Python 3.8+
- PyTorch 2.0+
- 16GB RAM (minimum)
- GPU recommended (CUDA-capable)
- 5GB disk space for models and dataset

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

## 🤖 Medical Report LLM Training

The system includes a custom fine-tuned GPT-2 model for generating medical radiology reports.

### Training the Report Generator

```bash
cd LLM
python train_model.py
```

**Training Details:**
- **Base Model:** GPT-2 (distilgpt2)
- **Dataset:** 5,000+ medical radiology reports
- **Training Time:** ~2-3 hours (GPU)
- **Output:** Professional structured medical reports
- **Format:** Clinical History, Technique, Findings, Impression

**Dataset Format (`medical_report_dataset.json`):**
```json
{
  "diagnosis": "PNEUMONIA",
  "confidence": 92.5,
  "exam_type": "Chest X-Ray",
  "report": "CLINICAL HISTORY:\n...\nFINDINGS:\n...\nIMPRESSION:\n..."
}
```

The trained model automatically generates comprehensive medical reports based on:
- AI diagnosis and confidence scores
- Patient information (age, gender, symptoms)
- Ensemble model consensus
- Clinical correlations and recommendations

---

## 🚀 Web Application

```bash
python app.py
```

**Features:**
- 🖼️ Drag-and-drop image upload
- 🤖 3-Model ensemble prediction
- 📊 Individual model confidence scores
- 📄 **AI-powered medical reports** using custom fine-tuned LLM (GPT-2 based)
- 👤 Patient information collection (age, gender, symptoms, medical history)
- ⚡ Fast inference (~125ms GPU, ~210ms CPU)

**Medical Report Generation:**

The system uses a **custom fine-tuned medical language model** (GPT-2 based) trained specifically on medical radiology reports:
- **Model Location:** `LLM/medical_report_model/`
- **Training Data:** 5,000+ real medical radiology reports
- **Output:** Professional structured medical reports with:
  - Clinical History
  - Technique
  - Findings
  - Impression with recommendations

No external API keys required - the model runs locally!

---

## 💻 Python Usage

### Disease Detection

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

### Generate Medical Report

```python
from LLM.report_generator import MedicalReportGenerator

# Initialize report generator
generator = MedicalReportGenerator(model_path="LLM/medical_report_model")

# Prepare patient data
patient_info = {
    'diagnosis': 'PNEUMONIA',
    'confidence': 92.5,
    'exam_type': 'Chest X-Ray',
    'patient_info': {'age': '45', 'gender': 'Male'},
    'model_consensus': [
        {'model': 'ResNet50', 'prediction': 'PNEUMONIA', 'confidence': 91.2},
        {'model': 'DenseNet121', 'prediction': 'PNEUMONIA', 'confidence': 93.1},
        {'model': 'EfficientNetB0', 'prediction': 'PNEUMONIA', 'confidence': 93.2}
    ]
}

# Generate report
report = generator.generate_report(patient_info)
print(report)
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
| LLM Model Not Found | Ensure `LLM/medical_report_model/` exists with model files |
| Report Generation Empty | Check debug output in terminal; model may need retraining |
| Web App Port Conflict | Change `PORT` in `config.py` to different value |

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

## ✨ Key Features

### 🎯 Disease Detection
- **8 unified disease classes** across chest and bone conditions
- **3-model ensemble** for improved accuracy and reliability
- **Real-time predictions** with confidence scores
- Individual model breakdowns for transparency

### 📝 Report Generation
- **Custom fine-tuned LLM** (GPT-2 based)
- Professional radiology report structure
- Clinical correlations and recommendations
- Patient information integration
- **No external API dependencies**

### 🌐 Web Interface
- Clean, responsive design
- Drag-and-drop file upload
- Real-time progress indicators
- Detailed probability distributions
- Visual confidence meters
- Downloadable reports

### 🔧 Development Features
- Modular architecture
- Easy model switching
- Comprehensive error handling
- Debug mode with detailed logging
- Extensible for additional diseases

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
├── LLM/                   # Medical report generation
│   ├── medical_report_model/  # Custom fine-tuned GPT-2 model
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── vocab.json
│   ├── report_generator.py    # Report generation logic
│   ├── train_model.py          # LLM training script
│   └── medical_report_dataset.json  # Training data
├── templates/             # Web UI
│   └── index.html
├── uploads/               # Uploaded images
├── app.py                 # Flask application
├── config.py              # Configuration
├── prepare_unified_dataset.py
├── unified_model_training.ipynb
└── requirements.txt
```

---

## 📧 Contact

**Repository:** [GitHub](https://github.com/CodeLegend55/Unified-Medical-X-Ray-Disease-Detection-System)  
**License:** MIT

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

### Citation
If you use this project in your research, please cite:
```
@software{unified_medical_xray_2025,
  title={Unified Medical X-Ray Disease Detection System},
  author={CodeLegend55},
  year={2025},
  url={https://github.com/CodeLegend55/Unified-Medical-X-Ray-Disease-Detection-System}
}
```

---

## ⚠️ Medical Disclaimer

This system is designed for **research and educational purposes only**. It should NOT be used as a substitute for professional medical diagnosis or treatment. All AI predictions and generated reports must be reviewed and validated by qualified healthcare professionals. In case of medical emergencies, always seek immediate professional medical attention.

---

*Made with ❤️ for advancing AI in medical diagnostics*


