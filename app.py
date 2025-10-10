from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import json
import config
import requests

# Try to import Hugging Face InferenceClient (optional)
try:
    from huggingface_hub import InferenceClient
    hf_client = InferenceClient(
        model=config.HUGGINGFACE_MODEL,
        token=config.HUGGINGFACE_API_KEY
    ) if config.HUGGINGFACE_API_KEY != "your-huggingface-api-key-here" else None
    HUGGINGFACE_AVAILABLE = bool(hf_client)
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    hf_client = None
    print("⚠️ Hugging Face Hub package not available. Using fallback report generation.")
except Exception as e:
    HUGGINGFACE_AVAILABLE = False
    hf_client = None
    print(f"⚠️ Hugging Face client initialization failed: {e}. Using fallback report generation.")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UNIFIED_CLASSES = config.UNIFIED_CLASSES
CHEST_CONDITIONS = config.CHEST_CONDITIONS
FRACTURE_CONDITIONS = config.FRACTURE_CONDITIONS
BONE_CONDITIONS = config.BONE_CONDITIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing for unified model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class UnifiedMedicalModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet_model = None
        self.densenet_model = None
        self.efficientnet_model = None
        self.load_models()
    
    def load_models(self):
        """Load ResNet50, DenseNet121, and EfficientNetB0 models"""
        # Load ResNet50
        resnet_path = 'models/unified_ResNet50.pth'
        if os.path.exists(resnet_path):
            try:
                self.resnet_model = models.resnet50(weights=None)
                num_features = self.resnet_model.fc.in_features
                
                # IMPORTANT: Match the exact architecture from training
                self.resnet_model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, len(UNIFIED_CLASSES))
                )
                
                state_dict = torch.load(resnet_path, map_location=self.device, weights_only=False)
                
                # Handle state_dict with 'backbone.' prefix
                if any(key.startswith('backbone.') for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('backbone.'):
                            new_key = key.replace('backbone.', '', 1)
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    state_dict = new_state_dict
                
                self.resnet_model.load_state_dict(state_dict, strict=False)
                self.resnet_model.to(self.device)
                self.resnet_model.eval()
                
                print(f"✓ Loaded ResNet50 model with {len(UNIFIED_CLASSES)} classes")
            except Exception as e:
                print(f"✗ Error loading ResNet50 model: {e}")
                import traceback
                traceback.print_exc()
                self.resnet_model = None
        else:
            print(f"✗ ResNet50 model file not found: {resnet_path}")
        
        # Load DenseNet121
        densenet_path = 'models/unified_DenseNet121.pth'
        if os.path.exists(densenet_path):
            try:
                self.densenet_model = models.densenet121(weights=None)
                num_features = self.densenet_model.classifier.in_features
                
                # IMPORTANT: Match the exact architecture from training
                self.densenet_model.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, len(UNIFIED_CLASSES))
                )
                
                state_dict = torch.load(densenet_path, map_location=self.device, weights_only=False)
                
                # Handle state_dict with 'backbone.' prefix
                if any(key.startswith('backbone.') for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('backbone.'):
                            new_key = key.replace('backbone.', '', 1)
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    state_dict = new_state_dict
                
                self.densenet_model.load_state_dict(state_dict, strict=False)
                self.densenet_model.to(self.device)
                self.densenet_model.eval()
                
                print(f"✓ Loaded DenseNet121 model with {len(UNIFIED_CLASSES)} classes")
            except Exception as e:
                print(f"✗ Error loading DenseNet121 model: {e}")
                import traceback
                traceback.print_exc()
                self.densenet_model = None
        else:
            print(f"✗ DenseNet121 model file not found: {densenet_path}")
        
        # Load EfficientNetB0
        efficientnet_path = 'models/unified_EfficientNetB0.pth'
        if os.path.exists(efficientnet_path):
            try:
                self.efficientnet_model = models.efficientnet_b0(weights=None)
                num_features = self.efficientnet_model.classifier[1].in_features
                
                # IMPORTANT: Match the exact architecture from training
                self.efficientnet_model.classifier[1] = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, len(UNIFIED_CLASSES))
                )
                
                state_dict = torch.load(efficientnet_path, map_location=self.device, weights_only=False)
                
                # Handle state_dict with 'backbone.' prefix
                if any(key.startswith('backbone.') for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('backbone.'):
                            new_key = key.replace('backbone.', '', 1)
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    state_dict = new_state_dict
                
                self.efficientnet_model.load_state_dict(state_dict, strict=False)
                self.efficientnet_model.to(self.device)
                self.efficientnet_model.eval()
                
                print(f"✓ Loaded EfficientNetB0 model with {len(UNIFIED_CLASSES)} classes")
            except Exception as e:
                print(f"✗ Error loading EfficientNetB0 model: {e}")
                import traceback
                traceback.print_exc()
                self.efficientnet_model = None
        else:
            print(f"✗ EfficientNetB0 model file not found: {efficientnet_path}")
        
        print(f"✓ Device: {self.device}")
    
    def predict_single_model(self, image_tensor, model, model_name):
        """Make prediction using a single model"""
        try:
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            all_probabilities = {
                UNIFIED_CLASSES[i]: round(probabilities[i].item() * 100, 2) 
                for i in range(len(UNIFIED_CLASSES))
            }
            
            predicted_class = max(all_probabilities.keys(), key=lambda x: all_probabilities[x])
            confidence = all_probabilities[predicted_class]
            
            return {
                'model': model_name,
                'class': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
        except Exception as e:
            print(f"Error in {model_name} prediction: {e}")
            return None
    
    def predict(self, image_path):
        """Make prediction using all three models and provide ensemble result"""
        if self.resnet_model is None and self.densenet_model is None and self.efficientnet_model is None:
            return None, "No models loaded"
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            results = {}
            
            # Get predictions from ResNet50
            if self.resnet_model is not None:
                resnet_result = self.predict_single_model(input_tensor, self.resnet_model, 'ResNet50')
                if resnet_result:
                    results['resnet50'] = resnet_result
            
            # Get predictions from DenseNet121
            if self.densenet_model is not None:
                densenet_result = self.predict_single_model(input_tensor, self.densenet_model, 'DenseNet121')
                if densenet_result:
                    results['densenet121'] = densenet_result
            
            # Get predictions from EfficientNetB0
            if self.efficientnet_model is not None:
                efficientnet_result = self.predict_single_model(input_tensor, self.efficientnet_model, 'EfficientNetB0')
                if efficientnet_result:
                    results['efficientnetb0'] = efficientnet_result
            
            # Create ensemble prediction (average probabilities)
            if len(results) > 0:
                ensemble_probs = {}
                for cls in UNIFIED_CLASSES:
                    probs = []
                    if 'resnet50' in results:
                        probs.append(results['resnet50']['all_probabilities'][cls])
                    if 'densenet121' in results:
                        probs.append(results['densenet121']['all_probabilities'][cls])
                    if 'efficientnetb0' in results:
                        probs.append(results['efficientnetb0']['all_probabilities'][cls])
                    ensemble_probs[cls] = round(sum(probs) / len(probs), 2)
                
                ensemble_class = max(ensemble_probs.keys(), key=lambda x: ensemble_probs[x])
                ensemble_confidence = ensemble_probs[ensemble_class]
                
                results['ensemble'] = {
                    'model': 'Ensemble (Average)',
                    'class': ensemble_class,
                    'confidence': ensemble_confidence,
                    'all_probabilities': ensemble_probs
                }
            
            return results, None
            
        except Exception as e:
            return None, f"Error during prediction: {str(e)}"

def generate_medical_report(predictions):
    """Generate medical report using Hugging Face API or fallback"""
    
    # Try Hugging Face first if available
    if HUGGINGFACE_AVAILABLE:
        try:
            return generate_huggingface_report(predictions)
        except Exception as e:
            print(f"Hugging Face API failed: {e}. Using fallback report.")
            import traceback
            traceback.print_exc()
    
    # Use fallback report generation
    return generate_fallback_report(predictions)

def generate_huggingface_report(predictions):
    """Generate medical report using Hugging Face Inference API"""
    # Use ensemble prediction for the main diagnosis
    ensemble = predictions.get('ensemble', predictions.get('resnet50', predictions.get('densenet121')))
    diagnosis = ensemble['class']
    confidence = ensemble['confidence']
    
    # Build the analysis summary
    analysis_summary = f"""Analysis Type: Unified Multi-Disease Detection (Ensemble Model)

Primary Diagnosis (Ensemble): {diagnosis}
Confidence Level: {confidence:.1f}%

Model Predictions:"""
    
    # Add individual model predictions
    for model_name, result in predictions.items():
        if model_name != 'ensemble':
            analysis_summary += f"\n- {result['model']}: {result['class']} ({result['confidence']:.1f}%)"
    
    analysis_summary += f"""

Top Ensemble Probabilities:
"""
    
    # Add top 3 probabilities from ensemble
    sorted_probs = sorted(ensemble['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
    for cls, prob in sorted_probs:
        analysis_summary += f"- {cls}: {prob:.1f}%\n"
    
    # Use chat completion format for conversational models
    messages = [
        {
            "role": "user",
            "content": f"""You are a medical AI assistant. Generate a structured medical report based on X-ray analysis results.

{analysis_summary}

Please provide a professional medical report with the following sections:
1. CLINICAL FINDINGS
2. MODEL CONSENSUS ANALYSIS
3. DIAGNOSTIC IMPRESSION  
4. RECOMMENDATIONS
5. IMPORTANT NOTES

Keep the language clear and professional. Include appropriate medical disclaimers."""
        }
    ]
    
    # Call Hugging Face Inference API using chat completion
    response = hf_client.chat_completion(
        messages=messages,
        max_tokens=800,
        temperature=0.3,
        top_p=0.9
    )
    
    # Extract the response text
    if hasattr(response, 'choices') and len(response.choices) > 0:
        return response.choices[0].message.content.strip()
    else:
        # Fallback if response format is different
        return str(response).strip()

def generate_fallback_report(predictions):
    """Generate a comprehensive report when OpenAI API is not available"""
    # Use ensemble prediction for the main diagnosis
    ensemble = predictions.get('ensemble', predictions.get('resnet50', predictions.get('densenet121', predictions.get('efficientnetb0'))))
    diagnosis = ensemble['class']
    confidence = ensemble['confidence']
    
    report = f"""
UNIFIED MEDICAL IMAGING ANALYSIS REPORT
=========================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Type: UNIFIED MULTI-DISEASE DETECTION (ENSEMBLE)

IMAGE ANALYSIS DETAILS:
Models Used: """
    
    # List all models used
    if 'resnet50' in predictions:
        report += "\n• ResNet50"
    if 'densenet121' in predictions:
        report += "\n• DenseNet121"
    if 'efficientnetb0' in predictions:
        report += "\n• EfficientNetB0"
    if 'ensemble' in predictions:
        num_models = sum(1 for k in ['resnet50', 'densenet121', 'efficientnetb0'] if k in predictions)
        report += f"\n• Ensemble (Average of {num_models} models)"
    
    report += f"\nTotal Classes: {len(UNIFIED_CLASSES)}"
    
    report += f"""

PRIMARY DIAGNOSTIC IMPRESSION (ENSEMBLE):
Main Finding: {diagnosis}
Confidence Level: {confidence:.1f}%

INDIVIDUAL MODEL PREDICTIONS:"""
    
    # Show predictions from each model
    if 'resnet50' in predictions:
        resnet = predictions['resnet50']
        report += f"\n\n{resnet['model']}:"
        report += f"\n  Prediction: {resnet['class']}"
        report += f"\n  Confidence: {resnet['confidence']:.1f}%"
    
    if 'densenet121' in predictions:
        densenet = predictions['densenet121']
        report += f"\n\n{densenet['model']}:"
        report += f"\n  Prediction: {densenet['class']}"
        report += f"\n  Confidence: {densenet['confidence']:.1f}%"
    
    if 'efficientnetb0' in predictions:
        efficientnet = predictions['efficientnetb0']
        report += f"\n\n{efficientnet['model']}:"
        report += f"\n  Prediction: {efficientnet['class']}"
        report += f"\n  Confidence: {efficientnet['confidence']:.1f}%"
    
    # Show ensemble probabilities
    report += "\n\nENSEMBLE PROBABILITY DISTRIBUTION:"
    report += "\nAll Classes (Sorted by Confidence):"
    
    sorted_probs = sorted(ensemble['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        report += f"\n• {cls}: {prob:.1f}%"
    
    # Model agreement analysis
    report += "\n\nMODEL CONSENSUS ANALYSIS:"
    model_predictions = []
    if 'resnet50' in predictions:
        model_predictions.append(('ResNet50', predictions['resnet50']['class'], predictions['resnet50']['confidence']))
    if 'densenet121' in predictions:
        model_predictions.append(('DenseNet121', predictions['densenet121']['class'], predictions['densenet121']['confidence']))
    if 'efficientnetb0' in predictions:
        model_predictions.append(('EfficientNetB0', predictions['efficientnetb0']['class'], predictions['efficientnetb0']['confidence']))
    
    if len(model_predictions) >= 2:
        # Check if all models agree
        all_agree = all(pred[1] == model_predictions[0][1] for pred in model_predictions)
        if all_agree:
            report += f"\n✓ High Confidence: All {len(model_predictions)} models agree on the diagnosis"
        else:
            report += f"\n⚠ Models Disagree:"
            for name, pred_class, conf in model_predictions:
                report += f"\n  • {name}: {pred_class} ({conf:.1f}%)"
            report += f"\n  • Ensemble Decision: {diagnosis} ({confidence:.1f}%)"
    
    # Add specific recommendations based on diagnosis
    report += "\n\nCLINICAL RECOMMENDATIONS:"
    
    if diagnosis == 'COVID19':
        report += """
• Immediate medical consultation recommended
• Consider PCR/RT-PCR testing for COVID-19 confirmation
• Follow local COVID-19 protocols and isolation guidelines
• Monitor symptoms closely (fever, cough, shortness of breath)"""
    
    elif diagnosis == 'PNEUMONIA':
        report += """
• Medical consultation recommended within 24 hours
• Clinical correlation with patient symptoms advised
• Consider sputum culture and blood tests
• Monitor respiratory symptoms and vital signs"""
    
    elif diagnosis == 'TUBERCULOSIS':
        report += """
• Urgent medical consultation required
• Sputum examination for AFB (Acid-Fast Bacilli) recommended
• Contact tracing and isolation precautions necessary
• Follow TB treatment protocols if confirmed"""
    
    elif diagnosis == 'FRACTURED':
        report += """
• Orthopedic consultation recommended immediately
• Immobilization may be required pending clinical evaluation
• Pain management as appropriate
• Follow-up imaging may be necessary to monitor healing"""
    
    elif diagnosis == 'OSTEOPOROSIS':
        report += """
• Endocrinology or orthopedic consultation recommended
• DEXA scan may be needed for bone density confirmation
• Consider calcium and vitamin D supplementation
• Evaluate for underlying metabolic bone disorders"""
    
    elif diagnosis in ['NORMAL_CHEST', 'NORMAL_BONE', 'NON_FRACTURED']:
        report += """
• No acute findings detected on current imaging
• Routine follow-up as clinically indicated
• Continue regular health monitoring"""
    
    report += f"""

IMPORTANT MEDICAL DISCLAIMERS:
⚠️ This AI analysis is for screening and research purposes only.
⚠️ Results should not replace professional medical diagnosis or clinical judgment.
⚠️ Always consult with qualified healthcare professionals for final diagnosis and treatment decisions.
⚠️ Clinical correlation with patient symptoms, history, and physical examination is essential.
⚠️ In case of emergency or acute symptoms, seek immediate medical attention regardless of AI results.

TECHNICAL NOTES:
• Ensemble approach combines predictions from multiple neural network architectures
• Total training samples: 39,818 images across 8 disease classes
• Model architectures: ResNet50 and DenseNet121 with transfer learning
• Ensemble method: Probability averaging for improved accuracy and reliability
• Analysis confidence represents model certainty within clinical context

Disclaimer: This automated analysis uses ensemble AI models trained on diverse medical imaging data. 
Results should be interpreted by qualified medical professionals in conjunction with clinical findings.
"""
    
    return report

# Initialize model
unified_model = UnifiedMedicalModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction using both models
            predictions, error = unified_model.predict(filepath)
            
            if error:
                return jsonify({'error': error}), 500
            
            if not predictions:
                return jsonify({'error': 'Prediction failed'}), 500
            
            # Generate medical report
            report = generate_medical_report(predictions)
            
            result = {
                'filename': filename,
                'predictions': predictions,
                'report': report,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Clean up uploaded file after a delay (optional)
            # You might want to keep files for review
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG images.'}), 400

@app.route('/health')
def health_check():
    resnet_loaded = unified_model.resnet_model is not None
    densenet_loaded = unified_model.densenet_model is not None
    efficientnet_loaded = unified_model.efficientnet_model is not None
    
    return jsonify({
        'status': 'healthy' if (resnet_loaded or densenet_loaded or efficientnet_loaded) else 'unhealthy',
        'models_loaded': {
            'resnet50': resnet_loaded,
            'densenet121': densenet_loaded,
            'efficientnetb0': efficientnet_loaded
        },
        'num_classes': len(UNIFIED_CLASSES),
        'classes': UNIFIED_CLASSES
    })

if __name__ == '__main__':
    print("🚀 Starting Unified Medical Imaging Analysis Web Application...")
    print(f"📂 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🤖 Models:")
    print(f"   • ResNet50: {'✓ Loaded' if unified_model.resnet_model else '✗ Not loaded'}")
    print(f"   • DenseNet121: {'✓ Loaded' if unified_model.densenet_model else '✗ Not loaded'}")
    print(f"   • EfficientNetB0: {'✓ Loaded' if unified_model.efficientnet_model else '✗ Not loaded'}")
    print(f"🏥 Classes: {len(UNIFIED_CLASSES)} ({', '.join(UNIFIED_CLASSES)})")
    print(f"🔗 Hugging Face API: {'Available' if HUGGINGFACE_AVAILABLE else 'Not available (using fallback reports)'}")
    if HUGGINGFACE_AVAILABLE:
        print(f"   Model: {config.HUGGINGFACE_MODEL}")
    print("🌐 Access the application at: http://localhost:5000")
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
