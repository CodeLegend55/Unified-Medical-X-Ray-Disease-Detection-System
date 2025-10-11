# Configuration file for the Unified Medical X-Ray Analysis Web Application

# Hugging Face API Configuration (Optional)
# Get your API key from: https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY = "API_KEY_HERE"  # Replace with your actual API key or leave as is for no key

# Hugging Face Model Configuration
# You can use different medical/general language models:
# - "mistralai/Mistral-7B-Instruct-v0.2" (general purpose, good for medical text) ⭐ RECOMMENDED
# - "microsoft/BioGPT-Large" (biomedical text generation)
# - "meta-llama/Llama-2-7b-chat-hf" (general purpose chat model)
# - "google/flan-t5-large" (instruction-tuned model)
HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# If you don't have a Hugging Face API key, leave it as is
# The app will use a fallback report generation system

# Flask Configuration
UPLOAD_FOLDER = "uploads"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Model Configuration
MODEL_PATH = "models"

# Unified Model Classes (8 classes total)
# IMPORTANT: This order MUST match the training notebook exactly!
UNIFIED_CLASSES = [
    "COVID19",
    "PNEUMONIA",
    "TUBERCULOSIS",
    "NORMAL_CHEST",
    "OSTEOPOROSIS",
    "NORMAL_BONE",
    "FRACTURED",
    "NON_FRACTURED"
]

# Class groupings for different problem types
CHEST_CONDITIONS = ["COVID19", "PNEUMONIA", "TUBERCULOSIS", "NORMAL_CHEST"]
FRACTURE_CONDITIONS = ["FRACTURED", "NON_FRACTURED"]
BONE_CONDITIONS = ["OSTEOPOROSIS", "NORMAL_BONE"]

# Server Configuration
DEBUG = True
HOST = "0.0.0.0"
PORT = 5000