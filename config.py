# Configuration file for the Unified Medical X-Ray Analysis Web Application

# ═══════════════════════════════════════════════════════════════════════════════
# AI API CONFIGURATION - Choose your preferred API for medical report generation
# ═══════════════════════════════════════════════════════════════════════════════

# Report Generation API Selection
# Options: "huggingface", "gemini", "fallback"
# - "huggingface": Use Hugging Face Inference API (requires HUGGINGFACE_API_KEY)
# - "gemini": Use Google Gemini API (requires GEMINI_API_KEY) ⭐ RECOMMENDED
# - "fallback": Use template-based report generation (no API key needed)
REPORT_API = "fallback"  # Change this to your preferred API

# ───────────────────────────────────────────────────────────────────────────────
# Hugging Face API Configuration (Optional)
# ───────────────────────────────────────────────────────────────────────────────
# Get your API key from: https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY = "API_KEY_HERE"  # Replace with your actual API key

# Hugging Face Model Configuration
# You can use different medical/general language models:
# - "mistralai/Mistral-7B-Instruct-v0.2" (general purpose, good for medical text)
# - "microsoft/BioGPT-Large" (biomedical text generation)
# - "meta-llama/Llama-2-7b-chat-hf" (general purpose chat model)
# - "google/flan-t5-large" (instruction-tuned model)
HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ───────────────────────────────────────────────────────────────────────────────
# Google Gemini API Configuration (Optional)
# ───────────────────────────────────────────────────────────────────────────────
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = "API_KEY_HERE"  # Replace with your actual Gemini API key

# Gemini Model Configuration
# Available models:
# - "gemini-1.5-pro" (most capable, best for complex medical reasoning) ⭐ RECOMMENDED
# - "gemini-1.5-flash" (faster, good balance of speed and quality)
# - "gemini-pro" (previous generation, still very capable)
GEMINI_MODEL = "gemini-1.5-pro"

# ───────────────────────────────────────────────────────────────────────────────
# If you don't have API keys, set REPORT_API = "fallback"
# The app will use a template-based report generation system
# ───────────────────────────────────────────────────────────────────────────────

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