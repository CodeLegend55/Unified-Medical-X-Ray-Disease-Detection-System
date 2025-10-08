# Configuration file for the Unified Medical X-Ray Analysis Web Application

# OpenAI API Configuration (Optional)
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY = "your-openai-api-key-here"

# If you don't have an OpenAI API key, leave it as is
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