# Configuration file for the Unified Medical X-Ray Analysis Web Application

# LLM Model Configuration
LLM_MODEL_PATH = "LLM/medical_report_model"  # Path to our custom trained medical report model

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