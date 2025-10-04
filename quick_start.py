"""
Quick Start Script for Unified Model Training
==============================================

This script helps you set up everything needed for training the unified model.
"""

import os
import subprocess
import sys
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")

def check_kaggle_api():
    """Check if Kaggle API is installed and configured."""
    try:
        import kaggle
        print("✓ Kaggle API is installed")
        return True
    except ImportError:
        print("✗ Kaggle API is not installed")
        return False

def install_kaggle_api():
    """Install Kaggle API."""
    print("Installing Kaggle API...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✓ Kaggle API installed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to install Kaggle API: {e}")
        return False

def download_datasets():
    """Download datasets using Kaggle API."""
    datasets = [
        {
            'name': 'Chest X-ray (COVID, Pneumonia, TB)',
            'slug': 'tawsifurrahman/covid19-radiography-database',
            'path': 'datasets/chest_xray_merged'
        },
        {
            'name': 'Osteoporosis',
            'slug': 'stevepython/osteoporosis-knee-xray-dataset',
            'path': 'datasets/osteoporosis'
        },
        {
            'name': 'Bone Fracture Multi-Region',
            'slug': 'pkdarabi/bone-fracture-detection-computer-vision-project',
            'path': 'datasets/Bone_Fracture_Binary_Classification'
        }
    ]
    
    print_header("DOWNLOADING DATASETS FROM KAGGLE")
    
    for dataset in datasets:
        print(f"\n📥 Downloading: {dataset['name']}")
        print(f"   Dataset: {dataset['slug']}")
        print(f"   Destination: {dataset['path']}")
        
        # Create directory
        Path(dataset['path']).mkdir(parents=True, exist_ok=True)
        
        try:
            # Download using kaggle API
            cmd = f"kaggle datasets download -d {dataset['slug']} -p {dataset['path']} --unzip"
            subprocess.check_call(cmd, shell=True)
            print(f"✓ Downloaded {dataset['name']}")
        except Exception as e:
            print(f"✗ Failed to download {dataset['name']}: {e}")
            print(f"   Please download manually from: https://www.kaggle.com/datasets/{dataset['slug']}")
            return False
    
    return True

def setup_environment():
    """Set up the environment for training."""
    print_header("ENVIRONMENT SETUP")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️ Python 3.8 or higher is recommended")
    else:
        print("✓ Python version is compatible")
    
    # Check for requirements.txt
    if Path('requirements.txt').exists():
        print("\n✓ Found requirements.txt")
        response = input("Do you want to install required packages? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            print("\nInstalling packages...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                print("✓ Packages installed successfully")
            except Exception as e:
                print(f"✗ Failed to install packages: {e}")
                return False
    else:
        print("⚠️ requirements.txt not found")
        print("   Please ensure all required packages are installed:")
        print("   - torch, torchvision")
        print("   - numpy, pandas, pillow")
        print("   - scikit-learn, matplotlib, seaborn")
        print("   - tqdm, flask")
    
    return True

def main():
    """Main function."""
    print_header("🚀 UNIFIED MODEL TRAINING - QUICK START")
    
    print("""
This script will help you:
1. Set up the Python environment
2. Download required datasets from Kaggle
3. Prepare the unified dataset
4. Guide you to start training

Prerequisites:
- Kaggle account (https://www.kaggle.com)
- Kaggle API credentials (kaggle.json)
  Download from: https://www.kaggle.com/settings -> API -> Create New API Token
  Place in: C:\\Users\\YourUsername\\.kaggle\\kaggle.json (Windows)
           ~/.kaggle/kaggle.json (Linux/Mac)
    """)
    
    response = input("Do you want to continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Setup cancelled.")
        return
    
    # Step 1: Set up environment
    if not setup_environment():
        print("\n⚠️ Environment setup failed. Please install packages manually.")
        return
    
    # Step 2: Check Kaggle API
    print_header("KAGGLE API SETUP")
    
    has_kaggle = check_kaggle_api()
    if not has_kaggle:
        response = input("Do you want to install Kaggle API? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            if not install_kaggle_api():
                print("\n⚠️ Please install Kaggle API manually: pip install kaggle")
                return
        else:
            print("\n⚠️ Kaggle API is required to download datasets automatically.")
            print("   You can download datasets manually from the URLs in README.md")
            return
    
    # Check kaggle.json
    kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_config.exists():
        print("\n⚠️ Kaggle API credentials not found!")
        print(f"   Expected location: {kaggle_config}")
        print("\nTo get your credentials:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Save kaggle.json to the location above")
        return
    else:
        print(f"✓ Found Kaggle credentials at {kaggle_config}")
    
    # Step 3: Download datasets
    response = input("\nDo you want to download datasets now? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        if not download_datasets():
            print("\n⚠️ Dataset download incomplete. Please download manually.")
            return
    else:
        print("\n⚠️ Skipping dataset download.")
        print("   Please download datasets manually before training.")
    
    # Step 4: Prepare unified dataset
    print_header("DATASET PREPARATION")
    
    response = input("Do you want to prepare the unified dataset now? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        print("\nRunning dataset preparation script...")
        try:
            subprocess.check_call([sys.executable, "prepare_unified_dataset.py"])
            print("✓ Dataset prepared successfully")
        except Exception as e:
            print(f"✗ Dataset preparation failed: {e}")
            print("   Please run: python prepare_unified_dataset.py")
            return
    
    # Final instructions
    print_header("✅ SETUP COMPLETE!")
    
    print("""
Next Steps:
1. Review the prepared dataset in 'unified_dataset/' folder
2. Open the training notebook:
   
   jupyter notebook unified_model_training.ipynb
   
   or
   
   jupyter lab unified_model_training.ipynb

3. Run all cells in the notebook to train the models

4. After training, the models will be saved in 'models/' folder:
   - unified_ResNet50.pth
   - unified_DenseNet121.pth
   - unified_EfficientNetB0.pth

5. Update your Flask application to use the unified models

Training Tips:
- Training time: 2-4 hours per model (with GPU)
- Use GPU for faster training (CPU will be very slow)
- Monitor the training progress and adjust hyperparameters if needed
- Check the confusion matrix and classification report for each model

For detailed instructions, see: README.md

Happy Training! 🚀
    """)

if __name__ == '__main__':
    main()
