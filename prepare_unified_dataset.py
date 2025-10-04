"""
Unified Dataset Preparation Script
===================================
This script combines multiple X-ray datasets into a single unified dataset for training
a multi-class classification model that can detect all diseases.

Datasets to combine:
1. Chest X-ray (COVID, TB, Pneumonia, Normal) - https://www.kaggle.com/datasets/jeevanrushi/chest-xray
2. Osteoporosis - https://www.kaggle.com/datasets/mrmann007/osteoporosis
3. Bone Fracture Multi-Region X-ray - https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

Final Classes:
- COVID19 (Chest X-ray)
- PNEUMONIA (Chest X-ray)
- TUBERCULOSIS (Chest X-ray)
- NORMAL_CHEST (Chest X-ray - Normal)
- OSTEOPOROSIS (Knee/Bone X-ray)
- NORMAL_BONE (Bone X-ray - Normal/No Osteoporosis)
- FRACTURED (Hand/Leg/Other bone X-rays)
- NON_FRACTURED (Hand/Leg/Other bone X-rays - Normal)
"""

import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import json

class UnifiedDatasetPreparator:
    """
    Prepare a unified dataset from multiple X-ray disease detection datasets.
    """
    
    # Define the unified class labels
    UNIFIED_CLASSES = [
        'COVID19',           # 0
        'PNEUMONIA',         # 1
        'TUBERCULOSIS',      # 2
        'NORMAL_CHEST',      # 3
        'OSTEOPOROSIS',      # 4
        'NORMAL_BONE',       # 5
        'FRACTURED',         # 6
        'NON_FRACTURED'      # 7
    ]
    
    def __init__(self, base_dir='datasets'):
        """
        Initialize the dataset preparator.
        
        Args:
            base_dir: Base directory where datasets are stored
        """
        self.base_dir = Path(base_dir)
        self.unified_dir = Path('unified_dataset')
        
        # Source dataset paths (actual structure)
        self.chest_xray_path = self.base_dir / 'chest_xray_merged'
        self.osteoporosis_path = self.base_dir / 'osteoporosis'
        self.fracture_path = self.base_dir / 'Bone_Fracture_Binary_Classification' / 'Bone_Fracture_Binary_Classification'
        
        # Output paths
        self.train_dir = self.unified_dir / 'train'
        self.val_dir = self.unified_dir / 'val'
        self.test_dir = self.unified_dir / 'test'
        
        # Create output directories
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            for class_name in self.UNIFIED_CLASSES:
                (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    def copy_images(self, source_files, target_dir, class_name):
        """Copy images to target directory with class name."""
        copied = 0
        for src_file in source_files:
            if src_file.exists():
                try:
                    # Verify it's a valid image
                    img = Image.open(src_file)
                    img.verify()
                    
                    # Create unique filename
                    dest_file = target_dir / class_name / f"{class_name.lower()}_{copied:05d}{src_file.suffix}"
                    shutil.copy2(src_file, dest_file)
                    copied += 1
                except Exception as e:
                    print(f"  ⚠️ Skipping invalid image {src_file.name}: {e}")
        return copied
    
    def prepare_chest_xray_dataset(self):
        """
        Prepare chest X-ray dataset (COVID, Pneumonia, TB, Normal).
        
        Expected structure:
        chest_xray_merged/
            train/
                covid/
                normal/
                pneumonia/
                tb/
            val/
                covid/
                normal/
                pneumonia/
                tb/
            test/
                covid/
                normal/
                pneumonia/
                tb/
        """
        print("\n📁 Processing Chest X-ray Dataset...")
        
        if not self.chest_xray_path.exists():
            print(f"  ⚠️ Chest X-ray dataset not found at {self.chest_xray_path}")
            print(f"  📥 Please ensure the dataset is in: datasets/chest_xray_merged/")
            return
        
        # Mapping from source folders to unified classes
        chest_mapping = {
            'covid': 'COVID19',
            'pneumonia': 'PNEUMONIA',
            'tb': 'TUBERCULOSIS',
            'normal': 'NORMAL_CHEST'
        }
        
        # Process each split (train, val, test)
        for split in ['train', 'val', 'test']:
            split_path = self.chest_xray_path / split
            
            if not split_path.exists():
                print(f"  ⚠️ Split '{split}' not found at {split_path}")
                continue
            
            for source_folder, unified_class in chest_mapping.items():
                source_class_path = split_path / source_folder
                
                if source_class_path.exists():
                    # Get all image files
                    image_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                        image_files.extend(list(source_class_path.glob(ext)))
                    
                    if image_files:
                        # Determine target directory based on split
                        if split == 'train':
                            target_dir = self.train_dir
                        elif split == 'val':
                            target_dir = self.val_dir
                        else:
                            target_dir = self.test_dir
                        
                        # Copy files
                        count = self.copy_images(image_files, target_dir, unified_class)
                        print(f"  Found {len(image_files)} images in {split}/{source_folder} -> {unified_class}: {count} copied")
    
    def prepare_osteoporosis_dataset(self):
        """
        Prepare osteoporosis dataset.
        
        Expected structure:
        osteoporosis/
            normal/
            osteoporosis/
        """
        print("\n📁 Processing Osteoporosis Dataset...")
        
        if not self.osteoporosis_path.exists():
            print(f"  ⚠️ Osteoporosis dataset not found at {self.osteoporosis_path}")
            print(f"  📥 Please ensure the dataset is in: datasets/osteoporosis/")
            return
        
        # Mapping from source folders to unified classes
        osteo_mapping = {
            'osteoporosis': 'OSTEOPOROSIS',
            'normal': 'NORMAL_BONE'
        }
        
        for source_folder, unified_class in osteo_mapping.items():
            source_path = self.osteoporosis_path / source_folder
            
            if source_path.exists():
                # Get all image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(list(source_path.glob(ext)))
                
                if image_files:
                    print(f"  Found {len(image_files)} images in {source_folder} -> {unified_class}")
                    
                    # Split into train/val/test (70/15/15)
                    train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
                    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
                    
                    # Copy files
                    train_count = self.copy_images(train_files, self.train_dir, unified_class)
                    val_count = self.copy_images(val_files, self.val_dir, unified_class)
                    test_count = self.copy_images(test_files, self.test_dir, unified_class)
                    
                    print(f"    ✓ Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    def prepare_fracture_dataset(self):
        """
        Prepare fracture dataset.
        
        Expected structure:
        Bone_Fracture_Binary_Classification/
            Bone_Fracture_Binary_Classification/
                train/
                    fractured/
                    not fractured/
                val/
                    fractured/
                    not fractured/
                test/
                    fractured/
                    not fractured/
        """
        print("\n📁 Processing Fracture Dataset...")
        
        if not self.fracture_path.exists():
            print(f"  ⚠️ Fracture dataset not found at {self.fracture_path}")
            print(f"  📥 Please ensure the dataset is in: datasets/Bone_Fracture_Binary_Classification/")
            return
        
        # Mapping from source folders to unified classes
        fracture_mapping = {
            'fractured': 'FRACTURED',
            'not fractured': 'NON_FRACTURED'
        }
        
        # Process each split (train, val, test)
        for split in ['train', 'val', 'test']:
            split_path = self.fracture_path / split
            
            if not split_path.exists():
                print(f"  ⚠️ Split '{split}' not found at {split_path}")
                continue
            
            for source_folder, unified_class in fracture_mapping.items():
                source_class_path = split_path / source_folder
                
                if source_class_path.exists():
                    # Get all image files (including subdirectories for multi-region)
                    image_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                        image_files.extend(list(source_class_path.rglob(ext)))
                    
                    if image_files:
                        # Determine target directory based on split
                        if split == 'train':
                            target_dir = self.train_dir
                        elif split == 'val':
                            target_dir = self.val_dir
                        else:
                            target_dir = self.test_dir
                        
                        # Copy files
                        count = self.copy_images(image_files, target_dir, unified_class)
                        print(f"  Found {len(image_files)} images in {split}/{source_folder} -> {unified_class}: {count} copied")
    
    def generate_dataset_info(self):
        """Generate dataset information and statistics."""
        print("\n📊 Generating Dataset Statistics...")
        
        stats = {
            'classes': self.UNIFIED_CLASSES,
            'num_classes': len(self.UNIFIED_CLASSES),
            'splits': {}
        }
        
        for split_name, split_dir in [('train', self.train_dir), 
                                       ('val', self.val_dir), 
                                       ('test', self.test_dir)]:
            split_stats = {}
            total = 0
            
            for class_name in self.UNIFIED_CLASSES:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    count = len(list(class_dir.glob('*.*')))
                    split_stats[class_name] = count
                    total += count
                else:
                    split_stats[class_name] = 0
            
            split_stats['total'] = total
            stats['splits'][split_name] = split_stats
            
            print(f"\n  {split_name.upper()} Set:")
            for class_name, count in split_stats.items():
                if class_name != 'total':
                    print(f"    {class_name}: {count}")
            print(f"    TOTAL: {total}")
        
        # Save statistics to JSON
        stats_file = self.unified_dir / 'dataset_info.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ Dataset info saved to {stats_file}")
        
        return stats
    
    def prepare_all(self):
        """Prepare the complete unified dataset."""
        print("=" * 70)
        print("🚀 UNIFIED MEDICAL X-RAY DATASET PREPARATION")
        print("=" * 70)
        print(f"\nOutput directory: {self.unified_dir.absolute()}")
        print(f"Unified classes: {', '.join(self.UNIFIED_CLASSES)}")
        
        # Process each dataset
        self.prepare_chest_xray_dataset()
        self.prepare_osteoporosis_dataset()
        self.prepare_fracture_dataset()
        
        # Generate statistics
        stats = self.generate_dataset_info()
        
        print("\n" + "=" * 70)
        print("✅ DATASET PREPARATION COMPLETE!")
        print("=" * 70)
        print(f"\nTotal samples: {sum([stats['splits'][split]['total'] for split in stats['splits']])}")
        print(f"  - Train: {stats['splits']['train']['total']}")
        print(f"  - Validation: {stats['splits']['val']['total']}")
        print(f"  - Test: {stats['splits']['test']['total']}")
        
        print("\n📝 Next Steps:")
        print("  1. Review the dataset structure in 'unified_dataset/' folder")
        print("  2. Open and run 'unified_model_training.ipynb' to train the model")
        print("  3. The trained model will detect all 8 disease classes automatically")
        
        return stats


def main():
    """Main function to run dataset preparation."""
    print("\n" + "=" * 70)
    print("INSTRUCTIONS FOR DATASET PREPARATION")
    print("=" * 70)
    print("""
Your datasets should be organized as follows:

datasets/
    chest_xray_merged/
        train/
            covid/
            normal/
            pneumonia/
            tb/
        val/
            covid/
            normal/
            pneumonia/
            tb/
        test/
            covid/
            normal/
            pneumonia/
            tb/
    
    osteoporosis/
        normal/
        osteoporosis/
    
    Bone_Fracture_Binary_Classification/
        Bone_Fracture_Binary_Classification/
            train/
                fractured/
                not fractured/
            val/
                fractured/
                not fractured/
            test/
                fractured/
                not fractured/

Note: The script will process these datasets and create a unified structure.
    """)
    
    response = input("\nHave you organized all datasets in the correct structure? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        preparator = UnifiedDatasetPreparator()
        stats = preparator.prepare_all()
    else:
        print("\n⚠️ Please organize the datasets first and then run this script again.")
        print("   Make sure your datasets folder follows the structure shown above.")


if __name__ == '__main__':
    main()
