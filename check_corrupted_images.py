"""
Script to check for corrupted images in the unified_dataset folder.
Scans all image files and reports any that cannot be opened or are corrupted.
"""

import os
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

def check_image(image_path):
    """
    Check if an image file is corrupted.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Try to open and verify the image
        with Image.open(image_path) as img:
            # Verify by loading the image data
            img.verify()
        
        # Re-open to check if we can actually load the data
        with Image.open(image_path) as img:
            img.load()
            
        return True, None
    except Exception as e:
        return False, str(e)

def scan_dataset(dataset_path):
    """
    Scan the dataset for corrupted images.
    
    Args:
        dataset_path: Path to the unified_dataset folder
        
    Returns:
        dict: Results containing valid and corrupted image information
    """
    results = {
        'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': 0,
        'valid_images': 0,
        'corrupted_images': 0,
        'corrupted_files': [],
        'summary_by_category': {}
    }
    
    # Image extensions to check
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    print(f"Scanning dataset at: {dataset_path}")
    print("=" * 70)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Check if file has an image extension
            if Path(file).suffix.lower() in image_extensions:
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(image_path, dataset_path)
                
                # Extract category from path
                path_parts = relative_path.split(os.sep)
                if len(path_parts) >= 2:
                    split = path_parts[0]  # train/val/test
                    category = path_parts[1]  # disease category
                    category_key = f"{split}/{category}"
                else:
                    category_key = "unknown"
                
                # Initialize category counter if needed
                if category_key not in results['summary_by_category']:
                    results['summary_by_category'][category_key] = {
                        'total': 0,
                        'valid': 0,
                        'corrupted': 0
                    }
                
                results['total_images'] += 1
                results['summary_by_category'][category_key]['total'] += 1
                
                # Check if image is valid
                is_valid, error_msg = check_image(image_path)
                
                if is_valid:
                    results['valid_images'] += 1
                    results['summary_by_category'][category_key]['valid'] += 1
                else:
                    results['corrupted_images'] += 1
                    results['summary_by_category'][category_key]['corrupted'] += 1
                    
                    corrupted_info = {
                        'path': relative_path,
                        'absolute_path': image_path,
                        'category': category_key,
                        'error': error_msg,
                        'size_bytes': os.path.getsize(image_path) if os.path.exists(image_path) else 0
                    }
                    results['corrupted_files'].append(corrupted_info)
                    
                    print(f"[CORRUPTED] {relative_path}")
                    print(f"            Error: {error_msg}")
                    print(f"            Size: {corrupted_info['size_bytes']} bytes")
                    print("-" * 70)
                
                # Progress indicator
                if results['total_images'] % 100 == 0:
                    print(f"Processed {results['total_images']} images...", end='\r')
    
    return results

def save_results(results, output_file='corrupted_images_report.json'):
    """Save scan results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to: {output_file}")

def print_summary(results):
    """Print a summary of the scan results."""
    print("\n" + "=" * 70)
    print("SCAN SUMMARY")
    print("=" * 70)
    print(f"Total images scanned: {results['total_images']}")
    print(f"Valid images: {results['valid_images']}")
    print(f"Corrupted images: {results['corrupted_images']}")
    print(f"Success rate: {(results['valid_images']/results['total_images']*100):.2f}%")
    
    if results['corrupted_images'] > 0:
        print("\n" + "-" * 70)
        print("CORRUPTED IMAGES BY CATEGORY:")
        print("-" * 70)
        for category, stats in sorted(results['summary_by_category'].items()):
            if stats['corrupted'] > 0:
                print(f"{category}: {stats['corrupted']} corrupted out of {stats['total']} images")
        
        print("\n" + "-" * 70)
        print("CORRUPTED FILES LIST:")
        print("-" * 70)
        for i, file_info in enumerate(results['corrupted_files'], 1):
            print(f"{i}. {file_info['path']}")
            print(f"   Error: {file_info['error']}")
    else:
        print("\n✓ No corrupted images found!")

def main():
    """Main function to run the corrupted image checker."""
    # Path to unified_dataset
    dataset_path = Path(__file__).parent / 'unified_dataset'
    
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    print("Starting corrupted image scan...")
    print(f"Dataset path: {dataset_path}\n")
    
    # Scan the dataset
    results = scan_dataset(str(dataset_path))
    
    # Print summary
    print_summary(results)
    
    # Save detailed results
    save_results(results)
    
    # Optionally create a list of corrupted files for easy deletion
    if results['corrupted_images'] > 0:
        corrupted_paths_file = 'corrupted_images_list.txt'
        with open(corrupted_paths_file, 'w') as f:
            for file_info in results['corrupted_files']:
                f.write(f"{file_info['absolute_path']}\n")
        print(f"List of corrupted file paths saved to: {corrupted_paths_file}")

if __name__ == "__main__":
    main()
