"""
Script to remove corrupted images from the unified_dataset.
This script reads the list of corrupted images and removes them safely.
"""

import os
import json
from pathlib import Path

def load_corrupted_files(report_file='corrupted_images_report.json'):
    """Load the list of corrupted files from the report."""
    if not os.path.exists(report_file):
        print(f"Error: Report file '{report_file}' not found.")
        print("Please run 'check_corrupted_images.py' first.")
        return None
    
    with open(report_file, 'r') as f:
        data = json.load(f)
    
    return data

def remove_corrupted_images(corrupted_files, dry_run=True):
    """
    Remove corrupted images from the dataset.
    
    Args:
        corrupted_files: List of corrupted file information
        dry_run: If True, only show what would be deleted without actually deleting
    """
    if not corrupted_files:
        print("No corrupted files to remove.")
        return
    
    deleted_count = 0
    failed_count = 0
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Removing corrupted images...")
    print("=" * 70)
    
    for i, file_info in enumerate(corrupted_files, 1):
        file_path = file_info['absolute_path']
        
        if os.path.exists(file_path):
            try:
                if not dry_run:
                    os.remove(file_path)
                    status = "[DELETED]"
                    deleted_count += 1
                else:
                    status = "[WOULD DELETE]"
                    deleted_count += 1
                
                print(f"{status} ({i}/{len(corrupted_files)}) {file_info['path']}")
                
            except Exception as e:
                status = "[FAILED]"
                failed_count += 1
                print(f"{status} ({i}/{len(corrupted_files)}) {file_info['path']}")
                print(f"          Error: {e}")
        else:
            print(f"[SKIP] ({i}/{len(corrupted_files)}) {file_info['path']} (file not found)")
    
    print("=" * 70)
    if dry_run:
        print(f"\nDRY RUN SUMMARY:")
        print(f"Would delete: {deleted_count} files")
    else:
        print(f"\nCLEANUP SUMMARY:")
        print(f"Successfully deleted: {deleted_count} files")
    
    if failed_count > 0:
        print(f"Failed to delete: {failed_count} files")
    
    print(f"Total processed: {len(corrupted_files)} files")

def update_dataset_info(deleted_count):
    """Update the dataset_info.json file after cleanup."""
    info_file = Path(__file__).parent / 'unified_dataset' / 'dataset_info.json'
    
    if not info_file.exists():
        print("\nWarning: dataset_info.json not found. Skipping update.")
        return
    
    try:
        with open(info_file, 'r') as f:
            dataset_info = json.load(f)
        
        # Add cleanup information
        if 'cleanup_history' not in dataset_info:
            dataset_info['cleanup_history'] = []
        
        from datetime import datetime
        cleanup_record = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'corrupted_images_removed': deleted_count,
            'reason': 'Truncated/corrupted JPEG files'
        }
        dataset_info['cleanup_history'].append(cleanup_record)
        
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n✓ Updated dataset_info.json with cleanup record")
        
    except Exception as e:
        print(f"\nWarning: Could not update dataset_info.json: {e}")

def main():
    """Main function to run the cleanup."""
    print("=" * 70)
    print("CORRUPTED IMAGE CLEANUP TOOL")
    print("=" * 70)
    
    # Load the corrupted files report
    report = load_corrupted_files()
    
    if report is None:
        return
    
    corrupted_files = report['corrupted_files']
    
    if len(corrupted_files) == 0:
        print("\n✓ No corrupted files found in the report. Nothing to clean up!")
        return
    
    # Display summary
    print(f"\nFound {len(corrupted_files)} corrupted images to remove:")
    print(f"- Test set: {sum(1 for f in corrupted_files if 'test' in f['path'])} files")
    print(f"- Train set: {sum(1 for f in corrupted_files if 'train' in f['path'])} files")
    print(f"- Validation set: {sum(1 for f in corrupted_files if 'val' in f['path'])} files")
    
    # First, do a dry run
    print("\n" + "=" * 70)
    print("STEP 1: DRY RUN (showing what would be deleted)")
    print("=" * 70)
    remove_corrupted_images(corrupted_files, dry_run=True)
    
    # Ask for confirmation
    print("\n" + "=" * 70)
    response = input("\nDo you want to proceed with deleting these files? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\n" + "=" * 70)
        print("STEP 2: ACTUAL DELETION")
        print("=" * 70)
        remove_corrupted_images(corrupted_files, dry_run=False)
        
        # Update dataset info
        update_dataset_info(len(corrupted_files))
        
        print("\n✓ Cleanup completed successfully!")
        print("\nRECOMMENDATION: Run 'check_corrupted_images.py' again to verify all corrupted files were removed.")
    else:
        print("\n✗ Cleanup cancelled. No files were deleted.")

if __name__ == "__main__":
    main()
