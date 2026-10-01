# region ABOUT
# ===================================================================================================
# > This script separates RGB images from AGGREGATED_RGB_IMAGES into training/validation and test
# > sets based on source_ids text files. Test sources are ONLY ORIGINAL images from RADCAT_F_UNPACKED
# > (NO augmentations), while trainval sources INCLUDE their augmentations. Results are organized
# > into TRAINVAL_SOURCES_RGB and TEST_SOURCES_RGB folders with file statistics displayed.
# ===================================================================================================
# endregion

# region IMPORTS
import os
import shutil
from pathlib import Path
from collections import defaultdict
# endregion

# region PATHS
AGGREGATED_RGB_IMAGES_DIRECTORY = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/AGGREGATED_RGB_IMAGES"
RADCAT_F_UNPACKED_DIRECTORY = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F_UNPACKED"
SOURCE_IDS_TRAINVAL_FILE = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Image Augmentation/source_ids_val_train.txt"
SOURCE_IDS_TEST_FILE = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Image Augmentation/source_ids_test.txt"

TRAINVAL_OUTPUT_DIRECTORY = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TRAINVAL_SOURCES_RGB"
TEST_OUTPUT_DIRECTORY = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TEST_SOURCES_RGB"
# endregion

# region FUNCTIONS
def load_source_ids_from_file(filepath):
    """
    Load source IDs from a text file (one ID per line).
    
    Args:
        filepath (str): Path to the source IDs text file
    
    Returns:
        set: Set of source IDs as integers
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return set()
    
    source_ids = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                source_id = line.strip()
                if source_id:  # Skip empty lines
                    try:
                        source_ids.add(int(source_id))
                    except ValueError:
                        print(f"Warning: Could not convert '{source_id}' to integer")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return source_ids


def extract_source_id_from_filename(filename):
    """
    Extract source ID from filename.
    Handles both original files (e.g., "263_RGB.png") and augmented files (e.g., "263_0.png").
    
    Args:
        filename (str): Filename to parse
    
    Returns:
        tuple: (source_id as int, is_augmented as bool)
    """
    # Remove .png extension
    base_name = filename.replace('.png', '')
    
    # Check for augmentation suffix (e.g., "_0", "_1")
    if '_' in base_name:
        parts = base_name.rsplit('_', 1)
        last_part = parts[-1]
        
        # Check if it's a digit (augmentation index)
        if last_part.isdigit():
            # This is an augmented file
            try:
                source_id = int(parts[0])
                return source_id, True
            except ValueError:
                return None, False
        
        # Check if it's "_RGB" suffix (original RADCAT_F_UNPACKED file)
        if last_part.lower() == 'rgb':
            try:
                source_id = int(parts[0])
                return source_id, False
            except ValueError:
                return None, False
    
    # Try to parse entire base_name as source ID
    try:
        source_id = int(base_name)
        return source_id, False
    except ValueError:
        return None, False


def cleanup_and_create_output_directories():
    """
    Remove existing output directories and create fresh ones.
    """
    for output_dir in [TRAINVAL_OUTPUT_DIRECTORY, TEST_OUTPUT_DIRECTORY]:
        if os.path.exists(output_dir):
            print(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")


def copy_files_to_folders(trainval_source_ids, test_source_ids):
    """
    Copy PNG files from AGGREGATED_RGB_IMAGES to appropriate folders based on source ID.
    
    Args:
        trainval_source_ids (set): Source IDs for training/validation
        test_source_ids (set): Source IDs for testing (original images only)
    
    Returns:
        dict: Statistics about copied files
    """
    stats = {
        'trainval_total': 0,
        'trainval_original': 0,
        'trainval_augmented': 0,
        'test_total': 0,
        'test_original': 0,
        'skipped': 0
    }
    
    if not os.path.exists(AGGREGATED_RGB_IMAGES_DIRECTORY):
        print(f"Error: {AGGREGATED_RGB_IMAGES_DIRECTORY} not found")
        return stats
    
    print(f"\nProcessing images from {AGGREGATED_RGB_IMAGES_DIRECTORY}...\n")
    
    for filename in os.listdir(AGGREGATED_RGB_IMAGES_DIRECTORY):
        if not filename.lower().endswith('.png'):
            continue
        
        source_id, is_augmented = extract_source_id_from_filename(filename)
        
        if source_id is None:
            print(f"Skipped: Could not parse source ID from {filename}")
            stats['skipped'] += 1
            continue
        
        source_path = os.path.join(AGGREGATED_RGB_IMAGES_DIRECTORY, filename)
        
        # Handle training/validation set (includes augmented versions)
        if source_id in trainval_source_ids:
            output_path = os.path.join(TRAINVAL_OUTPUT_DIRECTORY, filename)
            shutil.copy2(source_path, output_path)
            stats['trainval_total'] += 1
            if is_augmented:
                stats['trainval_augmented'] += 1
            else:
                stats['trainval_original'] += 1
            print(f"  TrainVal: {filename}")
        
        # Handle test set (ONLY original, non-augmented files)
        elif source_id in test_source_ids:
            if not is_augmented:
                # Only copy original (non-augmented) test images
                output_path = os.path.join(TEST_OUTPUT_DIRECTORY, filename)
                shutil.copy2(source_path, output_path)
                stats['test_total'] += 1
                stats['test_original'] += 1
                print(f"  Test: {filename}")
            else:
                # Skip augmented versions for test set
                print(f"  Skipped (augmented test file): {filename}")
                stats['skipped'] += 1
    
    return stats


def count_files_in_directory(directory):
    """
    Count PNG files in a directory.
    
    Args:
        directory (str): Directory path
    
    Returns:
        int: Number of PNG files
    """
    if not os.path.exists(directory):
        return 0
    
    return len([f for f in os.listdir(directory) if f.lower().endswith('.png')])


def display_summary_statistics(stats, trainval_count, test_count):
    """
    Display summary statistics of the splitting operation.
    
    Args:
        stats (dict): Statistics from copy_files_to_folders
        trainval_count (int): Final count of files in TRAINVAL directory
        test_count (int): Final count of files in TEST directory
    """
    print("\n" + "="*80)
    print("TRAIN/VAL/TEST SPLIT SUMMARY")
    print("="*80)
    
    print("\nProcessing Results:")
    print(f"  Training/Validation files copied: {stats['trainval_total']}")
    print(f"    - Original images: {stats['trainval_original']}")
    print(f"    - Augmented images: {stats['trainval_augmented']}")
    print(f"  Test files copied: {stats['test_total']}")
    print(f"    - Original images (only): {stats['test_original']}")
    print(f"  Skipped: {stats['skipped']}")
    
    print("\nFinal Directory Counts:")
    print(f"  {TRAINVAL_OUTPUT_DIRECTORY}")
    print(f"    - Total files: {trainval_count}")
    
    print(f"\n  {TEST_OUTPUT_DIRECTORY}")
    print(f"    - Total files: {test_count}")
    
    print(f"\n  {AGGREGATED_RGB_IMAGES_DIRECTORY}")
    print(f"    - Total files: {count_files_in_directory(AGGREGATED_RGB_IMAGES_DIRECTORY)}")
    
    print(f"\n  {RADCAT_F_UNPACKED_DIRECTORY}")
    print(f"    - Total files: {count_files_in_directory(RADCAT_F_UNPACKED_DIRECTORY)}")
    
    print("\n" + "="*80 + "\n")

# endregion

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("TRAIN/VALIDATION/TEST SPLIT WITH AUGMENTATIONS")
    print("="*80)
    
    # Load source IDs
    print("\nLoading source IDs from text files...")
    trainval_source_ids = load_source_ids_from_file(SOURCE_IDS_TRAINVAL_FILE)
    test_source_ids = load_source_ids_from_file(SOURCE_IDS_TEST_FILE)
    print(f"  Trainval source IDs: {len(trainval_source_ids)}")
    print(f"  Test source IDs: {len(test_source_ids)}")
    
    # Clean up and create output directories
    print("\nPreparing output directories...")
    cleanup_and_create_output_directories()
    
    # Copy files to appropriate directories
    stats = copy_files_to_folders(trainval_source_ids, test_source_ids)
    
    # Get final counts
    trainval_count = count_files_in_directory(TRAINVAL_OUTPUT_DIRECTORY)
    test_count = count_files_in_directory(TEST_OUTPUT_DIRECTORY)
    
    # Display summary
    display_summary_statistics(stats, trainval_count, test_count)
