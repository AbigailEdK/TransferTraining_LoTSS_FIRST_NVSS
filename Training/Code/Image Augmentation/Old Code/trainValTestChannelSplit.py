# region ABOUT
# ===================================================================================================
# > This script separates channel data (.npy files) from SPLIT_SOURCES into training/validation and
# > test sets based on source_ids text files. Each survey channel (FIRST, NVSS, LoTSS) is organized
# > into separate folders. Test sources include ONLY original files (NO augmentations), while trainval
# > sources INCLUDE their augmentations. Results are organized into 6 folders with statistics displayed.
# ===================================================================================================
# endregion

# region IMPORTS
import os
import shutil
from pathlib import Path
from collections import defaultdict
# endregion

# region PATHS
SPLIT_SOURCES_DIRECTORY = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/SPLIT_SOURCES"

SOURCE_IDS_TRAINVAL_FILE = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Image Augmentation/source_ids_val_train.txt"
SOURCE_IDS_TEST_FILE = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Image Augmentation/source_ids_test.txt"

# Output directories (one for each survey per split)
TRAINVAL_SOURCES_FIRST = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TRAINVAL_SOURCES_FIRST"
TRAINVAL_SOURCES_NVSS = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TRAINVAL_SOURCES_NVSS"
TRAINVAL_SOURCES_LoTSS = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TRAINVAL_SOURCES_LoTSS"

TEST_SOURCES_FIRST = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TEST_SOURCES_FIRST"
TEST_SOURCES_NVSS = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TEST_SOURCES_NVSS"
TEST_SOURCES_LoTSS = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TEST_SOURCES_LoTSS"

# Mapping: Color folder → Survey names and output directories
CHANNEL_MAPPING = {
    "Red": {
        "survey": "FIRST",
        "trainval_dir": TRAINVAL_SOURCES_FIRST,
        "test_dir": TEST_SOURCES_FIRST
    },
    "Green": {
        "survey": "LoTSS",
        "trainval_dir": TRAINVAL_SOURCES_LoTSS,
        "test_dir": TEST_SOURCES_LoTSS
    },
    "Blue": {
        "survey": "NVSS",
        "trainval_dir": TRAINVAL_SOURCES_NVSS,
        "test_dir": TEST_SOURCES_NVSS
    }
}
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
    Extract source ID and augmentation status from channel filename.
    Handles both original files (e.g., "263_RGB_R.npy") and augmented files (e.g., "263_0_R.npy").
    
    Args:
        filename (str): Filename to parse (e.g., "263_RGB_R.npy" or "263_0_R.npy")
    
    Returns:
        tuple: (source_id as int or None, is_augmented as bool)
    """
    # Remove .npy extension
    base_name = filename.replace('.npy', '')
    
    # Remove color suffix (R, G, or B)
    if base_name and base_name[-1] in ['R', 'G', 'B']:
        base_name = base_name[:-2]  # Remove "_X" where X is R, G, or B
    
    # Now parse the remaining part to extract source ID and augmentation status
    if '_' in base_name:
        parts = base_name.rsplit('_', 1)
        last_part = parts[-1]
        
        # Check if it's a digit (augmentation index)
        if last_part.isdigit():
            # This is an augmented file (e.g., "263_0")
            try:
                source_id = int(parts[0])
                return source_id, True
            except ValueError:
                return None, False
        
        # Check if it's "RGB" suffix (original file)
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
    all_dirs = [TRAINVAL_SOURCES_FIRST, TRAINVAL_SOURCES_NVSS, TRAINVAL_SOURCES_LoTSS,
                TEST_SOURCES_FIRST, TEST_SOURCES_NVSS, TEST_SOURCES_LoTSS]
    
    for output_dir in all_dirs:
        if os.path.exists(output_dir):
            print(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)


def copy_channel_files(trainval_source_ids, test_source_ids):
    """
    Copy channel (.npy) files from SPLIT_SOURCES to appropriate channel-specific folders.
    
    Args:
        trainval_source_ids (set): Source IDs for training/validation
        test_source_ids (set): Source IDs for testing
    
    Returns:
        dict: Statistics about copied files
    """
    stats = defaultdict(lambda: {
        'trainval_total': 0,
        'trainval_original': 0,
        'trainval_augmented': 0,
        'test_total': 0,
        'test_original': 0,
        'skipped': 0
    })
    
    # Check if SPLIT_SOURCES exists
    if not os.path.exists(SPLIT_SOURCES_DIRECTORY):
        print(f"Error: {SPLIT_SOURCES_DIRECTORY} not found")
        return stats
    
    # Process each color channel folder
    for color_folder, channel_info in CHANNEL_MAPPING.items():
        color_path = os.path.join(SPLIT_SOURCES_DIRECTORY, color_folder)
        survey_name = channel_info["survey"]
        trainval_dir = channel_info["trainval_dir"]
        test_dir = channel_info["test_dir"]
        
        if not os.path.exists(color_path):
            print(f"Warning: {color_path} not found, skipping")
            continue
        
        print(f"\nProcessing {survey_name} channel ({color_folder})...")
        
        for filename in os.listdir(color_path):
            if not filename.lower().endswith('.npy'):
                continue
            
            source_id, is_augmented = extract_source_id_from_filename(filename)
            
            if source_id is None:
                print(f"  Skipped: Could not parse source ID from {filename}")
                stats[survey_name]['skipped'] += 1
                continue
            
            source_path = os.path.join(color_path, filename)
            
            # Handle training/validation set (includes augmented versions)
            if source_id in trainval_source_ids:
                output_path = os.path.join(trainval_dir, filename)
                shutil.copy2(source_path, output_path)
                stats[survey_name]['trainval_total'] += 1
                if is_augmented:
                    stats[survey_name]['trainval_augmented'] += 1
                else:
                    stats[survey_name]['trainval_original'] += 1
                print(f"  TrainVal ({survey_name}): {filename}")
            
            # Handle test set (ONLY original, non-augmented files)
            elif source_id in test_source_ids:
                if not is_augmented:
                    # Only copy original (non-augmented) test files
                    output_path = os.path.join(test_dir, filename)
                    shutil.copy2(source_path, output_path)
                    stats[survey_name]['test_total'] += 1
                    stats[survey_name]['test_original'] += 1
                    print(f"  Test ({survey_name}): {filename}")
                else:
                    # Skip augmented versions for test set
                    print(f"  Skipped (augmented test file): {filename}")
                    stats[survey_name]['skipped'] += 1
    
    return stats


def count_files_in_directory(directory):
    """
    Count .npy files in a directory.
    
    Args:
        directory (str): Directory path
    
    Returns:
        int: Number of .npy files
    """
    if not os.path.exists(directory):
        return 0
    
    return len([f for f in os.listdir(directory) if f.lower().endswith('.npy')])


def display_summary_statistics(stats):
    """
    Display summary statistics of the channel splitting operation.
    
    Args:
        stats (dict): Statistics from copy_channel_files
    """
    print("\n" + "="*80)
    print("TRAIN/VAL/TEST CHANNEL SPLIT SUMMARY")
    print("="*80)
    
    print("\nProcessing Results by Survey:")
    total_trainval = 0
    total_test = 0
    
    for survey in ["FIRST", "LoTSS", "NVSS"]:
        if survey in stats:
            survey_stats = stats[survey]
            print(f"\n{survey}:")
            print(f"  Training/Validation files: {survey_stats['trainval_total']}")
            print(f"    - Original images: {survey_stats['trainval_original']}")
            print(f"    - Augmented images: {survey_stats['trainval_augmented']}")
            print(f"  Test files: {survey_stats['test_total']}")
            print(f"    - Original images (only): {survey_stats['test_original']}")
            print(f"  Skipped: {survey_stats['skipped']}")
            total_trainval += survey_stats['trainval_total']
            total_test += survey_stats['test_total']
    
    print("\n" + "-"*80)
    print("\nFinal Directory Counts:")
    
    print(f"\nTrainVal Directories:")
    for survey, channel_info in zip(["FIRST", "LoTSS", "NVSS"], 
                                    [CHANNEL_MAPPING["Red"], CHANNEL_MAPPING["Green"], CHANNEL_MAPPING["Blue"]]):
        directory = channel_info["trainval_dir"]
        count = count_files_in_directory(directory)
        print(f"  {survey}: {count} files")
    
    print(f"\nTest Directories:")
    for survey, channel_info in zip(["FIRST", "LoTSS", "NVSS"], 
                                    [CHANNEL_MAPPING["Red"], CHANNEL_MAPPING["Green"], CHANNEL_MAPPING["Blue"]]):
        directory = channel_info["test_dir"]
        count = count_files_in_directory(directory)
        print(f"  {survey}: {count} files")
    
    print(f"\nSource Directory (SPLIT_SOURCES):")
    for color in ["Red", "Green", "Blue"]:
        color_path = os.path.join(SPLIT_SOURCES_DIRECTORY, color)
        count = count_files_in_directory(color_path)
        print(f"  {color}: {count} files")
    
    print("\n" + "="*80 + "\n")

# endregion

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("TRAIN/VALIDATION/TEST CHANNEL SPLIT")
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
    print(f"Created 6 output directories (TrainVal + Test for FIRST, LoTSS, NVSS)")
    
    # Copy channel files to appropriate directories
    stats = copy_channel_files(trainval_source_ids, test_source_ids)
    
    # Display summary
    display_summary_statistics(stats)
