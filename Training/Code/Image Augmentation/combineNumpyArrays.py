# region ABOUT
# ===================================================================================================
# > This script loads the individual 3-channel numpy arrays and combines them into
# > train/test datasets in the format expected by get_dataset().
#
# > Saves X, y, and names arrays for both train and test sets.
# ===================================================================================================
# endregion

# region IMPORTS
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
# endregion

# region PATHS
COMBINED_NUMPY_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F_COMBINED_NUMPY"
JSON_MAPPING_PATH = "/home/abigaildeklerk/Desktop/DeKlerk_Models/source_id_classification_mapping.json"
OUTPUT_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/COMBINED_DATASETS"
# endregion

# region FUNCTIONS

def load_radcat_mapping(json_path):
    """
    Load source ID to classification mapping from JSON file.
    
    Args:
        json_path (str): Path to source_id_classification_mapping.json
        
    Returns:
        dict: Mapping from source_id (str) to class label
    """
    with open(json_path, 'r') as f:
        mapping_data = json.load(f)
    
    mapping = mapping_data['id_to_classification']
    print(f"Loaded classification mapping for {len(mapping)} sources from JSON")
    return mapping


def combine_arrays_into_dataset(combined_dir, radcat_mapping, test_size=0.2, random_state=42):
    """
    Load all individual .npy files and combine into X, y, names arrays.
    Then split into train/test sets.
    
    Args:
        combined_dir (str): Directory containing individual .npy files
        radcat_mapping (dict): Mapping from source_id to class label
        test_size (float): Proportion of data to use as test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, names_train, names_test)
    """
    
    # Get all .npy files sorted by source ID
    npy_files = sorted([f for f in os.listdir(combined_dir) if f.endswith('.npy')])
    print(f"Found {len(npy_files)} combined numpy files")
    
    if len(npy_files) == 0:
        print("ERROR: No .npy files found in combined directory")
        return None
    
    # Load all arrays and create lists
    X_list = []
    y_list = []
    names_list = []
    
    for idx, npy_file in enumerate(npy_files):
        source_id = npy_file.replace('.npy', '')
        
        # Load the combined array
        array_path = os.path.join(combined_dir, npy_file)
        combined_array = np.load(array_path)
        
        # Check if this source has a label
        if source_id not in radcat_mapping:
            print(f"Warning: Source {source_id} not found in RADCAT, skipping")
            continue
        
        X_list.append(combined_array)
        y_list.append(radcat_mapping[source_id])
        names_list.append(source_id)
        
        if (idx + 1) % 500 == 0:
            print(f"Loaded {idx + 1}/{len(npy_files)} arrays")
    
    print(f"Successfully loaded {len(X_list)} arrays with labels")
    
    # Stack into single array
    X = np.array(X_list)
    y = np.array(y_list)
    names = np.array(names_list)
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {y.shape}")
    print(f"Names shape: {names.shape}")
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for class_name, count in zip(unique, counts):
        print(f"  {class_name}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split into train/test
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Stratified split to maintain class distribution
    )
    
    print(f"\nTrain/test split (test_size={test_size}):")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, names_train, names_test


def save_datasets(output_dir, X_train, X_test, y_train, y_test, names_train, names_test):
    """
    Save train/test datasets to disk.
    
    Args:
        output_dir (str): Directory to save files
        X_train, X_test: Image arrays
        y_train, y_test: Label arrays
        names_train, names_test: Source name arrays
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image arrays (no clipping version)
    np.save(os.path.join(output_dir, "X_train_no_clip.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test_no_clip.npy"), X_test)
    
    # Save label arrays
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    # Save name arrays
    np.save(os.path.join(output_dir, "names_train.npy"), names_train)
    np.save(os.path.join(output_dir, "names_test.npy"), names_test)
    
    print(f"\nDatasets saved to: {output_dir}")
    print(f"  - X_train_no_clip.npy: {X_train.shape}")
    print(f"  - X_test_no_clip.npy: {X_test.shape}")
    print(f"  - y_train.npy: {y_train.shape}")
    print(f"  - y_test.npy: {y_test.shape}")
    print(f"  - names_train.npy: {names_train.shape}")
    print(f"  - names_test.npy: {names_test.shape}")


# endregion

if __name__ == "__main__":
    
    print("="*60)
    print("COMBINING INDIVIDUAL ARRAYS INTO DATASET")
    print("="*60)
    
    # Load classification mapping from JSON
    radcat_mapping = load_radcat_mapping(JSON_MAPPING_PATH)
    
    # Combine arrays
    result = combine_arrays_into_dataset(COMBINED_NUMPY_DIR, radcat_mapping)
    
    if result is None:
        print("ERROR: Failed to combine arrays")
        sys.exit(1)
    
    X_train, X_test, y_train, y_test, names_train, names_test = result
    
    # Save datasets
    save_datasets(OUTPUT_DIR, X_train, X_test, y_train, y_test, names_train, names_test)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
