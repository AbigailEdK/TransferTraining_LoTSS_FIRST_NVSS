# region ABOUT
# ===================================================================================================
# > This script reads augmented images from a binary file (augmentations.txt) and saves them as PNG
# > files in a specified output directory. It uses the augmentations_metadata.json file to maintain
# > proper source ID traceability, ensuring every augmented image can be traced back to its original source.
# ===================================================================================================
# endregion

# region IMPORTS
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
import shutil
# endregion

# region PATHS
AUGMENTATIONS_FILE = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Image Augmentation/augmentations.txt"
METADATA_FILE = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Image Augmentation/augmentations_metadata.json"
SOURCES_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F_UNPACKED"
OUTPUT_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/AUGMENTED_SOURCES"
# endregion

# region CLEANUP
def cleanup_output_directory():
    '''
    DESCRIPTION:
    -----------
        Removes the output directory if it already exists.
    '''
    if os.path.exists(OUTPUT_DIR):
        print(f"Existing output directory found. Deleting: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
        print("Old data cleaned up.")
# endregion

# region FUNCTIONS
def load_augmented_images():
    '''
    DESCRIPTION:
    -----------
        Reads augmented images sequentially from the binary augmentations.txt file.
    
    RETURNS:
    --------
        > augmented_images: list of np.arrays, augmented images
    '''
    augmented_images = []
    
    if not os.path.exists(AUGMENTATIONS_FILE):
        print(f"Error: {AUGMENTATIONS_FILE} not found")
        return augmented_images
    
    print(f"Loading augmented images from {AUGMENTATIONS_FILE}...")
    
    with open(AUGMENTATIONS_FILE, "rb") as f:
        count = 0
        while True:
            try:
                augmented_images.append(np.load(f, allow_pickle=True))
                count += 1
                print(f"  Loaded: {count} augmentations", end='\r')
            except EOFError:
                break
            except Exception as e:
                print(f"Warning: Error loading augmentation: {e}")
                break
    
    print(f"\nLoaded {len(augmented_images)} augmented image arrays")
    return augmented_images


def load_augmentation_metadata():
    '''
    DESCRIPTION:
    -----------
        Load augmentation metadata from JSON file for source traceability.
    
    RETURNS:
    --------
        > metadata: dict, containing augmentation mapping information
    '''
    if not os.path.exists(METADATA_FILE):
        print(f"Error: {METADATA_FILE} not found")
        return None
    
    try:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        print(f"Loaded augmentation metadata from {METADATA_FILE}")
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def load_source_ids_from_file():
    '''
    DESCRIPTION:
    -----------
        Load source IDs from source_ids_val_train.txt file.
    
    RETURNS:
    --------
        > source_ids: list of str, source IDs from the file
    '''
    source_ids_file = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Image Augmentation/source_ids_val_train.txt"
    
    if not os.path.exists(source_ids_file):
        print(f"Error: {source_ids_file} not found")
        return []
    
    try:
        source_ids = sorted(np.loadtxt(source_ids_file, dtype=str))
        print(f"Loaded {len(source_ids)} source IDs from {source_ids_file}")
        return source_ids
    except Exception as e:
        print(f"Error loading source IDs: {e}")
        return []


def get_source_ids():
    '''
    DESCRIPTION:
    -----------
        Get list of source IDs from the RADCAT_F_UNPACKED folder that match
        the IDs in source_ids_val_train.txt.
    
    RETURNS:
    --------
        > source_ids: list of str, source ID filenames matching the file
    '''
    if not os.path.exists(SOURCES_DIR):
        print(f"Error: {SOURCES_DIR} not found")
        return []
    
    # Load valid source IDs from file
    valid_source_ids = load_source_ids_from_file()
    if not valid_source_ids:
        return []
    
    # Filter to only those that exist in SOURCES_DIR (handling both old _RGB.png and new formats)
    valid_set = set(valid_source_ids)
    files = os.listdir(SOURCES_DIR)
    source_ids = sorted([os.path.splitext(f)[0].replace('_RGB', '') for f in files 
                        if f.endswith(('.png', '.fits')) and (os.path.splitext(f)[0].replace('_RGB', '') in valid_set)])
    
    print(f"Found {len(source_ids)} matching sources in {SOURCES_DIR}")
    return source_ids


def save_augmented_png_files(augmented_images, metadata):
    '''
    ARGUMENTS:
    ----------
        > augmented_images: list of np.arrays, augmented images
        > metadata: dict, metadata from load_augmentation_metadata()
    
    DESCRIPTION:
    -----------
        Saves augmented images as PNG files to OUTPUT_DIR with naming:
        {source_id}_{aug_index}.png
        
        Uses metadata to ensure proper source ID traceability for each augmentation.
        Assumes augmented_images array order matches metadata sources order.
    '''
    
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if len(augmented_images) == 0:
        print("No augmented images to save")
        return 0
    
    if metadata is None or "sources" not in metadata:
        print("Error: Invalid metadata structure")
        return 0
    
    total_saved = 0
    total_augmentations = sum(aug_array.shape[0] if aug_array.ndim > 0 else 1 for aug_array in augmented_images)
    
    print(f"Saving augmented PNG files to {OUTPUT_DIR}...")
    print(f"Total augmentations to save: {total_augmentations}\n")
    
    # Iterate through metadata to save augmentations with proper source IDs
    # Assume augmented_images array order matches metadata sources order
    for array_idx, source_info in enumerate(metadata["sources"]):
        if array_idx >= len(augmented_images):
            print(f"Warning: Not enough augmented_images arrays for all metadata sources")
            break
        
        source_id = source_info["source_id"]
        num_augs = source_info["num_augmentations"]
        Aug_array = augmented_images[array_idx]
        
        try:
            # Verify the array has expected number of augmentations
            if Aug_array.shape[0] != num_augs:
                print(f"Warning: Source {source_id} expects {num_augs} augmentations but array has {Aug_array.shape[0]}")
            
            # Save each augmentation
            for aug_idx in range(Aug_array.shape[0]):
                if Aug_array.ndim == 4:
                    aug_image = Aug_array[aug_idx]  # (height, width, 3)
                elif Aug_array.ndim == 3:
                    aug_image = Aug_array[aug_idx]
                else:
                    aug_image = Aug_array
                
                # Convert to uint8 if needed
                if aug_image.dtype == np.float32 or aug_image.dtype == np.float64:
                    aug_image = (np.clip(aug_image, 0, 1) * 255).astype(np.uint8)
                else:
                    aug_image = aug_image.astype(np.uint8)
                
                # Handle grayscale (2D) or RGB (3D with 3 channels)
                if aug_image.ndim == 2:
                    img_pil = Image.fromarray(aug_image, mode='L')
                elif aug_image.ndim == 3 and aug_image.shape[2] == 3:
                    img_pil = Image.fromarray(aug_image, mode='RGB')
                else:
                    print(f"Warning: Unexpected shape for {source_id}_{aug_idx}: {aug_image.shape}")
                    continue
                
                # Save PNG file with source ID and augmentation index
                output_file = os.path.join(OUTPUT_DIR, f"{source_id}_{aug_idx}.png")
                img_pil.save(output_file)
                total_saved += 1
                print(f"  Saved: {total_saved}/{total_augmentations} files", end='\r')
        
        except Exception as e:
            print(f"Error saving augmentations for source {source_id}: {e}")
            continue
    
    print(f"\n\nCompleted: {total_saved}/{total_augmentations} files")
    print(f"Files saved with full source ID traceability from metadata")
    return total_saved


def main():
    '''
    DESCRIPTION:
    -----------
        Main function that loads augmented images from binary file and metadata,
        then saves them as PNG files with proper source ID traceability.
    '''
    
    print("=" * 80)
    print("LOADING AUGMENTATIONS AND SAVING AS PNG FILES")
    print("=" * 80 + "\n")
    
    # Clean up existing output directory if it exists
    print("Checking for existing output directory...")
    cleanup_output_directory()
    print()
    
    # Load augmented images from binary file
    augmented_images = load_augmented_images()
    
    if not augmented_images:
        print("No augmented images loaded. Exiting.")
        return
    
    print()
    
    # Load metadata for source ID traceability
    metadata = load_augmentation_metadata()
    
    if metadata is None:
        print("No metadata loaded. Exiting.")
        return
    
    print()
    
    # Save augmented images as PNG files with metadata-based source IDs
    num_saved = save_augmented_png_files(augmented_images, metadata)
    
    print(f"\n" + "=" * 80)
    print(f"SUCCESS: Saved {num_saved} augmented PNG files to {OUTPUT_DIR}")
    print(f"All files have full source ID traceability from metadata")
    print("=" * 80)

# endregion

if __name__ == "__main__":
    main()
