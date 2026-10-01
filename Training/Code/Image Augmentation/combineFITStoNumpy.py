# region ABOUT
# ===================================================================================================
# > This script combines FITS files from three surveys (FIRST, LoTSS, NVSS) into multi-channel
# > numpy arrays, where each survey occupies a separate channel.
#
# > For each source ID, the script loads the corresponding FITS file from each survey folder,
# > normalizes the data, and stacks them into a single 3-channel (height, width, 3) array.
# ===================================================================================================
# endregion

# region IMPORTS
import os
import sys

sys.path.insert(0, '/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Management')

import numpy as np
from pathlib import Path
from astropy.io import fits
import warnings
from myUtils import check_folder_existence 
warnings.filterwarnings('ignore')
# endregion

# region PATHS
DATA_BASE_PATH = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F"

FIRST_FOLDER = os.path.join(DATA_BASE_PATH, "FIRST")
LOTSS_FOLDER = os.path.join(DATA_BASE_PATH, "LoTSS")
NVSS_FOLDER = os.path.join(DATA_BASE_PATH, "NVSS")

OUTPUT_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F_COMBINED_NUMPY"
# endregion

# region FUNCTIONS

def load_fits_data(fits_path):
    """
    Load data from a FITS file.
    
    Args:
        fits_path (str): Path to the FITS file
        
    Returns:
        np.ndarray: Image data, or None if unable to load
    """
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            
            if data is None:
                # Try first extension if primary is empty
                for hdu in hdul[1:]:
                    if hdu.data is not None:
                        data = hdu.data
                        break
            
            # Handle 3D data (take first channel)
            if data is not None and data.ndim == 3:
                data = data[0]
            
            return data
    except Exception as e:
        print(f"Warning: Could not load {fits_path}: {e}")
        return None


def normalize_image(data, percentile_range=(1, 99)):
    """
    Normalize image data to 0-255 range using percentile clipping.
    
    Args:
        data (np.ndarray): Image data
        percentile_range (tuple): Percentile range for clipping
        
    Returns:
        np.ndarray: Normalized 8-bit image data
    """
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    
    if data.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    
    v_min, v_max = np.percentile(data, percentile_range)
    
    if v_min == v_max:
        normalized = np.zeros_like(data, dtype=np.uint8)
    else:
        normalized = np.clip((data - v_min) / (v_max - v_min) * 255, 0, 255).astype(np.uint8)
    
    return normalized


def resize_to_match(data, target_shape):
    """
    Resize image data to match target shape using simple interpolation.
    
    Args:
        data (np.ndarray): Image data
        target_shape (tuple): Target (height, width)
        
    Returns:
        np.ndarray: Resized image data
    """
    from scipy import ndimage
    if data.shape == target_shape:
        return data
    
    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
    return ndimage.zoom(data, zoom_factors, order=1)


def combine_fits_to_numpy():
    """
    Iterate through FIRST folder, match files across all three surveys,
    combine into multi-channel numpy arrays, and save.
    """
    
    # Create output directory
    check_folder_existence(OUTPUT_DIR)
    
    print(f"Output directory ready: {OUTPUT_DIR}")
    
    # Get all FITS files from FIRST folder (as reference)
    first_files = sorted([f for f in os.listdir(FIRST_FOLDER) if f.endswith('.fits')])
    print(f"Found {len(first_files)} files in FIRST folder")
    
    successful_combines = 0
    failed_combines = 0
    missing_files = 0
    
    # Target shape (will use first valid image to determine)
    target_shape = None
    
    for idx, first_file in enumerate(first_files):
        source_id = first_file.replace('.fits', '')
        
        # Construct paths for all three surveys
        first_path = os.path.join(FIRST_FOLDER, first_file)
        lotss_path = os.path.join(LOTSS_FOLDER, first_file)  # Same filename
        nvss_path = os.path.join(NVSS_FOLDER, first_file)    # Same filename
        
        # Check if all files exist
        if not all(os.path.exists(p) for p in [first_path, lotss_path, nvss_path]):
            print(f"[{idx+1}/{len(first_files)}] ✗ Missing files for source {source_id}")
            missing_files += 1
            continue
        
        # Load FITS data from all three surveys
        first_data = load_fits_data(first_path)
        lotss_data = load_fits_data(lotss_path)
        nvss_data = load_fits_data(nvss_path)
        
        # Check if all loaded successfully
        if first_data is None or lotss_data is None or nvss_data is None:
            print(f"[{idx+1}/{len(first_files)}] ✗ Failed to load FITS data for source {source_id}")
            failed_combines += 1
            continue
        
        # Determine target shape from first successful image
        if target_shape is None:
            target_shape = first_data.shape
            print(f"Using target shape: {target_shape}")
        
        # Resize all images to match target shape if needed
        if first_data.shape != target_shape:
            first_data = resize_to_match(first_data, target_shape)
        if lotss_data.shape != target_shape:
            lotss_data = resize_to_match(lotss_data, target_shape)
        if nvss_data.shape != target_shape:
            nvss_data = resize_to_match(nvss_data, target_shape)
        
        # Normalize each channel
        first_norm = normalize_image(first_data)
        lotss_norm = normalize_image(lotss_data)
        nvss_norm = normalize_image(nvss_data)
        
        # Stack into 3-channel array (height, width, 3)
        # Channel 0: FIRST, Channel 1: LoTSS, Channel 2: NVSS
        combined_array = np.stack([first_norm, lotss_norm, nvss_norm], axis=2)
        
        # Save as numpy array
        output_path = os.path.join(OUTPUT_DIR, f"{source_id}.npy")
        np.save(output_path, combined_array)
        
        successful_combines += 1
        if (idx + 1) % 100 == 0:
            print(f"[{idx+1}/{len(first_files)}] ✓ Processed {successful_combines} sources")
    
    # Print summary
    print("\n" + "="*60)
    print("COMBINATION SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(first_files)}")
    print(f"Successfully combined: {successful_combines}")
    print(f"Failed combinations: {failed_combines}")
    print(f"Missing files: {missing_files}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"File format: {successful_combines} x .npy files (shape: {target_shape[0]}, {target_shape[1]}, 3)")
    print("="*60)


# endregion

if __name__ == "__main__":
    combine_fits_to_numpy()
