# region ABOUT
# ===================================================================================================
# > This script combines FITS files from different surveys (FIRST, LoTSS, NVSS) into RGB images.

# > FIRST data is mapped to the red channel, LoTSS to green, and NVSS to blue.

# > For each source, the script finds the corresponding FITS files from each survey, normalizes them, and creates a combined RGB image.
# ===================================================================================================
# endregion

# region IMPORTS
import os
import shutil
import numpy as np
from pathlib import Path
from astropy.io import fits
from PIL import Image
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
# endregion

# region PATHS
RADCAT_F_PATH = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F"
OUTPUT_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F_UNPACKED"
# endregion

# region FUNCTIONS

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


def get_source_files():
    """
    Discover all FITS files organized by source ID and survey.
    
    Returns:
        dict: Dictionary mapping source_id to {'FIRST': path, 'LoTSS': path, 'NVSS': path}
    """
    source_files = defaultdict(dict)
    surveys = ['FIRST', 'LoTSS', 'NVSS']
    
    for survey in surveys:
        survey_path = os.path.join(RADCAT_F_PATH, survey)
        
        if not os.path.exists(survey_path):
            print(f"Warning: {survey_path} does not exist")
            continue
        
        for file in os.listdir(survey_path):
            if file.lower().endswith(('.fits', '.fit')):
                # Extract source ID from filename (remove extension)
                source_id = file.replace('.fits', '').replace('.fit', '')
                file_path = os.path.join(survey_path, file)
                source_files[source_id][survey] = file_path
    
    return source_files


def create_rgb_image(first_data, lotss_data, nvss_data):
    """
    Combine normalized FITS data into an RGB image.
    
    Args:
        first_data (np.ndarray): FIRST survey data (red channel)
        lotss_data (np.ndarray): LoTSS survey data (green channel)
        nvss_data (np.ndarray): NVSS survey data (blue channel)
        
    Returns:
        PIL.Image: RGB image
    """
    # Normalize each channel
    r_channel = normalize_image(first_data) if first_data is not None else np.zeros_like(first_data or lotss_data or nvss_data, dtype=np.uint8)
    g_channel = normalize_image(lotss_data) if lotss_data is not None else np.zeros_like(lotss_data or first_data or nvss_data, dtype=np.uint8)
    b_channel = normalize_image(nvss_data) if nvss_data is not None else np.zeros_like(nvss_data or first_data or lotss_data, dtype=np.uint8)
    
    # Ensure all channels have the same shape by padding
    max_height = max(r_channel.shape[0], g_channel.shape[0], b_channel.shape[0])
    max_width = max(r_channel.shape[1], g_channel.shape[1], b_channel.shape[1])
    
    r_resized = np.zeros((max_height, max_width), dtype=np.uint8)
    g_resized = np.zeros((max_height, max_width), dtype=np.uint8)
    b_resized = np.zeros((max_height, max_width), dtype=np.uint8)
    
    r_resized[:r_channel.shape[0], :r_channel.shape[1]] = r_channel
    g_resized[:g_channel.shape[0], :g_channel.shape[1]] = g_channel
    b_resized[:b_channel.shape[0], :b_channel.shape[1]] = b_channel
    
    # Combine into RGB image
    rgb_array = np.stack([r_resized, g_resized, b_resized], axis=2)
    rgb_image = Image.fromarray(rgb_array, mode='RGB')
    
    return rgb_image


def combine_fits_to_rgb():
    """
    Main function that combines FITS files from different surveys into RGB images.
    """
    cleanup_output_directory()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all source files
    source_files = get_source_files()
    
    if not source_files:
        print("No FITS files found in RADCAT_F")
        return
    
    print(f"Found {len(source_files)} sources with FITS data")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    success_count = 0
    partial_count = 0
    failed_count = 0
    
    for source_id, files in sorted(source_files.items()):
        # Load data from each survey
        first_data = load_fits_data(files.get('FIRST')) if 'FIRST' in files else None
        lotss_data = load_fits_data(files.get('LoTSS')) if 'LoTSS' in files else None
        nvss_data = load_fits_data(files.get('NVSS')) if 'NVSS' in files else None
        
        # Check if we have at least some data
        data_count = sum([first_data is not None, lotss_data is not None, nvss_data is not None])
        
        if data_count == 0:
            print(f"✗ {source_id}: No valid data found")
            failed_count += 1
            continue
        
        # Create RGB image
        rgb_image = create_rgb_image(first_data, lotss_data, nvss_data)
        
        # Save image
        output_path = os.path.join(OUTPUT_DIR, f"{source_id}_RGB.png")
        rgb_image.save(output_path)
        
        # Report status
        surveys_used = []
        if first_data is not None:
            surveys_used.append('FIRST')
        if lotss_data is not None:
            surveys_used.append('LoTSS')
        if nvss_data is not None:
            surveys_used.append('NVSS')
        
        if data_count == 3:
            print(f"✓ {source_id}: Created RGB image ({', '.join(surveys_used)})")
            success_count += 1
        else:
            print(f"~ {source_id}: Created partial RGB image ({', '.join(surveys_used)})")
            partial_count += 1
    
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Total sources: {len(source_files)}")
    print(f"  Complete RGB (all 3 surveys): {success_count}")
    print(f"  Partial RGB (1-2 surveys): {partial_count}")
    print(f"  Failed: {failed_count}")
    print(f"{'='*70}")

# endregion

# Run the script
if __name__ == '__main__':
    combine_fits_to_rgb()
