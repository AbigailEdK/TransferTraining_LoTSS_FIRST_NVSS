# region ABOUT
# ===================================================================================================
# > This script loads RGB image files (.png) from AGGREGATED_RGB_IMAGES folder and separates them 
# > into individual color channels (R, G, B). Each channel is saved as a separate numpy array in 
# > the SPLIT_SOURCES folder organized by color.
# ===================================================================================================
# endregion

# region IMPORTS
import os
from PIL import Image
import numpy as np
import shutil
# endregion

# region PATHS
AGGREGATED_FILES_FOLDER = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/AGGREGATED_RGB_IMAGES"

SEPARATED_SOURCES_FOLDER = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/SPLIT_SOURCES"

color_folders = {"R": "Red", "G": "Green", "B": "Blue"}

# endregion

# Clean up existing output directory
if os.path.exists(SEPARATED_SOURCES_FOLDER):
    print(f"Removing existing directory: {SEPARATED_SOURCES_FOLDER}")
    shutil.rmtree(SEPARATED_SOURCES_FOLDER)

# Create output directories
for folder in color_folders.values():
    os.makedirs(os.path.join(SEPARATED_SOURCES_FOLDER, folder), exist_ok=True)

print(f"Output directories created: {SEPARATED_SOURCES_FOLDER}\n")

# region PROCESSING
# Process each PNG file
for filename in os.listdir(AGGREGATED_FILES_FOLDER):
    if filename.lower().endswith(".png"):
        filepath = os.path.join(AGGREGATED_FILES_FOLDER, filename)
        
        # Read PNG image as RGB
        try:
            img = Image.open(filepath)
            # Convert to RGB if grayscale
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array (height, width, 3)
            data = np.array(img)
            
            # Assume data is (height, width, 3) for RGB channels
            if len(data.shape) == 3 and data.shape[2] == 3:
                channels = [data[:, :, 0], data[:, :, 1], data[:, :, 2]]  # R, G, B
                colors = ["R", "G", "B"]
                
                for channel, color in zip(channels, colors):
                    # Save as numpy array
                    base_name = filename.replace(".png", "")
                    output_filename = f"{base_name}_{color}.npy"
                    output_path = os.path.join(SEPARATED_SOURCES_FOLDER, color_folders[color], output_filename)
                    
                    np.save(output_path, channel)
                    print(f"Saved: {output_path}")
            else:
                print(f"Warning: Unexpected shape for {filename}: {data.shape}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("\nProcessing complete!")
# endregion