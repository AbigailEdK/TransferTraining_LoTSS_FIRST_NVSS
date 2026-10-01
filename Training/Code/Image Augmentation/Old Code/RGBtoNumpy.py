import sys
sys.path.insert(0, '/home/abigaildeklerk/Desktop/DeKlerk_Models')
import numpy as np
from PIL import Image
import os
from pathlib import Path
import pandas as pd
from Code.Management.myUtils import check_folder_existence

# region PATHS
RGB_IMAGES_FOLDER = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/AGGREGATED_RGB_IMAGES"

TEST_IMAGES_FOLDER = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TESTING/TEST_SOURCES_RGB"

TRAINVAL_IMAGES_FOLDER = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TRAINING AND VALIDATION/TRAINVAL_SOURCES_RGB"

OUTPUT_FOLDER = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/NUMPY_SOURCES"

RADCAT_PATH = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT.csv"
# endregion

def get_galaxy_class(filename, radcat_df):
    """
    Extract ID from filename (number before first underscore) and look up class in RADCAT.csv
    
    Args:
        filename: Image filename (e.g., '209_B_0.png')
        radcat_df: Loaded RADCAT DataFrame with ID column and 'Type' column
        
    Returns:
        Class name (e.g., 'FRI' or 'FRII') or None if ID not found
    """
    try:
        # Extract ID from filename (number before first underscore)
        source_id = int(filename.split('_')[0])
        
        # Look up ID in RADCAT (search by the first column which contains IDs)
        id_column = radcat_df.columns[0]  # First column should be the ID
        match = radcat_df[radcat_df[id_column] == source_id]
        if not match.empty:
            return match['Type'].values[0]
        else:
            print(f"Warning: ID {source_id} from file {filename} not found in RADCAT.csv")
            return None
    except (ValueError, IndexError) as e:
        print(f"Error processing filename {filename}: {e}")
        return None
    

def png_folder_to_npy(png_folder, output_path, target_height=300, target_width=300, kind='train_val'):
    """
    Convert a folder of RGB PNGs to the format expected by get_dataset()
    
    Args:
        png_folder: Path to folder containing PNG images
        output_path: Where to save the .npy file (e.g., 'ConstructData/results/same_pixels/')
        target_height: Resize height (default 300 to match RADCAT)
        target_width: Resize width (default 300 to match RADCAT)
        kind: 'train_val', 'test'
        filekind: 'clip' or 'no_clip'
    """
    
    if check_folder_existence(output_path) is False:
        print(f"Output folder {output_path} already exists and was not replaced. Exiting.")
        return
    else:
        print(f"Output folder {output_path} is ready for saving .npy files.")

    # Load RADCAT.csv to get galaxy classes
    # Use engine='python' and on_bad_lines='skip' to handle CSV formatting issues
    radcat_df = pd.read_csv(RADCAT_PATH, engine='python', on_bad_lines='skip')
    # Strip whitespace from column names to fix access issues
    radcat_df.columns = radcat_df.columns.str.strip()
    print(f"Loaded RADCAT.csv with {len(radcat_df)} entries")
    print(f"Columns: {radcat_df.columns.tolist()}")

    # Get sorted list of PNG files
    png_files = sorted([f for f in os.listdir(png_folder) if f.lower().endswith('.png')])
    
    # Initialize arrays
    images = np.zeros((len(png_files), target_height, target_width, 3), dtype=np.float32)
    labels = []  # Store filenames without .png extension
    source_ids = []  # Store source IDs for reference
    
    print(f"Processing {len(png_files)} PNG files from {png_folder}...")
    # Load and resize each image
    for n in range(0, len(png_files), 100):  # Process in batches of 100
        batch_files = png_files[n:n+100]
        print(f"Processing batch {n//100 + 1} ({len(batch_files)} images)...")
        for i, filename in enumerate(batch_files):
            img_path = os.path.join(png_folder, filename)
            img = Image.open(img_path).convert('RGB')  # Ensure RGB
            img = img.resize((target_width, target_height))
            images[i] = np.array(img)
            # Extract filename without .png extension
            file_label = filename[:-4]  # Remove .png extension
            # Get galaxy class from RADCAT
            galaxy_class = get_galaxy_class(filename, radcat_df)
            labels.append(galaxy_class)
            source_ids.append(file_label)  # Store the filename as label (or you could store the class instead)
            print(f"Processed {i+1}/100 images", end='\r')
    
    # Create output directory if needed
    print(f"Saving {len(png_files)} images to {output_path} as .npy file...")
    
    # Save in expected format
    output_file = os.path.join(output_path, f"X_{kind}.npy")
    np.save(output_file, images)
    print(f"Saved {len(png_files)} images to {output_file}")
    print(f"Shape: {images.shape}")
    
    # Save labels
    labels_array = np.array(labels, dtype=object)
    labels_file = os.path.join(output_path, f"y_{kind}.npy")
    np.save(labels_file, labels_array)
    print(f"Saved {len(labels)} labels to {labels_file}")

    # save source IDs    source_ids_array = np.array(source_ids, dtype=object)
    source_ids_file = os.path.join(output_path, f"names_{kind}.npy")
    np.save(source_ids_file, np.array(source_ids, dtype=object))
    print(f"Saved {len(source_ids)} source IDs to {source_ids_file}")

def main():
    print("Converting trainval images...")
    png_folder_to_npy(TRAINVAL_IMAGES_FOLDER, OUTPUT_FOLDER + "/trainval", kind='train_val')
    print("Converting test images...")
    png_folder_to_npy(TEST_IMAGES_FOLDER, OUTPUT_FOLDER + "/test", kind='test')

if __name__ == "__main__":
    main()