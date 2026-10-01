# region ABOUT
# ===================================================================================================
# > This script aggregates RGB images from multiple source directories (RADCAT_F_UNPACKED and AUGMENTED_SOURCES) into a single output directory (AGGREGATED_RGB_IMAGES). It ensures that all .png files from the specified source directories are copied to the output directory, and it counts the total number of files aggregated at the end.
# ===================================================================================================
# endregion


# region IMPORTS
import os
import shutil
from pathlib import Path
# endregion

# region PATHS
RADCAT_F_UNPACKED_DIRECTORY = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F_UNPACKED"

AUGMENTED_SOURCES_DIRECTORY = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/AUGMENTED_SOURCES"

AGGREGATED_RGB_IMAGES_DIRECTORY = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/AGGREGATED_RGB_IMAGES"
# endregion

# region FUNCTIONS
def aggregate_rgb_images(source_dirs, output_dir):
    """
    Aggregates .png RGB images from multiple source directories into a single output directory.
    
    Args:
        source_dirs (list): List of source directory paths containing .png files
        output_dir (str): Output directory path where aggregated files will be copied
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    file_count = 0
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} does not exist")
            continue
        
        print(f"\nProcessing directory: {source_dir}")
        
        # Find all .png files in the source directory (non-recursive)
        for file in os.listdir(source_dir):
            if file.lower().endswith('.png'):
                source_path = os.path.join(source_dir, file)
                
                # Skip if it's a directory
                if not os.path.isfile(source_path):
                    continue
                
                output_path = os.path.join(output_dir, file)
                
                # Copy file to output directory
                shutil.copy2(source_path, output_path)
                file_count += 1
                print(f"  Copied: {file}")
    
    print(f"\nTotal files aggregated: {file_count}")

# endregion
aggregate_rgb_images([RADCAT_F_UNPACKED_DIRECTORY, AUGMENTED_SOURCES_DIRECTORY], AGGREGATED_RGB_IMAGES_DIRECTORY)

# Count and print final number of files in output directory
final_file_count = len([f for f in os.listdir(AGGREGATED_RGB_IMAGES_DIRECTORY) if os.path.isfile(os.path.join(AGGREGATED_RGB_IMAGES_DIRECTORY, f))])
print(f"\nFinal number of files in {AGGREGATED_RGB_IMAGES_DIRECTORY}: {final_file_count}")

