# region ABOUT
# ===================================================================================================
# > This script reads all .png files in the AGGREGATED_RGB_IMAGES directory and prints the final distribution of classes (FRI/FRII/COMPACT). It maps each file to its class by extracting the source ID from the filename (removing any "_X" augmentation suffix or "_RGB" suffix) and looking it up in RADCAT.csv.
# ===================================================================================================
# endregion


# region IMPORTS
import os
import pandas as pd
from collections import Counter
# endregion

# region PATHS
AGGREGATED_RGB_IMAGES_DIRECTORY = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/AGGREGATED_RGB_IMAGES"

RADCAT_CSV_PATH = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT.csv"
# endregion

# region FUNCTIONS
def print_class_distribution(output_dir, radcat_csv_path):
    """
    Prints the final distribution of classes for all PNG files in the output directory.
    
    This function reads the RADCAT.csv file to map source IDs to their classes (FRI/FRII/COMPACT),
    then iterates through all .png files in the output directory. For each file, it extracts
    the source ID (removing any "_X" suffix indicating augmented sources or "_RGB" suffix) and tallies the
    class distribution.
    
    Args:
        output_dir (str): Directory containing the aggregated .png files
        radcat_csv_path (str): Path to the RADCAT.csv file containing source class information
    """
    # Read RADCAT.csv to map source IDs to classes
    try:
        radcat_df = pd.read_csv(radcat_csv_path, skipinitialspace=True, quoting=1)  # quoting=1 is csv.QUOTE_ALL
        # Find the correct column name for Type (strip whitespace from column names)
        radcat_df.columns = radcat_df.columns.str.strip()
        # Use the first column (index 0) as source ID and the "Type" column for class
        source_to_class = dict(zip(radcat_df.iloc[:, 0], radcat_df['Type']))
    except Exception as e:
        print(f"Error reading RADCAT.csv: {e}")
        return
    
    # Count class distribution
    class_distribution = Counter()
    file_to_class = {}
    unmatched_sources = set()
    
    # Iterate through all .png files in the output directory
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        return
    
    png_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"No .png files found in {output_dir}")
        return
    
    for filename in png_files:
        # Extract source ID: remove .png extension and any "_X" suffix or "_RGB" suffix
        source_id_str = filename.replace('.png', '')
        
        # Remove augmentation suffix (e.g., "_0", "_1", etc.) or "_RGB" suffix
        if '_' in source_id_str:
            parts = source_id_str.rsplit('_', 1)
            # Check if the last part is a digit (augmentation indicator) or "RGB"
            if parts[-1].isdigit() or parts[-1].lower() == 'rgb':
                source_id_str = parts[0]
        
        # Convert to integer to match the RADCAT.csv source ID format
        try:
            source_id = int(source_id_str)
            if source_id in source_to_class:
                class_type = source_to_class[source_id]
                class_distribution[class_type] += 1
                file_to_class[filename] = class_type
            else:
                unmatched_sources.add(source_id)
        except ValueError:
            unmatched_sources.add(source_id_str)
    
    # Print results
    print("\n" + "="*70)
    print("FINAL CLASS DISTRIBUTION (AGGREGATED RGB IMAGES)")
    print("="*70)
    print(f"Total files in output directory: {len(png_files)}")
    print("\nClass Distribution:")
    for class_type, count in sorted(class_distribution.items()):
        percentage = (count / len(png_files)) * 100
        print(f"  {class_type}: {count} files ({percentage:.2f}%)")
    
    if unmatched_sources:
        print(f"\nWarning: {len(unmatched_sources)} source(s) not found in RADCAT.csv:")
        for source in sorted(unmatched_sources):
            print(f"  - {source}")
    
    print("="*70 + "\n")

# endregion

# Print the class distribution
print_class_distribution(AGGREGATED_RGB_IMAGES_DIRECTORY, RADCAT_CSV_PATH)
