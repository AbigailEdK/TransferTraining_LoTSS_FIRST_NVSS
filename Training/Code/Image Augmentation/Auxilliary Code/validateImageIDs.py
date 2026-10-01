import sys
sys.path.insert(0, '/home/abigaildeklerk/Desktop/DeKlerk_Models')
import pandas as pd
import os
from collections import defaultdict

# Paths
AGGREGATED_RGB_FOLDER = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/AGGREGATED_RGB_IMAGES"
RADCAT_PATH = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT.csv"

def validate_image_ids():
    """
    Compare IDs found in AGGREGATED_RGB_IMAGES against RADCAT.csv
    and report unknown IDs
    """
    
    # Load RADCAT.csv
    radcat_df = pd.read_csv(RADCAT_PATH, engine='python', on_bad_lines='skip')
    radcat_df.columns = radcat_df.columns.str.strip()
    radcat_ids = set(radcat_df.iloc[:, 0].values)  # First column contains IDs
    
    print(f"Loaded RADCAT.csv with {len(radcat_ids)} unique source IDs")
    print(f"Known ID range: {min(radcat_ids)} to {max(radcat_ids)}\n")
    
    # Extract IDs from image filenames
    image_files = [f for f in os.listdir(AGGREGATED_RGB_FOLDER) if f.lower().endswith('.png')]
    image_ids = defaultdict(int)  # Count occurrences of each ID
    
    for filename in image_files:
        try:
            source_id = int(filename.split('_')[0])
            image_ids[source_id] += 1
        except (ValueError, IndexError):
            print(f"Warning: Could not parse filename: {filename}")
    
    print(f"Found {len(image_ids)} unique source IDs in AGGREGATED_RGB_IMAGES")
    print(f"Image ID range: {min(image_ids)} to {max(image_ids)}\n")
    
    # Find unknown IDs
    unknown_ids = set(image_ids.keys()) - radcat_ids
    known_ids = set(image_ids.keys()) & radcat_ids
    
    print(f"Known IDs in images: {len(known_ids)}")
    print(f"Unknown IDs in images: {len(unknown_ids)}\n")
    
    if unknown_ids:
        print("=" * 60)
        print("UNKNOWN SOURCE IDs (not in RADCAT.csv):")
        print("=" * 60)
        unknown_ids_sorted = sorted(unknown_ids)
        for source_id in unknown_ids_sorted:
            count = image_ids[source_id]
            print(f"  ID {source_id}: {count} image(s)")
        
        print(f"\nTotal unknown IDs: {len(unknown_ids)}")
        print(f"Total unknown images: {sum(image_ids[sid] for sid in unknown_ids)}")
    else:
        print("✓ All image IDs are found in RADCAT.csv")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Total unique IDs in images: {len(image_ids)}")
    print(f"  - Known IDs: {len(known_ids)}")
    print(f"  - Unknown IDs: {len(unknown_ids)}")

if __name__ == "__main__":
    validate_image_ids()
