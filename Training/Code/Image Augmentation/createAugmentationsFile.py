# region ABOUT
# ===================================================================================================
# > This script generates a text file that contains augmented images for each source in the training/validation set. The augmented images are generated using random flips and rotations, and they are saved sequentially to a binary file (augmentations.txt) that can be read with np.load(). The script also ensures that the output directory is cleaned up before saving new files, and it matches the augmented images with source IDs from a text file to maintain proper naming conventions.
# ===================================================================================================
# endregion

# region IMPORTS
import os, sys
import json
import pandas as pd
from astropy.io import fits
import numpy as np
from PIL import Image
print("Setting up tf layers...")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
# endregion

# region GLOBAL VARIABLES
AUGMENTED_TEXT_FILE_SAVE_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Image Augmentation"

TEST_SOURCES_SAVE_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/TEST_SOURCES"

RADCAT_F_UNPACKED = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F_UNPACKED"

RADCAT_F = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F"

CATALOG_FILE = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT.csv"

# endregion

# region FUNCTIONS
def load_source_ids_from_files(directory):
    '''
    ARGUMENTS:
    ----------
        > directory: str, directory containing source_ids_test.txt and source_ids_val_train.txt

    RETURNS:
    --------
        > test_ids: list of str, source IDs for test set
        > train_val_ids: list of str, source IDs for train/val set

    DESCRIPTION:
    -----------
        Loads source IDs from files in the specified directory.
    '''
    test_file = os.path.join(directory, "source_ids_test.txt")
    train_val_file = os.path.join(directory, "source_ids_val_train.txt")
    
    test_ids = []
    train_val_ids = []
    
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_ids = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(test_ids)} test source IDs")
    else:
        print(f"Warning: {test_file} not found")
    
    if os.path.exists(train_val_file):
        with open(train_val_file, 'r') as f:
            train_val_ids = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(train_val_ids)} train/val source IDs")
    else:
        print(f"Warning: {train_val_file} not found")
    
    return test_ids, train_val_ids


def get_indices_for_source_ids(names, source_ids):
    '''
    ARGUMENTS:
    ----------
        > names: np.array of str, source IDs loaded from data
        > source_ids: list of str, source IDs to find indices for

    RETURNS:
    --------
        > indices: np.array of int, indices in names array matching source_ids

    DESCRIPTION:
    -----------
        Finds indices in the names array that correspond to the given source IDs.
    '''
    indices = []
    for source_id in source_ids:
        matching_indices = np.where(names == source_id)[0]
        if len(matching_indices) > 0:
            indices.append(matching_indices[0])
        else:
            print(f"Warning: Source ID {source_id} not found in loaded data")
    
    return np.array(indices)

def count_directory_files(directory, pattern="augmentations.txt"):
    '''
    DESCRIPTION:
    -----------
        Check if the augmentations.txt file exists.
    '''
    if not os.path.exists(directory):
        return 0
    
    augmentations_file = os.path.join(directory, pattern)
    return 1 if os.path.exists(augmentations_file) else 0

def print_file_counts(message=""):
    '''
    DESCRIPTION:
    -----------
        Check if the augmentations.txt file exists.
    '''
    if message:
        print(f"\n{message}")
    
    augmentations_file = os.path.join(AUGMENTED_TEXT_FILE_SAVE_DIR, "augmentations.txt")
    exists = "✓ Generated" if os.path.exists(augmentations_file) else "✗ Not found"
    
    print(f"augmentations.txt: {exists}\n")

def select_directory(survey):
    '''
    ARGUMENTS:
    ----------
        > survey: str, the name of the survey for which the augmented
          images will be saved (either LoTSS, FIRST, or NVSS).

    RETURNS:
    --------
        > save_dir: str, the directory to get the images from.

    DESCRIPTION:
    ------------
        This function determines which directory the images to be augmented comes from based on the survey name provided. 
    '''

    survey_directory = RADCAT_F + "/" + survey
    
    return survey_directory

def print_distribution(dataset, dataset_name, total_images=None, spacing=20):
    total_images = len(dataset) if total_images is None else total_images
    print(f"\n--- {dataset_name} Distribution ---")
    unique, counts = np.unique(dataset, return_counts=True)
    print(f"{'Morphology':^{spacing}}|{'Count':^{spacing}}|{'Percentage':^{spacing}}")
    print(f"{'-'*spacing}|{'-'*spacing}|{'-'*spacing}")
    for i in range(len(unique)):
        print(f"{unique[i]:^{spacing}}|{counts[i]:^{spacing}}|{f'{counts[i]/len(dataset)*100:.2f}%':^{spacing}}")
    print(f"{'-'*spacing}|{'-'*spacing}|{'-'*spacing}")
    print(f"Total Images: {len(dataset)}/{total_images}\t{len(dataset)/total_images*100:.2f}%\n")

def get_augmentation_metadata(metadata_dir):
    '''
    DESCRIPTION:
    -----------
        Load augmentation metadata from JSON file for traceability.
    
    RETURNS:
    --------
        > metadata: dict, containing augmentation mapping information
    '''
    metadata_file = os.path.join(metadata_dir, "augmentations_metadata.json")
    if not os.path.exists(metadata_file):
        print(f"Warning: Metadata file not found: {metadata_file}")
        return None
    
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

def get_source_for_augmentation(aug_index, metadata):
    '''
    DESCRIPTION:
    -----------
        Find which source ID an augmentation index belongs to.
    
    ARGUMENTS:
    ----------
        > aug_index: int, augmentation index
        > metadata: dict, metadata from get_augmentation_metadata()
    
    RETURNS:
    --------
        > source_info: dict, information about the source (None if not found)
    '''
    if metadata is None:
        return None
    
    for source in metadata["sources"]:
        if source["augmentation_start_index"] <= aug_index <= source["augmentation_end_index"]:
            return source
    
    return None

def generate_augmented_data(Xs, y, names, save_dir, save_files):
    '''
    ARGUMENTS:
    ----------
        > Xs: list of numpy arrays, the original images.
        > y: numpy array, the labels corresponding to each image.
        > names: numpy array, the names of the sources corresponding to
          each image.
        > save_dir: str, the directory where the augmentation files will be
          saved.
        > save_files: bool, whether to save the augmentation data to disk.

    RETURNS:
    --------
        > total_augmentations: int, total number of augmentations saved.

    DESCRIPTION:
    ------------
        This function generates augmented images for each source and saves them
        sequentially to augmentations.txt as a binary file that can be read with np.load().
        It also saves metadata (augmentations_metadata.json) that maps each augmentation
        back to its source ID, enabling full traceability.
    '''
    print("Calculating augmentation details...")
    details = {}
    for morph in np.unique(y):
        num_of_sources = len(np.where(y == morph)[0])
        num_to_augment = 3000 - num_of_sources
        num_per_source = int(np.ceil(num_to_augment / num_of_sources))

        print(f"Augmenting {morph}: {num_of_sources}")
        print(f"\tNeed to augment at least {num_to_augment} images")
        print(f"\tAugmenting {num_per_source} images per source")
        print(f"\tThus, {num_per_source * num_of_sources} images will be added")

        details[morph] = num_per_source

    same_dimension = True if Xs[0].shape[-1] == 3 else False

    print("Generating augmented image data...")
    
    total_augmentations = 0
    metadata = {
        "version": 1,
        "description": "Mapping of augmentation indices to source IDs for traceability",
        "sources": []
    }
    augmentation_index = 0
    
    # Save augmented images to binary file
    if save_files:
        augmentations_file = os.path.join(save_dir, "augmentations.txt")
        with open(augmentations_file, "wb") as f:
            for i in range(len(names)):
                print(f"Augmenting... {i+1}/{len(names)}", end='\r')

                source_id = names[i]
                source_label = y[i]
                num_augs_for_source = details[source_label]
                source_start_index = augmentation_index

                aug_images = []
                starting_seed = np.random.randint(2**32 - 1)
                for X in Xs:
                    np.random.seed(starting_seed)
                    aug_images.append([])
                    min_value = np.min(X[i])
                    for n in range(details[y[i]]):
                        seed = np.random.randint(2**32 - 1)
                        if same_dimension:
                            # (n, p, p, c)
                            aug_images[-1].append(
                                RandomFlip()(
                                RandomRotation(1, fill_mode="constant", fill_value=min_value)(X[i]) # 1 being -360 to 360 degrees
                                ))
                            
                        else:
                            # (n,) (p, p, 1)
                            aug_images[-1].append(
                                RandomFlip(seed=seed)(
                                RandomRotation(1,seed=seed, fill_mode="constant", fill_value=min_value)(X[i])) # 1 being -360 to 360 degrees
                                )
                            
                    aug_images[-1] = np.array(aug_images[-1])
                    # Save each augmented image array to the binary file
                    np.save(f, aug_images[-1])
                    num_augs_saved = aug_images[-1].shape[0]
                    total_augmentations += num_augs_saved
                
                # Record metadata for this source
                metadata["sources"].append({
                    "source_id": str(source_id),
                    "label": str(source_label),
                    "augmentation_start_index": source_start_index,
                    "augmentation_end_index": total_augmentations - 1,
                    "num_augmentations": num_augs_for_source,
                    "cumulative_total": total_augmentations
                })
                
                augmentation_index = total_augmentations
        
        print(f"\nSaved augmented image data to: {augmentations_file}")
        
        # Save metadata JSON file for traceability
        metadata_file = os.path.join(save_dir, "augmentations_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved augmentation metadata to: {metadata_file}")
    
    return total_augmentations

def get_data():
    '''
    ARGUMENTS:
    ----------
        None

    RETURNS:
    --------
        > data_images: list of numpy arrays, the original images.
        > labels: numpy array, the labels corresponding to each image.
        > names: numpy array, the names of the sources corresponding to each image. 

    DESCRIPTION:
    ------------
        This function reads in the catalog and gets the names of relevant sources from either FITS files
        or RGB PNG images in the RADCAT_F_UNPACKED directory. RGB images should have "_RGB" suffix.
        The function loops through the relevant sources and loads their images and labels from the catalog.
        The images are stored in a 4D numpy array where the dimensions are (number of sources, height, width, 3).
        The labels are stored in a 1D numpy array where each element corresponds to the label of the source
        at the same index in the names array.
    '''

    # * Read in the catalog
    print("Reading in catalog...")
    df = pd.read_csv(CATALOG_FILE, engine='python', on_bad_lines='skip', skipinitialspace=True)
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    # Create a mapping from source ID (first column) to Type
    id_column = df.iloc[:, 0]
    type_column = df['Type']
    source_to_type = dict(zip(id_column, type_column))

    # * Get all image files (FITS or RGB PNG) from the unpacked directory
    relevant_files = {}
    for f in os.listdir(RADCAT_F_UNPACKED):
        if f.lower().endswith(('.fits', '.fit')):
            # FITS file: use filename without extension as source ID
            source_id = os.path.splitext(f)[0]
            relevant_files[source_id] = f
        elif f.endswith('_RGB.png'):
            # RGB PNG file: remove _RGB suffix to get source ID
            source_id = f.replace('_RGB.png', '')
            relevant_files[source_id] = f
    
    relevant_sources = sorted(relevant_files.keys())

    names, labels = [],[]

    data_images = [np.zeros((len(relevant_sources), 128, 128, 3))] 
    
    # * Loop through all available sources and load their images and labels from CSV
    print("Loading images and labels...")
    for i, source in enumerate(relevant_sources):
        try:
            source_int = int(source)
        except ValueError:
            print(f"Warning: Could not convert source {source} to int")
            continue
        
        if source_int not in source_to_type:
            print(f"Warning: Source {source} not in RADCAT.csv")
            continue
        
        names.append(source)
        labels.append(source_to_type[source_int])
        
        try:
            filename = relevant_files[source]
            file_path = os.path.join(RADCAT_F_UNPACKED, filename)
            
            if filename.endswith('_RGB.png'):
                # Load RGB PNG image
                img = Image.open(file_path).convert('RGB')
                img = img.resize((128, 128), Image.LANCZOS)
                img_data = np.array(img, dtype=np.float32)
                # Normalize to 0-1 range for consistency
                img_data = img_data / 255.0
            else:
                # Load FITS file
                img_data = fits.getdata(file_path, memmap=False)
                # Assuming single channel data, replicate across 3 channels
                if img_data.ndim == 2:
                    img_data = np.stack([img_data, img_data, img_data], axis=-1)
                elif img_data.ndim == 3 and img_data.shape[-1] != 3:
                    print(f"Warning: Unexpected shape for {source}: {img_data.shape}")
                    names.pop()
                    labels.pop()
                    continue
            
            # Ensure image is 128x128
            if img_data.shape[0] != 128 or img_data.shape[1] != 128:
                # Convert to PIL, resize, and convert back
                if img_data.ndim == 2:
                    img_pil = Image.fromarray(img_data, mode='L')
                else:
                    # Convert float to uint8 for PIL
                    if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                        img_uint8 = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
                    else:
                        img_uint8 = img_data.astype(np.uint8)
                    img_pil = Image.fromarray(img_uint8, mode='RGB' if img_uint8.ndim == 3 else 'L')
                img_pil = img_pil.resize((128, 128), Image.LANCZOS)
                img_data = np.array(img_pil, dtype=np.float32) / 255.0
                if img_data.ndim == 2:
                    img_data = np.stack([img_data, img_data, img_data], axis=-1)
            
            data_images[0][len(names)-1,:,:,:] = img_data
            
        except Exception as e:
            print(f"Error loading image for source {source}: {e}")
            names.pop()
            labels.pop()

    # Trim arrays to actual loaded data
    if len(names) < len(relevant_sources):
        data_images[0] = data_images[0][:len(names)]
    
    return data_images, np.array(labels), np.array(names)

# endregion

# region MAIN
def main():

    # * Set to True to save the augmented images to disk, False to not save them
    SAVE_FILES = True
    
    # Clean up all existing output directories and files
    # cleanup_output_directories()
    
    # Create fresh output directory
    os.makedirs(AUGMENTED_TEXT_FILE_SAVE_DIR, exist_ok=True)

    # Check if augmentations.txt already exists and delete it
    augmentations_file = os.path.join(AUGMENTED_TEXT_FILE_SAVE_DIR, "augmentations.txt")
    if os.path.exists(augmentations_file):
        print(f"Existing augmentations.txt found. Deleting: {augmentations_file}")
        os.remove(augmentations_file)

    Xs, y, names = get_data() # | Get the original images, labels, and names of the sources.
    
    # * Load source IDs from files and get indices
    script_directory = os.path.dirname(os.path.abspath(__file__))
    test_ids, train_val_ids = load_source_ids_from_files(script_directory)
    
    test_index = get_indices_for_source_ids(names, test_ids)
    train_val_index = get_indices_for_source_ids(names, train_val_ids)
    
    print(f"Using {len(test_index)} sources for test set")
    print(f"Using {len(train_val_index)} sources for train/val set")

    # * Get the training/validation set (these are the sources that will be augmented)
    Xs_train_val = [X[train_val_index] for X in Xs]
    y_train_val = y[train_val_index]
    names_train_val = names[train_val_index]

    # * Print the distribution of the training/validation and testing sets
    for key, val in {"TrainVal": y_train_val, "Testing": y[test_index]}.items():
        print_distribution(val, key, total_images=len(y))

    del Xs, y, names
    
    # * Generate augmented image data and save to binary augmentations.txt file
    print("Generating augmented image data file...")
    total_count = generate_augmented_data(Xs_train_val, y_train_val, names_train_val, AUGMENTED_TEXT_FILE_SAVE_DIR, SAVE_FILES)

    print("\nAugmentation complete!")
    print(f"Binary file saved to: {AUGMENTED_TEXT_FILE_SAVE_DIR}/augmentations.txt")
    print(f"Metadata file saved to: {AUGMENTED_TEXT_FILE_SAVE_DIR}/augmentations_metadata.json")
    print(f"Total augmentations saved: {total_count}")
    print("\n" + "="*80)
    print("METADATA FILE STRUCTURE:")
    print("  Each augmentation can be traced back to its source ID using augmentations_metadata.json")
    print("  Fields per source:")
    print("    - source_id: Original source identifier")
    print("    - label: Morphological class (FRI, FRII, COMPACT)")
    print("    - augmentation_start_index: First augmentation index for this source")
    print("    - augmentation_end_index: Last augmentation index for this source")
    print("    - num_augmentations: Number of augmentations created for this source")
    print("="*80)

if __name__ == '__main__': main()