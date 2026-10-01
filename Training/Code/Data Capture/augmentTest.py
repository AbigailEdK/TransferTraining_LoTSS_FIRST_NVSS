# Admin
print("\n\n" + "="*160 + "="*160 + "\n\n")
## Imports
import shutil
import numpy as np
import os
import json
import pandas as pd
from glob import glob
from astropy.io import fits
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import RandomFlip, RandomRotation
## Output directories

# region PATHS
HOME_DIR = os.path.expanduser("~")
PROJECT_ROOT = os.path.join(HOME_DIR, "DeKlerk_Models") # ! CLUSTER USE
# PROJECT_ROOT = os.path.join(HOME_DIR, "Desktop", "DeKlerk_Models") # ! LOCAL USE

FITS_DIR = os.path.join(PROJECT_ROOT, "Data/FITS_BY_CHANNEL")
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Data/DATA_SOURCES/")
CATALOG_FILE = os.path.join(PROJECT_ROOT, "Data/RADCAT.csv")
# endregion 

# region FUNCTIONS
def check_directory(output_folder, overwrite=True):
    if os.path.exists(output_folder):
        if overwrite:
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)
            return True
        else:
            print(f"Folder {output_folder} already exists. Replace? (y/n)")
            choice = input().lower()

            if choice == 'y':
                shutil.rmtree(output_folder)
                os.makedirs(output_folder)
                return True
            else:
                return False

    else:
        os.makedirs(output_folder)
        return True
    
def count_folder_files(folder, extension=".fits"):
    if extension: 
        number = len([f for f in os.listdir(folder) if f.lower().endswith(extension)])
    else:
        number = len([f for f in os.listdir(folder)])

    print(f"Total FITS files in {folder}: {number}")
    # return number

def get_RADCAT_data(catalog_file=CATALOG_FILE, fits_folder=FITS_DIR):
    '''
    ARGS:
        catalog_file: path to the csv file containing the catalog data. It should have a "Type" column with the morphology type of each source, and the index should be the source names.
        fits_folder: path to the folder containing the fits files. The files should be named in the format "source_channel.fits", where channel is one of "FIRST", "LOFAR" or "NVSS". The folder should contain only the relevant fits files.
    RETURNS:
        data_images: a list of numpy arrays containing the image data for each channel. The shape of each array is (num_sources, width, height, num_channels). If the images have the same dimensions, then num_channels is 3 and the channels are in the last dimension. If the images have different dimensions, then num_channels is 1 and the channels are in separate arrays.
        labels: a numpy array containing the morphology type of each source.
        names: a numpy array containing the names of each source.
    '''
    same_dimensions = True 
    df = pd.read_csv(catalog_file, index_col=0)
    df = df[df["Type"].isin(["COMPACT", "FRI", "FRII"])]

    relevant_sources = sorted(df.index.to_list())
    relevant_files = os.listdir(fits_folder)

    names, labels = [],[]

    if same_dimensions:
        data_images = [np.zeros((len(relevant_sources), 128, 128, 3))] 
    else:
        data_images = [np.zeros((len(relevant_sources), 250, 250, 1)),  # FIRST
                       np.zeros((len(relevant_sources), 300, 300, 1)),  # LOFAR
                       np.zeros((len(relevant_sources), 30, 30, 1))]    # NVSS

    for i, source in enumerate(relevant_sources):

        names.append(source)
        labels.append(df.loc[source,"Type"])
        files = sorted([x for x in relevant_files if x.startswith(f"{source}_")])

        for j, file in enumerate(files):
            
            try:
                print(f"File {j}/{len(relevant_sources)}", end="\r")
                if same_dimensions:
                    data_images[0][i,:,:,j] = fits.getdata(fits_folder + "/" + file, memmap=False)
                else:
                    data_images[j][i,:,:,0] = fits.getdata(fits_folder + "/" + file, memmap=False)
            except Exception as e:
                print(f"Error for {source} in {file}")
                raise e

    return data_images, np.array(labels), np.array(names)

def generate_augmented_data(Xs, y, names, save_dir, save_files):

    '''
    ARGS:
        Xs: a list of numpy arrays containing the image data for each channel. The shape of each array is (num_sources, width, height, num_channels). If the images have the same dimensions, then num_channels is 3 and the channels are in the last dimension. If the images have different dimensions, then num_channels is 1 and the channels are in separate arrays.
        y: a numpy array containing the morphology type of each source.
        names: a numpy array containing the names of each source.
        save_dir: the directory where the augmented data should be saved. The function will save two files in this directory: "augmented_images_no_clip.txt" and "augmented_images_clip.txt". Each file will contain the augmented images without and with sigma clipping respectively. The shape of the augmented images will be (num_augmented_images, width, height, num_channels). The function will also save a file "augmented_labels.txt" containing the labels of the augmented images, and a file "augmented_names.txt" containing the names of the augmented images.
        save_files: a boolean indicating whether the augmented data should be saved to disk. If False, the function will still generate the augmented data but will not save it to disk. This can be useful for testing the function without creating large files on disk.
    RETURNS:
        augmented_images_no_clip: a numpy array containing the augmented images without sigma clipping. The shape of the augmented images will be (num_augmented_images, width, height, num_channels).
        augmented_images_clip: a numpy array containing the augmented images with sigma clipping. The shape of the augmented images will be (num_augmented_images, width, height, num_channels).
        augmented_labels: a numpy array containing the labels of the augmented images.
        augmented_names: a numpy array containing the names of the augmented images.
    '''
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

    same_dimension = True if len(Xs) == 1 else False

    augmented_images_no_clip_channels = [[] for _ in Xs]
    augmented_labels = []
    augmented_names = []

    if os.path.exists(save_dir + "augmented_images.txt"):
        raise FileExistsError("Files already exist. Please remove them before running this function again.")

    with open(save_dir + "augmented_images.txt", "wb") as f_no_clip:
        for i in range(len(names)):
            print(f"Augmenting... {i+1}/{len(names)}", end='\r')

            aug_images = [[] for _ in Xs]

            starting_seed = np.random.randint(2**32 - 1)
            for x_idx, X in enumerate(Xs):
                np.random.seed(starting_seed)
                min_value = np.min(X[i])
                for n in range(details[y[i]]):
                    seed = np.random.randint(2**32 - 1)
                    transformed = RandomFlip(seed=seed)(
                        RandomRotation(1, seed=seed, fill_mode="constant", fill_value=min_value)(X[i])
                    )
                    aug_images[x_idx].append(transformed)
                    augmented_names.append(names[i])
                    augmented_labels.append(y[i])

                aug_images[x_idx] = np.array(aug_images[x_idx])
                augmented_images_no_clip_channels[x_idx].append(aug_images[x_idx])

            # images = crop_normalize(aug_images)   
            [np.save(f_no_clip, X) if save_files else None for X in aug_images]
            # images = sigma_clipping(images, 3)
            # [np.save(f_clip, X) if save_files else None for X in images]

    if not save_files:
        # remove the files if they were not saved
        os.remove(save_dir + "augmented_images.txt")

    augmented_images_no_clip_channels = [
        np.concatenate(channel_batches, axis=0) for channel_batches in augmented_images_no_clip_channels
    ]
    augmented_images_no_clip = augmented_images_no_clip_channels[0] if same_dimension else augmented_images_no_clip_channels

    return augmented_images_no_clip, np.array(augmented_labels), np.array(augmented_names)

def crop_normalize(Xs):
    cropped_images = []

    for X in Xs:
    
        width, height, channel = X[0].shape
        crop_size = int(width * 128/300)
        cropped = X[:, width//2 - crop_size//2:width//2 + crop_size//2, height//2 - crop_size//2:height//2 + crop_size//2, :]

        # Makes sure that there are no blank images before continuing.
        for i, image in enumerate(cropped):
            maxi = np.max(image)
            mini = np.min(image)
            if maxi == mini:
                raise ValueError(f"Image has no variation in pixel values after cropping it. Batch image index={i}, shape={image.shape}, min={mini}, max={maxi}")
            
        minis = np.min(cropped, axis=(1,2), keepdims=True)
        maxis = np.max(cropped, axis=(1,2), keepdims=True)

        for i in range(cropped.shape[0]):
            cropped[i] = (cropped[i] - minis[i]) / (maxis[i] - minis[i])
        
        cropped_images.append(cropped)

    return cropped_images

def check_empty_image_error(Xs, sources):
    for X in Xs:
        empty_indices = []
        len_channels = X.shape[-1]
        for i, image in enumerate(X):
            for channel in range(len_channels):
                if np.isnan(image[:,:,channel]).all():
                    empty_indices.append(i)
                    break
    
        if len(empty_indices) > 0:
            raise ValueError(f"Found {len(empty_indices)} empty images. These sources are : {sources[empty_indices]}")
    
def fill_partial_images(Xs, sources, save_dir, save_files):
    cut_off_sources = set()
    for X in Xs:
        for i in range(X.shape[0]):
            for channel in range(X.shape[-1]):
                im = X[i,:,:,channel]
                if np.isnan(im).any():
                    min_val = np.nanmin(im)
                    X[i,:,:,channel] = np.nan_to_num(im, nan=min_val)
                    cut_off_sources.add(sources[i])
        
    if len(cut_off_sources) > 0:
        print(f"Filled {len(cut_off_sources)} partial images. These sources are : {cut_off_sources}")
        if save_files:
            np.save(save_dir + "filled_partial_names.npy", list(cut_off_sources))

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

def add_fused_channels(X, fusion_specs):
    """
    Add multiple fused channels to a 3-channel image array.

    ARGS:
        X: numpy array of shape (num_sources, width, height, 3)
        fusion_specs: list of tuples like:
            [
                ("mean", None),
                ("max", None),
                ("weighted_mean", [0.5, 0.3, 0.2]),
            ]

    RETURNS:
        numpy array of shape (num_sources, width, height, 3 + len(fusion_specs))
    """
    if X.ndim != 4 or X.shape[-1] != 3:
        raise ValueError(f"Expected input shape (N, H, W, 3), got {X.shape}")

    fused_channels = [X]

    for method, weights in fusion_specs:
        if method == "mean":
            fused = np.mean(X, axis=-1, keepdims=True)
        elif method == "weighted_mean":
            if weights is None:
                raise ValueError("weights must be provided when method='weighted_mean'")
            weights = np.asarray(weights, dtype=X.dtype)
            if weights.shape != (3,):
                raise ValueError(f"weights must have shape (3,), got {weights.shape}")
            if np.isclose(np.sum(weights), 0):
                raise ValueError("Sum of weights must not be zero")
            weights = weights / np.sum(weights)
            fused = np.tensordot(X, weights, axes=([-1], [0]))[..., np.newaxis]
        elif method == "max":
            fused = np.max(X, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unsupported fusion method: {method}")

        fused_channels.append(fused.astype(X.dtype, copy=False))

    return np.concatenate(fused_channels, axis=-1)

# endregion

# region MAIN
def main():
    check_directory(SOURCE_DIR)

    SAVE_FILES = True
    use_fused_channel = True
    
    print("\nLoading data...\n")
    Xs, y, names = get_RADCAT_data()

    print("\n" + "="*80)
    print(f"Xs shape: {np.array(Xs).shape}")
    print(f"y shape: {np.array(y).shape}")
    print(f"names shape: {np.array(names).shape}\n")
    print("="*80 + "\n")

    if use_fused_channel:
        print("\nFusing channels...\n")
        X_rgb = Xs[0][..., :3]   # always start from the original 3 channels
        Xs[0] = add_fused_channels(
            X_rgb,
            [
                ("mean", None),
                ("weighted_mean", [0.2, 0.6, 0.2]),
                # ("max", None),
            ]
        )
    else:
        print("Skipping fused channel creation. Using original 3 channels.")

    check_empty_image_error(Xs, names)
    fill_partial_images(Xs, names, SOURCE_DIR, save_files=SAVE_FILES)
    
    print("Splitting data into training/validation and testing sets...")
    train_val_index, test_index = train_test_split(range(len(y)), test_size=0.2, random_state=100, stratify=y)

    X_test = [X[test_index] for X in Xs]

    if SAVE_FILES:
        np.save(SOURCE_DIR + "X_test.npy", X_test[0])
        np.save(SOURCE_DIR + "y_test.npy", y[test_index])
        np.save(SOURCE_DIR + "names_test.npy", names[test_index])

    Xs_train_val = [X[train_val_index] for X in Xs]
    y_train_val = y[train_val_index]
    names_train_val = names[train_val_index]

    print("\nPreparing training and validation data...\n")
    for key, val in {"TrainVal": y_train_val, "Testing": y[test_index]}.items():
        print_distribution(val, key, total_images=len(y))

    # del Xs, y, names
    print("\n" + "="*80)
    print(f"Xs_train_val shape: {np.array(Xs_train_val).shape}")
    print(f"y_train_val shape: {np.array(y_train_val).shape}")
    print(f"names_train_val shape: {np.array(names_train_val).shape}")
    print("="*80 + "\n")

    print("\nGenerating augmented data...\n\n")
    X_aug, y_aug, names_aug = generate_augmented_data(Xs_train_val, y_train_val, names_train_val, save_dir=SOURCE_DIR, save_files=SAVE_FILES)

    print("\n" + "="*80)
    print(f"X_aug shape: {np.array(X_aug).shape}")
    print(f"y_aug shape: {np.array(y_aug).shape}")
    print(f"names_aug shape: {np.array(names_aug).shape}")
    print("="*80 + "\n")

    if SAVE_FILES:
        np.save(SOURCE_DIR + "X_train_val.npy", Xs_train_val[0])    
        np.save(SOURCE_DIR + "y_train_val.npy", y_train_val)
        np.save(SOURCE_DIR + "names_train_val.npy", names_train_val)

        np.save(SOURCE_DIR + "X_aug.npy", X_aug[0])    
        np.save(SOURCE_DIR + "y_aug.npy", y_aug)
        np.save(SOURCE_DIR + "names_aug.npy", names_aug)

    print("Finished!")

if __name__ == "__main__": main()
# endregion