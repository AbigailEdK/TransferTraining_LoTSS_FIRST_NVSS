# region ABOUT
# ===================================================================================================
# > This script handles loading in and augmenting the data needed for training and testing of models. It includes functions to load the RADCAT data, generate augmented data through random flips and rotations, crop and normalize images, check for empty images, fill partial images, print class distributions, add fused channels, and add channels of zeros or white noise. The script also includes a function to split the original data into training/validation and testing sets, and a function to copy the split data into experiment directories for further processing. The data augmentation process is designed to ensure that each class has at least 3000 images after augmentation, and the script includes checks to verify the success of the augmentation process before proceeding to model training.

# > Adjusted from construct_data.py by Dylan Farge.
# ===================================================================================================
# endregion

# region IMPORTS
import shutil
import numpy as np
import os
import pandas as pd
from glob import glob
from astropy.io import fits
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import RandomFlip, RandomRotation
# endregion

# region PATHS
from paths import CATALOG_FILE, FITS_DIR
# endregion 

# region CONSTANTS
from constants import NUM_TO_AUGMENT
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
    ARGUMENTS:
    ----------
        > catalog_file: the path to the RADCAT catalog file (CSV format) containing the metadata for the sources, including their names and morphology types.
        > fits_folder: the directory where the FITS files for each source are stored. (../DATA/FITS_BY_CHANNEL/). Each source is named as "source_channel.fits", where "source" is the name of the source and "channel" is the survey channel (e.g., FIRST, LOFAR, NVSS). 

    RETURNS:
    --------    
        > data_images: a list of numpy arrays containing the image data for each channel. The shape of each array is (num_sources, width, height, num_channels), where num_channels = 3 and size = 128x128. 
        > labels: a numpy array containing the morphology type of each source.
        > names: a numpy array containing the names (IDs) of each source.

    DESCRIPTION:
    ------------
    The function loads the RADCAT data by reading the catalog file to get the list of relevant sources and their morphology types, and then iterates through the FITS files in the specified directory to load the image data for each source. The function checks that each source has three corresponding FITS files (one for each survey) and loads the image data into numpy arrays. The resulting data is returned as a list of numpy arrays for the images, along with numpy arrays for the labels and names of the sources. 
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
        print(f"File {i+1}/{len(relevant_sources)}", end="\r")

        for j, file in enumerate(files):
            
            try:
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
    ARGUMENTS:
    ----------
        > Xs: a list of numpy arrays containing the image data for each channel. The shape of each array is (num_sources, width, height, num_channels), where num_channels = 3 and size = 128x128. If the images have different dimensions, then num_channels is 1 and the channels are in separate arrays.
        > y: a numpy array containing the morphology type of each source.
        > names: a numpy array containing the names of each source.
        > save_dir: the directory where the augmented data should be saved. The shape of the augmented images will be (num_augmented_images, width, height, num_channels). The function will also save a file "augmented_labels.txt" containing the labels of the augmented images, and a file "augmented_names.txt" containing the names of the augmented images.
        > save_files: a boolean indicating whether the augmented data should be saved to disk. If False, the function will still generate the augmented data but will not save it to disk.
    
    RETURNS:
    --------    
        > augmented_images: a numpy array containing the augmented images. The shape of the array is (num_augmented_images, width, height, num_channels).
        > augmented_labels: a numpy array containing the labels of the augmented images. The shape of the array is (num_augmented_images,).
        > augmented_names: a numpy array containing the names of the augmented images. The shape of the array is (num_augmented_images,).
    
    DESCRIPTION:
    ------------
    The function generates augmented data by applying random flips and rotations to the original images. The number of augmented images is determined based on the class distribution in the original training/validation set to ensure that each class has at least 3000 images after augmentation. The function iterates through each source in the original dataset, applies the specified augmentations, and collects the augmented images, labels, and names. If save_files is True, the function saves the augmented data to disk in the specified directory. Finally, the function returns the augmented images along with their corresponding labels and names.
    '''
    details = {}

    for morph in np.unique(y):
        num_of_sources = len(np.where(y == morph)[0])
        num_to_augment = NUM_TO_AUGMENT - num_of_sources
        num_per_source = int(np.ceil(num_to_augment / num_of_sources))

        print("\n" + "="*80)
        print(f"Augmenting {morph}: {num_of_sources}")
        print(f"\tNeed to augment at least {num_to_augment} images")
        print(f"\tAugmenting {num_per_source} images per source")
        print(f"\tThus, {num_per_source * num_of_sources} images will be added")
        print("="*80 + "\n")

        details[morph] = num_per_source

    same_dimension = True if len(Xs) == 1 else False

    augmented_images_channels = [[] for _ in Xs]
    augmented_labels = []
    augmented_names = []

    if os.path.exists(save_dir + "augmented_images.txt"):
        raise FileExistsError("Files already exist. Please remove them before running this function again.")

    with open(save_dir + "augmented_images.txt", "wb") as f_no_clip:
        for i in range(len(names)):
            print(f"Augmenting... {i+1}/{len(names)}", end='\r')

            aug_images = [[] for _ in Xs]

            starting_seed = int(np.random.randint(0, 2**32, dtype=np.uint32))
            for x_idx, X in enumerate(Xs):
                np.random.seed(starting_seed)
                min_value = np.min(X[i])
                for n in range(details[y[i]]):
                    seed = int(np.random.randint(0, 2**32, dtype=np.uint32))
                    transformed = RandomFlip(seed=seed)(
                        RandomRotation(1, seed=seed, fill_mode="constant", fill_value=min_value)(X[i])
                    )
                    aug_images[x_idx].append(transformed)
                    augmented_names.append(names[i])
                    augmented_labels.append(y[i])

                aug_images[x_idx] = np.array(aug_images[x_idx])
                augmented_images_channels[x_idx].append(aug_images[x_idx])

            [np.save(f_no_clip, X) if save_files else None for X in aug_images]

    if not save_files:
        os.remove(save_dir + "augmented_images.txt")

    augmented_images_channels = [
        np.concatenate(channel_batches, axis=0) for channel_batches in augmented_images_channels
    ]
    augmented_images = augmented_images_channels[0] if same_dimension else augmented_images_channels

    return augmented_images, np.array(augmented_labels), np.array(augmented_names)

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

# def add_zeros_channel(Xs):
#     '''
#     ARGUMENTS:
#     ----------
#         > Xs: a list of numpy arrays containing the image data for each channel. The shape of each array is (num_sources, width, height, num_channels), where num_channels = 3 and size = 128x128. 

#     RETURNS:
#     --------    
#         > Xs: the input list of numpy arrays with an additional channel of zeros added to the -1 axis. The shape of each array will be (num_sources, width, height, num_channels + 1) after the function is applied.

#     DESCRIPTION:
#     ------------    
#     This function takes a list of numpy arrays representing image data and adds a channel of zeros to the end of it. This is for use in the RADCAT model testing phase. 
#     '''

#     print("Xs shape before adding zeros channel: ", Xs.shape)
#     zeros_channel = np.zeros_like(Xs[:][..., :1])  # Create a zeros channel with the same shape as one of the existing channels
#     print("Zeros shape: ", zeros_channel.shape)
#     Xs = np.concatenate([Xs[:], zeros_channel], axis=-1)  # Add the zeros channel to the end of the existing channels
#     print("New Xs shape: ", Xs.shape)
#     return Xs

# def add_white_noise_channel(Xs):
#     '''
#     ARGUMENTS:
#     ----------
#         > Xs: a list of numpy arrays containing the image data for each channel. The shape of each array is (num_sources, width, height, num_channels), where num_channels = 3 and size = 128x128.

#     RETURNS:
#     --------    
#         > Xs: the input list of numpy arrays with an additional channel of white noise added to the -1 axis. The shape of each array will be (num_sources, width, height, num_channels + 1) after the function is applied.

#     DESCRIPTION:
#     ------------    
#     This function takes a list of numpy arrays representing image data and adds a channel of white noise to the end of it. This is for use in the RADCAT model testing phase. The white noise is generated as a Gaussian distribution with mean 0 and standard deviation 1, and it has the same shape as one of the existing channels in the input data. 
#     '''

#     print("xs shape before adding white noise channel:", Xs.shape)
#     noise_channel = np.random.normal(0, 1, size=Xs[:][..., :1].shape)  # Create a Gaussian noise channel with the same shape as one of the existing channels
#     print("noise shape:", noise_channel.shape)
#     Xs = np.concatenate([Xs[:], noise_channel], axis=-1)  # Add the noise channel to the end of the existing channels
#     print("New Xs shape:", Xs.shape)
#     return Xs

def add_zeros_channel(Xs):
    # Ensure Xs is a numpy array (handles cases where a list is passed)
    Xs = np.asarray(Xs) 
    
    # Create the extra channel, preserving the first 3 dims (N, H, W)
    zeros_channel = np.zeros((*Xs.shape[:-1], 1), dtype=Xs.dtype)
    
    # Direct concatenation
    return np.concatenate([Xs, zeros_channel], axis=-1)

def add_white_noise_channel(Xs):
    Xs = np.asarray(Xs)
    
    # Generate noise matching the batch size and spatial dimensions
    noise_channel = np.random.normal(0, 1, size=(*Xs.shape[:-1], 1)).astype(Xs.dtype)
    
    return np.concatenate([Xs, noise_channel], axis=-1)
# endregion

# region MAINS
def splitData(data_dir):
    '''
    ARGUMENTS:
    ----------
        > data_dir: the directory where the split data will be stored. (../Results/SourceData/)

    RETURNS:
    --------    
        > None. The function saves the split data to disk (X_test.npy, y_test.npy, names_test.npy, X_train_val.npy, y_train_val.npy, names_train_val.npy).

    DESCRIPTION:
    ------------    
    The function calls get_RADCAT_data() to load the original data from the specified directory, splits it into training/validation and testing sets using an 80-20 split, and saves the resulting datasets back to disk. The split is stratified based on the labels to ensure that the distribution of classes is maintained in both sets. The function also prints the distribution of classes in the training/validation and testing sets for verification.
    '''
    # check_directory(SOURCE_DIR)

    SAVE_FILES = True
    
    print(f"\nLoading data from {data_dir}...\n")
    Xs, y, names = get_RADCAT_data()
    # Xs = Xs_list[0]

    # print("\n" + "="*80)
    # print(f"Xs shape: {np.array(Xs).shape}")
    # print(f"y shape: {np.array(y).shape}")
    # print(f"names shape: {np.array(names).shape}\n")
    # print("="*80 + "\n")

    check_empty_image_error(Xs, names)
    fill_partial_images(Xs, names, data_dir, save_files=SAVE_FILES)

    print("Splitting data into training/validation and testing sets...")
    train_val_index, test_index = train_test_split(range(len(y)), test_size=0.2, random_state=100, stratify=y)

    X_test = [X[test_index] for X in Xs]
    X_train_val = [X[train_val_index] for X in Xs]
    
    # > Add a channel of zeros and a channel of white noise to the test set for use in the RADCAT model testing phase.
    X_test = add_zeros_channel(X_test[0])
    X_test = add_white_noise_channel(X_test)

    if SAVE_FILES:
        np.save(data_dir + "/X_test.npy", X_test)
        np.save(data_dir + "/y_test.npy", y[test_index])
        np.save(data_dir + "/names_test.npy", names[test_index])
        np.save(data_dir + "/X_train_val.npy", X_train_val[0])
        np.save(data_dir + "/y_train_val.npy", y[train_val_index])
        np.save(data_dir + "/names_train_val.npy", names[train_val_index])

    # print("\nPreparing training and validation data...\n")
    for key, val in {"TrainVal": y[train_val_index], "Testing": y[test_index]}.items():
        print_distribution(val, key, total_images=len(y))

    # del Xs, y, names
    # print("\n" + "="*80)
    # print(f"X_train_val shape: {np.array(X_train_val).shape}")
    # print(f"y_train_val shape: {np.array(y_train_val).shape}")
    # print(f"names_train_val shape: {np.array(names_train_val).shape}")
    # print("="*80 + "\n")

def copySplitData(data_dir, results_dir):
    '''
    ARGUMENTS:
    ----------
        > data_dir: the directory where the original split data is stored. (../Results/SourceData/)
        > results_dir: the directory where the split data should be copied to. (../Results/Experiment_X/Data/)

    RETURNS:
    --------    
        > None. The function copies the split data from the source directory to the experiment directory.

    DESCRIPTION:
    ------------    
    This function copies the original split data (X_train_val.npy, y_train_val.npy, names_train_val.npy, X_test.npy, y_test.npy, names_test.npy) from the source directory to the experiment directory. This ensures the data only has to be loaded once at the beginning of all experiments. Thereafter, each experiment can load the data itself and augment it separately. 
    '''

    src = data_dir
    dst = os.path.join(results_dir, "Data")
    os.makedirs(dst, exist_ok=True)

    npy_files = glob(os.path.join(src, "*.npy"))
    if not npy_files:
        print(f"No .npy files found in {src}")
        return

    for fp in npy_files:
        shutil.copy2(fp, dst)

    print(f"Copied {len(npy_files)} .npy files from {src} to {dst}.")

def augmentData(data_dir):
    '''
    ARGUMENTS:
    ----------
        > data_dir: the directory where the split data will be stored. (../Results/SourceData/)
    
    RETURNS:
    --------    
        > None. The function saves the augmented data to disk (X_train_val.npy, y_train_val.npy, names_train_val.npy after augmentation).  
    
    DESCRIPTION:
    ------------    
    This function loads the training and validation data from the specified directory, generates augmented data using the generate_augmented_data() function, adds a channel of zeros and a channel of white noise to the augmented data, and saves the resulting augmented dataset back to disk. The augmented data is generated by applying random flips and rotations to the original images, and the number of augmented images is determined based on the class distribution in the original training/validation set to ensure that each class has at least 3000 images after augmentation.
    '''

    SAVE_FILES = True

    # > Load the training and validation data from disk
    X_train_val = [np.load(os.path.join(data_dir, "X_train_val.npy"), allow_pickle=True)]
    y_train_val = np.load(os.path.join(data_dir, "y_train_val.npy"), allow_pickle=True)
    names_train_val = np.load(os.path.join(data_dir, "names_train_val.npy"), allow_pickle=True)

    print("\n" + "="*80)
    print(f"X_train_val shape: {np.array(X_train_val).shape}")
    print(f"y_train_val shape: {np.array(y_train_val).shape}")
    print(f"names_train_val shape: {np.array(names_train_val).shape}")
    print("="*80 + "\n")

    print("\nGenerating augmented data...\n\n")
    X_aug, y_aug, names_aug = generate_augmented_data(X_train_val, y_train_val, names_train_val, save_dir=data_dir + "/", save_files=SAVE_FILES)

    X_aug = add_zeros_channel(X_aug)
    X_aug = add_white_noise_channel(X_aug)

    print("\n" + "="*80)
    print(f"X_aug shape: {np.array(X_aug).shape}")
    print(f"y_aug shape: {np.array(y_aug).shape}")
    print(f"names_aug shape: {np.array(names_aug).shape}")
    print("="*80 + "\n")

    if SAVE_FILES:
        for filename, array in [
            ("X_train_val.npy", X_aug),
            ("y_train_val.npy", y_aug),
            ("names_train_val.npy", names_aug),
        ]:
            file_path = os.path.join(data_dir, filename)
            os.remove(file_path)
            np.save(file_path, array)

    # print("Finished!")
