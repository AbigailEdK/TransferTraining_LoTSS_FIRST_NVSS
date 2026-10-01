# region IMPORTS
import shutil
import numpy as np
import os
import pandas as pd
from astropy.io import fits
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import RandomFlip, RandomRotation
# endregion

# region PATHS
FITS_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/FITS_BY_CHANNEL"
NUMPY_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/NUMPY_SOURCES/"
SOURCE_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/DATA_SOURCES"

output_dirs = [FITS_DIR, NUMPY_DIR, SOURCE_DIR]

FIRST_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F/FIRST"
LOFAR_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F/LoTSS"
NVSS_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F/NVSS"

CATALOG_FILE = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT.csv"

# region FUNCTIONS
def check_directory(output_folder):
    if os.path.exists(output_folder):
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
    check_directory(save_dir)

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

    if os.path.exists(save_dir + "augmented_images_no_clip.txt"):
        raise FileExistsError("Files already exist. Please remove them before running this function again.")

    with open(save_dir + "augmented_images_no_clip.txt", "wb") as f_no_clip:
        for i in range(len(names)):
            print(f"Augmenting... {i+1}/{len(names)}", end='\r')

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
                
            images = crop_normalize(aug_images)   
            [np.save(f_no_clip, X) if save_files else None for X in images]
            # images = sigma_clipping(images, 3)
            # [np.save(f_clip, X) if save_files else None for X in images]
    
    if not save_files:
        # remove the files if they were not saved
        os.remove(save_dir + "augmented_images_no_clip.txt")
        os.remove(save_dir + "augmented_images_clip.txt")

def crop_normalize(Xs):
    cropped_images = []

    for X in Xs:
    
        width, height, channel = X[0].shape
        print(f"Cropping and normalizing images of shape ({width}, {height}, {channel})...")

        crop_size = int(width * 128/300)
        cropped = X[:, width//2 - crop_size//2:width//2 + crop_size//2, height//2 - crop_size//2:height//2 + crop_size//2, :]
        

        # Makes sure that there are no blank images before continuing.
        for i, image in enumerate(cropped):
            maxi = np.max(image)
            mini = np.min(image)
            if maxi == mini:
                print(f"Image {i} has no variation in pixel values after cropping it.")
            
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

# endregion


for directory in output_dirs:
    check_directory(directory)
# endregion

# region MAIN
def main():
    SAVE_FILES = True
    pixel_type = "same_pixels/"
    save_dir = NUMPY_DIR 

    check_directory(save_dir)

    Xs, y, names = get_RADCAT_data()

    check_empty_image_error(Xs, names)

    fill_partial_images(Xs, names, NUMPY_DIR, save_files=SAVE_FILES)

    train_val_index, test_index = train_test_split(range(len(y)), test_size=0.2, random_state=100, stratify=y)

    # X_test = crop_normalize([X[test_index] for X in Xs])
    X_test = [X[test_index] for X in Xs]

    if SAVE_FILES:
        save_dir = NUMPY_DIR + "TEST_DATA/"
        check_directory(save_dir)

        if pixel_type == "different_pixels/":
            np.save(save_dir + "X_test_no_clip_FIRST.npy", X_test[0])
            np.save(save_dir + "X_test_no_clip_LOFAR.npy", X_test[1])
            np.save(save_dir + "X_test_no_clip_NVSS.npy", X_test[2])
            # X_test = sigma_clipping(X_test, 3)
            np.save(save_dir + "X_test_clip_FIRST.npy", X_test[0])
            np.save(save_dir + "X_test_clip_LOFAR.npy", X_test[1])
            np.save(save_dir + "X_test_clip_NVSS.npy", X_test[2])
        else:
            np.save(save_dir + "X_test.npy", X_test[0])
            # X_test = sigma_clipping(X_test, 3)
            # np.save(save_dir + "X_test_clip.npy", X_test[0])
        
        np.save(save_dir + "y_test.npy", y[test_index])
        np.save(save_dir + "names_test.npy", names[test_index])

    # del X_test

    Xs_train_val = [X[train_val_index] for X in Xs]
    y_train_val = y[train_val_index]
    names_train_val = names[train_val_index]

    for key, val in {"TrainVal": y_train_val, "Testing": y[test_index]}.items():
        print_distribution(val, key, total_images=len(y))

    # del Xs, y, names

    save_dir = NUMPY_DIR + "TRAINVAL_DATA/"
    check_directory(save_dir)

    generate_augmented_data(Xs_train_val, y_train_val, names_train_val, save_dir, SAVE_FILES)

    # Xs_train_val = crop_normalize(Xs_train_val)

    if SAVE_FILES:
        if pixel_type == "different_pixels/":
            np.save(save_dir + "X_train_val_no_clip_FIRST.npy", Xs_train_val[0])
            np.save(save_dir + "X_train_val_no_clip_LOFAR.npy", Xs_train_val[1])
            np.save(save_dir + "X_train_val_no_clip_NVSS.npy", Xs_train_val[2])
            # Xs_train_val = sigma_clipping(Xs_train_val, 3)
            np.save(save_dir + "X_train_val_clip_FIRST.npy", Xs_train_val[0])
            np.save(save_dir + "X_train_val_clip_LOFAR.npy", Xs_train_val[1])
            np.save(save_dir + "X_train_val_clip_NVSS.npy", Xs_train_val[2])
        else:
            np.save(save_dir + "X_train_val.npy", Xs_train_val[0])
            # Xs_train_val = sigma_clipping(Xs_train_val, 3)
            # np.save(save_dir + "X_train_val_clip.npy", Xs_train_val[0])
        
        np.save(save_dir + "y_train_val.npy", y_train_val)
        np.save(save_dir + "names_train_val.npy", names_train_val)

    # del Xs_train_val, y_train_val, names_train_val

    print("Finished!")

    TEST_DIR = NUMPY_DIR + "TEST_DATA/"
    TRAINVAL_DIR = NUMPY_DIR + "TRAINVAL_DATA/"

    for f in os.listdir(TEST_DIR):
        if f.endswith(".npy"):
            shutil.copy2(os.path.join(TEST_DIR, f), os.path.join(SOURCE_DIR, f))

    for f in os.listdir(TRAINVAL_DIR):
        if f.endswith(".npy"):
            shutil.copy2(os.path.join(TRAINVAL_DIR, f), os.path.join(SOURCE_DIR, f))

if __name__ == "__main__":
    main()
