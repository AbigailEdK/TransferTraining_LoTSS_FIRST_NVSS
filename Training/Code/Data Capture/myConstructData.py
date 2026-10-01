# region ABOUT
# ===================================================================================================
# > DONT NEED THIS FILE

# > Adjusted from construct_data.py by Dylan Farge.
# ===================================================================================================
# endregion

# region IMPORTS
import os, sys
import pandas as pd
from astropy.io import fits
import numpy as np
from sklearn.model_selection import train_test_split
print("Setting up tf layers...")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
# endregion

# region GLOBAL VARIABLES

# endregion

# region FUNCTIONS
def print_distribution(dataset, dataset_name, total_images=None, spacing=20):
    '''
    ARGUMENTS:
    ----------
        > dataset: array-like, the dataset for which to print the
          distribution.
        > dataset_name: str, the name of the dataset to be printed in the
          header.
        > total_images: int, optional, the total number of images in the
          dataset (if not provided, it will be calculated from the dataset).
        > spacing: int, optional, the spacing for the printed table
          columns (default is 20).
    
    RETURNS:
    --------
        > None, prints the distribution of the dataset in a formatted
          table.

    DESCRIPTION:
    ------------
        This function calculates the unique values and their counts in the provided dataset, and then prints a formatted table showing the morphology, count, and percentage of each unique value in the dataset. It also prints the total number of images and the percentage of the total that each unique value represents.    
    '''

    total_images = len(dataset) if total_images is None else total_images
    print(f"\n--- {dataset_name} Distribution ---")
    unique, counts = np.unique(dataset, return_counts=True)
    print(f"{'Morphology':^{spacing}}|{'Count':^{spacing}}|{'Percentage':^{spacing}}")
    print(f"{'-'*spacing}|{'-'*spacing}|{'-'*spacing}")
    for i in range(len(unique)):
        print(f"{unique[i]:^{spacing}}|{counts[i]:^{spacing}}|{f'{counts[i]/len(dataset)*100:.2f}%':^{spacing}}")
    print(f"{'-'*spacing}|{'-'*spacing}|{'-'*spacing}")
    print(f"Total Images: {len(dataset)}/{total_images}\t{len(dataset)/total_images*100:.2f}%\n")

def get_RADCAT_data(fits_folder, catalog_file = "/home/abigaildeklerk/Downloads/DeKlerk_Models/Data/RADCAT.csv"):
    '''
    ARGUMENTS:
    ----------
        > catalog_file: str, the path to the CSV file containing the
          catalog data.
        > fits_folder: str, the path to the folder containing the FITS
          files.   
    
    RETURNS:
    --------        
        > data_images: list of numpy arrays, the images extracted from the
          FITS files, organized in a list where each element corresponds to a different survey (e.g., FIRST, LOFAR, NVSS).
        > labels: numpy array, the labels corresponding to each image,
          extracted from the catalog file.
        > names: numpy array, the names of the sources corresponding to
          each image, extracted from the catalog file.
    
    DESCRIPTION:
    ------------
        This function reads the catalog data from the provided CSV file and filters it to include only sources of types "COMPACT", "FRI", and "FRII". It then iterates through the relevant sources and their corresponding FITS files, extracting the image data and organizing it into a list of numpy arrays. Each element in the list corresponds to a different survey (e.g., FIRST, LOFAR, NVSS). The function also extracts the labels and names of the sources from the catalog file and returns them as numpy arrays. The images are stored in a 4D array format, where the dimensions correspond to (number of sources, image height, image width, number of channels). The function handles both cases where the images have the same dimensions across surveys and where they have different dimensions.    
    '''

    same_dimensions = True 
    df = pd.read_csv(catalog_file, index_col=0)
    df = df[df["Type"].isin(["COMPACT", "FRI", "FRII"])]

    relevant_sources = sorted(df.index.to_list())
    relevant_files = os.listdir(fits_folder)

    names, labels = [],[]

    if same_dimensions:
        data_images = [np.zeros((len(relevant_sources), 300, 300, 3))] 
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
                    data_images[0][i,:,:,j] = fits.getdata(fits_folder + file, memmap=False)
                else:
                    data_images[j][i,:,:,0] = fits.getdata(fits_folder + file, memmap=False)
            except Exception as e:
                print(f"Error for {source} in {file}")
                raise e

    return data_images, np.array(labels), np.array(names)

def sigma_clipping(Xs, sigma):
    '''
    ARGUMENTS:
    ----------
        > Xs: list of numpy arrays, the images to be sigma clipped.
        > sigma: float, the sigma value to be used for clipping.

    RETURNS:
    --------        
        > clipped_images: list of numpy arrays, the sigma clipped
          images.

    DESCRIPTION:
    ------------
        This function applies sigma clipping to a list of images. For each image in the list, it calculates a mask using the sigma_clip function from the astropy library, which identifies the pixels that are considered outliers based on the specified sigma value. The function then sets the pixel values that are not masked (i.e., the outliers) to zero. The resulting sigma clipped images are returned as a list of numpy arrays. The sigma clipping is performed along the specified axes (in this case, the last two dimensions of the image), and the process is repeated for a specified number of iterations (in this case, 5 iterations). The function modifies the input images in place and returns the clipped images.    
    '''
    clipped_images = []
    masks = []
    
    for X in Xs:
        mask = sigma_clip(X, maxiters=5, sigma=sigma, axis= (1,2) ).mask
        X[~ mask] = 0
        clipped_images.append(X)
        masks.append(mask)

    return clipped_images, masks

def crop_normalize(Xs):
    '''
    ARGUMENTS:
    ----------
        > Xs: list of numpy arrays, the images to be cropped and normalized.
    
    RETURNS:
    --------
        > cropped_images: list of numpy arrays, the cropped and normalized
          images.
    
    DESCRIPTION:
    ------------
        This function crops and normalizes a list of images. For each image in the list, it calculates the width, height, and number of channels. It then determines the crop size based on the width of the image and crops the image to a square centered on the middle of the original image. After cropping, the function checks if there are any blank images (i.e., images with no variation in pixel values) and raises a ValueError if any are found. If the images are valid, the function proceeds to normalize the pixel values of each cropped image by scaling them to the range [0, 1] using the minimum and maximum pixel values of each image. The resulting cropped and normalized images are returned as a list of numpy arrays. The cropping and normalization process is applied to each image in the input list, and the function ensures that the output images are in a consistent format suitable for further processing or analysis.
    '''
    
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
                raise ValueError("Image has no variation in pixel values after cropping it.")
            
        minis = np.min(cropped, axis=(1,2), keepdims=True)
        maxis = np.max(cropped, axis=(1,2), keepdims=True)

        for i in range(cropped.shape[0]):
            cropped[i] = (cropped[i] - minis[i]) / (maxis[i] - minis[i])
        
        cropped_images.append(cropped)

    return cropped_images


def generate_augmented_data(Xs, y, names, save_dir, save_files):
    '''
    ARGUMENTS:
    ----------
        > Xs: list of numpy arrays, the original images.
        > y: numpy array, the labels corresponding to each image.
        > names: numpy array, the names of the sources corresponding to
          each image.
        > save_dir: str, the directory where the augmented images will be
          saved.
        > save_files: bool, whether to save the augmented images to disk.

    RETURNS:
    --------
        None

    DESCRIPTION:
    ------------
        This function generates augmented images for each source in the dataset. It calculates the number of images to augment for each morphology and then applies random flips and rotations to create new images. The augmented images are then saved to disk if specified. The function handles both cases where the images have the same dimensions across surveys and where they have different dimensions. It also applies sigma clipping to the augmented images before saving them if specified. If the augmented images are not saved to disk, the function ensures that any temporary files created during the process are removed.
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

    same_dimension = True if Xs[0].shape[-1] == 3 else False

    if os.path.exists(save_dir + "augmented_images.txt"):
        raise FileExistsError("Files already exist. Please remove them before running this function again.")

    with open(save_dir + "augmented_images.txt", "wb") as f:
    
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
            [np.save(f, X) if save_files else None for X in images]
    
    if not save_files:
        # remove the files if they were not saved
        os.remove(save_dir + "augmented_images.txt")

# endregion
