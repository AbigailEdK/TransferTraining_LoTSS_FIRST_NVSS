# region ABOUT
# ===================================================================================================
# > This script contains functions for loading and processing the dataset, including building the dataset with data augmentation and checking for previously interrupted experiments. The functions are designed to work with the specific structure of the dataset and the configuration settings used in the project. The main functions include:
    # - get_dataset: loads the dataset based on the specified kind (train_val or test) and the configuration settings, returning the input data, labels, and names.
    # - get_channels: maps survey names to their corresponding channel indices and returns the list of channels to be used based on the surveys specified in the configuration.
    # - get_folds: uses StratifiedKFold to create folds for cross-validation, ensuring that the class distribution is preserved in each fold, and returns a list of train and validation indices for each fold.
    # - folder_construct: constructs the expected folder path for saving results based on the model, epochs, surveys, learning rate, regularization parameter, and save directory specified in the configuration.

# > Adjusted from process.py by Dylan Farge.
# ===================================================================================================
# endregion

# region IMPORTS
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from astropy.io import fits
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
# endregion

# region GLOBAL VARIABLES
HOME_DIR = os.path.expanduser("~")
PROJECT_ROOT = os.path.join(HOME_DIR, "DeKlerk_Models")
# DATA_FOLDER = os.path.join(PROJECT_ROOT, "Data", "NUMPY_SOURCES")
# TEST_SOURCES = os.path.join(PROJECT_ROOT, "Data", "TESTING")
# TRAINVAL_SOURCES = os.path.join(PROJECT_ROOT, "Data", "TRAINING AND VALIDATION")
RESULTSDIRECTORY = os.path.join(PROJECT_ROOT, "Results")
DATA_SOURCES = os.path.join(PROJECT_ROOT, "Data", "DATA_SOURCES")
# endregion

# region FUNCTIONS
def get_dataset(kind, cf, chunk_size=500):
    '''
    ARGUMENTS:
    ----------
        > kind: a string indicating the type of dataset to load ('train_val' or
          'test')
        > cf: configuration object containing settings for loading the dataset
        > chunk_size: number of samples to load at a time to avoid memory issues
          (default: 100). Reduce this if memory crashes still occur.
    
    RETURNS:
    --------        
        > X: the input data (images) for the specified dataset loaded from .npy
          files in chunks based on the configuration settings
        > y: the corresponding labels for the specified dataset loaded from .npy
          files in chunks based on the configuration settings
        > names: the IDs of the samples corresponding to the specified dataset
          loaded from .npy files in chunks based on the configuration settings
    
    DESCRIPTION:
    ------------        
        This function loads the dataset in chunks to avoid memory issues with large files.
        It uses memory mapping to read file metadata without loading data, then sequentially
        loads chunk_size samples at a time, selects the appropriate channels, and combines
        all chunks at the end into a single array.
    '''

    # ! OLD CODE
    '''
    channels = get_channels(cf.SURVEYS)

    print(f"Loading {kind} images in chunks (chunk_size={chunk_size})...")
    
    # Use memmap for X (large numeric array), but load y and names directly
    # (they're typically much smaller and contain Python objects)
    X_full = np.load(f"{DATA_FOLDER}/{kind}/X_{kind}.npy", mmap_mode='r')
    y = np.load(f"{DATA_FOLDER}/{kind}/y_{kind}.npy", allow_pickle=True)
    names = np.load(f"{DATA_FOLDER}/{kind}/names_{kind}.npy", allow_pickle=True)
    
    num_samples = X_full.shape[0]
    
    # Load X data in chunks and combine
    X_chunks = []
    
    for i in range(0, num_samples, chunk_size):
        end_idx = min(i + chunk_size, num_samples)
        print(f"  Loading samples {i} to {end_idx} of {num_samples}...")
        
        X_chunks.append(X_full[i:end_idx, :, :, channels])
    
    print("Combining all chunks...")
    X = np.concatenate(X_chunks, axis=0)

    return [
        X,
        y,
        names
    ]
    '''

    # ! NEW CODE
    channels = get_channels(cf.SURVEYS)

    print("Loading images for {cf.SURVEYS}...")

    X = np.load(f"{DATA_SOURCES}/X_{kind}.npy")[:,:,:,channels]

    return [
        X,
        np.load(f"{DATA_SOURCES}/y_{kind}.npy"),
        np.load(f"{DATA_SOURCES}/names_{kind}.npy")
    ]

def get_channels(surveys):
    '''
    ARGUMENTS:
    ----------
        > surveys: a list of survey names specified in the configuration

    RETURNS:
    --------        
        > channels: a list of channel indices corresponding to the 
          specified surveys

    DESCRIPTION:
    ------------
        This function maps survey names to their corresponding channel indices and returns the list of channels to be used based on the surveys specified in the configuration. The mapping is as follows:
        - FIRST: channel 0
        - LOFAR: channel 1
        - NVSS: channel 2
        - RADCAT: channel 3 
    '''

    return [v for k,v in {"FIRST": 0,"LOFAR": 1,"NVSS": 2}.items() if k in surveys]

def get_folds(X, y, val_split, random_state):
    '''
    ARGUMENTS:
    ----------
        > X: the input data (images) for which to create folds
        > y: the corresponding labels for the input data
        > val_split: the validation split ratio to determine the number of folds
        > random_state: the random seed for reproducibility when shuffling the data
    
    RETURNS:
    --------        
        > folds: a list of train and validation indices for each fold created using StratifiedKFold
    
    DESCRIPTION:
    ------------        
        This function uses StratifiedKFold to create folds for cross-validation, ensuring that the class distribution is preserved in each fold. The number of splits is determined by the validation split ratio (e.g., if val_split is 0.2, then n_splits will be 5). The function returns a list of train and validation indices for each fold created using StratifiedKFold.
    '''

    stratified_kfold = StratifiedKFold(n_splits=int(1/val_split), random_state=random_state, shuffle=True)
    return [[t,v] for t,v in stratified_kfold.split(X, y)]

def get_model_path(cf, fold=None):

    model_str = str(cf.MODEL)
    surveys_str = ''.join(x[0] for x in cf.MODEL)

    return cf.MODEL_DIR + "model" + model_str + "_set" + surveys_str + "_" + "20e" + "_" + str(cf.LEARNING_RATE) + "l_" + str(cf.REGULARISATION) + "r/" 

    # if fold is None:
    #     return RESULTSDIRECTORY + "/Training/model" + model_str + "_set" + surveys_str + "_" + "20e" + "_" + str(cf.LEARNING_RATE) + "l_" + str(cf.REGULARISATION) + "r/"
    # else:
    #     return RESULTSDIRECTORY + "/Training/model" + model_str + "_set" + surveys_str + "_" + "20e" + "_" + str(cf.LEARNING_RATE) + "l_" + str(cf.REGULARISATION) + "r/" + "model_fold_" + fold + ".keras"

def get_weights(path):    
    if os.path.exists(path):
        print(f"Loading model from {path}...")
        weights = tf.keras.models.load_weights(path)
    else:
        raise ValueError(f"Model file not found at {path}. Please check the path and ensure the model has been trained and saved correctly.")

    return weights


def build_dataset(X, y, names, indices, cf, size_per_type):
    '''
    ARGUMENTS:
    ----------
        > X: the full input data (images) already loaded and pre-augmented
        > y: the full labels already loaded
        > names: the full source IDs already loaded
        > indices: indices to select from the full dataset for this fold
        > cf: configuration object containing settings
        > size_per_type: target size per morphological type (for reference/logging)
    
    RETURNS:
    --------
        > X_subset: the selected input data for the specified indices
        > y_subset: the selected labels for the specified indices
        > names_subset: the selected source IDs for the specified indices
    
    DESCRIPTION:
    ------------
        This function selects a subset of the dataset based on the provided indices
        and returns the selected data, labels, and names. Since X, y, and names
        are already loaded from NUMPY_SOURCES and contain pre-augmented data,
        no additional augmentation is needed. The data is simply shuffled before
        returning to ensure randomness within each fold.
    '''
    
    print("Building Dataset...")
    print(f"Selecting {len(indices)} samples for this fold (target size per type: {size_per_type})")

    X_subset = X[indices]
    y_subset = y[indices]
    names_subset = names[indices]
    
    print("Shuffling...")
    X_subset, y_subset, names_subset = shuffle(X_subset, y_subset, names_subset, random_state=cf.SEED)

    return X_subset, y_subset, names_subset

def folder_construct(model, epochs, surveys, lr, reg, save_dir=RESULTSDIRECTORY):
    '''
    ARGUMENTS:
    ----------
        > save_dir: the directory where the results will be saved
        > model: the name of the model
        > epochs: the number of epochs for training
        > surveys: a list of survey names
        > lr: the learning rate
        > reg: the regularization parameter

    RETURNS:
    --------
        > folder_path: the path to the folder where the results will be
          saved

    DESCRIPTION:
    ------------
        This function constructs the expected folder path for saving results based on the model, epochs, surveys, learning rate, regularization parameter, and save directory specified in the configuration. The folder path is constructed in the format:
        "{save_dir}/{model}_{surveys_str}_{epochs}e_{lr}l_{reg}r/"
        where surveys_str is a concatenation of the survey names in the surveys list. The function returns the constructed folder path.
    '''
    model_str = ''.join(x[0] for x in model)

    surveys_str = ''.join(x[0] for x in surveys)

    return f"{save_dir.rstrip('/')}/model{model_str}_set{surveys_str}_{epochs}e_{lr}l_{reg}r/"

def check_if_experiments_were_interrupted_previously(save_dir, experiments):
    '''
    ARGUMENTS:
    ----------
        > save_dir: the directory where the results are saved
        > experiments: a list of experiments to be conducted, where each
          experiment is a tuple containing the model, isVis, clip, epochs,
          surveys, learning rate, and regularization parameter

    RETURNS:
    --------        
        > remaining_experiments: a list of experiments that have not been
          completed yet, based on the existence of corresponding folders in
          the save directory

    DESCRIPTION:
    ------------
        This function checks if any of the experiments in the provided list have been interrupted previously by checking for the existence of corresponding folders in the save directory. It iterates through the experiments and constructs the expected folder path for each experiment using the folder_construct function. If a folder exists, it assumes that the experiment has been completed and continues to the next one. If it encounters a folder that does not exist after having found existing folders, it assumes that the previous experiment was interrupted and deletes the folder of the last completed experiment to ensure a clean state. The function then returns the list of remaining experiments that have not been completed yet.
    '''

    counter = 0
    prev_model = ""
    prev_epochs = 0
    prev_surveys = []
    prev_lr = 0
    prev_reg = 0
    
    for epochs, surveys, lr, reg in experiments:  

        saved_dir = folder_construct(save_dir,  epochs, surveys, lr, reg)
        
        if os.path.exists(saved_dir):
            
            prev_epochs = epochs
            prev_surveys = surveys
            prev_lr = lr
            prev_reg = reg
            
            print("\nPath exists:", saved_dir)
            counter += 1

        elif counter > 0:
            print("\nPath does not exist:", saved_dir)
            saved_dir = folder_construct(save_dir, prev_epochs, prev_surveys, prev_lr, prev_reg)
            os.system(f"rm -r {saved_dir}")
            print("Deleted:", saved_dir)
            counter -= 1
            break

        else:break

    if counter == len(experiments):
        # If this point was reached, then the last experiment was not completed
        return experiments[-1:]

    return experiments[counter:]

# endregion