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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import shuffle
from astropy.io import fits
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import re
import shutil
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sys import argv
import gc
from keras import backend as K
from pathlib import Path
# endregion

# region PATHS
from paths import RESULTSDIRECTORY
# endregion

# region FUNCTIONS
def find_data(experiment_number):
    path = os.path.join(RESULTSDIRECTORY, f"Experiment_{experiment_number}", "Data")
    print(f"Finding data for Experiment {experiment_number} in {path}...")
    return path

def getFoldModels(models_root):
    """
    Returns:
        dict: {fold_number: model_path}
    """
    pattern = re.compile(r"^model_fold_([1-5])\.keras$")

    models = {}

    for file in Path(models_root).glob("*.keras"):
        match = pattern.match(file.name)
        if match:
            fold = int(match.group(1))
            models[fold] = str(file)

    return models

def get_dataset(kind, cf):
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
    channels = get_channels(cf.SURVEYS)
    print(f"Loading images for channel(s) {channels}, from survey(s) {cf.SURVEYS}...")

    DATA_SOURCES = find_data(cf.EXPERIMENT_NUMBER)

    # > float32 halves the memory of the arrays (and of the tensors copied to the GPU); Keras trains in float32 anyway
    X = np.load(f"{DATA_SOURCES}/X_{kind}.npy")[:, :, :, channels].astype(np.float32)

    y = np.load(f"{DATA_SOURCES}/y_{kind}.npy")

    names = np.load(f"{DATA_SOURCES}/names_{kind}.npy")

    return [
        X,
        y,
        names
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
        - RADCAT : combined channels 0, 1 and 2.
        - RADCAT_Rep : channel 0 repeated across all three channels (for single-channel models)
        - RADCAT_0s : channel 0 with channels 1 and 2 filled with 0s (for single-channel models)
        - RADCAT_WN : channel 0 with channels 1 and 2 filled with white noise (for single-channel models)
    '''

    '''
    CHANNELS:
    - 0 = FIRST
    - 1 = LOFAR
    - 2 = NVSS
    - 3 = Zeros
    - 4 = White Noise

    SELECTIONS:
    - FL: FIRST and LOFAR
    - LF: LOFAR and FIRST
    - FN: FIRST and NVSS
    - NF: NVSS and FIRST
    - LN: LOFAR and NVSS
    - NL: NVSS and LOFAR
    - FLN: FIRST, LOFAR and NVSS
    - FNL: FIRST, NVSS and LOFAR
    - NLF: NVSS, LOFAR and FIRST
    - NFL: NVSS, FIRST and LOFAR
    - LNF: LOFAR, NVSS and FIRST
    - LFN: LOFAR, FIRST and NVSS
    ABLATION SELECTIONS (one survey only, the other channels filled): <S><FILL><N>[_CH<k>]
    - S: F = FIRST, L = LOFAR, N = NVSS
    - FILL: REP = the survey repeated in every channel, ZER = zeros (channel 3), NSE = white noise (channel 4)
    - N: number of channels (2 or 3)
    - _CH<k>: the channel (1-based) that holds the survey; the remaining channels hold the fill
    e.g. FZER2_CH1 = [FIRST, zeros], LNSE2_CH2 = [noise, LOFAR], NZER3_CH3 = [zeros, zeros, NVSS]
    '''

    channel_map = {"FIRST": [0], "LoTSS": [1], "NVSS": [2], "ZEROS": [3], "WN": [4], 
                   # > Duo channel selections
                   "FL": [0, 1], 
                   "LF": [1, 0], 
                   "FN": [0, 2], 
                   "NF": [2, 0], 
                   "LN": [1, 2], 
                   "NL": [2, 1], 

                   # > Trio channel selections
                   "FLN": [0, 1, 2], 
                   "FNL": [0, 2, 1], 
                   "NLF": [2, 0, 1], 
                   "NFL": [2, 1, 0], 
                   "LNF": [1, 2, 0], 
                   "LFN": [1, 0, 2], 

                    # = TESTING FOR FIRST =
                    # > Testing for FIRST repeated
                   "FREP2": [0, 0], 
                   "FREP3": [0, 0, 0], 

                    # > Testing for FIRST with zeros
                   "FZER2_CH1": [0, 3], 
                   "FZER2_CH2": [3, 0], 
                   "FZER3_CH1": [0, 3, 3], 
                   "FZER3_CH2": [3, 0, 3], 
                   "FZER3_CH3": [3, 3, 0], 

                    # > Testing for FIRST with white noise
                   "FNSE2_CH1": [0, 4], 
                   "FNSE2_CH2": [4, 0], 
                   "FNSE3_CH1": [0, 4, 4], 
                   "FNSE3_CH2": [4, 0, 4], 
                   "FNSE3_CH3": [4, 4, 0], 

                    # = TESTING FOR LOFAR =
                    # > Testing for LOFAR repeated
                   "LREP2": [1, 1], 
                   "LREP3": [1, 1, 1], 

                   # > Testing for LOFAR with zeros
                   "LZER2_CH1": [1, 3], 
                   "LZER2_CH2": [3, 1], 
                   "LZER3_CH1": [1, 3, 3], 
                   "LZER3_CH2": [3, 1, 3], 
                   "LZER3_CH3": [3, 3, 1], 

                   # > Testing for LOFAR with white noise
                   "LNSE2_CH1": [1, 4], 
                   "LNSE2_CH2": [4, 1], 
                   "LNSE3_CH1": [1, 4, 4], 
                   "LNSE3_CH2": [4, 1, 4], 
                   "LNSE3_CH3": [4, 4, 1], 

                   # = TESTING FOR NVSS =
                   # > Testing for NVSS repeated
                   "NREP2": [2, 2], 
                   "NREP3": [2, 2, 2], 

                   # > Testing for NVSS with zeros
                   "NZER2_CH1": [2, 3], 
                   "NZER2_CH2": [3, 2], 
                   "NZER3_CH1": [2, 3, 3], 
                   "NZER3_CH2": [3, 2, 3], 
                   "NZER3_CH3": [3, 3, 2], 

                   # > Testing for NVSS with white noise
                   "NNSE2_CH1": [2, 4], 
                   "NNSE2_CH2": [4, 2], 
                   "NNSE3_CH1": [2, 4, 4], 
                   "NNSE3_CH2": [4, 2, 4], 
                   "NNSE3_CH3": [4, 4, 2], 
                   }

    print(f"Selected surveys: {surveys}")
    print(f"Corresponding channels: {channel_map[surveys]}")

    x = channel_map[surveys]

    return x

def get_folds(X, y, groups, val_split, random_state):
    '''
    ARGUMENTS:
    ----------
        > X: the input data (images) for which to create folds
        > y: the corresponding labels for the input data
        > groups: the source name of each sample, so that every augmented copy of a source lands in the same fold
        > val_split: the validation split ratio to determine the number of folds
        > random_state: the random seed for reproducibility when shuffling the data
    
    RETURNS:
    --------        
        > folds: a list of train and validation indices for each fold created using StratifiedKFold
    
    DESCRIPTION:
    ------------        
        This function uses StratifiedGroupKFold to create folds for cross-validation, ensuring that the class distribution is preserved in each fold and that all augmented copies of a source stay in the same fold (no train/validation leakage). The number of splits is determined by the validation split ratio (e.g., if val_split is 0.2, then n_splits will be 5). The function returns a list of train and validation indices for each fold created using StratifiedKFold.
    '''

    stratified_kfold = StratifiedGroupKFold(n_splits=int(1/val_split), random_state=random_state, shuffle=True)
    return [[t,v] for t,v in stratified_kfold.split(X, y, groups)]

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



def folder_construct(base_model, model, surveys, lr, base_reg, reg, wd, epoch_ceiling, save_dir=RESULTSDIRECTORY):
    '''
    ARGUMENTS:
    ----------
        > save_dir: the directory where the results will be saved
        > model: the name of the model
        > epochs: the number of epochs for training
        > surveys: a list of survey names
        > lr: the learning rate
        > reg: the regularization parameter
        > wd: the weight decay parameter

    RETURNS:
    --------
        > folder_path: the path to the folder where the results will be
          saved

    DESCRIPTION:
    ------------
        This function constructs the expected folder path for saving results based on the model, epochs, surveys, learning rate, regularization parameter, and save directory specified in the configuration. The folder path is constructed in the format:
        "{save_dir}/base{base_model}_target{model}_dataset{surveys}_{lr}l_base{base_reg}r_{reg}r_ceiling{epoch_ceiling}e/"
        where surveys_str is a concatenation of the survey names in the surveys list. The function returns the constructed folder path.
    '''

    # return f"{save_dir.rstrip('/')}/base{base_model}_model{model}_set{surveys}_base{base_epochs}e_{epochs}e_{lr}l_base{base_reg}r_{reg}r/"

    return f"{save_dir.rstrip('/')}/base{base_model}_target{model}_dataset{surveys}_{lr}l_base{base_reg}r_{reg}r_ceiling{epoch_ceiling}e_wd{wd}/"

def clearTransferLearningFolders(folders):

    filtered_folders = []

    for folder in folders:
        if 'NONE' in folder.name:
            filtered_folders.append(folder)
        else:
            print(
                f"Removing folder '{folder.name}'."
            )
            shutil.rmtree(folder)

    return filtered_folders

def extract_parameters_from_folder_name(folder_name):
    '''
    ARGUMENTS:
    ----------
        > folder_name: the name of the folder from which to extract parameters

    RETURNS:
    --------
        > parameters: a dictionary containing the extracted parameters (model,
          surveys, epochs, learning rate, regularization parameter)

    DESCRIPTION:
    ------------
        This function extracts the model, surveys, epochs, learning rate, and regularization parameter from the given folder name using regular expressions. The expected format of the folder name is:
        "base{base_model}_model{model}_set{surveys}_base{base_epochs}e_{epochs}e_{lr}l_base{base_reg}r_{reg}r/"
        The function returns a dictionary containing the extracted parameters.
    '''

    # pattern = re.compile(
    #     r'base(\w+)_model(\w+)_set(\w+)_base(\d+)e_(\d+)e_([\d.]+(?:e-?\d+)?)l_base([\d.]+)r_([\d.]+)r'
    # )
    
    pattern = re.compile(
        r'base(\w+)_target(\w+)_dataset(\w+)_([\d.]+(?:e-?\d+)?)l_base([\d.]+(?:e-?\d+)?)r_([\d.]+(?:e-?\d+)?)r_ceiling(\d+)e_wd([\d.]+(?:e-?\d+)?)$'
    )    
    
    match = pattern.match(folder_name)
    if not match:
        raise ValueError(f"Folder name '{folder_name}' does not match the expected format.")
    
    parameters = {
        "base_model": match.group(1),
        "model": match.group(2),
        "surveys": match.group(3),
        "lr": float(match.group(4)),          # Python float() natively parses '3.9e-05' perfectly
        "base_reg": float(match.group(5)),
        "reg": float(match.group(6)),
        "epoch_ceiling": int(match.group(7)),
        "wd": float(match.group(8))
    }

    # | MODEL = model, 
    # | SURVEYS = dataset,
    # | LEARNING_RATE = 0.00039, 
    # | REGULARISATION = reg, 
    # | IGNORE_WARNINGS = config["IGNORE_WARNINGS"],
    # | SAVE_XFOLD_DATA= config["SAVE_XFOLD_DATA"],
    # | VAL_SPLIT = config["VAL_SPLIT"],
    # | SEED =config["SEED"],
    # | SAVE_DIR = config["SAVE_DIR"],
    # | TESTING = True, 
    # | DATASET = dataset, 
    # | EXPERIMENT_NUMBER = experiment_number,
    # | MODEL_DIR = os.path.join(RESULTSDIRECTORY, "Training/"), 
    # | EPOCHS = epoch,
    # | BASE_MODEL = base_model,
    # | BASE_EPOCH = base_epoch,
    
    return parameters

# endregion