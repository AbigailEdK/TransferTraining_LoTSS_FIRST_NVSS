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

    X = np.load(f"{DATA_SOURCES}/X_{kind}.npy")[:, :, :, channels]

    y = np.load(f"{DATA_SOURCES}/y_{kind}.npy")

    names = np.load(f"{DATA_SOURCES}/names_{kind}.npy")

    return [
        X,
        y,
        names
    ]

def _get_available_base_epochs(base_model, training_dir):
    pattern = re.compile(rf'baseNONE_model{base_model}_set{base_model}_base(\d+)e_(\d+)e_.*')
    epochs = []
    try:
        for folder in Path(training_dir).iterdir():
            if folder.is_dir():
                m = pattern.match(folder.name)
                if m and m.group(1) == m.group(2):
                    epochs.append(int(m.group(1)))
    except FileNotFoundError:
        print(f"Warning: Training directory {training_dir} not found.")
    return sorted(set(epochs))

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
    - FREP = FIRST repeated across all three channels (for single-channel models) {FREP}
    - FZER = FIRST with channels 1 and 2 filled with zeros {FZER}
    - FNSE = FIRST with channels 1 and 2 filled with white noise {FNSE}
    - LREP = LOFAR repeated across all three channels (for single-channel models) {LREP}
    - LZER = LOFAR with channels 0 and 2 filled with zeros {LZER}
    - LNSE = LOFAR with channels 0 and 2 filled with white noise {LNSE}
    - NREP = NVSS repeated across all three channels (for single-channel models) {NREP}
    - NZER = NVSS with channels 0 and 1 filled with zeros {NZER}
    - NNSE = NVSS with channels 0 and 1 filled with white noise {NNSE}
    - RADCAT = RADCAT (combined channels 0, 1 and 2) {RADCAT}
    '''

    channel_map = {"FIRST": [0], "LOFAR": [1], "NVSS": [2], "ZEROS": [3], "WN": [4], "FREP": [0, 0, 0], "FZER": [0, 3, 3], "FNSE": [0, 4, 4], "LREP": [1, 1, 1], "LZER": [3, 1, 3], "LNSE": [4, 1, 4], "NREP": [2, 2, 2], "NZER": [3, 3, 2], "NNSE": [4, 4, 2], "RADCAT": [0, 1, 2]}

    print(f"Selected surveys: {surveys}")
    print(f"Corresponding channels: {channel_map[surveys]}")

    x = channel_map[surveys]

    return x

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
    surveys_str = ''.join(x[:4] for x in cf.MODEL)
    epoch_str = f"{cf.EPOCHS}e"

    return cf.MODEL_DIR + "model" + model_str + "_set" + surveys_str + "_" + epoch_str + "_" + str(cf.LEARNING_RATE) + "l_" + str(cf.REGULARISATION) + "r/" 

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



def folder_construct(base_model, model, surveys, lr, base_reg, reg, epoch_ceiling, save_dir=RESULTSDIRECTORY):
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

    # return f"{save_dir.rstrip('/')}/base{base_model}_model{model}_set{surveys}_base{base_epochs}e_{epochs}e_{lr}l_base{base_reg}r_{reg}r/"

    return f"{save_dir.rstrip('/')}/base{base_model}_target{model}_dataset{surveys}_{lr}l_base{base_reg}r_{reg}r_ceiling{epoch_ceiling}e/"

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

    #  e.g. baseNONE_modelRADCAT_setRADCAT_base25e_25e_0.00039l_base0.54r_0.54r
    # * e.g. baseNONE_modelRADCAT_setRADCAT_0.00039l_base0.54r_0.54r

    # pattern = re.compile(
    #     r'base(\w+)_model(\w+)_set(\w+)_base(\d+)e_(\d+)e_([\d.]+(?:e-?\d+)?)l_base([\d.]+)r_([\d.]+)r'
    # )
    
    pattern = re.compile(
        r'base(\w+)_target(\w+)_dataset(\w+)_([\d.]+(?:e-?\d+)?)l_base([\d.]+)r_([\d.]+)r_ceiling(\d+)e'
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
        "epoch_ceiling": int(match.group(7))
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

def restructure_results_by_epoch(results_dir):
    """
    Creates separate top-level experiment folders for each epoch threshold.
    Moves model files to their respective epoch folders, and sweeps all other 
    remaining files (logs, plots, metadata) into the maximum epoch directory.
    """
    base_path = Path(results_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory {results_dir} does not exist.")

    # Matches: (prefix)_base(digits)e_(digits)e_(suffix)
    exp_dir_pattern = re.compile(r'^(.*?)_base\d+e_\d+e_([^\/]+)$')
    checkpoint_pattern = re.compile(r'model_(\d+)e_fold_(\d+)\.keras')

    for exp_dir in list(base_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        match = exp_dir_pattern.match(exp_dir.name)
        if not match:
            continue
        
        dir_prefix = match.group(1)   # e.g., "baseCNN_modelV1_setLOFAR"
        dir_suffix = match.group(2)   # e.g., "0.001l_1e-4r"
        
        print(f"\nParsing parent experiment: {exp_dir.name}")

        # 1. Map model checkpoints and discover what epochs exist
        checkpoint_files = list(exp_dir.glob("model_*e_fold_*.keras"))
        found_epochs = sorted(list(set(
            int(checkpoint_pattern.match(f.name).group(1)) 
            for f in checkpoint_files if checkpoint_pattern.match(f.name)
        )))

        if not found_epochs:
            print(f"-> No valid model checkpoints found in {exp_dir.name}. Skipping.")
            continue

        max_epoch = max(found_epochs)
        
        # Determine the target directory path for the maximum epoch run
        max_epoch_dir = base_path / f"{dir_prefix}_base{max_epoch}e_{max_epoch}e_{dir_suffix}"

        # 2. PROCESS AND MOVE CHECKPOINTS
        for file in checkpoint_files:
            file_match = checkpoint_pattern.match(file.name)
            if not file_match:
                continue
                
            epoch = int(file_match.group(1))
            fold = file_match.group(2)
            
            new_dir_name = f"{dir_prefix}_base{epoch}e_{epoch}e_{dir_suffix}"
            new_exp_dir = base_path / new_dir_name
            new_exp_dir.mkdir(parents=True, exist_ok=True)
            
            dest_file = new_exp_dir / f"model_fold_{fold}.keras"
            file.rename(dest_file)

        # 3. SWEEP ALL REMAINING FILES INTO THE MAX EPOCH DIRECTORY
        # Ensure the max epoch directory is instantiated
        max_epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect everything left over (non-checkpoint files, logs, metadata, plots)
        remaining_files = [f for f in exp_dir.iterdir() if f.is_file()]
        
        for file in remaining_files:
            # If the source folder isn't already the max epoch folder, migrate the file
            if exp_dir != max_epoch_dir:
                file.rename(max_epoch_dir / file.name)

        # 4. TEARDOWN SOURCE FOLDER IF EMPTY
        if exp_dir != max_epoch_dir:
            if not any(exp_dir.iterdir()):
                exp_dir.rmdir()
                print(f"Cleaned up temporary source directory: {exp_dir.name}")
            else:
                print(f"Warning: Source directory {exp_dir.name} contains unexpected subdirectories and was not removed.")

    print("\nRestructuring execution complete.\n" + "="*80)

def _safe_load_model(model_path):
    '''
    ARGUMENTS:
    ----------
        > model_path: Path to the .keras model file to load.

    RETURN:
    -------
        > model: The loaded Keras model.

    DESCRIPTION:
    ------------
        Safely loads a Keras model that may have been saved with older versions of Keras/TensorFlow
        that used deprecated optimizer configuration parameters (jit_compile, is_legacy_optimizer).
        Loads without compiling first, then recompiles with a fresh Adam optimizer.
    '''
    import json
    import zipfile
    import tempfile
    import shutil
    
    try:
        # First attempt: try loading normally
        return tf.keras.models.load_model(model_path)
    except ValueError as e:
        if "Argument(s) not recognized" in str(e) and ("jit_compile" in str(e) or "is_legacy_optimizer" in str(e)):
            print(f"Detected deprecated optimizer parameters. Loading model without compilation...")
            
            # Extract and modify the model.json to remove problematic optimizer configs
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Modify the model config to remove optimizer compilation config
                config_path = os.path.join(tmpdir, 'model.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Remove the entire optimizer_config from compile_config to force fresh compilation
                    if 'config' in config and 'compile_config' in config['config']:
                        config['config']['compile_config'] = None
                    
                    with open(config_path, 'w') as f:
                        json.dump(config, f)
                
                # Re-create the zip file
                temp_model_path = os.path.join(tmpdir, 'model_fixed.keras')
                with zipfile.ZipFile(temp_model_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(tmpdir):
                        for file in files:
                            if file != 'model_fixed.keras':
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, tmpdir)
                                zipf.write(file_path, arcname)
                
                # Load model without compiling
                model = tf.keras.models.load_model(temp_model_path, compile=False)
                
                # Recompile with fresh optimizer
                model.compile(
                    optimizer=Adam(learning_rate=0.00039),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                print(f"Successfully loaded and recompiled model.")
                return model
        else:
            raise

# endregion