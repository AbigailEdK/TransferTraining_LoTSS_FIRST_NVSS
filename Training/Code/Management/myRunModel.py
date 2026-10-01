# region ABOUT
# ===================================================================================================
# > This script trains and tests multiple models on different surveys with varying hyperparameters, based on the experiments list passed from myManager.py. It loads the training and test datasets, prepares the data for training, and runs the training experiment for each set of hyperparameters and surveys specified in the experiments list. The results are saved in a structured format for later analysis.

# > Adjusted from run_model.py by Dylan Farge.
# ===================================================================================================
# endregion

import os

import tensorflow as tensorflow
# Add these lines right here:
gpus = tensorflow.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(f"Memory growth error: {e}")
        
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no info, 2=no warnings, 3=no errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Stops the numerical result warnings

# Find the home directory dynamically
home_dir = os.path.expanduser("~")
# Construct the project path
project_root = os.path.join(home_dir, "DeKlerk_Models")

import sys
# Add the project root to sys.path so it can find 'myProcess' and 'myModels'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # If they are inside a 'Code' subfolder, add that too:
    sys.path.insert(0, os.path.join(project_root, "Code"))

# region IMPORTS
import numpy as np
import myProcess as process
from tensorflow.keras.optimizers import Adam # type: ignore
from myModels import get_model
from sys import argv
import ast
import gc
import Code.Management.resultsDisplayOld as resultsDisplayOld
# endregion

# region PATHS
RESULTSDIRECTORY = os.path.join(project_root, "Results")
# endregion

# region FUNCTIONS
def _validate_directory(cf):
    '''
    ARGUMENTS:
    ----------
        > cf: Config object containing the SAVE_DIR attribute and
          IGNORE_WARNINGS flag.

    RETURN:
    -------
        > None. This function performs checks and creates directories as needed.

    DESCRIPTION:
    ------------
        Checks if the directory specified in cf.SAVE_DIR already exists and contains files. If it does, it prompts the user to confirm whether they want to overwrite the existing directory. If the user chooses not to overwrite, the program exits. If the directory does not exist, it creates the directory.
    '''
    if os.path.exists(cf.SAVE_DIR) and len(os.listdir(cf.SAVE_DIR)) > 0:
        
        # 1. If we are on the cluster (IGNORE_WARNINGS = True), just keep going
        if cf.IGNORE_WARNINGS:
            print(f"Directory {cf.SAVE_DIR} exists. IGNORE_WARNINGS is True, so overwriting...")
            return  # This exits the function and lets the script continue

        # 2. If we are running locally, ask the user
        while True:
            response = input(f"Directory {cf.SAVE_DIR} already exists. Do you want to overwrite it? (y/n): ")

            if response.lower() == "y":
                print("Proceeding with overwrite...")
                break # This breaks the 'while' loop and exits the function naturally

            elif response.lower() == "n":
                print("Operation cancelled by user.")
                exit()

            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    elif not os.path.exists(cf.SAVE_DIR):
        os.makedirs(cf.SAVE_DIR)

def _format_config(cf):
    '''
    ARGUMENTS:
    ----------
        > cf: Config object containing the SAVE_DIR attribute and
          IGNORE_WARNINGS flag.

    RETURN:
    -------
        > None. This function formats the configuration parameters.

    DESCRIPTION:
    ------------
        Formats the SAVE_DIR attribute by constructing the directory path based on the model and training parameters.
    '''
    print(f"Formatting save directory based on configuration...")

    if cf.TESTING == False:
        cf.SAVE_DIR = process.folder_construct(model=cf.SURVEYS, epochs=cf.EPOCHS, surveys=cf.SURVEYS, lr=cf.LEARNING_RATE, reg=cf.REGULARISATION, save_dir=cf.SAVE_DIR + "/Training/")
    else:
        cf.SAVE_DIR = process.folder_construct(model=cf.MODEL, epochs=20, surveys=[cf.DATASET], lr=cf.LEARNING_RATE, reg=cf.REGULARISATION, save_dir=cf.SAVE_DIR + "/Testing/")

    print(f"Formatted directory: {cf.SAVE_DIR}")
    

def print_distribution(dataset, dataset_name, total_images=None, spacing=20):
    '''
    ARGUMENTS:
    ----------
        > dataset: The dataset for which to print distribution.
        > dataset_name: The name of the dataset.
        > total_images: The total number of images in the dataset.
        > spacing: The spacing for formatting the output.

    RETURN:
    -------
        > None. This function prints the distribution of the dataset.

    DESCRIPTION:
    ------------
        Prints the distribution of the dataset with morphology, count, and percentage information.
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

def print_train_val_distribution(X_train, y_train, X_val, y_val, spacing=15, name='Train and Validation Data Morphology Distribution'):
    '''
    ARGUMENTS:
    ----------
        > X_train: The training data.
        > y_train: The training labels.
        > X_val: The validation data.
        > y_val: The validation labels.
        > spacing: The spacing for formatting the output.
        > name: The name of the distribution.

    RETURN:
    -------
        > None. This function prints the distribution of the training and validation data.

    DESCRIPTION:
    ------------
        Prints the distribution of the training and validation data with morphology, count, and percentage information.
    '''

    # Train and Validation Final Morphology Distribution
    print(f"\n{'-'*spacing*5}----")
    print(f"{name:^{spacing*5}}")
    print(f"{'-'*spacing*5}----")

    unique, counts_train = np.unique(y_train, return_counts=True)
    unique, counts_val = np.unique(y_val, return_counts=True)

    print(f"{'Morphology':^{spacing}}|{'Count_Train':^{spacing}}|{'Perc_of_Train':^{spacing}}|", end='')
    print(f"{'Count_Val':^{spacing}}|{'Perc_of_Val':^{spacing}}")
    print(f"{'-'*spacing*5}----")

    for i in range(len(unique)):
        print(f"{unique[i]:^{spacing}}|{counts_train[i]:^{spacing}}|{f'{counts_train[i]/len(y_train)*100:.2f}%':^{spacing}}|", end='')
        print(f"{counts_val[i]:^{spacing}}|{f'{counts_val[i]/len(y_val)*100:.2f}%':^{spacing}}")

    print(f"{'-'*spacing*5}----")
    print(f"{'Totals':^{spacing}}|{len(y_train):^{spacing}}|{'100.00%':^{spacing}}|", end='')
    print(f"{len(y_val):^{spacing}}|{'100.00%':^{spacing}}")

    print(f"{'-'*spacing*5}----")
    print(f"{f'TOTAL TRAINVAL IMAGES: {len(X_train)+len(X_val)}':^{spacing*5}}")
    print(f"{'-'*spacing*5}----")

def run_experiment(X_train_val, y_train_val, X_test, names_train_val, folds, cf):
    '''
    ARGUMENTS:
    ----------
        > X_train_val: The combined training and validation data.
        > y_train_val: The combined training and validation labels.
        > X_test: The test data.
        > names_train_val: The names of the training and validation
          samples.
        > folds: The cross-validation folds.
        > cf: The configuration object containing training parameters.
   
    RETURN:
    -------
        > None. This function runs the training experiment.

    DESCRIPTION:
    ------------
        Runs the training experiment by iterating through the cross-validation folds, training the model, and saving the results for each fold.
    '''

    losses = []
    
    for kfold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n>>>>>Training Fold {kfold+1}<<<<<<")

        if np.isnan(X_train_val).any():
            raise ValueError("NaN values found in training/validation data")

        X_train, y_train, names_train = process.build_dataset(X_train_val, y_train_val, names_train_val, train_idx, cf, size_per_type= 3000 - int(3000 * cf.VAL_SPLIT))

        X_val, y_val, names_val = process.build_dataset(X_train_val, y_train_val, names_train_val,  val_idx, cf, size_per_type=int(3000 * cf.VAL_SPLIT))

        print_train_val_distribution(X_train, y_train, X_val, y_val)

        print("Creating New Model...")
        
        # * Get the model architecture based on the configuration, with input shape and number of classes determined by the training data.
        model = get_model(cf,input_shape=X_train.shape[1:],num_classes=len(np.unique(y_train_val)))

        # * Compile the model with the Adam optimizer, using the learning rate from the configuration, and set the loss function to categorical crossentropy with accuracy and F1 score as metrics.
        model.compile(
            optimizer=Adam(learning_rate=cf.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=["accuracy"]  
        )

        print("Training Model...")

        # * Encode the training and validation labels using the lookup dictionary from the configuration, converting them to one-hot encoded format for training.
        encoded_y_train = np.array([cf.LOOKUP[x] for x in y_train])
        encoded_y_val = np.array([cf.LOOKUP[x] for x in y_val])

        # * Train the model on the training data, validating on the validation set, for the number of epochs specified in the configuration, and store the training history.
        loss = model.fit(
                    X_train, encoded_y_train,
                    validation_data=(X_val, encoded_y_val),
                    epochs=cf.EPOCHS,
                    batch_size=4,
                ).history

        losses.append(loss)

        print(f"Finished Training Fold {kfold+1}.")

        # * After training, predict on the test set and save the predicted probabilities for the test set, training set, and validation set to .npy files in the save directory specified in the configuration. Also save the training and validation data if specified in the configuration.
        with tensorflow.device('/cpu:0'): encoded_y_pred_test = model.predict(X_test, batch_size=4)        
        
        np.save(f"{cf.SAVE_DIR}/y_pred_test_{kfold+1}", encoded_y_pred_test)
        
        # * Predict on the training set and save the predictions and training data if specified.
        with tensorflow.device('/cpu:0'): encoded_y_pred_train = model.predict(X_train, batch_size=4)
        np.save(f"{cf.SAVE_DIR}/y_pred_train_{kfold+1}", encoded_y_pred_train)
        np.save(f"{cf.SAVE_DIR}/X_train_fold_{kfold+1}", X_train) if cf.SAVE_XFOLD_DATA else None
        del X_train

        print(f"Finished Predicting on Test Set for Fold {kfold+1}, now saving validation predictions and data...")

        # * Predict on the validation set and save the predictions and validation data if specified.
        with tensorflow.device('/cpu:0'): encoded_y_pred_val = model.predict(X_val, batch_size=4)
        np.save(f"{cf.SAVE_DIR}/y_pred_val_{kfold+1}", encoded_y_pred_val)
        np.save(f"{cf.SAVE_DIR}/X_val_fold_{kfold+1}", X_val) if cf.SAVE_XFOLD_DATA else None
        del X_val

        print(f"Finished Predicting on Validation Set for Fold {kfold+1}, now saving training history and labels...")

        # * Save the training history, training labels, validation labels, and names of the training and validation samples to .npy files in the save directory. Also save the trained model in .keras format if specified in the configuration.
        np.save(f"{cf.SAVE_DIR}/loss_fold_{kfold+1}", loss)
        np.save(f"{cf.SAVE_DIR}/y_train_fold_{kfold+1}", y_train)
        np.save(f"{cf.SAVE_DIR}/y_val_fold_{kfold+1}", y_val)
        np.save(f"{cf.SAVE_DIR}/names_train_fold_{kfold+1}", names_train)
        np.save(f"{cf.SAVE_DIR}/names_val_fold_{kfold+1}", names_val)

        print(f"Finished Saving Training History and Labels for Fold {kfold+1}, now saving model to {cf.SAVE_DIR} and freeing up resources...")
        
        # * Save the trained model in .keras format if specified in the configuration, and then delete the model from memory to free up resources before the next fold.
        model.save(f"{cf.SAVE_DIR}/model_fold_{kfold+1}.keras") if cf.SAVE_XFOLD_DATA else None
        
        print(f"Finished Saving Model for Fold {kfold+1}, now freeing up resources...")

        # 1. DELETE the model and large prediction arrays FIRST
        del model
        del encoded_y_pred_test, encoded_y_pred_train, encoded_y_pred_val
        del encoded_y_train, encoded_y_val
        
        # 2. THEN clear the backend and reset the graph
        tensorflow.keras.backend.clear_session()
        from tensorflow.python.framework import ops
        ops.reset_default_graph()
        
        # 3. FINALLY force garbage collection
        gc.collect()

        print(f"Finished Fold {kfold+1}. Moving on to the next fold...\n" + "="*80)

def experiment(**kwargs):
    '''
    ARGUMENTS:
    ----------
        > kwargs: Keyword arguments containing the configuration parameters for the experiment.
    
    RETURN:
    -------
        > None. This function runs the experiment with the given configuration.
    
    DESCRIPTION:
    ------------
        This function initializes the configuration from the provided keyword arguments, sets the random seed for reproducibility, formats the configuration, validates the save directory, loads the training/validation and test datasets, prints the distribution of the datasets, prepares the cross-validation folds, and runs the training experiment for each fold.
    '''
    
    cf = Config(kwargs)

    # * Set the random seed for reproducibility based on the configuration.
    np.random.seed(cf.SEED)

    _format_config(cf)
    _validate_directory(cf)

    X_train_val, y_train_val, names_train_val = process.get_dataset("train_val", cf)
    X_test, y_test, names_test = process.get_dataset("test", cf)  

    print_distribution(y_train_val, "Training/Validation")
    print_distribution(y_test, "Test")

    folds = process.get_folds(X_train_val, y_train_val, cf.VAL_SPLIT, cf.SEED)

    for i, (train_idx, val_idx) in enumerate(folds):
        print_train_val_distribution(X_train_val[train_idx], y_train_val[train_idx], X_train_val[val_idx], y_train_val[val_idx], name=f"Fold {i+1}")

    cf.LOOKUP = {
        "FRI": [1, 0, 0],
        "FRII": [0, 1, 0],
        "COMPACT": [0, 0, 1],
    }

    np.save(f"{cf.SAVE_DIR}/lookup", cf.LOOKUP)
    
    run_experiment(X_train_val, y_train_val, X_test, names_train_val, folds, cf)

    del X_train_val, y_train_val, X_test, names_train_val, names_test
    gc.collect()

def run_test(**kwargs):
    '''
    ARGUMENTS:
    ----------
        > kwargs: Keyword arguments from Config containing SURVEYS, EPOCHS, LEARNING_RATE, REGULARISATION, etc.

    RETURN:
    -------
        > None. This function runs the test for the specified model and dataset.

    DESCRIPTION:
    ------------
        This function loads the test dataset, loads all trained models from each fold, makes predictions with each model, and averages the predictions to create an ensemble result. The averaged predictions are saved to .npy files in a structured format for later analysis.
    '''

    cf = Config(kwargs)

    # * Format the configuration to construct the correct save directory path for the trained models.

    # * Load the test dataset based on the specified dataset name.
    print(f"Loading test dataset for {cf.DATASET}...")

    X_test, y_test, names_test = process.get_dataset("test", cf)  

    # * Collect predictions from all folds
    print("Loading fold models and collecting predictions...")

    fold_predictions = []
    fold = 1

    models_root = process.get_model_path(cf)
    print(f"Looking for models in {models_root}...")

    while os.path.exists(models_root + "model_fold_" + str(fold) + ".keras"):

        print(f"Found model for fold {fold}, in {models_root}. Loading model and making predictions...")

        print(f" >>>>> Loading fold {fold} model ({models_root}_model_fold_{fold}.keras) and making predictions... <<<<< ")

        model = tensorflow.keras.models.load_model(models_root + "model_fold_" + str(fold) + ".keras")

        # * Make predictions with this fold model
        with tensorflow.device('/cpu:0'): 
            y_pred_fold = model.predict(X_test, batch_size=4)
        
        fold_predictions.append(y_pred_fold)
        
        # * Clean up this fold model
        del model
        tensorflow.keras.backend.clear_session()
        gc.collect()
        
        fold += 1
    
    if not fold_predictions:
        raise ValueError(f"No trained models found in {cf.MODEL_DIR}")
    
    print(f"Found {len(fold_predictions)} fold predictions. Averaging predictions...")

    # * Average predictions across all folds
    averaged_y_pred_test = np.mean(fold_predictions, axis=0)
    
    _format_config(cf)
    _validate_directory(cf)

    # * Save the averaged predictions and test data
    np.save(f"{cf.SAVE_DIR}y_pred_test_ensemble", averaged_y_pred_test)
    np.save(f"{cf.SAVE_DIR}y_test", y_test)
    np.save(f"{cf.SAVE_DIR}names_test", names_test)
    
    print(f"Saved ensemble predictions to {cf.SAVE_DIR}/y_pred_test_ensemble.npy")
    
    # * Clean up
    del fold_predictions, averaged_y_pred_test
    gc.collect()

# endregion 

# region CONFIG CLASS
class Config:
    def __init__(self, kwargs):
        # make keys properties
        self.__dict__.update(kwargs)
# endregion

# region MAIN
def main():
    '''
    ARGUMENTS:
    ----------
        > None.

    RETURN:
    -------
        > None. This function runs the main experiment.

    DESCRIPTION:
    ------------
        This function initializes the results directory, sets up the configuration, and runs the main experiment. It checks for command-line arguments to determine the experiments to run, and if not provided, it raises an error prompting the user to run the manager script.
    '''
    
    testing = True # | If TRUE, tests already trained models on each other's datasets. Set to FALSE to run training experiments.
    
    config = dict(
        IGNORE_WARNINGS = True,
        SAVE_XFOLD_DATA=True,
        VAL_SPLIT = 0.2,
        SEED = 100,
        SAVE_DIR = RESULTSDIRECTORY,
    )

    if testing == False:
        print("Testing mode is OFF. Training will be performed.") 
    

        if len(argv) > 1:
            experiments = ast.literal_eval(argv[1])
            experiments = process.check_if_experiments_were_interrupted_previously(config["SAVE_DIR"], experiments)
        else:
            raise ValueError("Please run manager.py to start experiments.")
        
        # - experiments:
        #   - ((20,     survey,  lr,      reg))
        #   - ((epochs, "FIRST", 0.00039, 0.40))  
        #   - ((epochs, "NVSS",  0.00039, 0.26))  
        #   - ((epochs, "LOFAR", 0.00039, 0.52))  
    
        for i, (epochs, surveys, lr, reg) in enumerate(experiments):
            
            print("\n" + "="*80)
            print(f"\nRunning experiment {i+1}/{len(experiments)} ({surveys} dataset) with learning rate {lr} and regularisation {reg}")

            print(surveys)

            experiment(
                EPOCHS = epochs, # 20
                LEARNING_RATE = lr,
                REGULARISATION = reg,
                SURVEYS = surveys, # ["FIRST"], ["NVSS"], or ["LOFAR"]
                IGNORE_WARNINGS = config["IGNORE_WARNINGS"],
                SAVE_XFOLD_DATA= config["SAVE_XFOLD_DATA"],
                VAL_SPLIT = config["VAL_SPLIT"],
                SEED =config["SEED"],
                SAVE_DIR = config["SAVE_DIR"],
                TESTING = False, # Tells the script to look in the Training subfolder and format the SAVE_DIR accordingly
            )

            print(f"Finished experiment {i+1}/{len(experiments)} ({surveys} dataset) with learning rate {lr} and regularisation {reg}\n" + "="*80)

        testing = True
    
    if testing == True:
        print("Testing mode is ON. No training will be performed. Models will be tested on each other's datasets.")

        models = ["F", "N", "L"] 
        datasets = ["FIRST", "NVSS", "LOFAR"]


        for model in models:
            for dataset in datasets:
                print(f"\n>>>>> Testing model {model} on dataset {dataset} <<<<<\n")

                if model == "F":
                    reg = 0.40
                elif model == "N":
                    reg = 0.26
                elif model == "L":
                    reg = 0.52
                else:
                    raise ValueError("Invalid model specified. Please choose from 'F', 'N', or 'L'.")
                
                run_test(
                    MODEL = model, # Tells which model to load for testing
                    SURVEYS = [dataset], # Tells which channel to retrieve for data
                    LEARNING_RATE = 0.00039, # Based on previous results
                    REGULARISATION = reg, # Based on previous results
                    IGNORE_WARNINGS = config["IGNORE_WARNINGS"],
                    SAVE_XFOLD_DATA= config["SAVE_XFOLD_DATA"],
                    VAL_SPLIT = config["VAL_SPLIT"],
                    SEED =config["SEED"],
                    SAVE_DIR = config["SAVE_DIR"],
                    TESTING = True, 
                    DATASET = dataset, # Tells the script to look in the Testing subfolder and format the SAVE_DIR accordingly
                    MODEL_DIR = os.path.join(RESULTSDIRECTORY, "Training/"), # Tells the script where to look for the trained models
                )

                print(f"\n>>>>> Finished testing model {model} on dataset {dataset} <<<<<\n" + "="*80)

    resultsDisplayOld.main()
            
if __name__ == "__main__": main()  
# endregion