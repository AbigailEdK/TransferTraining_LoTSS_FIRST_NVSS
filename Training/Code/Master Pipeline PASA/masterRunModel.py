# region ABOUT
# ===================================================================================================
# > This script trains and tests multiple models on different surveys with varying hyperparameters, based on the experiments list passed from masterManager.py. It loads the training and test datasets, prepares the data for training, and runs the training experiment for each set of hyperparameters and surveys specified in the experiments list. The results are saved in a structured format for later analysis.

# > Adjusted from run_model.py by Dylan Farge.
# ===================================================================================================
# endregion

import os
import tensorflow as tensorflow
from tensorflow.python.framework import ops
from time import perf_counter

gpus = tensorflow.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
        # print("GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(f"Memory growth error: {e}")
        
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no info, 2=no warnings, 3=no errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Stops the numerical result warnings

# region IMPORTS
import numpy as np
import masterProcess as process
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
import re
from masterModels import get_model, load_model_from_base
from sys import argv
import gc
from keras import backend as K
from pathlib import Path
from masterUtils import print_training_parameters, print_testing_parameters, print_transfer_learning_parameters, print_distribution, print_train_val_distribution, _format_config, _validate_directory
# endregion

# region PATHS
from paths import HOME_DIR, PROJECT_ROOT, RESULTSDIRECTORY
# endregion

# region CONSTANTS
from constants import BATCH_SIZE, SIZE_PER_TYPE
# endregion

# region BOOLS
from bools import TESTING, TRANSFER_LEARNING
# endregion

# region FUNCTIONS
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
    # print(f"X_train_val: {X_train_val.shape}, y_train_val: {y_train_val.shape}, names: {names_train_val.shape}")

    X_test, y_test, names_test = process.get_dataset("test", cf)
    # print(f"X_test: {X_test.shape}, y_test: {y_test.shape}, names: {names_test.shape}")
  
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
    K.clear_session()
    gc.collect()

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
        Runs the training experiment by iterating through the cross-validation folds, 
        training the model, and saving checkpoints and metadata for each fold.
    '''

    losses = []
    best_epochs = []
    
    for kfold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n>>>>>Training Fold {kfold+1}<<<<<<\n\n")

        if np.isnan(X_train_val).any():
            raise ValueError("NaN values found in training/validation data")

        X_train, y_train, names_train = process.build_dataset(X_train_val, y_train_val, names_train_val, train_idx, cf, size_per_type = (SIZE_PER_TYPE - int(SIZE_PER_TYPE * cf.VAL_SPLIT)))
        X_val, y_val, names_val = process.build_dataset(X_train_val, y_train_val, names_train_val,  val_idx, cf, size_per_type=int(SIZE_PER_TYPE * cf.VAL_SPLIT))

        print_train_val_distribution(X_train, y_train, X_val, y_val)
        
        # Get the model architecture based on the configuration
        model = get_model(cf, input_shape=X_train.shape[1:], num_classes=len(np.unique(y_train_val)))

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=cf.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=["accuracy"]  
        )

        print("Training Model...")

        # Encode labels using the configuration lookup dictionary
        encoded_y_train = np.array([cf.LOOKUP[x] for x in y_train])
        encoded_y_val = np.array([cf.LOOKUP[x] for x in y_val])

        # # Initialize the custom checkpointing callback
        # epoch_checkpoint = SpecificEpochCheckpoint(
        #     target_epochs=cf.CHECKPOINT_EPOCHS, 
        #     save_dir=cf.SAVE_DIR, 
        #     fold=kfold+1
        # )

        # # Train the model and store the training history
        # loss = model.fit(
        #             X_train, encoded_y_train,
        #             validation_data=(X_val, encoded_y_val),
        #             epochs=cf.EPOCH_CEILING,
        #             batch_size=BATCH_SIZE,
        #             callbacks=[epoch_checkpoint]
        #         ).history

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )

        training_start = perf_counter()
        loss = model.fit(
            X_train, encoded_y_train,
            validation_data=(X_val, encoded_y_val),
            epochs=cf.EPOCH_CEILING,          # Now a ceiling, not a target (40 epochs)
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping]
        ).history
        training_time_seconds = perf_counter() - training_start

        best_epoch = int(np.argmin(loss['val_loss'])) + 1
        best_epochs.append(best_epoch)
        print(f"\nEarly stopping selected epoch {best_epoch} for fold {kfold+1}.")
        # model.save(f"{cf.SAVE_DIR}model_{best_epoch}e_fold_{kfold+1}.keras")

        losses.append(loss)

        print(f"\n\nFinished Training Fold {kfold+1}, freeing up data splits...\n")
        del X_train
        del X_val

        print(f"\n\nSaving training history and fold tracking metadata...\n")
        np.save(f"{cf.SAVE_DIR}/loss_fold_{kfold+1}", loss)
        np.save(f"{cf.SAVE_DIR}/training_time_fold_{kfold+1}", training_time_seconds)
        np.save(f"{cf.SAVE_DIR}/names_train_fold_{kfold+1}", names_train)
        np.save(f"{cf.SAVE_DIR}/names_val_fold_{kfold+1}", names_val)

        print(f"\n\nSaving final model to {cf.SAVE_DIR} and freeing up resources...\n")
        model.save(f"{cf.SAVE_DIR}/model_fold_{kfold+1}.keras") if cf.SAVE_XFOLD_DATA else None
        
        # 1. CLEANUP: Delete remaining local variables to free memory
        del model
        del encoded_y_train, encoded_y_val
        
        # 2. Reset the Keras backend graph
        K.clear_session()
        ops.reset_default_graph()
        
        # 3. Force garbage collection
        gc.collect()

        print(f"\n\nFinished Fold {kfold+1}. Moving on to the next fold...\n" + "="*80)
    
    consensus_epoch = int(np.round(np.median(best_epochs)))
    print(f"\nConsensus best epoch: {consensus_epoch} (per-fold: {best_epochs})")
    np.save(f"{cf.SAVE_DIR}/best_epoch", consensus_epoch)

def transfer_learn(**kwargs):
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

    # for i, (train_idx, val_idx) in enumerate(folds):
    #     print_train_val_distribution(X_train_val[train_idx], y_train_val[train_idx], X_train_val[val_idx], y_train_val[val_idx], name=f"Fold {i+1}")

    cf.LOOKUP = {
        "FRI": [1, 0, 0],
        "FRII": [0, 1, 0],
        "COMPACT": [0, 0, 1],
    }

    np.save(f"{cf.SAVE_DIR}/lookup", cf.LOOKUP)
    
    run_transfer_learning(X_train_val, y_train_val, X_test, names_train_val, folds, cf)

    del X_train_val, y_train_val, X_test, names_train_val, names_test
    K.clear_session()
    gc.collect()

def run_transfer_learning(X_train_val, y_train_val, X_test, names_train_val, folds, cf):
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
    best_fine_tune_epochs = []
    
    for kfold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n>>>>>Training Fold {kfold+1}<<<<<<\n\n")

        if np.isnan(X_train_val).any():
            raise ValueError("NaN values found in training/validation data")

        X_train, y_train, names_train = process.build_dataset(X_train_val, y_train_val, names_train_val, train_idx, cf, size_per_type=(SIZE_PER_TYPE - int(SIZE_PER_TYPE * cf.VAL_SPLIT)))

        X_val, y_val, names_val = process.build_dataset(X_train_val, y_train_val, names_train_val,  val_idx, cf, size_per_type=int(SIZE_PER_TYPE * cf.VAL_SPLIT))

        encoded_y_train = np.array([cf.LOOKUP[x] for x in y_train])
        encoded_y_val = np.array([cf.LOOKUP[x] for x in y_val])
        
        model = load_model_from_base(cf, input_shape=X_train.shape[1:], num_classes=len(np.unique(y_train_val)), fold=kfold+1)

        # Freeze all layers except the last one for a few epocs to let it learn reasonable weights
        for layer in model.layers[:-1]:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=cf.LEARNING_RATE*10),
            loss='categorical_crossentropy',
            metrics=["accuracy"]  
        )

        print("Training model output layer...")

        loss = model.fit(
                    X_train, encoded_y_train,
                    validation_data=(X_val, encoded_y_val),
                    epochs=5,
                    batch_size=BATCH_SIZE,
                ).history
        
        # Unfreeze all layers and retrain 
        for layer in model.layers[:-1]:
            layer.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=cf.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=["accuracy"]  
        )

        print("Training entire model from base...")

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )

        loss = model.fit(
            X_train, encoded_y_train,
            validation_data=(X_val, encoded_y_val),
            epochs=cf.EPOCH_CEILING - 5,      # cf.EPOCH_CEILING is now a ceiling
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping]
        ).history

        best_phase2_epoch = int(np.argmin(loss['val_loss'])) + 1
        total_best_epoch = 5 + best_phase2_epoch
        best_fine_tune_epochs.append(total_best_epoch)
        print(f"\nFold {kfold+1}: best phase-2 epoch = {best_phase2_epoch} (total: {total_best_epoch})")

        # ----------------------------------------

        losses.append(loss)

        print(f"\n\nFinished Training Fold {kfold+1}, now saving validation metrics and data traces...\n")
        
        # 1. Save historical performance metrics (loss, accuracy vectors)
        np.save(f"{cf.SAVE_DIR}/loss_fold_{kfold+1}", loss)
        
        # 2. Save dynamic sample indexes to ensure deterministic offline validation slicing
        np.save(f"{cf.SAVE_DIR}/names_train_fold_{kfold+1}", names_train)
        np.save(f"{cf.SAVE_DIR}/names_val_fold_{kfold+1}", names_val)

        print(f"\n\nFinished Saving Training History and Sample Labels for Fold {kfold+1}, now saving model to {cf.SAVE_DIR}...\n")
        
        # 3. Save the final epoch weights
        if cf.SAVE_XFOLD_DATA:
            model.save(f"{cf.SAVE_DIR}/model_fold_{kfold+1}.keras")
        
        print(f"\n\nFinished Saving Model for Fold {kfold+1}, now freeing up VRAM resources...\n")

        # 4. Rigorous VRAM context destruction to prevent Fold-to-Fold memory leakage
        del model
        
        K.clear_session()
        ops.reset_default_graph()
        gc.collect()

        print(f"\n\nFinished Fold {kfold+1}. Moving on to the next fold...\n" + "="*80)
    
    consensus_epoch = int(np.round(np.median(best_fine_tune_epochs)))
    print(f"\nConsensus fine-tune epoch: {consensus_epoch} (per-fold: {best_fine_tune_epochs})")
    np.save(f"{cf.SAVE_DIR}/best_epoch", consensus_epoch)

    # old_path = cf.SAVE_DIR.rstrip('/')
    # new_folder_name = os.path.basename(old_path).replace(
    #     f"_base{cf.BASE_EPOCHS}e_{cf.EPOCH_CEILING}e_",
    #     f"_base{cf.BASE_EPOCHS}e_{consensus_epoch}e_"
    # )
    # new_path = os.path.join(os.path.dirname(old_path), new_folder_name)
    # if old_path != new_path and not os.path.exists(new_path):
    #     os.rename(old_path, new_path)
    #     print(f"Renamed: {os.path.basename(old_path)} -> {new_folder_name}")

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
        This function loads the test dataset, loads all trained models from each fold, 
        makes predictions with each model, and averages the predictions to create an ensemble result. 
        It saves both the individual per-fold predictions and the averaged ensemble predictions 
        to .npy files in a structured format for later analysis.
    '''

    cf = Config(kwargs)

    print(f"Loading test dataset for {cf.DATASET}...")  

    X_test, y_test, names_test = process.get_dataset("test", cf)  

    print("Loading fold models and collecting predictions...") 

    _format_config(cf)
    _validate_directory(cf)

    fold_predictions = []
    fold = 1

    models_root = cf.MODEL_DIR
    print(f"Looking for models in {models_root}...")

    fold_models = process.getFoldModels(models_root)

    for fold, model_path in sorted(fold_models.items()):
        print(f"Processing fold {fold}: {model_path}")

        # print(f"Found model for fold {fold}, in {models_root}. Loading model and making predictions...")
        # print(f" >>>>> Loading fold {fold} model ({models_root}model_fold_{fold}.keras) and making predictions... <<<<< ")

        model = process._safe_load_model(model_path)

        # Make predictions with this fold model
        with tensorflow.device('/cpu:0'): 
            y_pred_fold = model.predict(X_test, batch_size=BATCH_SIZE)
        
        fold_predictions.append(y_pred_fold)
        
        # SAVE PER-FOLD PREDICTIONS TO PREVENT MISSING DATA ERRORS DOWNSTREAM
        fold_save_path = f"{cf.SAVE_DIR}y_pred_test_fold_{fold}.npy"
        np.save(fold_save_path, y_pred_fold)
        print(f"Saved fold {fold} predictions to {fold_save_path}")
        
        # Clean up this fold model to isolate memory
        del model
        tensorflow.keras.backend.clear_session()
        gc.collect()
        
        fold += 1
    
    # if not os.path.exists(models_root + "model_fold_" + str(fold) + ".keras"):
    #     raise ValueError(f"No trained models found in {cf.MODEL_DIR}")
    
    print(f"Found {len(fold_predictions)} fold predictions. Averaging predictions...")
    
    # Average predictions across all folds
    averaged_y_pred_test = np.mean(fold_predictions, axis=0) 
    
    # Save the averaged predictions and structural test data
    np.save(f"{cf.SAVE_DIR}y_pred_test_ensemble", averaged_y_pred_test)
    np.save(f"{cf.SAVE_DIR}y_test", y_test)
    np.save(f"{cf.SAVE_DIR}names_test", names_test)
    
    print(f"Saved ensemble predictions to {cf.SAVE_DIR}y_pred_test_ensemble.npy")
    
    # Clean up
    del fold_predictions, averaged_y_pred_test
    gc.collect()
# endregion 

# region CLASSES
class Config:
    def __init__(self, kwargs):
        # make keys properties
        self.__dict__.update(kwargs)


class SpecificEpochCheckpoint(Callback):
    def __init__(self, target_epochs, save_dir, fold):
        super().__init__()
        self.target_epochs = target_epochs
        self.save_dir = save_dir
        self.fold = fold

    def on_epoch_end(self, epoch, logs=None):
        # Keras epochs are 0-indexed, so add 1 for standard 1-based logic
        current_epoch = epoch + 1
        if current_epoch in self.target_epochs:
            filepath = f"{self.save_dir}model_{current_epoch}e_fold_{self.fold}.keras"
            self.model.save(filepath)
            print(f"\nSaved intermediate model to {filepath}")
# endregion

# region MAIN
def runModel(experiments, experiment_number):
    '''
    ARGUMENTS:
    ----------
        > experiments: List of experiment parameters.
        > experiment_number: Number of the experiment to run.

    RETURN:
    -------
        > None. This function runs the main experiment.

    DESCRIPTION:
    ------------
        This function initializes the results directory, sets up the configuration, and runs the main experiment. It checks for command-line arguments to determine the experiments to run, and if not provided, it raises an error prompting the user to run the manager script.
    '''
    testing = TESTING
    transfer_learning = TRANSFER_LEARNING
    RESULTSDIRECTORY = os.path.join(PROJECT_ROOT, "Results", "Experiment_" + str(experiment_number))
    
    config = dict(
        IGNORE_WARNINGS = True,
        SAVE_XFOLD_DATA=True,
        VAL_SPLIT = 0.2,
        SEED = 100,
        SAVE_DIR = RESULTSDIRECTORY,
    )

# region TRAINING
    # * ======================================= TRAINING LOGIC =======================================
    if testing == False:
        print("Testing mode is OFF. Training will be performed.")
        
        if transfer_learning == False:
            for i, (epoch_ceiling, surveys, lr, reg) in enumerate(experiments):
                
                print("\n" + "="*80)
                print(f"\nRunning training for sub-experiment {i+1}/{len(experiments)}.")

                print_training_parameters(experiment_number, epoch_ceiling, surveys, lr, reg)

                experiment(
                    EPOCH_CEILING = epoch_ceiling, # Ceiling for training - 40 epoch_ceiling. Using early stopping. 
                    LEARNING_RATE = lr,
                    REGULARISATION = reg,
                    SURVEYS = surveys, # ["FIRST"], ["NVSS"], ["LOFAR"], or ["RADCAT"]
                    IGNORE_WARNINGS = config["IGNORE_WARNINGS"],
                    SAVE_XFOLD_DATA= config["SAVE_XFOLD_DATA"],
                    VAL_SPLIT = config["VAL_SPLIT"],
                    SEED =config["SEED"],
                    SAVE_DIR = config["SAVE_DIR"],
                    EXPERIMENT_NUMBER = experiment_number,
                    TESTING = False, 
                    BASE_MODEL = "NONE",
                    BASE_REGULARISATION = reg,
                    TRANSFER_LEARNING = False,
                )

                K.clear_session()
                gc.collect() 
                
                print(f"Finished sub-experiment {i+1}/{len(experiments)} ({surveys} dataset) with learning rate {lr} and regularisation {reg}\n" + "="*80)
            
            # Move all intermediate models into their own folders
            # process.restructure_results_by_epoch(os.path.join(RESULTSDIRECTORY, "Training/"))
            transfer_learning = True

# region TFER LEARN
    # * ================================== TRANSFER LEARNING LOGIC ==================================
        if transfer_learning == True:
            
            MODEL_DIRECTORY = os.path.join(RESULTSDIRECTORY, "Training/") 
            MODEL_PATH = Path(MODEL_DIRECTORY) 
            
            target_models = ["FIRST", "NVSS", "LOFAR"]
            transfer_counter = 1
            folders = list(MODEL_PATH.iterdir())
            print(f"Folders to parse: {[folder.name for folder in folders]}")

            folders = process.clearTransferLearningFolders(folders)

            for folder in folders:
                if folder.is_dir():
                    print(f"Processing transfer learning for: {folder.name} ({transfer_counter}/{len(folders)})")
                    print(f"Absolute path: {folder}")
                    current_dir = str(folder)
                    
                    transfer_parameters = process.extract_parameters_from_folder_name(folder.name) # | Extract parameters from the folder name for testing

                    # ! base_model = "NONE" -> the true base model will be transfer_parameters["model"]
                    # ! Note that surveys = model here. 
                    base_model = transfer_parameters["model"]
                    # | parameters = {
                    # |     "base_model": match.group(1),
                    # |     "model": match.group(2),
                    # |     "surveys": match.group(3),
                    # |     "base_epochs": float(match.group(4)),
                    # |     "epochs": float(match.group(5)),
                    # |     "lr": float(match.group(6)),
                    # |     "base_reg": float(match.group(7)),
                    # |     "reg": float(match.group(8))
                    # | }

                    for target_model in target_models:
                        if target_model != base_model: # if base model =! model to train
                            print(f"Preparing to train target model {target_model} using base model {base_model} ...")

                            print_transfer_learning_parameters(experiment_number=experiment_number, epoch_ceiling=transfer_parameters['epoch_ceiling'], surveys=target_model, lr=transfer_parameters['lr'], reg=transfer_parameters['reg'], base_model=base_model, base_reg=transfer_parameters['base_reg'])

                            transfer_learn(
                                EXPERIMENT_NUMBER = experiment_number,
                                BASE_MODEL = base_model,
                                BASE_REGULARISATION = transfer_parameters["base_reg"],
                                EPOCH_CEILING = transfer_parameters["epoch_ceiling"],   # Ceiling; EarlyStopping determines actual stop
                                LEARNING_RATE = transfer_parameters["lr"]/10,
                                REGULARISATION = transfer_parameters["reg"],
                                SURVEYS = target_model,
                                MODEL = target_model,
                                IGNORE_WARNINGS = config["IGNORE_WARNINGS"],
                                SAVE_XFOLD_DATA = config["SAVE_XFOLD_DATA"],
                                VAL_SPLIT = config["VAL_SPLIT"],
                                SEED = config["SEED"],
                                SAVE_DIR = config["SAVE_DIR"],
                                TESTING = False,
                                TRANSFER_LEARNING = True,
                                MODEL_DIR = MODEL_DIRECTORY,
                            )

                            K.clear_session()
                            gc.collect()

                            print(f"\n>>>>> Finished transfer learning run {transfer_counter}: target {target_model} from base {base_model} <<<<<\n" + "="*80)
                            
                            transfer_counter += 1
            
            testing = True

# region TESTING   
    # * ======================================= TESTING LOGIC =======================================
    if testing == True:

        MODEL_DIRECTORY = os.path.join(RESULTSDIRECTORY, "Training/") 
        MODEL_PATH = Path(MODEL_DIRECTORY) 
        
        print("Testing mode is ON. No training will be performed. Models will be tested on each other's datasets.")
        test_counter = 1
        
        # | folder format: baseMODEL_modelMODEL_setDATASET_baseXXe_XXe_0.00039l_baseXXr_XXr

        # Verify directory existence before initializing loop
        if not MODEL_PATH.exists() or not MODEL_PATH.is_dir():
            raise FileNotFoundError(f"The path {MODEL_DIRECTORY} is not a valid directory.")

        # .iterdir() yields both files and directories; .is_dir() filters for folders only
        for folder in MODEL_PATH.iterdir():
            if folder.is_dir():
                print(f"Processing testing for: {folder.name} ({test_counter}/{len(list(MODEL_PATH.iterdir()))})")
                print(f"Absolute path: {folder}")
                current_dir = str(folder)
                
                test_parameters = process.extract_parameters_from_folder_name(folder.name) # | Extract parameters from the folder name for testing

                # | parameters = {
                # |     "base_model": match.group(1),
                # |     "model": match.group(2),
                # |     "surveys": match.group(3),
                # |     "base_epochs": float(match.group(4)),
                # |     "epochs": float(match.group(5)),
                # |     "lr": float(match.group(6)),
                # |     "base_reg": float(match.group(7)),
                # |     "reg": float(match.group(8))
                # | }

                
                # ========================
                # * PERFORM OTHER TESTING
                # ========================
                for dataset in ["FIRST", "NVSS", "LOFAR"]:

                    print_testing_parameters(experiment_number, test_parameters, dataset)

                    run_test(
                        MODEL = test_parameters["model"], # | MODEL TO TEST
                        SURVEYS = dataset, # | DATASET TO TEST ON (for formatting reasons)
                        LEARNING_RATE = test_parameters["lr"], # | LEARNING RATE USED FOR THIS MODEL (for formatting reasons)
                        REGULARISATION = test_parameters["reg"], # | REGULARISATION USED FOR THIS MODEL (for formatting reasons)
                        BASE_REGULARISATION = test_parameters["base_reg"], # | REGULARISATION USED FOR THE BASE MODEL THIS ONE WAS TRAINED ON (for formatting reasons)
                        DATASET = dataset, # | DATASET TO TEST ON
                        EXPERIMENT_NUMBER = experiment_number,
                        MODEL_DIR = current_dir,
                        EPOCH_CEILING = test_parameters["epoch_ceiling"], # | HOW LONG MODEL WAS TRAINED FOR 
                        BASE_MODEL = test_parameters["base_model"], # | BASE MODEL THAT THIS ONE WAS TRAINED ON (IF TRANSFER LEARNING)
                        # BASE_EPOCHS = test_parameters["base_epochs"], # | HOW MANY EPOCHS THE BASE MODEL TRAINED FOR 
                        IGNORE_WARNINGS = config["IGNORE_WARNINGS"],
                        SAVE_XFOLD_DATA= config["SAVE_XFOLD_DATA"],
                        VAL_SPLIT = config["VAL_SPLIT"],
                        SEED =config["SEED"],
                        SAVE_DIR = config["SAVE_DIR"],
                        TESTING = True,
                    )

                    test_counter += 1
            
                    print(f"\n>>>>> Finished testing model {test_parameters['model']} trained on model {test_parameters['base_model']} on dataset {dataset} <<<<<\n" + "="*80)

                print(f"\nFinished processing testing for {folder.name} ({test_counter}/{len(list(MODEL_PATH.iterdir()))})\n" + "="*80)
            
        K.clear_session()
        gc.collect()

# endregion