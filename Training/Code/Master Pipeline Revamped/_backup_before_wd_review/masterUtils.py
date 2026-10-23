import os
import shutil
import masterProcess as process
import numpy as np
import os
import tensorflow as tensorflow

# region PATHS
from paths import HOME_DIR, PROJECT_ROOT, MANAGER, RUNMODEL, AUGMENTDATA, RESULTSDIRECTORY, SOURCEDATADIRECTORY
# endregion

# region DIRECTORY FUNCTIONS
def check_directory(folder_path, replace=True):            
    if os.path.exists(folder_path) and replace:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        print(f"Folder {folder_path} replaced successfully.")
        return True
        
    else:
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created successfully.")
        return True

def checkExist(folder_path, num_files=None):
    if not os.path.exists(folder_path):
        return False
    if num_files is not None:
        return len(os.listdir(folder_path)) >= num_files
    return True

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
    # print(f"Formatting save directory based on configuration...")

    # if cf.TESTING == False:
    #     cf.SAVE_DIR = process.folder_construct(base_model=cf.BASE_MODEL, model=cf.SURVEYS, base_epochs=cf.BASE_EPOCHS, epochs=cf.EPOCHS, surveys=cf.SURVEYS, lr=cf.LEARNING_RATE, base_reg=cf.BASE_REGULARISATION, reg=cf.REGULARISATION, save_dir=cf.SAVE_DIR + "/Training/")
    # else:
    #     cf.SAVE_DIR = process.folder_construct(base_model=cf.BASE_MODEL, model=cf.MODEL, base_epochs=cf.BASE_EPOCHS,  epochs=cf.EPOCHS, surveys=cf.DATASET, lr=cf.LEARNING_RATE, base_reg=cf.BASE_REGULARISATION,  reg=cf.REGULARISATION, save_dir=cf.SAVE_DIR + "/Testing/")

    if cf.TESTING == False:
        cf.SAVE_DIR = process.folder_construct(base_model=cf.BASE_MODEL, model=cf.SURVEYS, surveys=cf.SURVEYS, lr=cf.LEARNING_RATE, base_reg=cf.BASE_REGULARISATION, reg=cf.REGULARISATION, save_dir=cf.SAVE_DIR + "/Training/", epoch_ceiling=cf.EPOCH_CEILING)
    else:
        cf.SAVE_DIR = process.folder_construct(base_model=cf.BASE_MODEL, model=cf.MODEL, surveys=cf.DATASET, lr=cf.LEARNING_RATE, base_reg=cf.BASE_REGULARISATION,  reg=cf.REGULARISATION, save_dir=cf.SAVE_DIR + "/Testing/", epoch_ceiling=cf.EPOCH_CEILING)

    print(f"Formatted directory: {cf.SAVE_DIR}")
# endregion

# region PRINT FUNCTIONS
def print_training_parameters(experiment_number, epoch_ceiling, surveys, lr, reg):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("      >>> Training Parameters: <<< ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Experiment Number: {experiment_number}")
    print(f"Epoch ceiling: {epoch_ceiling}")
    print(f"Survey: {surveys}")
    print(f"Learning Rate: {lr}")
    print(f"Regularisation: {reg}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

def print_testing_parameters(experiment_number, test_parameters, dataset):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("      >>> Testing Parameters: <<< ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Experiment Number: {experiment_number}")
    print(f"Transfer Learning Base Model: {test_parameters['base_model']}")
    print(f"Model to test: {test_parameters['model']}")
    # print(f"Base Epochs: {test_parameters['base_epochs']}")
    print(f"Epoch ceiling: {test_parameters['epoch_ceiling']}")
    print(f"Test set: {dataset}")
    print(f"Learning Rate: {test_parameters['lr']}")
    print(f"Regularisation: {test_parameters['reg']}")
    print(f"Base Regularisation: {test_parameters['base_reg']}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

def print_transfer_learning_parameters(experiment_number, epoch_ceiling, surveys, lr, reg, base_model, base_reg):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("      >>> Training Parameters: <<< ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Experiment Number: {experiment_number}")
    print(f"Target model: {surveys}")
    print(f"Epoch ceiling: {epoch_ceiling}")
    print(f"Regularisation: {reg}")
    print(f"Base Model: {base_model}")
    # print(f"Base Epochs: {base_epoch}")
    print(f"Base Regularisation: {base_reg}")
    print(f"Learning Rate: {lr}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

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

# endregion