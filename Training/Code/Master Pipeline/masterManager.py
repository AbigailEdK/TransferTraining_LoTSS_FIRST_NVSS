# region ABOUT
# ===================================================================================================
# > This script handles running five experiments to verify the effect of random augmentations on training data. Each experiment involves splitting the data into training and testing sets, augmenting the training data, and then training a model using the augmented data. The script uses multiprocessing to run each experiment in a separate process, ensuring that memory is cleared between experiments. The results of each experiment are saved in separate directories for later analysis. The script also includes error handling to check for successful completion of each step and to catch any critical errors that may occur during the subprocess execution.

# > Adjusted from manager.py by Dylan Farge.
# ===================================================================================================
# endregion

print("\n\n" + "="*160 + "\n" + "="*160 + "\n\n")

# region IMPORTS
import sys
import os
import multiprocessing
import subprocess
from time import time
from datetime import datetime
from masterUtils import check_directory, checkExist
from masterRunModel import runModel
from masterDataAugmentation import augmentData, splitData, copySplitData
# endregion

# region PATHS
from paths import RESULTSDIRECTORY, SOURCEDATADIRECTORY
# endregion

# region BOOLS
from bools import TRAINING
# endregion

# region MAIN
def main():
    if TRAINING: check_directory(RESULTSDIRECTORY)
    if TRAINING: check_directory(SOURCEDATADIRECTORY, replace=True)
    
    num_experiments = 5
    start_time = datetime.now()
    
    print(f"Starting Master Manager at {start_time.strftime('%Y-%m-%d %H:%M:%S')} \n\n\n")
    tick = time()

    # === Experiment Setup ===
    experiments = []

    epochs = [60]
    for no_epochs in epochs:
        experiments.append((no_epochs, "FIRST", 0.00039, 0.40))  
        experiments.append((no_epochs, "NVSS", 0.00039, 0.26))  
        experiments.append((no_epochs, "LOFAR", 0.00039, 0.52))  

    # > Format for the sub-script
    experiments_formatted = [(ep, survey, lr, reg) for ep, survey, lr, reg in experiments]

    # === Run Experiments ===
    try:
        sourceDataExist = checkExist(SOURCEDATADIRECTORY, num_files=6)
        if TRAINING:
            if sourceDataExist == False:
                splitData(SOURCEDATADIRECTORY) # | Split data into training and testing sets, and do not repeat for every experiment. 
            else: print(f"Source data already exists in {SOURCEDATADIRECTORY}, proceeding with training.")

        print("~"*160 + "\n")
        print(f">>>>> Starting Training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} <<<<< \n")
        print("~"*160 + "\n")

        EXPERIMENTDIRECTORY = os.path.join(RESULTSDIRECTORY, f"Experiment_0")

        augmentDataExist = checkExist(os.path.join(EXPERIMENTDIRECTORY, "Data"), num_files=7)
        if TRAINING:
            if augmentDataExist == False:
                copySplitData(SOURCEDATADIRECTORY, EXPERIMENTDIRECTORY) # Copy source data into experiment folder
                augmentData(os.path.join(EXPERIMENTDIRECTORY, "Data")) # Augment the data in the experiment folder 
            else: print(f"Augmented data already exists in {os.path.join(EXPERIMENTDIRECTORY, 'Data')}, proceeding with training.")

        # > Check that the augmentation process completed successfully by verifying the existence of the expected output files. 
        required_file = os.path.join(EXPERIMENTDIRECTORY, "Data", "X_train_val.npy")
        if not os.path.exists(required_file):
            print(f"ERROR: Augmentation failed! {required_file} not found.")
            sys.exit(1)

        p = multiprocessing.Process(
            target=runModel, 
            args=(experiments_formatted, 0),
        )
        
        p.start()
        p.join()  

        # > Check if the process crashed
        if p.exitcode != 0:
            print(f"CRITICAL: Training crashed with exit code {p.exitcode}")
            sys.exit(p.exitcode)

        print("~"*160 + "\n")
        print(f">>>>> Finished Training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} <<<<< \n")
        print("~"*160 + "\n")

        print("=" * 160 + "\n" + "=" * 160 + "\n\n")
    
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL ERROR in myRunModel: {e}")
        sys.exit(1)

if __name__ == "__main__": 
    multiprocessing.set_start_method('spawn', force=True)
    main()