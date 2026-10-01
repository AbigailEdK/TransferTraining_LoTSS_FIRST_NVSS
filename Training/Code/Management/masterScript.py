# region ABOUT
# ===================================================================================================
# > This script manages the training of multiple models on different surveys, with varying hyperparameters, by running the run_model.py script in separate tmux sessions. It monitors the progress of the experiments and estimates the time remaining based on the number of completed experiments and the time taken for each. Once all experiments are completed, it sends an email notification with the total time taken.

# > Adjusted from manager.py by Dylan Farge.
# ===================================================================================================
# endregion

# region IMPORTS
import sys
import os

# Find the home directory dynamically
home_dir = os.path.expanduser("~")
# Construct the project path
project_root = os.path.join(home_dir, "DeKlerk_Models")

# Insert the dynamic path so 'Code.Management.myUtils' can be found
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import subprocess
from time import sleep, time
from datetime import datetime
# from Code.Management.myUtils import send_gmail
# endregion

# region PATHS
data_dir = os.path.join(project_root, "Data")
code_dir = os.path.join(project_root, "Code")
model_dir = os.path.join(project_root, "Models")

MANAGER = os.path.join(project_root, "Code", "Management", "myManager.py")
RUNMODEL = os.path.join(project_root, "Code", "Management", "myRunModel.py")
RESULTSDIRECTORY = os.path.join(project_root, "Results")

# Ensure output directories exist so the script doesn't crash during saving
for d in [RESULTSDIRECTORY, model_dir]:
    os.makedirs(d, exist_ok=True)
# endregion

config = dict(
    EPOCHS = [20],
    LEARNING_RATE_RANGE = np.linspace(1.000e-06, 2.751e-03, 8),  # IMG
    REGULARISATION_RANGE = np.linspace(0.0, 50.0, 8),            # IMG
    SURVEYS = [
        ["FIRST", "LOFAR", "NVSS", "RADCAT"],
        ],
)

experiments = []
epochs = 20
                 
# RADCAT: RD: 0.54; LR = 0.00039
# FIRST:  RD: 0.40; LR = 0.00039
# NVSS:   RD: 0.26; LR = 0.00039
# LOFAR:  RD: 0.52; LR = 0.00039

# > Set up to train 4 models, each on a different survey, with individual learning rates and regularisation values based on previous results.
# experiments.append((epochs, "RADCAT", 0.00039, 0.54))  
experiments.append((epochs, "FIRST", 0.00039, 0.40))  
experiments.append((epochs, "NVSS", 0.00039, 0.26))  
experiments.append((epochs, "LOFAR", 0.00039, 0.52))  

print("="*80)
print(f"Total number of experiments: {len(experiments)}")
print("="*80)

# Convert experiments format: surveys from string to list for myRunModel
experiments_formatted = [(ep, [survey], lr, reg) for ep, survey, lr, reg in experiments]

# Call myRunModel with experiments as string argument
print("Beginning subprocess...")
subprocess.run([
    sys.executable, RUNMODEL, str(experiments_formatted)
])





   
