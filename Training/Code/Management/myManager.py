# region ABOUT
# ===================================================================================================
# > 

# > Adjusted from manager.py by Dylan Farge.
# ===================================================================================================
# endregion

# region IMPORTS
import sys
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import subprocess
from time import time
from datetime import datetime

# Find the home directory dynamically
home_dir = os.path.expanduser("~")
project_root = os.path.join(home_dir, "DeKlerk_Models")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
# from Code.Management.myUtils import send_gmail
# endregion

# region PATHS
RUNMODEL = os.path.join(project_root, "Code", "Management", "myRunModel.py")
RESULTSDIRECTORY = os.path.join(project_root, "Results")
model_dir = os.path.join(project_root, "Models")

os.makedirs(RESULTSDIRECTORY, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# endregion

print("="*80)
print(f"MANAGER STARTING AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

tick = time()

# Experiment Setup
epochs = 20
experiments = []
experiments.append((epochs, "FIRST", 0.00039, 0.40))  
experiments.append((epochs, "NVSS", 0.00039, 0.26))  
experiments.append((epochs, "LOFAR", 0.00039, 0.52))  

# Format for the sub-script
experiments_formatted = [(ep, [survey], lr, reg) for ep, survey, lr, reg in experiments]

print(f"Total number of experiments to process: {len(experiments)}")

try:
    # Instead of tmux, we run the script directly. 
    # subprocess.run will wait here until myRunModel finishes all experiments.
    print(f"Launching Training Script: {RUNMODEL}")
    
    subprocess.run([
        sys.executable, 
        RUNMODEL, 
        str(experiments_formatted)
    ], check=True)

    print("Subprocess finished successfully.")

except subprocess.CalledProcessError as e:
    print(f"CRITICAL ERROR in myRunModel: {e}")
    # send_gmail("Manager Error", f"The training subprocess failed: {e}")
    sys.exit(1)

# region FINAL STATS
tock = time() - tick
hours = int(tock // 3600)
minutes = int((tock % 3600) // 60)
seconds = int(tock % 60)

time_str = f"{hours}h {minutes}m {seconds}s"
print(f"Total Time taken: {time_str}")

# send_gmail("Manager.py Done", f"All experiments (FIRST, NVSS, LOFAR) completed.\nTotal time: {time_str}")
# endregion