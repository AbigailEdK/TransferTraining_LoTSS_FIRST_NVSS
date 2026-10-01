"""
Master orchestration script for Image Augmentation pipeline
Runs all processing scripts in the correct order with error handling and reporting
"""

import sys

sys.path.insert(0, '/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Management')

import os
import subprocess
import time
from datetime import datetime
from myUtils import check_folder_existence


SCRIPTS_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Image Augmentation"
VENV_PYTHON = "/home/abigaildeklerk/Desktop/DeKlerk_Models/.venv/bin/python"
DATA_FOLDER = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data"

def main():
    # Check if data folder exists
    if not check_folder_existence(DATA_FOLDER):
        print(f"Folder not found. Creating folder at: {DATA_FOLDER}")
    else:
        print(f"✓ Data folder found at {DATA_FOLDER}. Proceeding with pipeline execution.")    
        
    scripts = [
        "combineFITStoNumpy.py", # Convert FITS from RADCAT_F to individual .npy files
        "combineNumpyArrays.py", # Combine individual .npy files into X, y, names 
        
    ]

    # > Ask user if they want to generate a new augmentation file
    while True:
        generate_aug = input("Would you like to generate a new augmentation file? (y/n): ").strip().lower()
        if generate_aug in {"y", "n"}:
            break
        print("Invalid input. Please enter 'y' or 'n'.")

    if generate_aug == "y":
        scripts.insert(0, "createAugmentationFile.py")
        print("New augmentation file generation will run first in the pipeline.")
    else:
        print("Skipping new augmentation file generation.")
    
    # > Run all scripts in order with error handling
    for script in scripts:
        script_path = os.path.join(SCRIPTS_DIR, script)
        print(f"\n>>>>> Running {script} <<<<<\n" + "="*80)
        
        try:
            result = subprocess.run([VENV_PYTHON, script_path], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Errors from {script}:\n{result.stderr}")
        except Exception as e:
            print(f"Error running {script}: {e}")
        
        print(f"\n>>>>> Finished {script} <<<<<\n" + "="*80)
        time.sleep(2)  # Brief pause between scripts
    
    print("\n" + "="*80 + "\n" + "=>>>>> IMAGE AUGMENTATION PIPELINE COMPLETE <<<<<" + "\n" + "="*80)

if __name__ == "__main__": main()