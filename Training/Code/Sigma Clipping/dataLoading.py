# region ABOUT
# ===================================================================================================
# > This script sigma clips the flux values of sources in the LOFAR survey and compares them to the corresponding flux values in the FIRST survey using Pearson correlation and Hellinger distance. It loads the data from FITS files, applies sigma clipping to the LOFAR flux values, and then calculates the Pearson correlation coefficient and Hellinger distance between the clipped LOFAR flux values and the corresponding FIRST flux values. The results are printed to the console.
# ===================================================================================================
# endregion

# region IMPORTS
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
from PIL import Image
import sys
sys.path.insert(0, '/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping')
import sigmaClipping
# endregion

# region MAIN
def main(debug=False, show_figs=False):
    # Define CSV output path
    csv_output_path = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping/Results/sigma_clip_results.csv"
    
    # Check if CSV file exists and ask user
    if os.path.exists(csv_output_path):
        print(f"CSV file found: {csv_output_path}")
        response = input("Would you like to (1) reload from CSV or (2) rerun the analysis? Enter 1 or 2: ").strip()
        
        if response == '1':
            print("Loading results from existing CSV file...")
            results = sigmaClipping.load_results_from_csv(csv_output_path)
        else:
            print("Rerunning analysis...")
            results = sigmaClipping.run_analysis(csv_output_path, debug=debug)
    else:
        print("CSV file not found. Running analysis...")
        results = sigmaClipping.run_analysis(csv_output_path, debug=debug)
    
    # Create visualizations
    # sigmaClipping.create_plots(results, show_figs=show_figs)
    
    # Display final summary
    sigmaClipping.display_summary(results, show_figs=show_figs)
    
    sigmaClipping.display_plots()


if __name__ == "__main__": 
    main()

# endregion