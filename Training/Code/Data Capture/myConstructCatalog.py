# region ABOUT
# ===================================================================================================
# > This script takes the RADCAT_nonunique.csv file and the downloaded FITS files from DATA_RESULTS (aquired from download_data.py), filters the sources based on their types (COMPACT, FRI, FRII) and download status (failed or succeeded), categorizes the FITS files by survey type (FIRST, LOFAR, NVSS), and saves the results to new CSV files for use in model training. 

# > File result: 4 CSV files containing the filtered sources and categorized FITS file paths, saved to CONSTRUCT_RESULTS.
    # > 1. FIRST_fits_files.csv
        # > Contains the paths to the FITS files for sources of type FIRST
    # > 2. LOFAR_fits_files.csv
        # > Contains the paths to the FITS files for sources of type LOFAR
    # > 3. NVSS_fits_files.csv
        # > Contains the paths to the FITS files for sources of type NVSS
    # > 4. filtered_sources.csv
        # > Contains the filtered source information (COMPACT, FRI, FRII) - only sources for which FITS files were successfully downloaded

# > Adjusted from construct_catalog.py by Dylan Farge.
# ===================================================================================================
# endregion

# region IMPORTS
import os
from turtle import pd
from astropy.io import fits
import numpy as np
# endregion

# region CLASS DEF
class ConstructSourceDatasets:
    def __init__(self, data_path, fits_path, output_path, debug=False):
        self.data_path = data_path # | Path to the CSV file containing source information
        self.fits_path = fits_path # | Path to the folder containing the FITS files (DATA_RESULTS)
        self.output_path = output_path # | Path to save the final filtered CSV files to

        self.df = pd.read_csv(self.data_path, index_col=0) # | Load the CSV file into a DataFrame, using the first column as the index

        self.corrupted_files = [] # | List to keep track of corrupted files

        self.FIRST_set = []
        self.LOFAR_set = []
        self.NVSS_set = []

        self.debug = debug # | Flag to enable or disable debug print statements

        if self.debug:
            print(self.df.shape[0],"\tNum of sources in original DataFrame")

    # > Filter the DataFrame to include only sources of types COMPACT, FRI, and FRII
    def filter_by_types(self, save_intermediate_dfs=False):
        mask = self.df["Type"].isin(["COMPACT", "FRI", "FRII"])

        # * Optionally save the intermediate DataFrame of excluded sources for analysis
        if save_intermediate_dfs:
            self.df[~mask].to_csv("ConstructCatalog/data_excluded/0_not_compact_fri_frii.csv")

        # * Update the main DataFrame to include only the filtered sources
        self.df = self.df[mask]
        
        if self.debug:
            print(self.df.shape[0],"\tNum of sources after filtering by types")

    # > Filter the DataFrame to include only sources for which FITS files were successfully downloaded
    def filter_by_downloads(self, save_intermediate_dfs=False):
        downloaded_sources = set()
        
        # * Iterate through the FITS files in the specified folder and extract the source indices from the filenames
        for path in os.listdir(self.fits_path):
            downloaded_sources.add(int(path.split('_')[0]))

        # * Optionally save the intermediate DataFrame of excluded sources for analysis
        if save_intermediate_dfs:
            self.df[~self.df.index.isin(downloaded_sources)].to_csv("ConstructCatalog/data_excluded/1_could_not_download.csv")

        # * Update the main DataFrame to include only the filtered sources
        self.df = self.df[self.df.index.isin(downloaded_sources)]

        if self.debug:
            print(self.df.shape[0],"\tNum of sources after filtering by downloads")

    # > Categorize the FITS files into separate lists based on their survey type (FIRST, LOFAR, NVSS)
    def categorize_fits_files(self):
        # * Iterate through the files in the data folder
        for file_name in os.listdir(self.fits_path):
            if file_name.endswith(".fits"):
                file_path = os.path.join(self.fits_path, file_name)
                
                # * Open the FITS file to determine its type
                with fits.open(file_path) as hdul:
                    # * Assuming the FITS file has a header keyword to identify its type
                    header = hdul[0].header
                    survey_type = header.get('SURVEY', '').upper()
                    
                    # * Categorize the file based on the survey type
                    if "FIRST" in survey_type:
                        self.FIRST_set.append(file_path)
                    elif "LOFAR" in survey_type:
                        self.LOFAR_set.append(file_path)
                    elif "NVSS" in survey_type:
                        self.NVSS_set.append(file_path)

        # * Print the results
        if self.debug:
            print("FIRST set:", self.FIRST_set)
            print("LOFAR set:", self.LOFAR_set)
            print("NVSS set:", self.NVSS_set)

    # > Remove corrupted FITS files and update the DataFrame accordingly
    def remove_corrupted_fits_files(self):
        for fits_list in [self.FIRST_set, self.LOFAR_set, self.NVSS_set]:
            for file_path in fits_list[:]:  # Iterate over a copy of the list to allow removal
                try:
                    with fits.open(file_path) as hdul:
                        hdul.verify('silentfix')  # Check for corruption
                except Exception as e:
                    print(f"Corrupted FITS file detected and removed: {file_path} | Error: {e}")
                    
                    # * Remove the corrupted file from the respective list 
                    fits_list.remove(file_path)
                    
                    # * Keep track of the corrupted file paths for reference
                    self.corrupted_files.append(file_path)

                    # * Remove the corrupted file from the main DataFrame and respectively from the categorized FITS file lists
                    self.df = self.df[self.df["FITS_File"] != file_path]
                    self.FIRST_set = [f for f in self.FIRST_set if f != file_path]
                    self.LOFAR_set = [f for f in self.LOFAR_set if f != file_path]
                    self.NVSS_set = [f for f in self.NVSS_set if f != file_path]

    # > Print the distribution of source types in the final filtered DataFrame    
    def print_distribution(self, spacing=20):
        dataset = self.df["Type"].values
        
        print(f"\n{'--- Morphology Distribution ---':^{spacing*3 + 2}}")
        unique, counts = np.unique(dataset, return_counts=True)
        
        print(f"{'Morphology':^{spacing}}|{'Count':^{spacing}}|{'Percentage':^{spacing}}")
        
        print(f"{'-'*spacing}|{'-'*spacing}|{'-'*spacing}")
        
        for i in range(len(unique)):
            print(f"{unique[i]:^{spacing}}|{counts[i]:^{spacing}}|{f'{counts[i]/len(dataset)*100:.2f}%':^{spacing}}")
        
        print(f"{'-'*spacing}|{'-'*spacing}|{'-'*spacing}")
        footer_text = f"Total Images: {len(dataset)}/{self.original_len}\t{len(dataset)/self.original_len*100:.2f}%\n"
        
        print(f"{footer_text:^{spacing*3 + 2}}\n")

    # > Save the final filtered DataFrames to new CSV files
    def save_to_CSVs(self):
        self.df.to_csv(os.path.join(self.output_path, "filtered_sources.csv"))
        
        self.FIRST_set_df = pd.DataFrame(self.FIRST_set, columns=["FITS_File"])
        self.FIRST_set_df.to_csv(os.path.join(self.output_path, "FIRST_fits_files.csv"), index=False)

        self.LOFAR_set_df = pd.DataFrame(self.LOFAR_set, columns=["FITS_File"])
        self.LOFAR_set_df.to_csv(os.path.join(self.output_path, "LOFAR_fits_files.csv"), index=False)

        self.NVSS_set_df = pd.DataFrame(self.NVSS_set, columns=["FITS_File"])
        self.NVSS_set_df.to_csv(os.path.join(self.output_path, "NVSS_fits_files.csv"), index=False)
        
        if self.debug:
            print(f"Filtered sources saved to {self.output_path}")

    # > Print a summary of the final results, including the number of sources in each category and the number of corrupted files removed
    def print_final_summary(self):
        print("\n--- Final Summary ---")
        print(f"Total sources in original DataFrame: {self.original_len}")
        print(f"Number of corrupted FITS files removed: {len(self.corrupted_files)}")
        print(f"Number of sources in FIRST set: {len(self.FIRST_set)}")
        print(f"Number of sources in LOFAR set: {len(self.LOFAR_set)}")
        print(f"Number of sources in NVSS set: {len(self.NVSS_set)}")
        print(f"Total sources after filtering: {self.df.shape[0]}")
# endregion


# region MAIN
def main():
    data_path = "/home/abigaildeklerk/Downloads/DeKlerk_Models/Code/DataCapture/RADCAT_nonunique.csv" # | Path to the CSV file containing source information
    fits_path = "/home/abigaildeklerk/Downloads/DeKlerk_Models/Data/DATA_RESULTS" # | Path to the folder containing the FITS files (DATA_RESULTS)
    output_path = "/home/abigaildeklerk/Downloads/DeKlerk_Models/Data/CONSTRUCT_RESULTS" # | Path to save the final filtered CSV files to

    save_intermediate_dfs = True # | Flag to enable or disable saving intermediate DataFrames of excluded sources for analysis

    data = ConstructSourceDatasets(data_path, fits_path, output_path, debug=True)

    data.filter_by_types(save_intermediate_dfs=save_intermediate_dfs) # | Filter the DataFrame to include only sources of types COMPACT, FRI, and FRII, and optionally save the intermediate DataFrame of excluded sources for analysis
    data.filter_by_downloads(save_intermediate_dfs=save_intermediate_dfs) # | Filter the DataFrame to include only sources for which FITS files were successfully downloaded, and optionally save the intermediate DataFrame of excluded sources for analysis
    data.print_distribution() # | Print the distribution of source types in the final filtered DataFrame
    data.categorize_fits_files() # | Categorize the FITS files into separate lists based on their survey type (FIRST, LOFAR, NVSS)
    data.remove_corrupted_fits_files() # | Remove corrupted FITS files and update the DataFrame accordingly
    data.save_to_CSVs() # | Save the final filtered DataFrames to new CSV files

    data.print_final_summary() # | Print a summary of the final results, including the number of sources in each category and the number of corrupted files removed

if __name__ == "__main__": main()














