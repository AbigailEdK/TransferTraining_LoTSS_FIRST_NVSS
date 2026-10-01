# region ABOUT
# ===================================================================================================
# > This script provides an interactive visual comparison tool for FITS files from FIRST and LoTSS
#   surveys. It also looks up each source in RADCAT.csv so the source origin is shown during review.
#   LoMORPH sources are deferred to the end of the queue so they can be checked in a separate final pass.
# ===================================================================================================
# endregion

# region IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import Normalize
import os
from astropy.io import fits
from sigmaClipping import apply_lofar_mask_to_both, get_display_images_lofar_mask, pearson_correlation, hellinger_distance
# endregion

# region PATHS
RADCAT_CSV_PATH = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT.csv"
OUTTAKES_CSV_PATH = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping/source_comparison_outtakes.csv"
# endregion

# region CLASS
class SourceComparisonTool:
    '''
    DESCRIPTION:
    -----------
        Interactive tool for comparing FIRST and LoTSS radio survey images before and after
        sigma clipping. Users rate whether sources appear similar or different.
    '''
    
    def __init__(self, first_folder, lofar_folder, results_csv_path, outtakes_csv_path):
        '''Initialize the comparison tool and load FITS files'''
        self.first_folder = first_folder
        self.lofar_folder = lofar_folder
        self.results_csv_path = results_csv_path
        self.outtakes_csv_path = outtakes_csv_path
        self.source_origins = self._load_source_origins(RADCAT_CSV_PATH)
        self._lomorph_section_announced = False
        self._finalized = False
        self.current_index = 0
        self.results = []
        self.session_results = []
        self.outtakes = []
        self.session_outtakes = []
        
        # Load file lists
        self.first_files = sorted([f for f in os.listdir(first_folder) if f.endswith('.fits')])
        self.lofar_files = sorted([f for f in os.listdir(lofar_folder) if f.endswith('.fits')])
        
        # Find sources that exist in both folders
        first_sources = set([f.replace('.fits', '') for f in self.first_files])
        lofar_sources = set([f.replace('.fits', '') for f in self.lofar_files])
        all_sources = sorted(list(first_sources & lofar_sources))
        self.regular_sources = [s for s in all_sources if not self._is_lomorph(s)]
        self.lomorph_sources = [s for s in all_sources if self._is_lomorph(s)]
        self.sources = self.regular_sources + self.lomorph_sources
        
        print(f"Found {len(self.sources)} sources in both FIRST and LoTSS folders")
        print(f"  Regular sources: {len(self.regular_sources)}")
        print(f"  LoMORPH sources deferred to the end: {len(self.lomorph_sources)}")
        
        # Load existing results if any
        if os.path.exists(results_csv_path):
            existing_df = pd.read_csv(results_csv_path, dtype={'source': str})
            self.results = existing_df.to_dict('records')
            print(f"Loaded {len(self.results)} existing results.")

        if os.path.exists(outtakes_csv_path):
            existing_outtakes_df = pd.read_csv(outtakes_csv_path, dtype={'source': str})
            self.outtakes = existing_outtakes_df.to_dict('records')
            print(f"Loaded {len(self.outtakes)} existing outtakes.")

        reviewed_sources = set([r['source'] for r in self.results]) | set([r['source'] for r in self.outtakes])
        self.regular_sources = [s for s in self.regular_sources if s not in reviewed_sources]
        self.lomorph_sources = [s for s in self.lomorph_sources if s not in reviewed_sources]
        self.sources = self.regular_sources + self.lomorph_sources

        if reviewed_sources:
            print(f"Skipping {len(reviewed_sources)} already-handled sources from previous runs.")

        print(f"{len(self.sources)} sources remaining for review.")
        
        if len(self.sources) == 0:
            print("All sources have been reviewed!")
            return
        
        # Create figure
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('FITS File Comparison: FIRST vs LoTSS (Before/After Sigma Clipping)', 
                         fontsize=14, fontweight='bold')
        
        # Create subplots
        self.ax_lofar_before = plt.subplot(2, 2, 1)
        self.ax_first_before = plt.subplot(2, 2, 2)
        self.ax_lofar_after = plt.subplot(2, 2, 3)
        self.ax_first_after = plt.subplot(2, 2, 4)
        
        # Create buttons
        ax_same = plt.axes([0.25, 0.05, 0.12, 0.04])
        ax_diff = plt.axes([0.40, 0.05, 0.12, 0.04])
        ax_skip = plt.axes([0.55, 0.05, 0.12, 0.04])
        
        self.btn_same = Button(ax_same, 'SAME SOURCE', color='lightgreen', hovercolor='green')
        self.btn_diff = Button(ax_diff, 'DIFFERENT SOURCE', color='lightcoral', hovercolor='red')
        self.btn_skip = Button(ax_skip, 'SKIP', color='lightyellow', hovercolor='gold')
        ax_remove = plt.axes([0.70, 0.05, 0.12, 0.04])
        self.btn_remove = Button(ax_remove, 'REMOVE', color='lightgray', hovercolor='gray')
        
        self.btn_same.on_clicked(lambda event: self.record_result('Same'))
        self.btn_diff.on_clicked(lambda event: self.record_result('Different'))
        self.btn_skip.on_clicked(lambda event: self.skip_source())
        self.btn_remove.on_clicked(lambda event: self.remove_source())

        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Display first source
        self.display_source(0)

    def _load_source_origins(self, catalog_csv_path):
        '''Load RADCAT source origins keyed by source ID.'''
        try:
            catalog_df = pd.read_csv(catalog_csv_path, index_col=0)
            catalog_df.columns = catalog_df.columns.str.strip()
        except Exception as e:
            print(f"Warning: could not read RADCAT catalog at {catalog_csv_path}: {e}")
            return {}

        if 'Catalog' not in catalog_df.columns:
            print(f"Warning: RADCAT catalog at {catalog_csv_path} does not contain a 'Catalog' column")
            return {}

        return {str(source_id): str(origin).strip() for source_id, origin in catalog_df['Catalog'].items()}

    def _get_source_origin(self, source):
        '''Return the origin catalog for a source.'''
        return self.source_origins.get(str(source), 'Unknown')

    def _is_lomorph(self, source):
        '''Check whether a source belongs to the LoMORPH catalog.'''
        return self._get_source_origin(source).lower() == 'lomorph'

    def _refresh_sources(self):
        '''Rebuild the combined queue so LoMORPH sources always remain at the end.'''
        self.sources = self.regular_sources + self.lomorph_sources
    
    def display_source(self, index):
        '''Display the current source'''
        if index >= len(self.sources):
            print("\nAll sources reviewed!")
            self.finalize_session(close_figure=True)
            return
        
        self.current_index = index
        source = self.sources[index]
        origin = self._get_source_origin(source)

        if origin.lower() == 'lomorph' and not self._lomorph_section_announced:
            print("\n--- Starting LoMORPH sources (reviewed at the end) ---")
            self._lomorph_section_announced = True
        
        # Load images
        first_file = os.path.join(self.first_folder, f"{source}.fits")
        lofar_file = os.path.join(self.lofar_folder, f"{source}.fits")
        
        try:
            first_data = fits.getdata(first_file, memmap=False)
            lofar_data = fits.getdata(lofar_file, memmap=False)
        except Exception as e:
            print(f"Error loading {source}: {e}")
            self.display_source(index + 1)
            return
        
        # Get display images with LoTSS mask applied to both
        first_clipped_img, lofar_clipped_img = get_display_images_lofar_mask(first_data, lofar_data)
        
        # Clear and update subplots
        for ax in [self.ax_first_before, self.ax_lofar_before, 
                   self.ax_first_after, self.ax_lofar_after]:
            ax.clear()
        
        # Normalize for display
        norm_first = Normalize(vmin=np.percentile(first_data, 5), 
                               vmax=np.percentile(first_data, 95))
        norm_lofar = Normalize(vmin=np.percentile(lofar_data, 5), 
                               vmax=np.percentile(lofar_data, 95))
        
        # Display images
        self.ax_lofar_before.imshow(lofar_data, cmap='viridis', norm=norm_lofar)
        self.ax_lofar_before.set_title('LoTSS (Before)', fontweight='bold')
        self.ax_lofar_before.axis('off')
        
        self.ax_first_before.imshow(first_data, cmap='viridis', norm=norm_first)
        self.ax_first_before.set_title('FIRST (Before)', fontweight='bold')
        self.ax_first_before.axis('off')
        
        self.ax_lofar_after.imshow(lofar_clipped_img, cmap='viridis', norm=norm_lofar)
        self.ax_lofar_after.set_title('LoTSS (After σ-clipping)', fontweight='bold')
        self.ax_lofar_after.axis('off')
        
        self.ax_first_after.imshow(first_clipped_img, cmap='viridis', norm=norm_first)
        self.ax_first_after.set_title('FIRST (After σ-clipping)', fontweight='bold')
        self.ax_first_after.axis('off')
        
        # Update title with progress
        total_reviewed = len(self.results)
        total_remaining = len(self.sources)
        self.fig.suptitle(f'Source {source} [{origin}] ({total_reviewed + 1}/{total_reviewed + total_remaining})', 
                         fontsize=14, fontweight='bold')
        
        plt.draw()
    
    def record_result(self, verdict):
        '''Record user's verdict and move to next source'''
        source = self.sources[self.current_index]
        origin = self._get_source_origin(source)
        result_row = {
            'source': source,
            'catalog': origin,
            'verdict': verdict
        }
        self.results.append(result_row)
        self.session_results.append(result_row)
        
        # Remove reviewed source from the list
        if source in self.regular_sources:
            self.regular_sources.remove(source)
        elif source in self.lomorph_sources:
            self.lomorph_sources.remove(source)
        self._refresh_sources()
        
        print(f"Source {source} [{origin}]: {verdict} | Progress: {len(self.results)}/{len(self.results) + len(self.sources)}")
        
        # Move to next source (don't increment since we popped element)
        self.display_source(self.current_index)

    def remove_source(self):
        '''Remove the current source from calculations and store it as an outtake.'''
        source = self.sources[self.current_index]
        origin = self._get_source_origin(source)
        outtake_row = {
            'source': source,
            'catalog': origin
        }
        self.outtakes.append(outtake_row)
        self.session_outtakes.append(outtake_row)

        if source in self.regular_sources:
            self.regular_sources.remove(source)
        elif source in self.lomorph_sources:
            self.lomorph_sources.remove(source)
        self._refresh_sources()

        print(f"Source {source} [{origin}]: REMOVED (added to outtakes)")

        # Move to next source without counting this one as a result.
        self.display_source(self.current_index)
    
    def skip_source(self):
        '''Skip current source and add it to the end of the list'''
        source = self.sources[self.current_index]
        origin = self._get_source_origin(source)
        # Move source to the end of its own queue so LoMORPH remains last overall.
        if source in self.regular_sources:
            self.regular_sources.remove(source)
            self.regular_sources.append(source)
        elif source in self.lomorph_sources:
            self.lomorph_sources.remove(source)
            self.lomorph_sources.append(source)
        self._refresh_sources()
        print(f"Source {source} [{origin}]: SKIPPED (moved to end of its queue)")
        
        # Display next source (don't increment current_index since we removed an element)
        self.display_source(self.current_index)
    
    def save_results(self):
        '''Save results to CSV'''
        if not self.session_results:
            print("No new results to save.")
            return
        
        df = pd.DataFrame(self.session_results)
        
        # Append to existing CSV if it exists
        if os.path.exists(self.results_csv_path):
            existing_df = pd.read_csv(self.results_csv_path, dtype={'source': str})
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(self.results_csv_path, index=False)
        print(f"\n✓ Results saved to: {self.results_csv_path}")
        
        # Print summary
        same_count = (df['verdict'] == 'Same').sum()
        diff_count = (df['verdict'] == 'Different').sum()
        print(f"\nSummary:")
        print(f"  Same Source: {same_count} ({100*same_count/len(df):.1f}%)")
        print(f"  Different Source: {diff_count} ({100*diff_count/len(df):.1f}%)")

    def save_outtakes(self):
        '''Save outtakes to CSV.'''
        if not self.session_outtakes:
            print("No new outtakes to save.")
            return

        df = pd.DataFrame(self.session_outtakes)

        if os.path.exists(self.outtakes_csv_path):
            existing_df = pd.read_csv(self.outtakes_csv_path, dtype={'source': str})
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(self.outtakes_csv_path, index=False)
        print(f"✓ Outtakes saved to: {self.outtakes_csv_path}")

    def finalize_session(self, close_figure=False):
        '''Persist the current session once, then close the figure if requested.'''
        if self._finalized:
            return

        self._finalized = True
        self.save_results()
        self.save_outtakes()

        if close_figure and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

    def on_close(self, event):
        '''Save any in-progress work if the user closes the window manually.'''
        if getattr(self, '_finalized', False):
            return
        self.finalize_session(close_figure=False)


# endregion

# region MAIN
def main():
    first_folder = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F/FIRST"
    lofar_folder = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F/LoTSS"
    results_csv_path = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping/source_comparison_results.csv"
    outtakes_csv_path = OUTTAKES_CSV_PATH
    
    tool = SourceComparisonTool(first_folder, lofar_folder, results_csv_path, outtakes_csv_path)
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()

# endregion
