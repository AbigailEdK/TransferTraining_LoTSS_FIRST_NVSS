# region ABOUT
# ===================================================================================================
# > This script provides an interactive visual comparison tool with AVERAGED MASKING.
#   Sigma clipping masks are independently derived for FIRST and LoTSS data, then averaged
#   and applied to both surveys. This approach balances the clipping criteria between both datasets.
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
from sigmaClipping import apply_averaged_mask_to_both, get_display_images_averaged_mask, pearson_correlation, hellinger_distance
# endregion

# region CLASS
class SourceComparisonToolAveragedMask:
    '''
    DESCRIPTION:
    -----------
        Interactive tool for comparing FIRST and LoTSS radio survey images before and after
        sigma clipping with AVERAGED masks. Two independent masks are computed, then averaged
        and applied to both images for balanced clipping.
    '''
    
    def __init__(self, first_folder, lofar_folder, results_csv_path):
        '''Initialize the comparison tool and load FITS files'''
        self.first_folder = first_folder
        self.lofar_folder = lofar_folder
        self.results_csv_path = results_csv_path
        self.current_index = 0
        self.results = []
        
        # Load file lists
        self.first_files = sorted([f for f in os.listdir(first_folder) if f.endswith('.fits')])
        self.lofar_files = sorted([f for f in os.listdir(lofar_folder) if f.endswith('.fits')])
        
        # Find sources that exist in both folders
        first_sources = set([f.replace('.fits', '') for f in self.first_files])
        lofar_sources = set([f.replace('.fits', '') for f in self.lofar_files])
        self.sources = sorted(list(first_sources & lofar_sources))
        
        print(f"Found {len(self.sources)} sources in both FIRST and LoTSS folders")
        print("Using AVERAGED mask approach (masks are averaged and applied to both surveys)")
        
        # Load existing results if any
        if os.path.exists(results_csv_path):
            existing_df = pd.read_csv(results_csv_path, dtype={'source': str})
            self.results = existing_df.to_dict('records')
            reviewed_sources = set([r['source'] for r in self.results])
            self.sources = [s for s in self.sources if s not in reviewed_sources]
            print(f"Loaded {len(self.results)} existing results. {len(self.sources)} sources remaining.")
        
        if len(self.sources) == 0:
            print("All sources have been reviewed!")
            return
        
        # Create figure
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('FITS File Comparison: LoTSS vs FIRST (Before/After Averaged Sigma Clipping)', 
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
        
        self.btn_same.on_clicked(lambda event: self.record_result('Same'))
        self.btn_diff.on_clicked(lambda event: self.record_result('Different'))
        self.btn_skip.on_clicked(lambda event: self.skip_source())
        
        # Display first source
        self.display_source(0)
    
    def display_source(self, index):
        '''Display the current source'''
        if index >= len(self.sources):
            print("\nAll sources reviewed!")
            plt.close(self.fig)
            self.save_results()
            return
        
        self.current_index = index
        source = self.sources[index]
        
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
        
        # Get display images with averaged mask applied to both
        first_clipped_img, lofar_clipped_img = get_display_images_averaged_mask(first_data, lofar_data)
        
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
        self.ax_lofar_after.set_title('LoTSS (After Averaged σ-clipping)', fontweight='bold')
        self.ax_lofar_after.axis('off')
        
        self.ax_first_after.imshow(first_clipped_img, cmap='viridis', norm=norm_first)
        self.ax_first_after.set_title('FIRST (After Averaged σ-clipping)', fontweight='bold')
        self.ax_first_after.axis('off')
        
        # Update title with progress
        total_reviewed = len(self.results)
        total_remaining = len(self.sources)
        self.fig.suptitle(f'Source {source} ({total_reviewed + 1}/{total_reviewed + total_remaining})', 
                         fontsize=14, fontweight='bold')
        
        plt.draw()
    
    def record_result(self, verdict):
        '''Record user's verdict and move to next source'''
        source = self.sources[self.current_index]
        self.results.append({
            'source': source,
            'verdict': verdict
        })
        
        # Remove reviewed source from the list
        self.sources.pop(self.current_index)
        
        print(f"Source {source}: {verdict} | Progress: {len(self.results)}/{len(self.results) + len(self.sources)}")
        
        # Move to next source (don't increment since we popped element)
        self.display_source(self.current_index)
    
    def skip_source(self):
        '''Skip current source and add it to the end of the list'''
        source = self.sources[self.current_index]
        # Move source to end of list
        self.sources.append(self.sources.pop(self.current_index))
        print(f"Source {source}: SKIPPED (moved to end of queue)")
        
        # Display next source (don't increment current_index since we removed an element)
        self.display_source(self.current_index)
    
    def save_results(self):
        '''Save results to CSV'''
        if not self.results:
            print("No results to save.")
            return
        
        df = pd.DataFrame(self.results)
        
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


# endregion

# region MAIN
def main():
    first_folder = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F/FIRST"
    lofar_folder = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F/LoTSS"
    results_csv_path = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/SigmaClip Checking/source_comparison_results_averaged_mask.csv"
    
    tool = SourceComparisonToolAveragedMask(first_folder, lofar_folder, results_csv_path)
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()

# endregion
