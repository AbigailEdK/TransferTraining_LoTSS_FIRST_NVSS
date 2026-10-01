# region ABOUT
# ===================================================================================================
# > Utility module for sigma clipping operations on FITS images.
#   Provides functions for masking, averaging masks, and computing correlation metrics.
# ===================================================================================================
# endregion

# region IMPORTS
import numpy as np
from scipy.stats import sigmaclip, pearsonr
from scipy.spatial.distance import jensenshannon
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
# endregion

# region FUNCTIONS
def get_lofar_mask(lofar_flat):
    '''
    ARGUMENTS:
    ----------
        > lofar_flat: numpy array, flattened LoTSS image data.
    
    RETURNS:
    --------        
        > lofar_mask: numpy array, boolean mask from sigma clipping.
    
    DESCRIPTION:
    ------------
        Applies sigma clipping (3-sigma) to LoTSS data and returns the mask
        of pixels that are outliers (sources).
    '''
    lofar_clipped, _, _ = sigmaclip(lofar_flat, low=3, high=3)
    return np.isin(lofar_flat, lofar_clipped)


def get_first_mask(first_flat):
    '''
    ARGUMENTS:
    ----------
        > first_flat: numpy array, flattened FIRST image data.
    
    RETURNS:
    --------        
        > first_mask: numpy array, boolean mask from sigma clipping.
    
    DESCRIPTION:
    ------------
        Applies sigma clipping (3-sigma) to FIRST data and returns the mask
        of pixels that are outliers (sources).
    '''
    first_clipped, _, _ = sigmaclip(first_flat, low=3, high=3)
    return np.isin(first_flat, first_clipped)


def average_masks(first_mask, lofar_mask):
    '''
    ARGUMENTS:
    ----------
        > first_mask: numpy array, boolean mask from FIRST.
        > lofar_mask: numpy array, boolean mask from LoTSS.
    
    RETURNS:
    --------        
        > averaged_mask: numpy array, boolean mask (threshold at 0.5).
    
    DESCRIPTION:
    ------------
        Averages two masks by converting to float, averaging, and thresholding at 0.5.
        A pixel is included if it's an outlier in at least one survey.
    '''
    first_float = first_mask.astype(float)
    lofar_float = lofar_mask.astype(float)
    averaged = (first_float + lofar_float) / 2.0
    return averaged >= 0.5


def apply_lofar_mask_to_both(first_data, lofar_data):
    '''
    ARGUMENTS:
    ----------
        > first_data: numpy array, 2D FIRST image.
        > lofar_data: numpy array, 2D LoTSS image.
    
    RETURNS:
    --------        
        > first_clipped: numpy array, FIRST data with LoTSS mask applied.
        > lofar_clipped: numpy array, LoTSS data with LoTSS mask applied.
        > num_pixels: int, number of pixels retained.
    
    DESCRIPTION:
    ------------
        Applies the LoTSS sigma-clipping mask to both FIRST and LoTSS images.
        Returns flattened arrays of pixels where sources are detected.
    '''
    lofar_flat = lofar_data.flatten()
    lofar_mask = get_lofar_mask(lofar_flat)
    
    first_clipped = first_data[lofar_mask.reshape(first_data.shape)]
    lofar_clipped = lofar_data[lofar_mask.reshape(lofar_data.shape)]
    
    return first_clipped, lofar_clipped, len(first_clipped)


def get_display_images_lofar_mask(first_data, lofar_data):
    '''
    ARGUMENTS:
    ----------
        > first_data: numpy array, 2D FIRST image.
        > lofar_data: numpy array, 2D LoTSS image.
    
    RETURNS:
    --------        
        > first_img: numpy array, FIRST image with background zeroed.
        > lofar_img: numpy array, LoTSS image with background zeroed.
    
    DESCRIPTION:
    ------------
        Creates display images by applying LoTSS mask to both surveys.
        Background pixels are set to zero, showing only sources.
    '''
    lofar_flat = lofar_data.flatten()
    lofar_mask = get_lofar_mask(lofar_flat)
    
    first_img = first_data.copy()
    first_img[lofar_mask.reshape(first_data.shape)] = 0
    
    lofar_img = lofar_data.copy()
    lofar_img[lofar_mask.reshape(lofar_data.shape)] = 0
    
    return first_img, lofar_img


def apply_averaged_mask_to_both(first_data, lofar_data):
    '''
    ARGUMENTS:
    ----------
        > first_data: numpy array, 2D FIRST image.
        > lofar_data: numpy array, 2D LoTSS image.
    
    RETURNS:
    --------        
        > first_clipped: numpy array, FIRST data with averaged mask applied.
        > lofar_clipped: numpy array, LoTSS data with averaged mask applied.
        > num_pixels: int, number of pixels retained.
    
    DESCRIPTION:
    ------------
        Computes independent masks for FIRST and LoTSS, averages them, and
        applies the averaged mask to both images.
        Returns flattened arrays of pixels where sources are detected.
    '''
    first_flat = first_data.flatten()
    lofar_flat = lofar_data.flatten()
    
    first_mask = get_first_mask(first_flat)
    lofar_mask = get_lofar_mask(lofar_flat)
    
    avg_mask = average_masks(first_mask, lofar_mask)
    
    first_clipped = first_data[avg_mask.reshape(first_data.shape)]
    lofar_clipped = lofar_data[avg_mask.reshape(lofar_data.shape)]
    
    return first_clipped, lofar_clipped, len(first_clipped)


def get_display_images_averaged_mask(first_data, lofar_data):
    '''
    ARGUMENTS:
    ----------
        > first_data: numpy array, 2D FIRST image.
        > lofar_data: numpy array, 2D LoTSS image.
    
    RETURNS:
    --------        
        > first_img: numpy array, FIRST image with background zeroed.
        > lofar_img: numpy array, LoTSS image with background zeroed.
    
    DESCRIPTION:
    ------------
        Creates display images by applying averaged mask to both surveys.
        Background pixels are set to zero, showing only sources.
    '''
    first_flat = first_data.flatten()
    lofar_flat = lofar_data.flatten()
    
    first_mask = get_first_mask(first_flat)
    lofar_mask = get_lofar_mask(lofar_flat)
    avg_mask = average_masks(first_mask, lofar_mask)
    
    first_img = first_data.copy()
    first_img[avg_mask.reshape(first_data.shape)] = 0
    
    lofar_img = lofar_data.copy()
    lofar_img[avg_mask.reshape(lofar_data.shape)] = 0
    
    return first_img, lofar_img


def pearson_correlation(X1, X2):
    '''
    ARGUMENTS:
    ----------
        > X1: numpy array, first set of values.
        > X2: numpy array, second set of values.
    
    RETURNS:
    --------        
        > pearson_corr: float, Pearson correlation coefficient.
        > pearson_pval: float, p-value associated with correlation.
    
    DESCRIPTION:
    ------------
        Calculates Pearson correlation coefficient and p-value between two arrays.
        Returns (NaN, NaN) if arrays have <2 values or zero variance.
    '''
    if len(X1) < 2 or len(X2) < 2:
        return np.nan, np.nan
    # Check for zero variance (constant arrays)
    if np.std(X1) == 0 or np.std(X2) == 0:
        return np.nan, np.nan
    pearson_corr, pearson_pval = pearsonr(X1, X2)
    return pearson_corr, pearson_pval


def hellinger_distance(X1, X2, bins=50):
    '''
    ARGUMENTS:
    ----------
        > X1: numpy array, first set of values.
        > X2: numpy array, second set of values.
        > bins: int, number of histogram bins.
    
    RETURNS:
    --------        
        > hellinger_dist: float, Hellinger distance between distributions.
    
    DESCRIPTION:
    ------------
        Calculates Hellinger distance between normalized histogram distributions.
        Returns NaN if arrays have <2 values or zero variance.
    '''
    if len(X1) < 2 or len(X2) < 2:
        return np.nan
    # Check for zero variance (constant arrays)
    if np.std(X1) == 0 or np.std(X2) == 0:
        return np.nan
    hist_X1, _ = np.histogram(X1, bins=bins, density=True)
    hist_X2, _ = np.histogram(X2, bins=bins, density=True)
    hellinger_dist = jensenshannon(hist_X1, hist_X2)
    return hellinger_dist


def load_results_from_csv(csv_path):
    '''
    ARGUMENTS:
    ----------
        > csv_path: str, the path to the CSV file to load.
    
    RETURNS:
    --------        
        > results: dict, containing lists of metrics for each source.
    
    DESCRIPTION:
    ------------
        This function loads the results from a CSV file and returns them as a dictionary.
        Handles both single mask format (pearson_corr, etc.) and dual mask format
        (pearson_corr_lofar_mask, pearson_corr_avg_mask, etc.).
        Automatically strips whitespace from column names.
    '''
    df = pd.read_csv(csv_path, dtype={'source': str})
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Check if this is the dual mask format (newer) or single mask format (older)
    if 'pearson_corr_lofar_mask' in df.columns:
        # Dual mask format - preserve all columns
        results = {
            'source': df['source'].tolist(),
            'pearson_corr_lofar_mask': df['pearson_corr_lofar_mask'].tolist(),
            'pearson_pval_lofar_mask': df['pearson_pval_lofar_mask'].tolist(),
            'hellinger_dist_lofar_mask': df['hellinger_dist_lofar_mask'].tolist(),
            'num_pixels_lofar_mask': df['num_pixels_lofar_mask'].tolist(),
            'pearson_corr_avg_mask': df['pearson_corr_avg_mask'].tolist(),
            'pearson_pval_avg_mask': df['pearson_pval_avg_mask'].tolist(),
            'hellinger_dist_avg_mask': df['hellinger_dist_avg_mask'].tolist(),
            'num_pixels_avg_mask': df['num_pixels_avg_mask'].tolist()
        }
    else:
        # Single mask format - original format
        results = {
            'source': df['source'].tolist(),
            'pearson_corr': df['pearson_corr'].tolist(),
            'pearson_pval': df['pearson_pval'].tolist(),
            'hellinger_dist': df['hellinger_dist'].tolist(),
            'num_pixels': df['num_pixels'].tolist()
        }
    return results


def save_results_to_csv(results, csv_path):
    '''
    ARGUMENTS:
    ----------
        > results: dict, containing lists of metrics for each source.
        > csv_path: str, the path to save the CSV file.
    
    RETURNS:
    --------        
        None
    
    DESCRIPTION:
    ------------
        This function saves the results dictionary to a CSV file.
    '''
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")


def get_FITS_files(first_folder, lofar_folder):
    '''
    ARGUMENTS:
    ----------
        > first_folder: str, path to FIRST FITS files directory.
        > lofar_folder: str, path to LoTSS FITS files directory.
    
    RETURNS:
    --------        
        > data_images: list of numpy arrays containing image data.
        > names: list of source names.
    
    DESCRIPTION:
    ------------
        This function loads FITS files from FIRST and LoTSS folders,
        stacks them into a single array, and returns the data with source names.
    '''
    from astropy.io import fits
    import os
    
    first_files = sorted([f for f in os.listdir(first_folder) if f.endswith('.fits')])
    lofar_files = sorted([f for f in os.listdir(lofar_folder) if f.endswith('.fits')])
    
    data_list = []
    names = []
    
    for first_file, lofar_file in zip(first_files, lofar_files):
        first_path = os.path.join(first_folder, first_file)
        lofar_path = os.path.join(lofar_folder, lofar_file)
        
        first_data = fits.getdata(first_path)
        lofar_data = fits.getdata(lofar_path)
        
        # Stack the two surveys: shape becomes (128, 128, 2)
        stacked = np.stack([first_data, lofar_data], axis=-1)
        data_list.append(stacked)
        names.append(first_file.replace('.fits', ''))
    
    # Stack all sources: shape becomes (num_sources, 128, 128, 2)
    all_data = np.stack(data_list, axis=0)
    
    return [all_data], names

def display_debug_masks(first_data, lofar_data, source_name):
    '''
    ARGUMENTS:
    ----------
        > first_data: numpy array, 2D FIRST image.
        > lofar_data: numpy array, 2D LoTSS image.
        > source_name: str, name of the source for display.
    
    RETURNS:
    --------        
        None (displays matplotlib figures)
    
    DESCRIPTION:
    ------------
        Displays before/after comparisons of mask application for debugging.
        Shows both LoTSS mask and averaged mask approaches.
    '''
    # ===== LoTSS Mask Approach =====
    fig_lofar, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig_lofar.suptitle(f'Debug: LoTSS Mask Application for {source_name}', fontsize=14, fontweight='bold')
    
    # Original images
    axes[0, 0].imshow(first_data, cmap='viridis')
    axes[0, 0].set_title('FIRST (Original)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    axes[0, 1].imshow(lofar_data, cmap='viridis')
    axes[0, 1].set_title('LoTSS (Original)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    
    # After LoTSS mask
    first_masked, lofar_masked = get_display_images_lofar_mask(first_data, lofar_data)
    
    axes[1, 0].imshow(first_masked, cmap='viridis')
    axes[1, 0].set_title('FIRST (LoTSS Mask Applied)')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    
    axes[1, 1].imshow(lofar_masked, cmap='viridis')
    axes[1, 1].set_title('LoTSS (LoTSS Mask Applied)')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    
    fig_lofar.tight_layout()
    fig_lofar.show()
    
    # ===== Averaged Mask Approach =====
    fig_avg, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig_avg.suptitle(f'Debug: Averaged Mask Application for {source_name}', fontsize=14, fontweight='bold')
    
    # Original images
    axes[0, 0].imshow(first_data, cmap='viridis')
    axes[0, 0].set_title('FIRST (Original)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    axes[0, 1].imshow(lofar_data, cmap='viridis')
    axes[0, 1].set_title('LoTSS (Original)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    
    # After averaged mask
    first_masked_avg, lofar_masked_avg = get_display_images_averaged_mask(first_data, lofar_data)
    
    axes[1, 0].imshow(first_masked_avg, cmap='viridis')
    axes[1, 0].set_title('FIRST (Averaged Mask Applied)')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    
    axes[1, 1].imshow(lofar_masked_avg, cmap='viridis')
    axes[1, 1].set_title('LoTSS (Averaged Mask Applied)')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    
    fig_avg.tight_layout()
    fig_avg.show()
    
    print(f"\n✓ Debug visualizations displayed for source: {source_name}")

    fig_lofar.savefig(f"/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping/Results/debug_masks_lofar_{source_name}.png")
    fig_avg.savefig(f"/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping/Results/debug_masks_avg_{source_name}.png")
    print(f"✓ Debug visualizations saved for source: {source_name}")
# endregion

def create_plots(results, show_figs=False):
    '''
    ARGUMENTS:
    ----------
        > results: dict, containing lists of metrics for each source.
    
    RETURNS:
    --------        
        None (displays plots)
    
    DESCRIPTION:
    ------------
        This function creates side-by-side visualizations comparing
        LoTSS mask approach vs averaged mask approach for all metrics.
    '''
    print("\nCreating visualizations comparing LoTSS mask vs Averaged mask approaches...")

    def sort_metric_desc(metric_values, source_labels):
        order = np.argsort(metric_values)[::-1]
        sorted_values = np.asarray(metric_values)[order]
        sorted_labels = [source_labels[i] for i in order]
        return sorted_values, sorted_labels

    sources_short = [s[-4:] for s in results['source']]  # Last 4 chars of source ID
    pearson_corr_lofar_sorted, sources_lofar = sort_metric_desc(results['pearson_corr_lofar_mask'], sources_short)
    pearson_corr_avg_sorted, sources_avg = sort_metric_desc(results['pearson_corr_avg_mask'], sources_short)
    hellinger_lofar_sorted, hellinger_lofar_labels = sort_metric_desc(results['hellinger_dist_lofar_mask'], sources_short)
    hellinger_avg_sorted, hellinger_avg_labels = sort_metric_desc(results['hellinger_dist_avg_mask'], sources_short)
    pval_lofar_sorted, pval_lofar_labels = sort_metric_desc(results['pearson_pval_lofar_mask'], sources_short)
    pval_avg_sorted, pval_avg_labels = sort_metric_desc(results['pearson_pval_avg_mask'], sources_short)
    pixels_lofar_sorted, pixels_lofar_labels = sort_metric_desc(results['num_pixels_lofar_mask'], sources_short)
    pixels_avg_sorted, pixels_avg_labels = sort_metric_desc(results['num_pixels_avg_mask'], sources_short)
    
    # Plot 1: Pearson Correlation (LoTSS mask vs Averaged mask)
    fig1, (ax1l, ax1r) = plt.subplots(1, 2, figsize=(16, 6))
    
    # LoTSS mask
    ax1l.bar(range(len(sources_lofar)), pearson_corr_lofar_sorted, color='steelblue', alpha=0.7)
    ax1l.axhline(y=np.mean(pearson_corr_lofar_sorted), color='red', linestyle='--', 
                 label=f"Mean: {np.mean(pearson_corr_lofar_sorted):.3f}")
    ax1l.set_xlabel('Source', fontsize=11)
    ax1l.set_ylabel('Pearson Correlation', fontsize=11)
    ax1l.set_title('LoTSS Mask Approach', fontsize=12, fontweight='bold')
    ax1l.set_xticks(range(len(sources_lofar)))
    ax1l.set_xticklabels(sources_lofar, rotation=45, fontsize=9)
    ax1l.legend(fontsize=10)
    ax1l.grid(axis='y', alpha=0.3)
    
    # Averaged mask
    ax1r.bar(range(len(sources_avg)), pearson_corr_avg_sorted, color='darkseagreen', alpha=0.7)
    ax1r.axhline(y=np.mean(pearson_corr_avg_sorted), color='darkgreen', linestyle='--',
                 label=f"Mean: {np.mean(pearson_corr_avg_sorted):.3f}")
    ax1r.set_xlabel('Source', fontsize=11)
    ax1r.set_ylabel('Pearson Correlation', fontsize=11)
    ax1r.set_title('Averaged Mask Approach', fontsize=12, fontweight='bold')
    ax1r.set_xticks(range(len(sources_avg)))
    ax1r.set_xticklabels(sources_avg, rotation=45, fontsize=9)
    ax1r.legend(fontsize=10)
    ax1r.grid(axis='y', alpha=0.3)
    
    fig1.suptitle('Pearson Correlation Coefficient (FIRST vs LOFAR)', fontsize=13, fontweight='bold', y=1.02)
    fig1.tight_layout()
    
    # Plot 2: Hellinger Distance (LoTSS mask vs Averaged mask)
    fig2, (ax2l, ax2r) = plt.subplots(1, 2, figsize=(16, 6))
    
    # LoTSS mask
    ax2l.bar(range(len(hellinger_lofar_labels)), hellinger_lofar_sorted, color='coral', alpha=0.7)
    ax2l.axhline(y=np.mean(hellinger_lofar_sorted), color='darkred', linestyle='--',
                 label=f"Mean: {np.mean(hellinger_lofar_sorted):.3f}")
    ax2l.set_xlabel('Source', fontsize=11)
    ax2l.set_ylabel('Hellinger Distance', fontsize=11)
    ax2l.set_title('LoTSS Mask Approach', fontsize=12, fontweight='bold')
    ax2l.set_xticks(range(len(hellinger_lofar_labels)))
    ax2l.set_xticklabels(hellinger_lofar_labels, rotation=45, fontsize=9)
    ax2l.legend(fontsize=10)
    ax2l.grid(axis='y', alpha=0.3)
    
    # Averaged mask
    ax2r.bar(range(len(hellinger_avg_labels)), hellinger_avg_sorted, color='lightsalmon', alpha=0.7)
    ax2r.axhline(y=np.mean(hellinger_avg_sorted), color='orangered', linestyle='--',
                 label=f"Mean: {np.mean(hellinger_avg_sorted):.3f}")
    ax2r.set_xlabel('Source', fontsize=11)
    ax2r.set_ylabel('Hellinger Distance', fontsize=11)
    ax2r.set_title('Averaged Mask Approach', fontsize=12, fontweight='bold')
    ax2r.set_xticks(range(len(hellinger_avg_labels)))
    ax2r.set_xticklabels(hellinger_avg_labels, rotation=45, fontsize=9)
    ax2r.legend(fontsize=10)
    ax2r.grid(axis='y', alpha=0.3)
    
    fig2.suptitle('Hellinger Distance (FIRST vs LOFAR)', fontsize=13, fontweight='bold', y=1.02)
    fig2.tight_layout()
    
    # Plot 3: p-values (log scale, LoTSS vs Averaged)
    fig3, (ax3l, ax3r) = plt.subplots(1, 2, figsize=(16, 6))
    
    # LoTSS mask
    ax3l.semilogy(range(len(pval_lofar_labels)), pval_lofar_sorted, marker='o', 
                  color='green', alpha=0.7, linestyle='-', linewidth=2, markersize=6)
    ax3l.axhline(y=0.05, color='red', linestyle='--', label='p=0.05 threshold', linewidth=2)
    ax3l.set_xlabel('Source', fontsize=11)
    ax3l.set_ylabel('p-value (log scale)', fontsize=11)
    ax3l.set_title('LoTSS Mask Approach', fontsize=12, fontweight='bold')
    ax3l.set_xticks(range(len(pval_lofar_labels)))
    ax3l.set_xticklabels(pval_lofar_labels, rotation=45, fontsize=9)
    ax3l.legend(fontsize=10)
    ax3l.grid(alpha=0.3)
    
    # Averaged mask
    ax3r.semilogy(range(len(pval_avg_labels)), pval_avg_sorted, marker='s',
                  color='purple', alpha=0.7, linestyle='-', linewidth=2, markersize=6)
    ax3r.axhline(y=0.05, color='red', linestyle='--', label='p=0.05 threshold', linewidth=2)
    ax3r.set_xlabel('Source', fontsize=11)
    ax3r.set_ylabel('p-value (log scale)', fontsize=11)
    ax3r.set_title('Averaged Mask Approach', fontsize=12, fontweight='bold')
    ax3r.set_xticks(range(len(pval_avg_labels)))
    ax3r.set_xticklabels(pval_avg_labels, rotation=45, fontsize=9)
    ax3r.legend(fontsize=10)
    ax3r.grid(alpha=0.3)
    
    fig3.suptitle('Pearson Correlation p-values', fontsize=13, fontweight='bold', y=1.02)
    fig3.tight_layout()
    
    # Plot 4: Number of pixels (LoTSS vs Averaged)
    fig4, (ax4l, ax4r) = plt.subplots(1, 2, figsize=(16, 6))
    
    # LoTSS mask
    ax4l.bar(range(len(pixels_lofar_labels)), pixels_lofar_sorted, color='mediumpurple', alpha=0.7)
    ax4l.axhline(y=np.mean(pixels_lofar_sorted), color='darkviolet', linestyle='--',
                 label=f"Mean: {np.mean(pixels_lofar_sorted):.0f}")
    ax4l.set_xlabel('Source', fontsize=11)
    ax4l.set_ylabel('Number of Pixels', fontsize=11)
    ax4l.set_title('LoTSS Mask Approach', fontsize=12, fontweight='bold')
    ax4l.set_xticks(range(len(pixels_lofar_labels)))
    ax4l.set_xticklabels(pixels_lofar_labels, rotation=45, fontsize=9)
    ax4l.legend(fontsize=10)
    ax4l.grid(axis='y', alpha=0.3)
    
    # Averaged mask
    ax4r.bar(range(len(pixels_avg_labels)), pixels_avg_sorted, color='plum', alpha=0.7)
    ax4r.axhline(y=np.mean(pixels_avg_sorted), color='indigo', linestyle='--',
                 label=f"Mean: {np.mean(pixels_avg_sorted):.0f}")
    ax4r.set_xlabel('Source', fontsize=11)
    ax4r.set_ylabel('Number of Pixels', fontsize=11)
    ax4r.set_title('Averaged Mask Approach', fontsize=12, fontweight='bold')
    ax4r.set_xticks(range(len(pixels_avg_labels)))
    ax4r.set_xticklabels(pixels_avg_labels, rotation=45, fontsize=9)
    ax4r.legend(fontsize=10)
    ax4r.grid(axis='y', alpha=0.3)
    
    fig4.suptitle('Pixels Retained After Sigma Clipping', fontsize=13, fontweight='bold', y=1.02)
    fig4.tight_layout()

    fig1.savefig("/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping/Results/pearson_correlation.png")
    print(f"✓ Saved Pearson correlation plot.")
    fig2.savefig("/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping/Results/hellinger_distance.png")
    print(f"✓ Saved Hellinger distance plot.")
    fig3.savefig("/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping/Results/p_values.png")
    print(f"✓ Saved p-value plot.")
    fig4.savefig("/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping/Results/num_pixels.png")
    print(f"✓ Saved number of pixels plot.")

    # Display only if requested
    if show_figs:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
        print(f"\n✓ Displayed 4 comparison plot windows (LoTSS mask vs Averaged mask)")
    else:
        # Close figures to prevent automatic display
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)

def run_analysis(csv_output_path, debug=False):
    '''Run the full sigma clipping analysis on all sources'''
    print("\nProcessing data...")
    # Load data from FIRST and LOFAR folders
    first_folder = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F/FIRST"
    lofar_folder = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Data/RADCAT_F/LoTSS"
    
    data_images, names = get_FITS_files(first_folder, lofar_folder)
    
    # Extract FIRST and LOFAR channels
    first_images = data_images[0][:,:,:,0]  # Shape: (num_sources, 128, 128)
    lofar_images = data_images[0][:,:,:,1]  # Shape: (num_sources, 128, 128)
    
    # Collect results for plotting
    results = {
        'source': [],
        'pearson_corr_lofar_mask': [],
        'pearson_pval_lofar_mask': [],
        'hellinger_dist_lofar_mask': [],
        'num_pixels_lofar_mask': [],
        'pearson_corr_avg_mask': [],
        'pearson_pval_avg_mask': [],
        'hellinger_dist_avg_mask': [],
        'num_pixels_avg_mask': []
    }

    if debug:
            print("\n" + "="*80)
            print("DEBUG MODE: Displaying mask application for first source")
            print("="*80)
            first_img_2d = first_images[0]
            lofar_img_2d = lofar_images[0]
            display_debug_masks(first_img_2d, lofar_img_2d, names[0])
            print("="*80 + "\n")
    else:
    # Process each source
        for i, source in enumerate(names):
            first_img = first_images[i]
            lofar_img = lofar_images[i]
            
            # Flatten for masking
            first_flat = first_img.flatten()
            lofar_flat = lofar_img.flatten()
            
            # Apply sigma clipping to get LOFAR mask
            lofar_clipped, _, _ = sigmaclip(lofar_flat, low=3, high=3)
            lofar_mask = np.isin(lofar_flat, lofar_clipped)
            
            # Apply sigma clipping to get FIRST mask
            first_clipped, _, _ = sigmaclip(first_flat, low=3, high=3)
            first_mask = np.isin(first_flat, first_clipped)
            
            # ===== LoTSS Mask Approach =====
            lofar_mask_reshaped = lofar_mask.reshape(first_img.shape)
            first_img_lofar_mask = first_img[~lofar_mask_reshaped]
            lofar_img_lofar_mask = lofar_img[~lofar_mask_reshaped]
            
            if len(first_img_lofar_mask) > 0:
                pearson_corr_lofar, pearson_pval_lofar = pearson_correlation(first_img_lofar_mask, lofar_img_lofar_mask)
                hellinger_dist_lofar = hellinger_distance(first_img_lofar_mask, lofar_img_lofar_mask)
                num_pixels_lofar = len(first_img_lofar_mask)
            else:
                pearson_corr_lofar, pearson_pval_lofar = np.nan, np.nan
                hellinger_dist_lofar = np.nan
                num_pixels_lofar = 0
            
            # ===== Averaged Mask Approach =====
            avg_mask = average_masks(first_mask, lofar_mask)
            avg_mask_reshaped = avg_mask.reshape(first_img.shape)
            first_img_avg_mask = first_img[~avg_mask_reshaped]
            lofar_img_avg_mask = lofar_img[~avg_mask_reshaped]
            
            if len(first_img_avg_mask) > 0:
                pearson_corr_avg, pearson_pval_avg = pearson_correlation(first_img_avg_mask, lofar_img_avg_mask)
                hellinger_dist_avg = hellinger_distance(first_img_avg_mask, lofar_img_avg_mask)
                num_pixels_avg = len(first_img_avg_mask)
            else:
                pearson_corr_avg, pearson_pval_avg = np.nan, np.nan
                hellinger_dist_avg = np.nan
                num_pixels_avg = 0
            
            # Store results
            results['source'].append(source)
            results['pearson_corr_lofar_mask'].append(pearson_corr_lofar)
            results['pearson_pval_lofar_mask'].append(pearson_pval_lofar)
            results['hellinger_dist_lofar_mask'].append(hellinger_dist_lofar)
            results['num_pixels_lofar_mask'].append(num_pixels_lofar)
            results['pearson_corr_avg_mask'].append(pearson_corr_avg)
            results['pearson_pval_avg_mask'].append(pearson_pval_avg)
            results['hellinger_dist_avg_mask'].append(hellinger_dist_avg)
            results['num_pixels_avg_mask'].append(num_pixels_avg)
            
            # Display progress
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(names)} sources...")
        
        # Save results to CSV
        save_results_to_csv(results, csv_output_path)
        return results

def display_summary(results, show_figs=False):
    '''Display final summary statistics'''
    df = pd.DataFrame({
        'source': results['source'],
        'pearson_corr_lofar_mask': results['pearson_corr_lofar_mask'],
        'pearson_corr_avg_mask': results['pearson_corr_avg_mask']
    })
    
    # Remove NaN values for summary
    df_lofar = df.dropna(subset=['pearson_corr_lofar_mask'])
    df_avg = df.dropna(subset=['pearson_corr_avg_mask'])
    
    print("\n" + "="*80)
    print("FINAL SUMMARY: FIRST vs LoTSS CORRELATION")
    print("="*80)
    
    print("\n--- LoTSS MASK APPROACH ---")
    print(f"Mean Pearson Correlation: {df_lofar['pearson_corr_lofar_mask'].mean():.6f}")
    print(f"Median Pearson Correlation: {df_lofar['pearson_corr_lofar_mask'].median():.6f}")
    print(f"Std Dev: {df_lofar['pearson_corr_lofar_mask'].std():.6f}")
    
    # Interpretation
    mean_corr_lofar = df_lofar['pearson_corr_lofar_mask'].mean()
    if mean_corr_lofar < 0.1:
        interp_lofar = "Sources appear VERY DIFFERENT between surveys (weak correlation)"
    elif mean_corr_lofar < 0.3:
        interp_lofar = "Sources appear SOMEWHAT DIFFERENT between surveys (weak-moderate correlation)"
    elif mean_corr_lofar < 0.5:
        interp_lofar = "Sources appear MODERATELY SIMILAR between surveys (moderate correlation)"
    else:
        interp_lofar = "Sources appear VERY SIMILAR between surveys (strong correlation)"
    print(f"Interpretation: {interp_lofar}")
    
    print("\n--- AVERAGED MASK APPROACH ---")
    print(f"Mean Pearson Correlation: {df_avg['pearson_corr_avg_mask'].mean():.6f}")
    print(f"Median Pearson Correlation: {df_avg['pearson_corr_avg_mask'].median():.6f}")
    print(f"Std Dev: {df_avg['pearson_corr_avg_mask'].std():.6f}")
    
    # Interpretation
    mean_corr_avg = df_avg['pearson_corr_avg_mask'].mean()
    if mean_corr_avg < 0.1:
        interp_avg = "Sources appear VERY DIFFERENT between surveys (weak correlation)"
    elif mean_corr_avg < 0.3:
        interp_avg = "Sources appear SOMEWHAT DIFFERENT between surveys (weak-moderate correlation)"
    elif mean_corr_avg < 0.5:
        interp_avg = "Sources appear MODERATELY SIMILAR between surveys (moderate correlation)"
    else:
        interp_avg = "Sources appear VERY SIMILAR between surveys (strong correlation)"
    print(f"Interpretation: {interp_avg}")
    
    print("="*80 + "\n")
    
    # Create visualizations
    create_plots(results, show_figs=show_figs)

def display_plots(results_dir="/home/abigaildeklerk/Desktop/DeKlerk_Models/Code/Sigma Clipping/Results"):
    '''
    ARGUMENTS:
    ----------
        > results_dir: str, path to the Results folder containing saved plot images.
    
    RETURNS:
    --------        
        None (displays plots)
    
    DESCRIPTION:
    ------------
        Finds and displays the four saved plot images in a 2x2 grid.
        Looks for: pearson_correlation.png, hellinger_distance.png, 
        p_values.png, and num_pixels.png
    '''
    plot_files = [
        'pearson_correlation.png',
        'hellinger_distance.png',
        'p_values.png',
        'num_pixels.png'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Analysis Results - Comparison of Masking Approaches', fontsize=16, fontweight='bold', y=0.995)
    
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for plot_file, (row, col) in zip(plot_files, positions):
        file_path = os.path.join(results_dir, plot_file)
        
        if os.path.exists(file_path):
            try:
                img = Image.open(file_path)
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error loading:\n{plot_file}\n\n{str(e)}',
                                   ha='center', va='center', fontsize=10, color='red')
                axes[row, col].axis('off')
        else:
            axes[row, col].text(0.5, 0.5, f'File not found:\n{plot_file}',
                               ha='center', va='center', fontsize=10, color='orange')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"✓ Displayed plots from {results_dir}")
    plt.savefig(os.path.join(results_dir, "combined_results_plots.png"))
    print(f"✓ Saved combined plot image to {results_dir}/combined_results_plots.png")
# endregion