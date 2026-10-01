import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, recall_score
import matplotlib.gridspec as gridspec

# --- CONFIGURATION ---
HOME_DIR = os.path.expanduser("~")
PROJECT_ROOT = os.path.join(HOME_DIR, "Desktop", "DeKlerk_Models", "Training")
RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, "Results")
CLASSES = ["FRI", "FRII", "COMPACT"]
DATARESULTS_DIR = os.path.join(PROJECT_ROOT, "DATARESULTS_GENERALIZATION")

EXPERIMENTS = [f"Experiment_{i}" for i in range(1, 6)]

plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({'font.size': 12, 'figure.facecolor': 'white'})

# Mapping folder substrings to readable labels
TEST_LABELS = {
    "FIRST": "FIRST", "LOFAR": "LOFAR", "NVSS": "NVSS", "RADCAT": "RADCAT",
    "FREP": "F-Repeated", "FZER": "F-Zeroed", "FNSE": "F-Noise",
    "LREP": "L-Repeated", "LZER": "L-Zeroed", "LNSE": "L-Noise",
    "NREP": "N-Repeated", "NZER": "N-Zeroed", "NNSE": "N-Noise"
}

def clean_dir():
    if os.path.exists(DATARESULTS_DIR):
        shutil.rmtree(DATARESULTS_DIR)
    os.makedirs(DATARESULTS_DIR)

# --- ANALYTICAL FUNCTIONS ---
def save_generalization_heatmap(df, suffix=""):
    core_surveys = ["FIRST", "LOFAR", "NVSS", "RADCAT"]
    subset = df[df['Dataset'].isin(core_surveys) & df['Model'].isin(core_surveys)]
    if subset.empty: 
        print("[WARN] No core survey data found for heatmap")
        return
    
    # Get unique epochs and sort them
    epochs = sorted(subset['Epochs'].unique())
    if len(epochs) == 0:
        print("[WARN] No epochs found in subset")
        return
    
    # For each metric, create a heatmap across epochs
    metrics = [("weighted_f1", "Weighted F1"), ("accuracy", "Accuracy"), ("weighted_recall", "Weighted Recall")]
    for metric_key, metric_label in metrics:
        fig, axes = plt.subplots(1, len(epochs), figsize=(6*len(epochs), 5.5))
        if len(epochs) == 1:
            axes = [axes]

        for idx, epoch in enumerate(epochs):
            epoch_data = subset[subset['Epochs'] == epoch]
            pivot = epoch_data.pivot_table(index="Model", columns="Dataset", values=metric_key, aggfunc='mean')
            pivot = pivot.reindex(index=core_surveys, columns=core_surveys)

            sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f", ax=axes[idx], 
                        cbar_kws={'label': metric_label}, vmin=0, vmax=1)
            axes[idx].set_title(f"{epoch} Epochs", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Dataset", fontsize=10)
            axes[idx].set_ylabel("Model", fontsize=10)

        fig.suptitle(f"Generalization Matrix ({metric_label}): {suffix}", fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        fname = f"01_heatmap_{metric_key}_{suffix.replace(' ', '_')}.png"
        plt.savefig(os.path.join(DATARESULTS_DIR, fname), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved generalization heatmap ({metric_label}) ({suffix}) with {len(epochs)} epoch subplot(s)")

def save_radcat_strategy_matrix(df, suffix=""):
    rad_data = df[df['Model'] == 'RADCAT'].copy()
    # Filter only for the strategy-based datasets (the ones with hyphens)
    strategies = ["F-Repeated", "F-Zeroed", "F-Noise", "L-Repeated", "L-Zeroed", "L-Noise", "N-Repeated", "N-Zeroed", "N-Noise"]
    subset = rad_data[rad_data['Dataset'].isin(strategies)].copy()
    
    if subset.empty: 
        print("[DEBUG] No RADCAT strategy data found for matrix.")
        return
    
    # Get unique epochs and sort them
    epochs = sorted(subset['Epochs'].unique())
    if len(epochs) == 0:
        print("[WARN] No epochs found in RADCAT subset")
        return
    
    # Create heatmaps for multiple metrics
    metrics = [("weighted_f1", "Weighted F1"), ("accuracy", "Accuracy"), ("weighted_recall", "Weighted Recall")]
    for metric_key, metric_label in metrics:
        fig, axes = plt.subplots(1, len(epochs), figsize=(8*len(epochs), 6))
        if len(epochs) == 1:
            axes = [axes]

        for idx, epoch in enumerate(epochs):
            epoch_subset = subset[subset['Epochs'] == epoch].copy()
            epoch_subset['Source'] = epoch_subset['Dataset'].apply(lambda x: x.split('-')[0] if '-' in x else x)
            epoch_subset['Method'] = epoch_subset['Dataset'].apply(lambda x: x.split('-')[1] if '-' in x else 'Native')

            pivot = epoch_subset.pivot_table(index="Source", columns="Method", values=metric_key, aggfunc='mean')

            if not pivot.empty:
                sns.heatmap(pivot, annot=True, cmap="rocket_r", fmt=".3f", ax=axes[idx], 
                            cbar_kws={'label': metric_label}, vmin=0, vmax=1)
                axes[idx].set_title(f"{epoch} Epochs", fontsize=12, fontweight='bold')
                axes[idx].set_xlabel("Padding Method", fontsize=10)
                axes[idx].set_ylabel("Input Survey", fontsize=10)

        fig.suptitle(f"RADCAT Robustness ({metric_label}): Input Survey vs. Padding Method ({suffix})", fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        fname = f"02_radcat_strategy_{metric_key}_{suffix.replace(' ', '_')}.png"
        plt.savefig(os.path.join(DATARESULTS_DIR, fname), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved RADCAT strategy matrix ({metric_label}) ({suffix}) with {len(epochs)} epoch subplot(s)")

def save_hyperparam_analysis(df, suffix=""):
    """Plot training epochs vs. weighted F1 performance, aggregated by model."""
    # Group by Model and Epochs, averaging F1 scores across all datasets and regularization constants
    epoch_stats = df.groupby(['Model', 'Epochs'])['weighted_f1'].agg(['mean', 'std', 'count']).reset_index()
    
    print(f"\n[DEBUG] Epoch stats:\n{epoch_stats}")
    print(f"[DEBUG] Unique epochs: {sorted(epoch_stats['Epochs'].unique())}")
    
    if epoch_stats.empty:
        print("[WARN] No epoch data to plot")
        return
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=epoch_stats, x='Epochs', y='mean', hue='Model', marker='o', markersize=8, linewidth=2.5)
    
    # Add error bars for standard deviation
    for model in epoch_stats['Model'].unique():
        model_data = epoch_stats[epoch_stats['Model'] == model].sort_values('Epochs')
        plt.errorbar(model_data['Epochs'], model_data['mean'], yerr=model_data['std'], 
                     fmt='none', capsize=5, alpha=0.3, capthick=2, label='_nolegend_')
    
    plt.title(f"Model Convergence: Impact of Training Epochs on Generalization ({suffix})", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Weighted F1-Score", fontsize=12)
    plt.xlabel("Training Epochs", fontsize=12)
    plt.xticks(sorted(epoch_stats['Epochs'].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model Type", fontsize=10, title_fontsize=11, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(DATARESULTS_DIR, f"03_epoch_trends_{suffix.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[INFO] Epoch trends plot saved.")
    print(f"Data summary:\n{epoch_stats.to_string(index=False)}")

def save_radcat_strategy_global_average(df):
    """Generate global RADCAT strategy matrix across all experiments."""
    rad_data = df[df['Model'] == 'RADCAT'].copy()
    strategies = ["F-Repeated", "F-Zeroed", "F-Noise", "L-Repeated", "L-Zeroed", "L-Noise", "N-Repeated", "N-Zeroed", "N-Noise"]
    subset = rad_data[rad_data['Dataset'].isin(strategies)].copy()
    
    if subset.empty:
        print("[DEBUG] No RADCAT strategy data found for global average matrix.")
        return
    
    epochs = sorted(subset['Epochs'].unique())
    if len(epochs) == 0:
        print("[WARN] No epochs found in RADCAT subset")
        return
    
    metrics = [("weighted_f1", "Weighted F1"), ("accuracy", "Accuracy"), ("weighted_recall", "Weighted Recall")]
    for metric_key, metric_label in metrics:
        fig, axes = plt.subplots(1, len(epochs), figsize=(8*len(epochs), 6))
        if len(epochs) == 1:
            axes = [axes]

        for idx, epoch in enumerate(epochs):
            epoch_subset = subset[subset['Epochs'] == epoch].copy()
            epoch_subset['Source'] = epoch_subset['Dataset'].apply(lambda x: x.split('-')[0] if '-' in x else x)
            epoch_subset['Method'] = epoch_subset['Dataset'].apply(lambda x: x.split('-')[1] if '-' in x else 'Native')
            pivot = epoch_subset.pivot_table(index="Source", columns="Method", values=metric_key, aggfunc='mean')

            if not pivot.empty:
                sns.heatmap(pivot, annot=True, cmap="rocket_r", fmt=".3f", ax=axes[idx], 
                            cbar_kws={'label': metric_label}, vmin=0, vmax=1)
                axes[idx].set_title(f"{epoch} Epochs", fontsize=12, fontweight='bold')
                axes[idx].set_xlabel("Padding Method", fontsize=10)
                axes[idx].set_ylabel("Input Survey", fontsize=10)

        fig.suptitle(f"RADCAT Robustness ({metric_label}): Input Survey vs. Padding Method (Global Average)", fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        fname = f"02_radcat_strategy_Global_Average_{metric_key}.png"
        plt.savefig(os.path.join(DATARESULTS_DIR, fname), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved global RADCAT strategy matrix ({metric_label}) with {len(epochs)} epoch subplot(s)")

def save_hyperparam_analysis_global_average(df):
    """Generate global epoch trends across all experiments."""
    epoch_stats = df.groupby(['Model', 'Epochs'])['weighted_f1'].agg(['mean', 'std', 'count']).reset_index()
    
    print(f"\n[DEBUG] Global epoch stats:\n{epoch_stats}")
    print(f"[DEBUG] Unique epochs: {sorted(epoch_stats['Epochs'].unique())}")
    
    if epoch_stats.empty:
        print("[WARN] No epoch data to plot for global average")
        return
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=epoch_stats, x='Epochs', y='mean', hue='Model', marker='o', markersize=8, linewidth=2.5)
    
    for model in epoch_stats['Model'].unique():
        model_data = epoch_stats[epoch_stats['Model'] == model].sort_values('Epochs')
        plt.errorbar(model_data['Epochs'], model_data['mean'], yerr=model_data['std'], 
                     fmt='none', capsize=5, alpha=0.3, capthick=2, label='_nolegend_')
    
    plt.title(f"Model Convergence: Impact of Training Epochs on Generalization (Global Average)", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Weighted F1-Score", fontsize=12)
    plt.xlabel("Training Epochs", fontsize=12)
    plt.xticks(sorted(epoch_stats['Epochs'].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model Type", fontsize=10, title_fontsize=11, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(DATARESULTS_DIR, f"03_epoch_trends_Global_Average.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[INFO] Global epoch trends plot saved.")
    print(f"Data summary:\n{epoch_stats.to_string(index=False)}")

# --- MAIN ENGINE ---

def main():
    clean_dir()
    all_results = []
    
    # Create mapping from class labels to indices
    CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(CLASSES)}
    
    # Regex for float/int extraction
    e_regex = re.compile(r'_(\d+)e_')
    l_regex = re.compile(r'_([\d\.]+)l_')
    r_regex = re.compile(r'_([\d\.]+)r')
    
    for exp in EXPERIMENTS:
        test_path = os.path.join(RESULTS_BASE_DIR, exp, "Testing")
        if not os.path.exists(test_path):
            print(f"[SKIP] {exp} Testing folder not found.")
            continue
        
        folders = [f for f in os.listdir(test_path) if "_set" in f]
        print(f"[INFO] Scanning {exp}: found {len(folders)} valid folders.")

        for folder in folders:
            folder_path = os.path.join(test_path, folder)
            try:
                # 1. Parsing with underscores
                parts = folder.split('_')
                raw_model = parts[0].replace('model', '')
                # Specifically grab the part containing 'set'
                raw_dataset = [p.replace('set', '') for p in parts if 'set' in p][0]
                
                # 2. Extract Hyperparams
                epochs = int(e_regex.search(folder).group(1))
                l_rate = float(l_regex.search(folder).group(1))
                r_const = float(r_regex.search(folder).group(1))
                
                model_label = TEST_LABELS.get(raw_model, raw_model)
                data_label = TEST_LABELS.get(raw_dataset, raw_dataset)
                
                # 3. Load & Score
                y_pred_path = os.path.join(folder_path, "y_pred_test_ensemble.npy")
                y_test_path = os.path.join(folder_path, "y_test.npy")
                
                if not os.path.exists(y_pred_path) or not os.path.exists(y_test_path):
                    continue

                y_pred_probs = np.load(y_pred_path)
                y_test = np.load(y_test_path)
                
                # Convert probabilities to numeric class indices (0, 1, or 2)
                y_pred_indices = np.argmax(y_pred_probs, axis=1)
                
                # Convert string labels to numeric indices
                if y_test.dtype.kind in ('U', 'O', 'S'):  # Unicode, Object, or Byte string
                    y_test = np.array([CLASS_TO_INDEX.get(label, -1) for label in y_test])
                else:
                    y_test = y_test.astype(int)
                
                f1 = f1_score(y_test, y_pred_indices, average='weighted', zero_division=0)
                acc = accuracy_score(y_test, y_pred_indices)
                # Compute recall (weighted and macro) to inspect sensitivity
                rec_weighted = recall_score(y_test, y_pred_indices, average='weighted', zero_division=0)
                rec_macro = recall_score(y_test, y_pred_indices, average='macro', zero_division=0)
                # Per-class recall (one value per class index)
                rec_per_class = recall_score(y_test, y_pred_indices, average=None, zero_division=0)
                
                entry = {
                    "Experiment": exp, "Model": model_label, "Dataset": data_label,
                    "Epochs": epochs, "Learn_Rate": l_rate, "Reg_Const": r_const,
                    "weighted_f1": f1, "accuracy": acc,
                    "weighted_recall": rec_weighted, "macro_recall": rec_macro
                }
                # add per-class recall with readable keys
                for idx, cls in enumerate(CLASSES):
                    entry[f"rec_{cls}"] = float(rec_per_class[idx]) if idx < len(rec_per_class) else 0.0
                all_results.append(entry)
            except Exception as e:
                print(f"[ERROR] Failed to process {folder}: {e}")
                continue

    if not all_results:
        print("[CRITICAL] No data loaded into DataFrame.")
        return

    df = pd.DataFrame(all_results)
    
    # Display heatmaps for each experiment separately
    for exp in EXPERIMENTS:
        exp_data = df[df['Experiment'] == exp]
        if not exp_data.empty:
            save_generalization_heatmap(exp_data, f"{exp}")
            # Display RADCAT strategy matrices for each epoch
            save_radcat_strategy_matrix(df, f"{exp}")
            
            # Display hyperparameter analysis
            save_hyperparam_analysis(df, f"{exp}")
    
    # Display global heatmap across all experiments
    save_generalization_heatmap(df, "Global Average")
    
    # Display global RADCAT strategy matrix
    save_radcat_strategy_global_average(df)
    
    # Display global epoch trends
    save_hyperparam_analysis_global_average(df)
    
    # Save raw results for inspection
    try:
        df.to_csv(os.path.join(DATARESULTS_DIR, 'all_results.csv'), index=False)
        print(f"[INFO] Saved raw results to {os.path.join(DATARESULTS_DIR, 'all_results.csv')}")
    except Exception as e:
        print(f"[WARN] Failed to save all_results.csv: {e}")

    # Aggregate metrics (mean and std) across Model / Epochs / Reg_Const
    agg = df.groupby(['Model', 'Epochs', 'Reg_Const']).agg({
        'weighted_f1': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'weighted_recall': ['mean', 'std']
    }).reset_index()

    # Flatten multiindex columns
    agg.columns = ['Model', 'Epochs', 'Reg_Const',
                   'weighted_f1_mean', 'weighted_f1_std',
                   'accuracy_mean', 'accuracy_std',
                   'weighted_recall_mean', 'weighted_recall_std']

    # Compute a simple stability metric from weighted F1 (mean - std)
    agg['Stability'] = agg['weighted_f1_mean'] - agg['weighted_f1_std'].fillna(0)
    agg = agg.sort_values(by='Stability', ascending=False)

    # Save aggregated summary
    try:
        agg.to_csv(os.path.join(DATARESULTS_DIR, 'summary_metrics.csv'), index=False)
        print(f"[INFO] Saved summary metrics to {os.path.join(DATARESULTS_DIR, 'summary_metrics.csv')}")
    except Exception as e:
        print(f"[WARN] Failed to save summary_metrics.csv: {e}")

    # Print stability ranking and key metric columns
    print("\n" + "="*85)
    print(f"{'STABILITY RANKING (F1 MEAN - STD DEV)':^85}")
    print("="*85)
    display_cols = ['Model', 'Epochs', 'Reg_Const', 'weighted_f1_mean', 'weighted_f1_std', 'accuracy_mean', 'weighted_recall_mean', 'Stability']
    print(agg[display_cols].to_string(index=False))
    print("="*85)
    # Plot mean per-class accuracy (mean of per-run per-class recall)
    per_class_cols = [f"rec_{cls}" for cls in CLASSES]
    if all(col in df.columns for col in per_class_cols):
        per_class_means = df[per_class_cols].mean()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=CLASSES, y=per_class_means.values, palette='muted')
        plt.ylim(0, 1)
        plt.ylabel('Mean Per-Class Accuracy')
        plt.xlabel('Class')
        plt.title('Mean Per-Class Accuracy (averaged across all runs)')
        for i, v in enumerate(per_class_means.values):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
        out_path = os.path.join(DATARESULTS_DIR, '04_mean_per_class_accuracy.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved mean per-class accuracy plot to {out_path}")
    else:
        print("[WARN] Per-class recall columns not found; skipping mean per-class plot")

    # Mean per-class accuracy for each Model
    models = df['Model'].unique()
    for model in models:
        model_df = df[df['Model'] == model]
        if all(col in model_df.columns for col in per_class_cols) and not model_df.empty:
            model_means = model_df[per_class_cols].mean()
            plt.figure(figsize=(8, 6))
            sns.barplot(x=CLASSES, y=model_means.values, palette='muted')
            plt.ylim(0, 1)
            plt.ylabel('Mean Per-Class Accuracy')
            plt.xlabel('Class')
            plt.title(f'Mean Per-Class Accuracy - {model}')
            for i, v in enumerate(model_means.values):
                plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
            fname = f"04_mean_per_class_accuracy_{str(model).replace(' ', '_')}.png"
            out_path = os.path.join(DATARESULTS_DIR, fname)
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Saved mean per-class accuracy plot for model {model} to {out_path}")
        else:
            print(f"[WARN] Skipping per-class plot for model {model} (missing columns or no data)")

    # Create a 2x2 layout for the core models (FIRST, LOFAR, NVSS, RADCAT)
    core_order = ["FIRST", "LOFAR", "NVSS", "RADCAT"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    for idx, model in enumerate(core_order):
        ax = axes_flat[idx]
        if model in models:
            model_df = df[df['Model'] == model]
            if all(col in model_df.columns for col in per_class_cols) and not model_df.empty:
                model_means = model_df[per_class_cols].mean()
            else:
                model_means = pd.Series([0.0]*len(CLASSES), index=per_class_cols)
        else:
            model_means = pd.Series([0.0]*len(CLASSES), index=per_class_cols)

        sns.barplot(x=CLASSES, y=model_means.values, palette='muted', ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title(f"{model}")
        ax.set_ylabel('Mean Per-Class Accuracy')
        for i, v in enumerate(model_means.values):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

    fig.suptitle('Mean Per-Class Accuracy by Model (2x2)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_path = os.path.join(DATARESULTS_DIR, '04_mean_per_class_accuracy_2x2.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved combined 2x2 mean per-class accuracy figure to {combined_path}")

    # Per-experiment plots: overall average + 2x2 per-model for each experiment
    for exp in EXPERIMENTS:
        exp_df = df[df['Experiment'] == exp]
        if exp_df.empty:
            print(f"[WARN] No data for {exp}; skipping per-experiment plots")
            continue

        # overall per-class average for this experiment
        if all(col in exp_df.columns for col in per_class_cols):
            exp_means = exp_df[per_class_cols].mean()
            plt.figure(figsize=(8, 6))
            sns.barplot(x=CLASSES, y=exp_means.values, palette='muted')
            plt.ylim(0, 1)
            plt.ylabel('Mean Per-Class Accuracy')
            plt.xlabel('Class')
            plt.title(f'Mean Per-Class Accuracy (Overall) - {exp}')
            for i, v in enumerate(exp_means.values):
                plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
            out_over = os.path.join(DATARESULTS_DIR, f'04_mean_per_class_accuracy_{exp.replace(" ", "_")}.png')
            plt.tight_layout()
            plt.savefig(out_over, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Saved per-experiment overall mean per-class accuracy to {out_over}")
        else:
            print(f"[WARN] Missing per-class columns for {exp}; skipping overall plot")

        # 2x2 per-model layout for this experiment
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()
        for idx, model in enumerate(core_order):
            ax = axes_flat[idx]
            model_df = exp_df[exp_df['Model'] == model]
            if all(col in model_df.columns for col in per_class_cols) and not model_df.empty:
                model_means = model_df[per_class_cols].mean()
            else:
                model_means = pd.Series([0.0]*len(CLASSES), index=per_class_cols)

            sns.barplot(x=CLASSES, y=model_means.values, palette='muted', ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title(f"{model}")
            ax.set_ylabel('Mean Per-Class Accuracy')
            for i, v in enumerate(model_means.values):
                ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

        fig.suptitle(f'Mean Per-Class Accuracy by Model - {exp}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_2x2 = os.path.join(DATARESULTS_DIR, f'04_mean_per_class_accuracy_{exp.replace(" ", "_")}_2x2.png')
        plt.savefig(out_2x2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved per-experiment 2x2 mean per-class accuracy figure to {out_2x2}")

        # Combined figure (3x2) for this experiment: F1 heatmap, Recall heatmap,
        # overall per-class, 2x2 per-model, epoch trends, RADCAT strategy
        try:
            fig = plt.figure(figsize=(22, 32))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

            # Top-left: weighted F1 generalization heatmap (avg across epochs)
            ax_f1 = fig.add_subplot(gs[0, 0])
            core_surveys = ["FIRST", "LOFAR", "NVSS", "RADCAT"]
            heat_df = exp_df.groupby(['Model', 'Dataset'])['weighted_f1'].mean().reset_index()
            pivot_f1 = heat_df.pivot(index='Model', columns='Dataset', values='weighted_f1')
            pivot_f1 = pivot_f1.reindex(index=core_surveys, columns=core_surveys)
            sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0, vmax=1, ax=ax_f1,
                        cbar_kws={'label': 'Weighted F1'})
            ax_f1.set_title(f"Weighted F1 (avg) - {exp}")

            # Top-right: weighted recall generalization heatmap
            ax_rec = fig.add_subplot(gs[0, 1])
            heat_df_r = exp_df.groupby(['Model', 'Dataset'])['weighted_recall'].mean().reset_index()
            pivot_rec = heat_df_r.pivot(index='Model', columns='Dataset', values='weighted_recall')
            pivot_rec = pivot_rec.reindex(index=core_surveys, columns=core_surveys)
            sns.heatmap(pivot_rec, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0, vmax=1, ax=ax_rec,
                        cbar_kws={'label': 'Weighted Recall'})
            ax_rec.set_title(f"Weighted Recall (avg) - {exp}")

            # Middle-left: overall per-class bar
            ax_overall = fig.add_subplot(gs[1, 0])
            if all(col in exp_df.columns for col in per_class_cols):
                exp_means = exp_df[per_class_cols].mean()
            else:
                exp_means = pd.Series([0.0]*len(CLASSES), index=per_class_cols)
            sns.barplot(x=CLASSES, y=exp_means.values, palette='muted', ax=ax_overall)
            ax_overall.set_ylim(0, 1)
            ax_overall.set_title('Overall Mean Per-Class Accuracy')
            for i, v in enumerate(exp_means.values):
                ax_overall.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

            # Middle-right: nested 2x2 per-model bars
            sub_gs = gs[1, 1].subgridspec(2, 2)
            for idx, model in enumerate(core_surveys):
                sub_ax = fig.add_subplot(sub_gs[idx])
                model_df = exp_df[exp_df['Model'] == model]
                if all(col in model_df.columns for col in per_class_cols) and not model_df.empty:
                    model_means = model_df[per_class_cols].mean()
                else:
                    model_means = pd.Series([0.0]*len(CLASSES), index=per_class_cols)
                sns.barplot(x=CLASSES, y=model_means.values, palette='muted', ax=sub_ax)
                sub_ax.set_ylim(0, 1)
                sub_ax.set_title(model)
                for i, v in enumerate(model_means.values):
                    sub_ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=8)

            # Bottom-left: epoch trends (weighted F1)
            ax_trend = fig.add_subplot(gs[2, 0])
            epoch_stats = exp_df.groupby(['Model', 'Epochs'])['weighted_f1'].agg(['mean', 'std']).reset_index()
            if not epoch_stats.empty:
                sns.lineplot(data=epoch_stats, x='Epochs', y='mean', hue='Model', marker='o', ax=ax_trend)
                for model in epoch_stats['Model'].unique():
                    model_data = epoch_stats[epoch_stats['Model'] == model].sort_values('Epochs')
                    ax_trend.errorbar(model_data['Epochs'], model_data['mean'], yerr=model_data['std'], fmt='none', capsize=3, alpha=0.3)
            ax_trend.set_title('Epoch Trends (Weighted F1)')
            ax_trend.set_ylabel('Mean Weighted F1')
            ax_trend.set_ylim(0, 1)

            # Bottom-right: RADCAT strategy matrix (weighted F1)
            ax_rad = fig.add_subplot(gs[2, 1])
            rad_subset = exp_df[exp_df['Model'] == 'RADCAT']
            strategies = ["F-Repeated", "F-Zeroed", "F-Noise", "L-Repeated", "L-Zeroed", "L-Noise", "N-Repeated", "N-Zeroed", "N-Noise"]
            rad_strat = rad_subset[rad_subset['Dataset'].isin(strategies)].copy()
            if not rad_strat.empty:
                rad_strat['Source'] = rad_strat['Dataset'].apply(lambda x: x.split('-')[0] if '-' in x else x)
                rad_strat['Method'] = rad_strat['Dataset'].apply(lambda x: x.split('-')[1] if '-' in x else 'Native')
                pivot_rad = rad_strat.pivot_table(index='Source', columns='Method', values='weighted_f1', aggfunc='mean')
                sns.heatmap(pivot_rad, annot=True, fmt='.3f', cmap='rocket_r', vmin=0, vmax=1, ax=ax_rad, cbar_kws={'label': 'Weighted F1'})
            ax_rad.set_title('RADCAT Strategy (Weighted F1)')

            fig.suptitle(f'Combined Summary - {exp}', fontsize=18)
            combined_out = os.path.join(DATARESULTS_DIR, f'05_combined_summary_{exp.replace(" ", "_")}.png')
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.savefig(combined_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Saved combined experiment figure to {combined_out}")
        except Exception as e:
            print(f"[WARN] Failed to create combined figure for {exp}: {e}")

    # ------------------------------------------------------------------
    # Combined Global-Average Summary (across all experiments)
    # ------------------------------------------------------------------
    try:
        fig = plt.figure(figsize=(22, 32))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

        # Top-left: weighted F1 generalization heatmap (avg across experiments)
        ax_f1 = fig.add_subplot(gs[0, 0])
        core_surveys = ["FIRST", "LOFAR", "NVSS", "RADCAT"]
        heat_df = df.groupby(['Model', 'Dataset'])['weighted_f1'].mean().reset_index()
        pivot_f1 = heat_df.pivot(index='Model', columns='Dataset', values='weighted_f1')
        pivot_f1 = pivot_f1.reindex(index=core_surveys, columns=core_surveys)
        if pivot_f1.isnull().all().all():
            ax_f1.text(0.5, 0.5, 'No data', ha='center', va='center')
        else:
            sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0, vmax=1, ax=ax_f1,
                        cbar_kws={'label': 'Weighted F1'})
        ax_f1.set_title(f"Weighted F1 (Global Avg)")

        # Top-right: weighted recall generalization heatmap
        ax_rec = fig.add_subplot(gs[0, 1])
        heat_df_r = df.groupby(['Model', 'Dataset'])['weighted_recall'].mean().reset_index()
        pivot_rec = heat_df_r.pivot(index='Model', columns='Dataset', values='weighted_recall')
        pivot_rec = pivot_rec.reindex(index=core_surveys, columns=core_surveys)
        if pivot_rec.isnull().all().all():
            ax_rec.text(0.5, 0.5, 'No data', ha='center', va='center')
        else:
            sns.heatmap(pivot_rec, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0, vmax=1, ax=ax_rec,
                        cbar_kws={'label': 'Weighted Recall'})
        ax_rec.set_title(f"Weighted Recall (Global Avg)")

        # Middle-left: overall per-class bar (global)
        ax_overall = fig.add_subplot(gs[1, 0])
        if all(col in df.columns for col in per_class_cols):
            global_means = df[per_class_cols].mean()
        else:
            global_means = pd.Series([0.0]*len(CLASSES), index=per_class_cols)
        sns.barplot(x=CLASSES, y=global_means.values, palette='muted', ax=ax_overall)
        ax_overall.set_ylim(0, 1)
        ax_overall.set_title('Overall Mean Per-Class Accuracy (Global Avg)')
        for i, v in enumerate(global_means.values):
            ax_overall.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

        # Middle-right: nested 2x2 per-model bars (global)
        sub_gs = gs[1, 1].subgridspec(2, 2)
        for idx, model in enumerate(core_surveys):
            sub_ax = fig.add_subplot(sub_gs[idx])
            model_df = df[df['Model'] == model]
            if all(col in model_df.columns for col in per_class_cols) and not model_df.empty:
                model_means = model_df[per_class_cols].mean()
            else:
                model_means = pd.Series([0.0]*len(CLASSES), index=per_class_cols)
            sns.barplot(x=CLASSES, y=model_means.values, palette='muted', ax=sub_ax)
            sub_ax.set_ylim(0, 1)
            sub_ax.set_title(model)
            for i, v in enumerate(model_means.values):
                sub_ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=8)

        # Bottom-left: global epoch trends (weighted F1)
        ax_trend = fig.add_subplot(gs[2, 0])
        epoch_stats = df.groupby(['Model', 'Epochs'])['weighted_f1'].agg(['mean', 'std']).reset_index()
        if not epoch_stats.empty:
            sns.lineplot(data=epoch_stats, x='Epochs', y='mean', hue='Model', marker='o', ax=ax_trend)
            for model in epoch_stats['Model'].unique():
                model_data = epoch_stats[epoch_stats['Model'] == model].sort_values('Epochs')
                ax_trend.errorbar(model_data['Epochs'], model_data['mean'], yerr=model_data['std'], fmt='none', capsize=3, alpha=0.3)
        ax_trend.set_title('Epoch Trends (Weighted F1) - Global Avg')
        ax_trend.set_ylabel('Mean Weighted F1')
        ax_trend.set_ylim(0, 1)

        # Bottom-right: RADCAT strategy matrix (weighted F1) - global
        ax_rad = fig.add_subplot(gs[2, 1])
        rad_subset = df[df['Model'] == 'RADCAT']
        strategies = ["F-Repeated", "F-Zeroed", "F-Noise", "L-Repeated", "L-Zeroed", "L-Noise", "N-Repeated", "N-Zeroed", "N-Noise"]
        rad_strat = rad_subset[rad_subset['Dataset'].isin(strategies)].copy()
        if not rad_strat.empty:
            rad_strat['Source'] = rad_strat['Dataset'].apply(lambda x: x.split('-')[0] if '-' in x else x)
            rad_strat['Method'] = rad_strat['Dataset'].apply(lambda x: x.split('-')[1] if '-' in x else 'Native')
            pivot_rad = rad_strat.pivot_table(index='Source', columns='Method', values='weighted_f1', aggfunc='mean')
            sns.heatmap(pivot_rad, annot=True, fmt='.3f', cmap='rocket_r', vmin=0, vmax=1, ax=ax_rad, cbar_kws={'label': 'Weighted F1'})
        else:
            ax_rad.text(0.5, 0.5, 'No RADCAT strategy data', ha='center', va='center')
        ax_rad.set_title('RADCAT Strategy (Weighted F1) - Global Avg')

        fig.suptitle('Combined Summary - Global Average', fontsize=18)
        combined_out = os.path.join(DATARESULTS_DIR, f'05_combined_summary_Global_Average.png')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(combined_out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved combined global-average summary to {combined_out}")
    except Exception as e:
        print(f"[WARN] Failed to create combined global-average figure: {e}")
if __name__ == "__main__":
    main()