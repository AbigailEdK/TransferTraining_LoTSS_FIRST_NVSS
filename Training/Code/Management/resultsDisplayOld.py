# region ABOUT
# ===================================================================================================
# > This script loads and displays test results from cross-validation ensemble predictions.
# > It scans Results/Testing for result folders and displays metrics in a readable format.
# ===================================================================================================
# endregion

import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns


# Find the home directory dynamically
home_dir = os.path.expanduser("~")
# Construct the project path
project_root = os.path.join(home_dir, "DeKlerk_Models")

# Scan all experiment folders
RESULTS_BASE_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Results_Run1"
EXPERIMENTS = [f"Experiment_{i}" for i in range(1, 6)]  # Experiment_1 through Experiment_5

# Mapping for abbreviations
ABBREV_TO_FULL = {"FIR": "FIRST", "NVS": "NVSS", "LOF": "LOFAR", "RME": "RADCAT_MEAN", "RWE": "RADCAT_WEIGHTED_MEAN"}
CLASSES = ["FRI", "FRII", "COMPACT"]

DATARESULTS_DIR = os.path.join(os.getcwd(), "DATARESULTS")
os.makedirs(DATARESULTS_DIR, exist_ok=True)


def save_accuracy_plot(accuracies, model_name):
    plt.figure()
    plt.plot(accuracies, label="Accuracy")
    plt.title(f"Accuracy Over Epochs for {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(DATARESULTS_DIR, f"accuracy_{model_name}.png"))
    plt.close()

def save_loss_plot(losses, model_name):
    plt.figure()
    plt.plot(losses, label="Loss")
    plt.title(f"Loss Over Epochs for {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(DATARESULTS_DIR, f"loss_{model_name}.png"))
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    # Convert probabilities to class labels if necessary
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.array([class_names.index(label) for label in y_true])

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig(os.path.join(DATARESULTS_DIR, f"confusion_matrix_{model_name}.png"))
    plt.close()

def save_model_comparison_plot(exp_name, model_data):
    """Create a comparison plot for models within an experiment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = sorted(model_data['Model'].unique())
    
    # Accuracy comparison
    for model in models:
        model_subset = model_data[model_data['Model'] == model]
        datasets = model_subset['Dataset'].values
        accuracies = model_subset['accuracy'].values
        ax1.plot(datasets, accuracies, marker='o', label=model, linewidth=2)
    
    ax1.set_title(f"{exp_name} - Accuracy by Model & Dataset")
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Weighted F1 comparison
    for model in models:
        model_subset = model_data[model_data['Model'] == model]
        datasets = model_subset['Dataset'].values
        wf1 = model_subset['weighted_f1'].values
        ax2.plot(datasets, wf1, marker='s', label=model, linewidth=2)
    
    ax2.set_title(f"{exp_name} - Weighted F1 by Model & Dataset")
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Weighted F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATARESULTS_DIR, f"comparison_{exp_name}.png"), dpi=100)
    plt.close()

def save_combined_confusion_matrices(experiment_results, exp_name):
    """Create a 5x5 grid of confusion matrices for all model-dataset combinations."""
    configs = sorted(experiment_results, key=lambda x: (x['Model'], x['Dataset']))
    
    if len(configs) > 25:
        configs = configs[:25]  # Limit to 25 for 5x5 grid
    
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes_flat = axes.flatten()
    
    for idx, result in enumerate(configs):
        ax = axes_flat[idx]
        
        exp_path = os.path.join(RESULTS_BASE_DIR, result["Experiment"], "Testing")
        folder_path = os.path.join(exp_path, result["Folder"])
        
        # Load test results
        y_pred, y_test, names_test = load_test_results(folder_path)
        if y_pred is None or y_test is None:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Create confusion matrix
        pred_indices = np.argmax(y_pred, axis=1)
        y_pred_labels = np.array([CLASSES[i] for i in pred_indices])
        
        cm = confusion_matrix(y_test, y_pred_labels, labels=CLASSES)
        
        # Normalize for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=CLASSES, yticklabels=CLASSES, cbar=False,
                    vmin=0, vmax=1)
        
        ax.set_title(f"{result['Model']} on {result['Dataset']}\nAcc: {result['accuracy']:.3f}", 
                    fontsize=9)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
    
    # Hide unused subplots
    for idx in range(len(configs), 25):
        axes_flat[idx].axis('off')
    
    plt.suptitle(f"{exp_name} - Confusion Matrices (All Configurations)", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(DATARESULTS_DIR, f"confusion_matrices_grid_{exp_name}.png"), dpi=100, bbox_inches='tight')
    plt.close()

def save_generalization_trends(df, all_experiments):
    """Create line plots showing generalization score trends across experiments."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Collect generalization scores for each model across experiments
    model_trends = {}
    
    for exp_name in all_experiments:
        exp_data = df[df["Experiment"] == exp_name]
        if exp_data.empty:
            continue
        
        for model in sorted(exp_data['Model'].unique()):
            gen_score, mean_wf1, std_wf1 = calculate_generalization_score(exp_data, model)
            
            if model not in model_trends:
                model_trends[model] = {
                    'experiments': [],
                    'gen_scores': [],
                    'mean_wf1s': [],
                    'std_wf1s': []
                }
            
            model_trends[model]['experiments'].append(exp_name)
            model_trends[model]['gen_scores'].append(gen_score)
            model_trends[model]['mean_wf1s'].append(mean_wf1)
            model_trends[model]['std_wf1s'].append(std_wf1)
    
    # Plot 1: Generalization Scores
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_trends)))
    
    for (model, data), color in zip(sorted(model_trends.items()), colors):
        exp_nums = [int(e.split('_')[1]) for e in data['experiments']]
        ax1.plot(exp_nums, data['gen_scores'], marker='o', label=model, linewidth=2.5, 
                markersize=8, color=color)
    
    ax1.set_xlabel("Experiment", fontsize=11)
    ax1.set_ylabel("Generalization Score (Mean WF1 - Std WF1)", fontsize=11)
    ax1.set_title("Model Generalization Score Trends Across Experiments", fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 2, 3, 4, 5])
    
    # Plot 2: Mean Weighted F1
    ax2 = axes[1]
    for (model, data), color in zip(sorted(model_trends.items()), colors):
        exp_nums = [int(e.split('_')[1]) for e in data['experiments']]
        ax2.plot(exp_nums, data['mean_wf1s'], marker='s', label=model, linewidth=2.5,
                markersize=8, color=color)
    
    ax2.set_xlabel("Experiment", fontsize=11)
    ax2.set_ylabel("Mean Weighted F1", fontsize=11)
    ax2.set_title("Mean Weighted F1 Trends Across Experiments", fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([1, 2, 3, 4, 5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATARESULTS_DIR, "generalization_trends.png"), dpi=100, bbox_inches='tight')
    plt.close()

def calculate_generalization_score(model_results, model_name):
    """
    Calculate generalization score for a model across datasets.
    For RADCAT_MEAN and RADCAT_WEIGHTED_MEAN, only keep the better score.
    Returns: mean_weighted_f1 - std_weighted_f1
    """
    model_data = model_results[model_results['Model'] == model_name].copy()
    
    # Group RADCAT datasets and keep only the best
    radcat_rows = model_data[model_data['Dataset'].str.contains('RADCAT', na=False)]
    other_rows = model_data[~model_data['Dataset'].str.contains('RADCAT', na=False)]
    
    if not radcat_rows.empty:
        # Keep only the better RADCAT variant (higher weighted_f1)
        best_radcat = radcat_rows.loc[radcat_rows['weighted_f1'].idxmax()]
        model_data_filtered = pd.concat([other_rows, best_radcat.to_frame().T], ignore_index=True)
    else:
        model_data_filtered = model_data
    
    if len(model_data_filtered) == 0:
        return 0, 0, 0
    
    mean_wf1 = model_data_filtered['weighted_f1'].mean()
    std_wf1 = model_data_filtered['weighted_f1'].std()
    generalization = mean_wf1 - std_wf1
    
    return generalization, mean_wf1, std_wf1

def load_test_results(folder_path):
    """
    Load test results from a folder.
    
    Returns:
        tuple: (y_pred, y_test, names_test) or (None, None, None) if files don't exist
    """
    pred_file = os.path.join(folder_path, "y_pred_test_ensemble.npy")
    test_file = os.path.join(folder_path, "y_test.npy")
    names_file = os.path.join(folder_path, "names_test.npy")
    
    if not all(os.path.exists(f) for f in [pred_file, test_file, names_file]):
        return None, None, None
    
    try:
        y_pred = np.load(pred_file, allow_pickle=True)
        y_test = np.load(test_file, allow_pickle=True)
        names_test = np.load(names_file, allow_pickle=True)
        return y_pred, y_test, names_test
    except Exception as e:
        print(f"Error loading files from {folder_path}: {e}")
        return None, None, None

def extract_model_dataset(folder_name):
    """
    Extract model and dataset from folder name.
    Expects format: modelXXX_setYYY_... where XXX and YYY are three letters
    Example: modelFIR_setNVS_20e_0.00039l_0.4r
    """
    if folder_name.startswith("model") and "_set" in folder_name:
        model = folder_name[5:8]  # Extract first three letters for model
        set_idx = folder_name.index("_set")
        dataset = folder_name[set_idx+4:set_idx+7]  # Extract first three letters for dataset
        return model, dataset

    parts = folder_name.split("_")
    if len(parts) >= 2:
        model = parts[0][:3]  # Extract first three letters for model
        dataset = parts[1][:3]  # Extract first three letters for dataset
        return model, dataset

    return None, None

def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels (class strings)
        y_pred: Predicted probabilities (shape: (N, num_classes))
    
    Returns:
        dict: Dictionary containing accuracy and weighted F1 (most relevant for imbalanced data)
    """
    pred_indices = np.argmax(y_pred, axis=1)
    y_pred_labels = np.array([CLASSES[i] for i in pred_indices])
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_labels),
        "macro_f1": f1_score(y_true, y_pred_labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred_labels, average="weighted", zero_division=0),
    }
    
    return metrics

def main():
    """Load and display all test results from all experiments."""
    
    all_results = []
    experiment_summary = []
    
    # Scan all experiments
    for exp_name in EXPERIMENTS:
        exp_path = os.path.join(RESULTS_BASE_DIR, exp_name, "Testing")
        
        if not os.path.exists(exp_path):
            continue
        
        folders = os.listdir(exp_path)
        exp_results = []
        
        for folder in sorted(folders):
            folder_path = os.path.join(exp_path, folder)
            
            if not os.path.isdir(folder_path):
                continue
            
            y_pred, y_test, names_test = load_test_results(folder_path)
            
            if y_pred is None:
                continue
            
            # Extract model and dataset from folder name
            model, dataset = extract_model_dataset(folder)
            
            if model is None or dataset is None:
                model = folder
                dataset = "Unknown"
            
            # Convert abbreviations if needed
            model_display = ABBREV_TO_FULL.get(model, model)
            dataset_display = ABBREV_TO_FULL.get(dataset, dataset)
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred)
            
            result = {
                "Experiment": exp_name,
                "Model": model_display,
                "Dataset": dataset_display,
                "Folder": folder,
                "Num_Samples": len(y_test),
                **metrics
            }
            all_results.append(result)
            exp_results.append(result)
        
        # Aggregate experiment summary
        if exp_results:
            exp_df = pd.DataFrame(exp_results)
            experiment_summary.append({
                "Experiment": exp_name,
                "Num_Configs": len(exp_results),
                "Mean_Accuracy": exp_df["accuracy"].mean(),
                "Std_Accuracy": exp_df["accuracy"].std(),
                "Best_Accuracy": exp_df["accuracy"].max(),
                "Mean_Weighted_F1": exp_df["weighted_f1"].mean(),
                "Std_Weighted_F1": exp_df["weighted_f1"].std(),
                "Best_Weighted_F1": exp_df["weighted_f1"].max(),
            })
    
    if not all_results:
        print("No test results found across any experiments.")
        return
    
    # Create DataFrames
    df = pd.DataFrame(all_results)
    exp_summary_df = pd.DataFrame(experiment_summary)
    
    # Display results
    print("\n" + "="*120)
    print("EXPERIMENT COMPARISON SUMMARY".center(120))
    print("="*120)
    summary_display = exp_summary_df[[
        "Experiment", "Num_Configs", "Mean_Accuracy", "Best_Accuracy", 
        "Mean_Weighted_F1", "Best_Weighted_F1"
    ]].copy()
    summary_display["Mean_Accuracy"] = summary_display["Mean_Accuracy"].round(4)
    summary_display["Best_Accuracy"] = summary_display["Best_Accuracy"].round(4)
    summary_display["Mean_Weighted_F1"] = summary_display["Mean_Weighted_F1"].round(4)
    summary_display["Best_Weighted_F1"] = summary_display["Best_Weighted_F1"].round(4)
    print(summary_display.to_string(index=False))
    
    # Detailed results per experiment
    print("\n" + "="*120)
    print("DETAILED RESULTS BY EXPERIMENT".center(120))
    print("="*120)
    
    for exp_name in EXPERIMENTS:
        exp_data = df[df["Experiment"] == exp_name]
        if exp_data.empty:
            continue
        
        print(f"\n{exp_name}:")
        print("-" * 120)
        display_cols = ["Model", "Dataset", "Num_Samples", "accuracy", "macro_f1", "weighted_f1"]
        available_cols = [col for col in display_cols if col in exp_data.columns]
        
        display_df = exp_data[available_cols].copy()
        display_df["accuracy"] = display_df["accuracy"].round(4)
        display_df["macro_f1"] = display_df["macro_f1"].round(4)
        display_df["weighted_f1"] = display_df["weighted_f1"].round(4)
        
        print(display_df.to_string(index=False))
    
    # Best results across all experiments
    print("\n" + "="*120)
    print("BEST RESULTS ACROSS ALL EXPERIMENTS".center(120))
    print("="*120)
    
    best_acc_idx = df["accuracy"].idxmax()
    best_acc = df.loc[best_acc_idx]
    print(f"\nBest Accuracy: {best_acc['Experiment']:12s} | {best_acc['Model']:10s} on {best_acc['Dataset']:12s} = {best_acc['accuracy']:.4f}")
    
    best_wf1_idx = df["weighted_f1"].idxmax()
    best_wf1 = df.loc[best_wf1_idx]
    print(f"Best Weighted_F1: {best_wf1['Experiment']:8s} | {best_wf1['Model']:10s} on {best_wf1['Dataset']:12s} = {best_wf1['weighted_f1']:.4f}")
    
    # Cross-experiment model performance
    print("\n" + "="*120)
    print("MODEL PERFORMANCE ACROSS EXPERIMENTS".center(120))
    print("="*120)
    
    model_performance = df.groupby("Model", as_index=False).agg({
        "accuracy": ["mean", "std", "count"],
        "weighted_f1": ["mean", "std"],
    }).round(4)
    model_performance.columns = ["Model", "Mean_Acc", "Std_Acc", "Num_Runs", "Mean_WeightedF1", "Std_WeightedF1"]
    print(model_performance.to_string(index=False))
    
    # Generalization scores per experiment
    print("\n" + "="*120)
    print("MODEL GENERALIZATION SCORES PER EXPERIMENT".center(120))
    print("="*120)
    print("(Calculated as: mean_weighted_f1 - std_weighted_f1, excluding worse RADCAT variant)")
    print("="*120)
    
    for exp_name in EXPERIMENTS:
        exp_data = df[df["Experiment"] == exp_name]
        if exp_data.empty:
            continue
        
        gen_scores = []
        for model in sorted(exp_data['Model'].unique()):
            gen_score, mean_wf1, std_wf1 = calculate_generalization_score(exp_data, model)
            gen_scores.append({
                "Model": model,
                "Generalization_Score": gen_score,
                "Mean_WeightedF1": mean_wf1,
                "Std_WeightedF1": std_wf1
            })
        
        gen_df = pd.DataFrame(gen_scores).sort_values("Generalization_Score", ascending=False)
        gen_df_display = gen_df.copy()
        gen_df_display["Generalization_Score"] = gen_df_display["Generalization_Score"].round(4)
        gen_df_display["Mean_WeightedF1"] = gen_df_display["Mean_WeightedF1"].round(4)
        gen_df_display["Std_WeightedF1"] = gen_df_display["Std_WeightedF1"].round(4)
        
        print(f"\n{exp_name}:")
        print("-" * 80)
        print(gen_df_display.to_string(index=False))
        
        best_model = gen_df.iloc[0]
        print(f"\n  ✓ BEST GENERALIZATION: {best_model['Model']:15s} (score: {best_model['Generalization_Score']:.4f})")
    
    print("\n" + "="*120)

    # Generate comparison plots for each experiment
    print("\nGenerating experiment comparison plots...")
    for exp_name in EXPERIMENTS:
        exp_data = df[df["Experiment"] == exp_name]
        if not exp_data.empty:
            save_model_comparison_plot(exp_name, exp_data)

    # Generate combined confusion matrix grids for each experiment
    print("Generating combined confusion matrix grids...")
    for exp_name in EXPERIMENTS:
        exp_results = [r for r in all_results if r["Experiment"] == exp_name]
        if exp_results:
            save_combined_confusion_matrices(exp_results, exp_name)
    
    # Generate generalization trend plots
    print("Generating generalization trend plots...")
    save_generalization_trends(df, EXPERIMENTS)

    print("All plots generated and saved in DATARESULTS directory.")

if __name__ == "__main__":
    main()
