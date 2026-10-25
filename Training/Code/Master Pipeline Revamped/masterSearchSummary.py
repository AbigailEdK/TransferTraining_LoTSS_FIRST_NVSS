# region ABOUT
# ===================================================================================================
# > This script summarises a hyperparameter search stage (COARSE or FINE) run by masterManager.py. It reads the saved training histories of every config, scores each fold by its lowest validation loss (the epoch EarlyStopping restores), averages over folds, and ranks the configs per survey. It saves the full ranking to a CSV. masterManager.py uses it between stages to pick the next stage's configs automatically. Only validation data is used, so the test set stays untouched until the FINAL stage.
# ===================================================================================================
# endregion

# region IMPORTS
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import masterProcess as process
# endregion

# region PATHS
from paths import RESULTSDIRECTORY
# endregion

# region FUNCTIONS
def load_history(path):
    history = np.load(path, allow_pickle=True)
    return history.item() if isinstance(history, np.ndarray) and history.shape == () else history

def summarise_stage(stage, experiment_number=0):
    '''
    ARGUMENTS:
    ----------
        > stage: "COARSE" or "FINE"
        > experiment_number: the experiment folder the search ran in

    RETURNS:
    --------
        > summary: a DataFrame with one row per (survey, config), ranked by mean validation loss within each survey

    DESCRIPTION:
    ------------
        For every config folder in Search_<stage>/Training, the lowest validation loss of each fold is taken, and the mean and standard deviation over folds are reported together with the validation accuracy at that epoch.
    '''
    training_dir = Path(RESULTSDIRECTORY) / f"Experiment_{experiment_number}" / f"Search_{stage}" / "Training"
    if not training_dir.is_dir():
        raise FileNotFoundError(f"No search results found in {training_dir}")

    rows = []
    for folder in sorted(training_dir.iterdir()):
        if not folder.is_dir():
            continue
        parameters = process.extract_parameters_from_folder_name(folder.name)
        if parameters["base_model"] != "NONE":
            continue

        fold_losses, fold_accuracies = [], []
        for history_path in sorted(folder.glob("loss_fold_*.npy")):
            history = load_history(history_path)
            best = int(np.nanargmin(history["val_loss"]))
            fold_losses.append(history["val_loss"][best])
            fold_accuracies.append(history["val_accuracy"][best])

        if len(fold_losses) == 0:
            print(f"WARNING: no finished folds in {folder.name}, skipping.")
            continue

        rows.append({
            "survey": parameters["model"],
            "lr": parameters["lr"],
            "reg": parameters["reg"],
            "wd": parameters["wd"],
            "n_folds": len(fold_losses),
            "val_loss_mean": np.mean(fold_losses),
            "val_loss_std": np.std(fold_losses),
            "val_accuracy_mean": np.mean(fold_accuracies),
            "folder": folder.name,
        })

    summary = pd.DataFrame(rows).sort_values(["survey", "val_loss_mean"]).reset_index(drop=True)
    summary["rank"] = summary.groupby("survey").cumcount() + 1
    return summary

def save_summary(summary, stage, experiment_number=0):
    '''
    Saves the full ranking to Search_<stage>/search_summary_<stage>.csv and prints the top 3 per survey.
    '''
    output_file = Path(RESULTSDIRECTORY) / f"Experiment_{experiment_number}" / f"Search_{stage}" / f"search_summary_{stage.lower()}.csv"
    summary.to_csv(output_file, index=False)
    print(f"Saved the full {stage} ranking ({len(summary)} configs) to {output_file}")

    print(f"\nTop 3 per survey ({stage}):")
    print(summary[summary["rank"] <= 3][["survey", "rank", "lr", "reg", "wd", "n_folds", "val_loss_mean", "val_loss_std", "val_accuracy_mean"]].to_string(index=False))
    warn_edges(summary)

def print_next_stage(summary, stage):
    best = summary[summary["rank"] == 1]
    name = "FINE_CENTRES" if stage == "COARSE" else "BEST_PARAMETERS"
    print(f"\n# > Best per survey (masterManager.py picks these automatically; paste into FINE_CENTRES / BEST_PARAMETERS to override):")
    print(f"    {name} = {{")
    for _, row in best.iterrows():
        print(f"        \"{row['survey']}\": ({row['lr']:.3g}, {row['reg']:.3g}, {row['wd']:.3g}), # val_loss {row['val_loss_mean']:.4f} +- {row['val_loss_std']:.4f}, val_acc {row['val_accuracy_mean']:.4f}")
    print("    }")

def warn_edges(summary):
    best = summary[summary["rank"] == 1]
    edge = []
    for _, row in best.iterrows():
        grid = summary[summary["survey"] == row["survey"]]
        for column in ["lr", "reg", "wd"]:
            # > Only flag ranges with at least 3 values, and not reg = 0 (there is nothing below it)
            if grid[column].nunique() > 2 and row[column] in (grid[column].min(), grid[column].max()) and not (column == "reg" and row[column] == 0):
                edge.append(f"{row['survey']} {column}={row[column]:.3g}")
    if len(edge) > 0:
        print(f"\nWARNING: these best values sit on the edge of the searched range, so the optimum may lie outside it: {', '.join(edge)}")
# endregion

# region MAIN
# > masterManager.py calls this automatically between stages. Run it by hand to inspect a stage: python masterSearchSummary.py COARSE
if __name__ == "__main__":
    stage = sys.argv[1].upper() if len(sys.argv) > 1 else "COARSE"
    summary = summarise_stage(stage)
    save_summary(summary, stage)
    print_next_stage(summary, stage)
# endregion
