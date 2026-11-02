# region ABOUT
# ===================================================================================================
# > This script summarises everything the revamped master pipeline produces, for writing up in the thesis:
    # - SEARCH: the COARSE and FINE hyperparameter searches and the FINAL configs, so the progression from the
    #   coarse grid to the chosen hyperparameters can be shown (rankings, heatmaps, sensitivity and spread plots).
    #   Experimental views of the full (lr, reg, wd) grid go under figures/search/experimental_3D (sliced heatmaps,
    #   parallel coordinates, 3D scatter, loss surfaces).
    # - TRAINING: learning curves, early-stopping epochs and training times of every FINAL model (scratch and transfer).
    # - TESTING: per-fold and fold-ensemble test metrics with bootstrap confidence intervals, cross-survey
    #   generalisation matrices, transfer-learning gains with McNemar tests, confusion matrices, ROC curves,
    #   channel-order effects and a generalisation leaderboard per channel group.
    # - ABLATIONS: how the duo- and trio-channel models cope when only one survey is available (the survey repeated in
    #   every channel, or with zeros / white noise in the other channels), which fill works best, whether the channel
    #   position matters, how much each model relies on each survey, and how they compare with single-channel models.
    # - SURVEY COMPARISON: per target (single, duo and trio), the scratch model against every model fine-tuned on it on each test set,
    #   and per survey (FIRST, LoTSS, NVSS) every model given only that survey (repeated, zeros and noise for duo/trio models),
    #   one chart per channel group and one master chart with every channel group.
    # - METRIC HEATMAPS: per channel group, every model against every in-domain metric, and every model against every test set
    #   for each of accuracy, balanced accuracy, macro F1, MCC, Cohen's kappa and ROC AUC.
    # - TEST SET METRICS: per test set (FIRST, LoTSS, NVSS), every model given it against each metric (accuracy, balanced
    #   accuracy, macro precision, recall and F1, weighted F1, MCC, Cohen's kappa, ROC AUC), one bar chart per metric and one grid:
    #   single-channel models with duo/trio models given the survey repeated, then duo and trio models each on their own with
    #   the survey repeated, with zeros and with white noise side by side.
# > Where plots belong together, every separate PNG is also combined into one figure (files ending in _combined.png).
# > Every result is saved as a CSV, every table also as a LaTeX (booktabs) file, and every chart as a PNG, under
#   Results/Experiment_<n>/Summary/. Include the tables with \input{...}; they need \usepackage{booktabs}.
# > Hyperparameters are only ever selected on validation loss. Test metrics are reported, never used for selection.
# > The COARSE search trains only the first folds of the same 5-fold split, so search stages are compared on those shared folds;
#   within a stage, configs are ranked only once all of that stage's folds have finished.
# ===================================================================================================
# endregion

# region IMPORTS
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
    roc_auc_score, roc_curve, confusion_matrix, cohen_kappa_score, log_loss,
)
# endregion

# region PATHS
# > Results are found relative to this script (Training/Code/Master Pipeline Revamped -> Training/Results), so the same file
# > works on the Desktop and on the cluster without editing paths.py
RESULTSDIRECTORY = Path(__file__).resolve().parents[2] / "Results"
SOURCEDATADIRECTORY = RESULTSDIRECTORY / "SourceData"
# endregion

# region SETTINGS
EXPERIMENT_NUMBER = 0
EXPERIMENT_DIRECTORY = Path(RESULTSDIRECTORY) / f"Experiment_{EXPERIMENT_NUMBER}"
SUMMARY_DIRECTORY = EXPERIMENT_DIRECTORY / "Summary"

STAGES = ["COARSE", "FINE", "FINAL"]
STAGE_DIRECTORIES = {
    "COARSE": EXPERIMENT_DIRECTORY / "Search_COARSE",
    "FINE": EXPERIMENT_DIRECTORY / "Search_FINE",
    "FINAL": EXPERIMENT_DIRECTORY,
}

CLASSES = ["FRI", "FRII", "COMPACT"] # | Same order as cf.LOOKUP in masterRunModel.py
SINGLE_CHANNEL = ["FIRST", "NVSS", "LoTSS"]
DUO_CHANNEL = ["FL", "LF", "FN", "NF", "LN", "NL"]
TRIO_CHANNEL = ["FLN", "FNL", "LNF", "LFN", "NLF", "NFL"]
CHANNEL_GROUPS = {"single": SINGLE_CHANNEL, "duo": DUO_CHANNEL, "trio": TRIO_CHANNEL}
ALL_SURVEYS = SINGLE_CHANNEL + DUO_CHANNEL + TRIO_CHANNEL
SURVEY_NAMES = {"F": "FIRST", "L": "LoTSS", "N": "NVSS"}

# > Single-survey ablation inputs, named <survey><fill><channels>[_CH<position>] (see masterProcess.get_channels)
ABLATION_PATTERN = re.compile(r"^([FLN])(REP|ZER|NSE)([23])(?:_CH([123]))?$")
FILLS = ["REP", "ZER", "NSE"]
FILL_NAMES = {"REP": "Repeated", "ZER": "Zeros", "NSE": "White noise"}

SAVE_INDIVIDUAL_PLOTS = True # | One learning-curve PNG per model and one confusion-matrix PNG per test (over a thousand files, slow); False skips only these
GROUP_CHANNELS = {"single": 1, "duo": 2, "trio": 3}
SURVEY_LETTERS = {"FIRST": "F", "LoTSS": "L", "NVSS": "N"}
RELATIONS = {"native": "Survey in its\nown channel", "moved": "Survey in a\ndifferent channel", "repeated": "Survey in\nevery channel", "absent": "Survey not in\nthe model"}
N_BOOTSTRAP = 1000
SEED = 100
LEADERBOARD_SIZE = 15

# > Mirrors masterProcess.folder_construct: base{base}_target{model}_dataset{dataset}_{lr}l_base{base_reg}r_{reg}r_ceiling{n}e_wd{wd}
NUMBER = r"[\d.]+(?:e-?\d+)?"
FOLDER_PATTERN = re.compile(
    rf"^base(\w+)_target(\w+)_dataset(\w+)_({NUMBER})l_base({NUMBER})r_({NUMBER})r_ceiling(\d+)e_wd({NUMBER})$"
)
# endregion

# region STYLE
# > Light, print-friendly palette. Categorical hues are assigned in a fixed order and follow the entity (a survey
#   keeps its colour in every chart). Sequential maps are one hue light->dark; diverging maps are blue<->red around grey.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURVEY_COLOURS = {"FIRST": SERIES[0], "NVSS": SERIES[1], "LoTSS": SERIES[2]}
STAGE_COLOURS = {"COARSE": SERIES[0], "FINE": SERIES[1], "FINAL": SERIES[2]}
SPLIT_COLOURS = {"train": SERIES[0], "val": SERIES[1]}
FILL_COLOURS = {"REP": SERIES[6], "ZER": SERIES[3], "NSE": SERIES[7]}
GROUP_COLOURS = {"single": SERIES[0], "duo": SERIES[1], "trio": SERIES[2]}
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("red_blue", ["#a32626", "#ec8b8b", "#f0efec", "#6da7ec", "#184f95"]) # | Negative red, positive blue

def set_style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": MUTED,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 6,
    })

def annotate_heatmap(ax, values, fmt="{:.3f}", cmap=SEQUENTIAL, norm=None):
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    for (i, j), value in np.ndenumerate(values):
        if not np.isfinite(value):
            ax.text(j, i, "-", ha="center", va="center", color=MUTED, fontsize=7)
            continue
        # > White text on dark cells, judged from the cell colour's luminance
        red, green, blue, _ = image.cmap(image.norm(value))
        dark = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
        ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=7, color="#ffffff" if dark else INK)
    ax.grid(False)
    return image
# endregion

# region HELPERS
def output_dirs():
    dirs = {name: SUMMARY_DIRECTORY / name for name in ["csv", "latex", "figures"]}
    for sub in ["search", "training", "testing", "ablation", "survey_comparison", "metric_heatmaps", "test_set_metrics", "base_vs_target", "learning_curves", "confusion_matrices"]:
        dirs[sub] = dirs["figures"] / sub
    dirs["search_heatmaps"] = dirs["search"] / "heatmaps"
    dirs["search_3d"] = dirs["search"] / "experimental_3D"
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs

def parse_folder(name):
    match = FOLDER_PATTERN.match(name)
    if match is None:
        return None
    return {
        "base_model": match.group(1), "model": match.group(2), "dataset": match.group(3),
        "lr": float(match.group(4)), "base_reg": float(match.group(5)), "reg": float(match.group(6)),
        "epoch_ceiling": int(match.group(7)), "wd": float(match.group(8)),
    }

def load_history(path):
    history = np.load(path, allow_pickle=True)
    return history.item() if isinstance(history, np.ndarray) and history.shape == () else history

def describe_dataset(name):
    '''
    Splits a test input name into its kind ("standard" survey combination or single-survey "ablation") and, for
    ablations, the survey present, the fill of the other channels, the number of channels and the survey's channel.
    '''
    match = ABLATION_PATTERN.match(name)
    if match is None:
        return {"test_kind": "standard", "ablation_survey": None, "fill": None, "n_channels": len(name) if name not in SINGLE_CHANNEL else 1, "position": None}
    position = int(match.group(4)) if match.group(4) else None
    return {"test_kind": "ablation", "ablation_survey": SURVEY_NAMES[match.group(1)], "fill": match.group(2), "n_channels": int(match.group(3)), "position": position}

def ablation_label(name):
    info = describe_dataset(name)
    if info["test_kind"] != "ablation":
        return name
    label = f"{info['ablation_survey']} {FILL_NAMES[info['fill']].lower()}"
    return label + (f" (ch{info['position']})" if info["position"] else "")

def ablation_short(name):
    info = describe_dataset(name)
    fill = {"REP": "rep", "ZER": "zero", "NSE": "noise"}[info["fill"]]
    return f"{SURVEY_LETTERS[info['ablation_survey']]} {fill}" + (f" ch{info['position']}" if info["position"] else "")

def ablation_datasets(n_channels):
    '''
    The single-survey test inputs of an n-channel model, in the order masterRunModel.py tests them.
    '''
    return [f"{s}{fill}{n_channels}{position}" for s in "FLN" for fill, position in [("REP", "")] + [(fill, f"_CH{k}") for fill in ["ZER", "NSE"] for k in range(1, n_channels + 1)]]

def survey_relation(model, survey, fill, position):
    '''
    Where the one survey left in an ablation input sits relative to the channels the model was trained on.
    '''
    letter = SURVEY_LETTERS[survey]
    if letter not in model:
        return "absent"
    if fill == "REP":
        return "repeated"
    return "native" if model.index(letter) + 1 == position else "moved"

def label_panels(axes):
    for k, ax in enumerate(axes):
        ax.text(0, 1.04, f"({chr(97 + k)})", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", color=INK)

def panel_grid(n, columns, width=7.5, height=2.3, **kwargs):
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(width, height * rows), squeeze=False, **kwargs)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return fig, axes

def model_order(model):
    return ALL_SURVEYS.index(model) if model in ALL_SURVEYS else len(ALL_SURVEYS)

def channel_group(survey):
    for group, surveys in CHANNEL_GROUPS.items():
        if survey in surveys:
            return group
    return "other"

def strategy_label(base_model, model):
    return model if base_model == "NONE" else f"{base_model}$\\rightarrow${model}"

def plain_label(base_model, model):
    return model if base_model == "NONE" else f"{base_model}->{model}"

def save_figure(fig, path, **kwargs):
    fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"Saved figure {path}")

def save_csv(df, dirs, name):
    path = dirs["csv"] / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved table {path}")
# endregion

# region LATEX
LATEX_FILES = []

def latex_escape(text):
    text = str(text)
    if "$" in text or "\\" in text:
        return text # | Already LaTeX
    return text.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")

def sci(value):
    '''
    Formats a hyperparameter as LaTeX scientific notation, e.g. 3.16e-4 -> $3.16\\times10^{-4}$.
    '''
    if value is None or not np.isfinite(value):
        return "--"
    if value == 0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    if np.isclose(mantissa, 1):
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"

def mean_std(mean, std, digits=3):
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"

def write_latex_table(dirs, name, df, caption, label, column_format=None, bold_max=(), bold_min=(), note=None, resize=False):
    '''
    Writes df (already formatted as strings, except columns in bold_max/bold_min, which are given as
    (display_column, numeric_column) pairs) to a booktabs table.
    '''
    display = df.copy()
    for column in display.columns:
        if df[column].dtype == object or pd.api.types.is_string_dtype(df[column]):
            display[column] = [latex_escape(v) for v in display[column]]
    for display_column, numeric_column in bold_max:
        best = df[numeric_column].max()
        display[display_column] = [f"\\textbf{{{v}}}" if np.isclose(n, best) else v for v, n in zip(df[display_column], df[numeric_column])]
    for display_column, numeric_column in bold_min:
        best = df[numeric_column].min()
        display[display_column] = [f"\\textbf{{{v}}}" if np.isclose(n, best) else v for v, n in zip(df[display_column], df[numeric_column])]
    numeric_columns = {numeric for _, numeric in list(bold_max) + list(bold_min)}
    display = display[[c for c in display.columns if c not in numeric_columns or c in [d for d, _ in list(bold_max) + list(bold_min)]]]

    column_format = column_format or ("l" + "c" * (len(display.columns) - 1))
    lines = [
        "% Generated by resultsSummary.py. Requires \\usepackage{booktabs}" + (" and \\usepackage{graphicx}." if resize else "."),
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%" if resize else "",
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
        " & ".join(latex_escape(c) for c in display.columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}" + ("}" if resize else "")]
    if note:
        lines.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\footnotesize {note}}}")
    lines.append("\\end{table}")

    path = dirs["latex"] / f"{name}.tex"
    path.write_text("\n".join(line for line in lines if line) + "\n", encoding="utf-8")
    LATEX_FILES.append(path.name)
    print(f"Saved LaTeX table {path}")

def write_latex_index(dirs):
    lines = ["% Generated by resultsSummary.py: \\input this file to include every table."]
    lines += [f"\\input{{{name}}}" for name in LATEX_FILES]
    (dirs["latex"] / "all_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
# endregion

# region LOADING
def load_training_runs(stage):
    '''
    RETURNS:
    --------
        > folds: one row per (config folder, fold) with the early-stopping epoch and the train/validation metrics at it
        > histories: {folder name: [history dict per fold]} (transfer runs have the phase-1 history prepended)
    '''
    training_dir = STAGE_DIRECTORIES[stage] / "Training"
    rows, histories = [], {}
    if not training_dir.is_dir():
        return pd.DataFrame(), histories

    for folder in sorted(training_dir.iterdir()):
        parameters = parse_folder(folder.name) if folder.is_dir() else None
        if parameters is None:
            continue
        histories[folder.name] = []
        for history_path in sorted(folder.glob("loss_fold_*.npy"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1])):
            fold = int(re.findall(r"\d+", history_path.stem)[-1])
            history = load_history(history_path)
            head_path = folder / f"head_loss_fold_{fold}.npy"
            head_epochs = 0
            if head_path.exists():
                head = load_history(head_path)
                head_epochs = len(head["val_loss"])
                history = {key: list(head[key]) + list(history[key]) for key in history if key in head}
            time_path = folder / f"training_time_fold_{fold}.npy"
            best = head_epochs + int(np.nanargmin(history["val_loss"][head_epochs:]))
            histories[folder.name].append({"fold": fold, "history": history, "head_epochs": head_epochs})
            rows.append({
                "stage": stage, "folder": folder.name, **parameters,
                "group": channel_group(parameters["model"]),
                "transfer_learning": parameters["base_model"] != "NONE",
                "fold": fold,
                "epochs_run": len(history["val_loss"]),
                "best_epoch": best + 1,
                "val_loss": history["val_loss"][best],
                "val_accuracy": history["val_accuracy"][best],
                "train_loss": history["loss"][best],
                "train_accuracy": history["accuracy"][best],
                "generalisation_gap": history["val_loss"][best] - history["loss"][best],
                "training_time_min": float(np.load(time_path)) / 60 if time_path.exists() else np.nan,
            })
    return pd.DataFrame(rows), histories

def aggregate_configs(folds):
    '''
    One row per config: fold means and standard deviations, ranked per (stage, base, model) by mean validation loss.
    Also adds:
        > complete: whether every fold of the config's stage has finished. Only complete configs are ranked, so a config
          still running (fewer folds) is never compared with finished ones.
        > val_loss_shared_mean / _std: validation loss over the folds every stage trains (the COARSE stage trains only the
          first folds of the same split, and the later folds are harder), so configs from different stages compare like for like.
    '''
    if folds.empty:
        return folds
    keys = ["stage", "folder", "base_model", "model", "group", "transfer_learning", "lr", "reg", "wd"]
    metrics = ["val_loss", "val_accuracy", "train_loss", "train_accuracy", "generalisation_gap", "best_epoch", "epochs_run", "training_time_min"]
    grouped = folds.groupby(keys)
    configs = grouped[metrics].mean().add_suffix("_mean").join(grouped[metrics].std().add_suffix("_std")).reset_index()
    configs["n_folds"] = grouped.size().values
    configs["complete"] = configs["n_folds"] == configs.groupby("stage")["n_folds"].transform("max")

    shared_folds = int(folds[folds["stage"] == "COARSE"]["fold"].max()) if (folds["stage"] == "COARSE").any() else int(folds["fold"].max())
    shared = folds[folds["fold"] <= shared_folds].groupby(["stage", "folder"])["val_loss"].agg(["mean", "std"])
    shared.columns = ["val_loss_shared_mean", "val_loss_shared_std"]
    configs = configs.merge(shared.reset_index(), on=["stage", "folder"], how="left")
    configs["shared_folds"] = shared_folds

    configs = configs.sort_values(["stage", "base_model", "model", "val_loss_mean"]).reset_index(drop=True)
    configs["rank"] = np.nan
    complete = configs["complete"]
    configs.loc[complete, "rank"] = configs[complete].groupby(["stage", "base_model", "model"]).cumcount() + 1
    return configs

def load_test_runs():
    '''
    RETURNS:
    --------
        > runs: one dict per Testing folder with the labels, per-fold and ensemble probabilities and metadata
    '''
    testing_dir = STAGE_DIRECTORIES["FINAL"] / "Testing"
    runs = []
    if not testing_dir.is_dir():
        return runs
    for folder in sorted(testing_dir.iterdir()):
        parameters = parse_folder(folder.name) if folder.is_dir() else None
        if parameters is None or not (folder / "y_test.npy").exists():
            continue
        y_true = np.array([CLASSES.index(str(label)) for label in np.load(folder / "y_test.npy", allow_pickle=True)])
        fold_probs = {}
        for path in sorted(folder.glob("y_pred_test_fold_*.npy")):
            fold_probs[int(re.findall(r"\d+", path.stem)[-1])] = np.load(path)
        if len(fold_probs) == 0:
            continue
        ensemble_path = folder / "y_pred_test_ensemble.npy"
        ensemble = np.load(ensemble_path) if ensemble_path.exists() else np.mean(list(fold_probs.values()), axis=0)
        names_path = folder / "names_test.npy"
        runs.append({
            "folder": folder.name, **parameters,
            "group": channel_group(parameters["model"]),
            "transfer_learning": parameters["base_model"] != "NONE",
            "in_domain": parameters["dataset"] == parameters["model"],
            **describe_dataset(parameters["dataset"]),
            "label": plain_label(parameters["base_model"], parameters["model"]),
            "y_true": y_true, "fold_probs": fold_probs, "ensemble": ensemble,
            "names": np.load(names_path, allow_pickle=True) if names_path.exists() else None,
        })
    return runs
# endregion

# region METRICS
def classification_metrics(y_true, probabilities):
    y_pred = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASSES)), zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
        "weighted_f1": np.average(f1, weights=support) if support.sum() else np.nan,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "log_loss": log_loss(y_true, np.clip(probabilities, 1e-7, 1) / np.clip(probabilities, 1e-7, 1).sum(axis=1, keepdims=True), labels=range(len(CLASSES))),
    }
    try:
        metrics["roc_auc_macro"] = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro", labels=range(len(CLASSES)))
    except ValueError:
        metrics["roc_auc_macro"] = np.nan
    for i, name in enumerate(CLASSES):
        metrics[f"precision_{name}"] = precision[i]
        metrics[f"recall_{name}"] = recall[i]
        metrics[f"f1_{name}"] = f1[i]
    return metrics

def bootstrap_ci(y_true, probabilities, n=N_BOOTSTRAP, seed=SEED):
    '''
    95% percentile bootstrap intervals of accuracy and macro F1 over test-set resamples (vectorised over resamples).
    RETURNS: (accuracy_low, accuracy_high, macro_f1_low, macro_f1_high)
    '''
    rng = np.random.default_rng(seed)
    y_pred = np.argmax(probabilities, axis=1)
    idx = rng.integers(0, len(y_true), (n, len(y_true)))
    truth, pred = y_true[idx], y_pred[idx]
    accuracy = np.mean(truth == pred, axis=1)
    f1 = []
    for c in range(len(CLASSES)):
        tp = np.sum((truth == c) & (pred == c), axis=1)
        fp = np.sum((truth != c) & (pred == c), axis=1)
        fn = np.sum((truth == c) & (pred != c), axis=1)
        f1.append(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0))
    macro_f1 = np.mean(f1, axis=0)
    return (*np.percentile(accuracy, [2.5, 97.5]), *np.percentile(macro_f1, [2.5, 97.5]))

def holm(p_values):
    '''
    Holm-Bonferroni adjusted p-values (controls the family-wise error rate over every test in the family).
    '''
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.maximum.accumulate((len(p_values) - np.arange(len(p_values))) * p_values[order])
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result

def mcnemar(y_true, probabilities_a, probabilities_b):
    '''
    Exact McNemar test on two models' predictions of the same test set. Returns (n A right & B wrong, n B right & A wrong, p).
    '''
    correct_a = np.argmax(probabilities_a, axis=1) == y_true
    correct_b = np.argmax(probabilities_b, axis=1) == y_true
    only_a, only_b = int(np.sum(correct_a & ~correct_b)), int(np.sum(~correct_a & correct_b))
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b > 0 else 1.0
    return only_a, only_b, p

def evaluate_test_runs(runs):
    fold_rows, ensemble_rows = [], []
    meta_keys = ["folder", "base_model", "model", "dataset", "lr", "reg", "wd", "group", "transfer_learning", "in_domain",
                 "test_kind", "ablation_survey", "fill", "n_channels", "position", "label"]
    for run in runs:
        meta = {key: run[key] for key in meta_keys}
        for fold, probabilities in sorted(run["fold_probs"].items()):
            fold_rows.append({**meta, "fold": fold, **classification_metrics(run["y_true"], probabilities)})
        ensemble_metrics = classification_metrics(run["y_true"], run["ensemble"])
        acc_low, acc_high, f1_low, f1_high = bootstrap_ci(run["y_true"], run["ensemble"])
        ensemble_rows.append({**meta, "n_test": len(run["y_true"]), "n_folds": len(run["fold_probs"]), **ensemble_metrics,
                              "accuracy_ci_low": acc_low, "accuracy_ci_high": acc_high,
                              "macro_f1_ci_low": f1_low, "macro_f1_ci_high": f1_high})
    folds = pd.DataFrame(fold_rows)
    ensemble = pd.DataFrame(ensemble_rows)
    if not folds.empty:
        stats = folds.groupby("folder")[["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
                                         "mcc", "cohen_kappa", "roc_auc_macro"]].agg(["mean", "std"])
        stats.columns = [f"fold_{metric}_{stat}" for metric, stat in stats.columns]
        ensemble = ensemble.merge(stats.reset_index(), on="folder", how="left")
    return folds, ensemble

def generalisation_scores(ensemble):
    '''
    One row per trained model: mean and worst-case ensemble macro F1 over every standard test set in its channel group
    (real survey combinations, not ablations), the in-domain score, the out-of-domain mean and retention (out-of-domain / in-domain).
    For duo and trio models the out-of-domain sets are of two kinds, reported separately: the same surveys in another channel
    order (channel-order robustness) and, for duo models, a different survey in a channel (cross-survey generalisation).
    Every trio test set holds all three surveys, so a trio model's out-of-domain score is channel-order robustness only.
    '''
    rows = []
    for (base_model, model), df in ensemble[ensemble["test_kind"] == "standard"].groupby(["base_model", "model"]):
        in_domain = df[df["in_domain"]]
        out_domain = df[~df["in_domain"]]
        reordered = out_domain[[set(d) == set(model) for d in out_domain["dataset"]]] if channel_group(model) != "single" else out_domain.iloc[:0]
        other_surveys = out_domain[[set(d) != set(model) for d in out_domain["dataset"]]] if channel_group(model) != "single" else out_domain
        rows.append({
            "base_model": base_model, "model": model, "label": plain_label(base_model, model),
            "group": channel_group(model), "transfer_learning": base_model != "NONE",
            "lr": df["lr"].iloc[0], "reg": df["reg"].iloc[0], "wd": df["wd"].iloc[0],
            "n_test_datasets": len(df),
            "mean_macro_f1": df["macro_f1"].mean(),
            "worst_macro_f1": df["macro_f1"].min(),
            "worst_dataset": df.loc[df["macro_f1"].idxmin(), "dataset"],
            "in_domain_macro_f1": in_domain["macro_f1"].mean() if len(in_domain) else np.nan,
            "out_domain_macro_f1": out_domain["macro_f1"].mean() if len(out_domain) else np.nan,
            "other_order_macro_f1": reordered["macro_f1"].mean() if len(reordered) else np.nan,
            "other_survey_macro_f1": other_surveys["macro_f1"].mean() if len(other_surveys) else np.nan,
            "mean_accuracy": df["accuracy"].mean(),
            "mean_mcc": df["mcc"].mean(),
        })
    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores["retention"] = scores["out_domain_macro_f1"] / scores["in_domain_macro_f1"]
        scores = scores.sort_values(["group", "mean_macro_f1"], ascending=[True, False]).reset_index(drop=True)
        scores["rank_in_group"] = scores.groupby("group").cumcount() + 1
    return scores
# endregion

# region DATASET
def dataset_summary(dirs):
    '''
    Class counts of the original train/validation split, the augmented train/validation set and the test set.
    '''
    sources = {
        "Train/val (sources)": Path(SOURCEDATADIRECTORY) / "y_train_val.npy",
        "Train/val (augmented)": EXPERIMENT_DIRECTORY / "Data" / "y_train_val.npy",
        "Test (sources)": EXPERIMENT_DIRECTORY / "Data" / "y_test.npy",
    }
    rows = []
    for split, path in sources.items():
        if not path.exists():
            continue
        labels = np.load(path, allow_pickle=True).astype(str)
        row = {"Split": split}
        for name in CLASSES:
            row[name] = int(np.sum(labels == name))
        row["Total"] = len(labels)
        rows.append(row)
    if len(rows) == 0:
        print("No data splits found, skipping the dataset table.")
        return
    df = pd.DataFrame(rows)
    save_csv(df, dirs, "dataset_class_counts")
    write_latex_table(dirs, "dataset_class_counts", df.astype(str), "Number of images per morphological class in each data split. The augmented set holds only the augmented copies of the train/validation sources.", "tab:dataset_class_counts")
# endregion

# region SEARCH
def search_progression(configs, dirs):
    '''
    Tables and charts of the COARSE -> FINE -> FINAL progression of the scratch (non-transfer) models.
    '''
    scratch = configs[~configs["transfer_learning"]]
    if scratch.empty:
        print("No search or training results found, skipping the search progression.")
        return
    save_csv(scratch, dirs, "search_all_configs")
    running = scratch[~scratch["complete"]]
    if len(running):
        print(f"Leaving out {len(running)} configs that have not finished all their folds yet: {', '.join(sorted(set(running['stage'] + ' ' + running['model'])))}")
    scratch = scratch[scratch["complete"]]
    best = scratch[scratch["rank"] == 1].copy()
    save_csv(best, dirs, "search_best_per_stage")
    shared = f"folds 1--{int(scratch['shared_folds'].iloc[0])}"

    # > Progression table: chosen hyperparameters and validation loss at each stage
    rows = []
    for survey in ALL_SURVEYS:
        row = {"Survey": survey}
        numeric = {}
        for stage in STAGES:
            entry = best[(best["stage"] == stage) & (best["model"] == survey)]
            if entry.empty:
                row[f"{stage.title()} (lr, reg, wd)"] = "--"
                row[f"{stage.title()} val. loss ({shared})"] = "--"
                if stage != "COARSE":
                    row[f"{stage.title()} val. loss (all folds)"] = "--"
                numeric[stage] = np.nan
                continue
            entry = entry.iloc[0]
            row[f"{stage.title()} (lr, reg, wd)"] = f"{sci(entry['lr'])}, {sci(entry['reg'])}, {sci(entry['wd'])}"
            row[f"{stage.title()} val. loss ({shared})"] = mean_std(entry["val_loss_shared_mean"], entry["val_loss_shared_std"])
            if stage != "COARSE":
                row[f"{stage.title()} val. loss (all folds)"] = mean_std(entry["val_loss_mean"], entry["val_loss_std"])
            numeric[stage] = entry["val_loss_mean"]
        if all(not np.isfinite(v) for v in numeric.values()):
            continue
        rows.append(row)
    table = pd.DataFrame(rows)
    # > FINAL uses the FINE best, so its hyperparameter column is dropped from the LaTeX table
    latex = table.drop(columns=[c for c in table.columns if c.startswith("Final (")])
    write_latex_table(dirs, "search_progression", latex,
        f"Hyperparameter search progression. For each input, the configuration with the lowest mean validation loss in the coarse and fine searches, and the final model trained with the fine-search optimum (mean $\\pm$ std over folds). The coarse search trains only {shared.replace('--', '--')} of the same 5-fold split, so every stage is also scored on those folds to compare like for like; the later folds are harder, so all-fold losses are higher. Inputs without a coarse entry took their fine-search centre from the coarse result of the same survey combination.",
        "tab:search_progression", column_format="l" + "c" * (len(latex.columns) - 1), resize=True)

    # > Search space table
    space = []
    for stage in ["COARSE", "FINE"]:
        df = scratch[scratch["stage"] == stage]
        if df.empty:
            continue
        space.append({
            "Stage": stage.title(), "Inputs": str(df["model"].nunique()),
            "Learning rates": ", ".join(sci(v) for v in sorted(df["lr"].unique())) if stage == "COARSE" else f"{df['lr'].nunique()} values, $\\pm$ half a decade",
            "L2 coefficients": ", ".join(sci(v) for v in sorted(df["reg"].unique())) if stage == "COARSE" else f"{df['reg'].nunique()} values, $\\pm$ half a decade",
            "Weight decays": ", ".join(sci(v) for v in sorted(df["wd"].unique())) if stage == "COARSE" else f"{df['wd'].nunique()} values, $\\pm$ half a decade",
            "Configs": str(len(df)), "Folds": str(int(df["n_folds"].max())),
        })
    if space:
        write_latex_table(dirs, "search_space", pd.DataFrame(space), "Hyperparameter search space per stage. Fine-search values are centred on each input's coarse optimum.", "tab:search_space", column_format="llp{3cm}p{3cm}p{2.5cm}cc")

    plot_search_progression(best, dirs)
    plot_search_spread(scratch, dirs)
    plot_search_overview(best, scratch, dirs)
    plot_search_spread_broken(scratch, dirs)
    plot_search_overview_broken(best, scratch, dirs)
    for stage in ["COARSE", "FINE"]:
        df = scratch[scratch["stage"] == stage]
        if not df.empty:
            plot_search_heatmaps(df, stage, dirs)
            plot_search_sensitivity(df, stage, dirs)
            plot_search_3d(df, stage, dirs)

def draw_search_progression(ax, best):
    '''
    Filled markers: the best config of each stage scored on the folds every stage shares (like for like). Hollow markers: the
    fine and final scores over all their folds, which are higher because the later folds are harder.
    '''
    surveys = [s for s in ALL_SURVEYS if s in set(best["model"])]
    offsets = {"COARSE": -0.2, "FINE": 0.0, "FINAL": 0.2}
    shared = int(best["shared_folds"].iloc[0]) if len(best) else 0
    for stage in STAGES:
        df = best[best["stage"] == stage].set_index("model")
        x = [i + offsets[stage] for i, s in enumerate(surveys) if s in df.index]
        if not x:
            continue
        y = [df.loc[s, "val_loss_shared_mean"] for s in surveys if s in df.index]
        e = [df.loc[s, "val_loss_shared_std"] for s in surveys if s in df.index]
        ax.errorbar(x, y, yerr=e, fmt="o", color=STAGE_COLOURS[stage], ecolor=STAGE_COLOURS[stage], elinewidth=1, capsize=0, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, label=stage.title())
        if stage != "COARSE":
            y_all = [df.loc[s, "val_loss_mean"] for s in surveys if s in df.index]
            ax.scatter(x, y_all, s=30, facecolors=SURFACE, edgecolors=STAGE_COLOURS[stage], linewidths=1.2, zorder=3)
    for boundary in [len(SINGLE_CHANNEL) - 0.5, len(SINGLE_CHANNEL) + len(DUO_CHANNEL) - 0.5]:
        if boundary < len(surveys):
            ax.axvline(boundary, color=GRIDLINE, linewidth=1)
    ax.set_xticks(range(len(surveys)))
    ax.set_xticklabels(surveys, rotation=45, ha="right")
    ax.set_ylabel("Best mean validation loss")
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", linestyle="", color=INK_SECONDARY, markersize=6, label=f"Folds 1-{shared} (all stages)"))
    handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_SECONDARY, markersize=6, label="All 5 folds"))
    ax.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.3))

def plot_search_progression(best, dirs):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    draw_search_progression(ax, best)
    ax.set_title("Validation loss of the best configuration at each search stage")
    save_figure(fig, dirs["search"] / "search_progression.png")

# > The spread plots compare stages, so they use the validation loss over the folds every stage shares
SPREAD_COLUMN = "val_loss_shared_mean"

def draw_search_spread(ax, search):
    surveys = [s for s in ALL_SURVEYS if s in set(search["model"])]
    rng = np.random.default_rng(SEED)
    offsets = {"COARSE": -0.18, "FINE": 0.18}
    for stage in ["COARSE", "FINE"]:
        df = search[search["stage"] == stage]
        for i, survey in enumerate(surveys):
            values = df[df["model"] == survey][SPREAD_COLUMN].values
            if len(values) == 0:
                continue
            xs = i + offsets[stage] + rng.uniform(-0.08, 0.08, len(values))
            ax.scatter(xs, values, s=10, color=STAGE_COLOURS[stage], alpha=0.55, linewidths=0)
    ax.set_xticks(range(len(surveys)))
    ax.set_xticklabels(surveys, rotation=45, ha="right")
    ax.set_ylabel(f"Mean validation loss, folds 1-{int(search['shared_folds'].iloc[0])}")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], marker="o", linestyle="", color=STAGE_COLOURS[s], label=s.title()) for s in ["COARSE", "FINE"]]
    ax.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))

def plot_search_spread(scratch, dirs):
    '''
    Every config's validation loss per input and stage: shows how the fine search narrows onto the good region.
    '''
    search = scratch[scratch["stage"].isin(["COARSE", "FINE"])]
    if search.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    draw_search_spread(ax, search)
    ax.set_title("Validation loss of every searched configuration (lower is better)")
    save_figure(fig, dirs["search"] / "search_spread.png")

def plot_search_overview(best, scratch, dirs):
    '''
    Combined figure: the best config per stage above every searched config.
    '''
    search = scratch[scratch["stage"].isin(["COARSE", "FINE"])]
    if search.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.4))
    draw_search_progression(axes[0], best)
    axes[0].set_title("Best configuration at each search stage")
    draw_search_spread(axes[1], search)
    axes[1].set_title("Every searched configuration (lower is better)")
    label_panels(axes)
    fig.tight_layout()
    save_figure(fig, dirs["search"] / "search_overview_combined.png")

def draw_search_spread_broken(top, bottom, search):
    '''
    Same as draw_search_spread, with the y axis broken at the widest gap between losses so the collapsed runs
    (stuck at ln 3, uniform predictions) sit in a thin top panel and the useful range fills the bottom one.
    '''
    draw_search_spread(top, search)
    draw_search_spread(bottom, search)
    top.get_legend().remove()
    values = np.sort(search[SPREAD_COLUMN].values)
    k = np.argmax(np.diff(values))
    pad = 0.05 * (values[k] - values[0])
    bottom.set_ylim(values[0] - pad, values[k] + pad)
    top.set_ylim(values[k + 1] - pad / 4, values[-1] + pad / 4)
    top.set_yticks([round(np.median(values[k + 1:]), 2)])
    top.set_ylabel("")
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    marks = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8, linestyle="none", color=MUTED, mec=MUTED, mew=1, clip_on=False)
    top.plot([0], [0], transform=top.transAxes, **marks)
    bottom.plot([0], [1], transform=bottom.transAxes, **marks)

def plot_search_spread_broken(scratch, dirs):
    search = scratch[scratch["stage"].isin(["COARSE", "FINE"])]
    if search.empty:
        return
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7.5, 3.6), gridspec_kw={"height_ratios": [1, 5], "hspace": 0.08})
    draw_search_spread_broken(top, bottom, search)
    top.set_title("Validation loss of every searched configuration (lower is better)")
    save_figure(fig, dirs["search"] / "search_spread_broken.png")

def plot_search_overview_broken(best, scratch, dirs):
    search = scratch[scratch["stage"].isin(["COARSE", "FINE"])]
    if search.empty:
        return
    fig = plt.figure(figsize=(7.5, 7.2))
    outer = fig.add_gridspec(2, 1, hspace=0.75)
    progression = fig.add_subplot(outer[0])
    inner = outer[1].subgridspec(2, 1, height_ratios=[1, 5], hspace=0.08)
    top = fig.add_subplot(inner[0])
    bottom = fig.add_subplot(inner[1], sharex=top)
    draw_search_progression(progression, best)
    progression.set_title("Best configuration at each search stage")
    draw_search_spread_broken(top, bottom, search)
    top.set_title("Every searched configuration (lower is better)")
    label_panels([progression, top])
    save_figure(fig, dirs["search"] / "search_overview_broken_combined.png")

SEARCH_AXES = {"lr": "Learning rate", "reg": "L2 coefficient", "wd": "Weight decay"}
SEARCH_PAIRS = [("lr", "reg"), ("reg", "wd"), ("lr", "wd")] # | (rows, columns): LR vs L2, L2 vs WD, LR vs WD

def tick_label(value):
    if value == 0:
        return "0"
    mantissa = value / 10**np.floor(np.log10(abs(value)))
    return f"{value:.0e}" if np.isclose(mantissa, 1) else f"{value:.1e}"

def draw_search_heatmap(ax, df, rows, columns, norm=None):
    '''
    Best mean validation loss for each (rows, columns) pair of hyperparameters, minimising over the third. The box marks the best cell.
    '''
    pivot = df.pivot_table(index=rows, columns=columns, values="val_loss_mean", aggfunc="min").sort_index(ascending=False)
    image = annotate_heatmap(ax, pivot.values, cmap=SEQUENTIAL.reversed(), norm=norm)
    best = np.unravel_index(np.nanargmin(pivot.values), pivot.values.shape)
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.5))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([tick_label(v) for v in pivot.columns], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([tick_label(v) for v in pivot.index], fontsize=6)
    ax.set_xlabel(SEARCH_AXES[columns])
    ax.set_ylabel(SEARCH_AXES[rows])
    return image

def survey_norm(df):
    return Normalize(df["val_loss_mean"].min(), df["val_loss_mean"].max())

def plot_search_heatmaps(df, stage, dirs):
    '''
    For every input, the three hyperparameter planes side by side (LR vs L2, L2 vs WD, LR vs WD), each showing the best mean
    validation loss over the hyperparameter left out, on one colour scale per input. Saves one figure per input, a combined
    figure with one row per input, and the LR vs L2 overview of every input.
    '''
    surveys = [s for s in ALL_SURVEYS if s in set(df["model"])]
    for survey in surveys:
        part = df[df["model"] == survey]
        norm = survey_norm(part)
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), layout="constrained")
        for ax, (rows, columns) in zip(axes, SEARCH_PAIRS):
            image = draw_search_heatmap(ax, part, rows, columns, norm=norm)
        fig.colorbar(image, ax=list(axes), label="Mean validation loss", shrink=0.9)
        fig.suptitle(f"{stage.title()} search, {survey}: best mean validation loss over the third hyperparameter (darker is lower, box is best)", fontsize=9)
        save_figure(fig, dirs["search_heatmaps"] / f"search_heatmaps_{stage.lower()}_{survey}.png")

    if len(surveys) > 1:
        fig, axes = plt.subplots(len(surveys), 3, figsize=(7.5, 2.3 * len(surveys)), layout="constrained", squeeze=False)
        for row, survey in zip(axes, surveys):
            part = df[df["model"] == survey]
            norm = survey_norm(part)
            for ax, (rows, columns) in zip(row, SEARCH_PAIRS):
                draw_search_heatmap(ax, part, rows, columns, norm=norm)
            row[1].set_title(survey, fontsize=10, fontweight="bold")
        fig.suptitle(f"{stage.title()} search: best mean validation loss over the third hyperparameter\n(colour scaled per input, darker is lower, box is best)", fontsize=10)
        save_figure(fig, dirs["search"] / f"search_heatmaps_{stage.lower()}_combined.png")

    # > Overview: LR vs L2 for every input in one grid
    columns = min(4, len(surveys))
    rows = int(np.ceil(len(surveys) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 2.1 * rows), squeeze=False)
    for ax, survey in zip(axes.flat, surveys):
        draw_search_heatmap(ax, df[df["model"] == survey], "lr", "reg")
        ax.set_title(survey)
    for ax in list(axes.flat)[len(surveys):]:
        ax.axis("off")
    fig.suptitle(f"{stage.title()} search: best mean validation loss over weight decay (darker is lower, box is best)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, dirs["search"] / f"search_heatmaps_{stage.lower()}.png")

SENSITIVITY_GROUPS = {"single": SINGLE_CHANNEL, "multichannel": DUO_CHANNEL + TRIO_CHANNEL}
SENSITIVITY_PARAMETERS = [("lr", "Learning rate", "log"), ("reg", "L2 coefficient", "symlog"), ("wd", "Weight decay", "log")]

def draw_sensitivity(axes, df, surveys, group):
    for ax, (column, title, scale) in zip(axes, SENSITIVITY_PARAMETERS):
        for k, survey in enumerate(surveys):
            curve = df[df["model"] == survey].groupby(column)["val_loss_mean"].min()
            # > A combined panel can have up to 12 inputs; colours are assigned in fixed order by input
            colour = SURVEY_COLOURS[survey] if group == "single" else SERIES[k % len(SERIES)]
            ax.plot(curve.index, curve.values, marker="o", markersize=4, linewidth=1.5, color=colour, label=survey,
                    linestyle="-" if k < len(SERIES) else "--")
        if scale == "symlog":
            ax.set_xscale("symlog", linthresh=1e-5)
        else:
            ax.set_xscale("log")
        ax.set_xlabel(title)
    axes[0].set_ylabel("Best mean validation loss")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1 if len(surveys) <= 8 else 2)

def plot_search_sensitivity(df, stage, dirs):
    '''
    Best validation loss reachable at each value of one hyperparameter, minimising over the other two.
    One figure per input group, plus a combined figure with one row per group.
    '''
    panels = []
    for group, members in SENSITIVITY_GROUPS.items():
        surveys = [s for s in members if s in set(df["model"])]
        if not surveys:
            continue
        panels.append((group, surveys))
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), sharey=True)
        draw_sensitivity(axes, df, surveys, group)
        fig.suptitle(f"{stage.title()} search sensitivity ({group} inputs)", fontsize=10)
        fig.tight_layout()
        save_figure(fig, dirs["search"] / f"search_sensitivity_{stage.lower()}_{group}.png")
    if len(panels) > 1:
        fig, axes = plt.subplots(len(panels), 3, figsize=(7.5, 2.5 * len(panels)), sharey="row", squeeze=False)
        for row, (group, surveys) in zip(axes, panels):
            draw_sensitivity(row, df, surveys, group)
            row[1].set_title(f"{group.title()} inputs")
        label_panels(axes[:, 0])
        fig.suptitle(f"{stage.title()} search sensitivity", fontsize=10)
        fig.tight_layout()
        save_figure(fig, dirs["search"] / f"search_sensitivity_{stage.lower()}_combined.png")

# > Experimental views of the full (lr, reg, wd) grid, saved under search/experimental_3D
SEARCH_GRID = [("lr", "Learning rate"), ("reg", "L2 coefficient"), ("wd", "Weight decay")]

def collapse_threshold(values):
    '''
    Loss above which runs count as collapsed (e.g. stuck at ln 3, uniform predictions): the midpoint of the widest gap
    between sorted losses, if that gap spans over half the range. None when there is no such gap.
    '''
    values = np.sort(values)
    if len(values) < 2:
        return None
    gaps = np.diff(values)
    k = np.argmax(gaps)
    return (values[k] + values[k + 1]) / 2 if gaps[k] > 0.5 * (values[-1] - values[0]) else None

def grid_norm(part):
    '''
    Colour scale over the losses that did not collapse, so the collapsed runs do not squash it.
    '''
    values = part["val_loss_mean"].values
    threshold = collapse_threshold(values)
    kept = values[values < threshold] if threshold is not None else values
    return Normalize(kept.min(), kept.max()), threshold

def grid_index(values):
    '''
    Position of each hyperparameter value on its grid (0, 1, 2, ...), so the axes are evenly spaced and reg = 0 has a place.
    '''
    levels = sorted(set(values))
    return np.array([levels.index(v) for v in values]), levels

def draw_parallel(ax, part, norm, threshold):
    cmap = SEQUENTIAL.reversed()
    span = norm.vmax - norm.vmin or 1
    coords = []
    for column, _ in SEARCH_GRID:
        index, levels = grid_index(part[column].values)
        coords.append(index / max(len(levels) - 1, 1))
        x = len(coords) - 1
        for i, level in enumerate(levels):
            ax.text(x - 0.05, i / max(len(levels) - 1, 1), tick_label(level), ha="right", va="center", fontsize=6, color=INK_SECONDARY,
                    zorder=4, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6, alpha=0.9))
    losses = part["val_loss_mean"].values
    collapsed = losses > threshold if threshold is not None else np.zeros(len(losses), bool)
    coords.append(np.where(collapsed, 1.15, (losses - norm.vmin) / span))
    coords = np.array(coords).T
    best = np.argmin(losses)
    for i in np.argsort(-losses):
        if collapsed[i]:
            ax.plot(range(4), coords[i], color=MUTED, linewidth=0.6, linestyle="--", alpha=0.5)
        elif i != best:
            ax.plot(range(4), coords[i], color=cmap(norm(losses[i])), linewidth=1, alpha=0.8)
    ax.plot(range(4), coords[best], color=INK, linewidth=2.2, zorder=3)
    for value in [norm.vmin, (norm.vmin + norm.vmax) / 2, norm.vmax]:
        ax.text(3.05, (value - norm.vmin) / span, f"{value:.3f}", ha="left", va="center", fontsize=6, color=INK_SECONDARY)
    if collapsed.any():
        ax.text(3.05, 1.15, f"{np.median(losses[collapsed]):.2f} (collapsed)", ha="left", va="center", fontsize=6, color=MUTED)
    for x in range(4):
        ax.axvline(x, color=MUTED, linewidth=0.8, zorder=0)
    ax.set_xlim(-0.35, 3.5)
    ax.set_ylim(-0.08, 1.25 if collapsed.any() else 1.08)
    ax.set_xticks(range(4))
    ax.set_xticklabels([label for _, label in SEARCH_GRID] + ["Val. loss"], fontsize=7)
    ax.set_yticks([])
    ax.grid(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(False)

def setup_grid_axes(ax, part):
    levels = {column: grid_index(part[column].values)[1] for column, _ in SEARCH_GRID}
    for (column, label), axis in zip(SEARCH_GRID, [ax.xaxis, ax.yaxis, ax.zaxis]):
        axis.set_ticks(range(len(levels[column])))
        axis.set_ticklabels([tick_label(v) for v in levels[column]], fontsize=6)
        axis.label.set_text(label)
        axis.label.set_fontsize(7)
    ax.view_init(elev=22, azim=-60)
    ax.set_box_aspect(None, zoom=0.85)

def draw_scatter3d(ax, part, norm, threshold):
    xs, ys, zs = (grid_index(part[column].values)[0] for column, _ in SEARCH_GRID)
    losses = part["val_loss_mean"].values
    collapsed = losses > threshold if threshold is not None else np.zeros(len(losses), bool)
    image = ax.scatter(xs[~collapsed], ys[~collapsed], zs[~collapsed], c=losses[~collapsed], cmap=SEQUENTIAL.reversed(), norm=norm, s=28, depthshade=False, edgecolors=SURFACE, linewidths=0.4)
    if collapsed.any():
        ax.scatter(xs[collapsed], ys[collapsed], zs[collapsed], marker="x", color=MUTED, s=18, depthshade=False)
    best = np.argmin(losses)
    ax.scatter([xs[best]], [ys[best]], [zs[best]], s=110, facecolors="none", edgecolors=INK, linewidths=1.2, depthshade=False)
    setup_grid_axes(ax, part)
    return image

def draw_surface3d(ax, part, norm, threshold):
    '''
    Loss over the learning rate x L2 grid, one surface per weight decay. Collapsed losses are clipped just above the colour scale
    so the cliff shows as a plateau without flattening the rest.
    '''
    cap = norm.vmax + 0.15 * (norm.vmax - norm.vmin) if threshold is not None else np.inf
    lr_levels = grid_index(part["lr"].values)[1]
    reg_levels = grid_index(part["reg"].values)[1]
    handles = []
    for k, wd in enumerate(sorted(set(part["wd"]))):
        pivot = part[part["wd"] == wd].pivot_table(index="reg", columns="lr", values="val_loss_mean", aggfunc="min").reindex(index=reg_levels, columns=lr_levels)
        x, y = np.meshgrid(range(len(lr_levels)), range(len(reg_levels)))
        colour = SERIES[k % len(SERIES)]
        ax.plot_surface(x, y, np.minimum(pivot.values, cap), color=colour, alpha=0.45, edgecolor=colour, linewidth=0.6)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colour, alpha=0.6, label=f"wd = {tick_label(wd)}"))
    for axis, levels, label in [(ax.xaxis, lr_levels, "Learning rate"), (ax.yaxis, reg_levels, "L2 coefficient")]:
        axis.set_ticks(range(len(levels)))
        axis.set_ticklabels([tick_label(v) for v in levels], fontsize=6)
        axis.label.set_text(label)
        axis.label.set_fontsize(7)
    ax.set_zlabel("Mean validation loss", fontsize=7, labelpad=8)
    ax.tick_params(axis="z", labelsize=6)
    ax.view_init(elev=24, azim=-130)
    ax.set_box_aspect(None, zoom=0.85)
    return handles

def plot_search_3d(df, stage, dirs):
    '''
    Four experimental views of the full three-hyperparameter grid, per input and combined over every input:
    lr x L2 heatmaps sliced by weight decay, parallel coordinates, a 3D scatter of the grid, and lr x L2 loss surfaces per weight decay.
    Colour scales leave out collapsed runs (see collapse_threshold), which are greyed out or clipped instead.
    '''
    surveys = [s for s in ALL_SURVEYS if s in set(df["model"])]
    if not surveys:
        return
    name = stage.lower()
    out = dirs["search_3d"]
    clip_note = lambda threshold: " (colour excludes collapsed runs)" if threshold is not None else ""
    columns = min(4, len(surveys))
    rows = int(np.ceil(len(surveys) / columns))
    wd_count = max(df[df["model"] == s]["wd"].nunique() for s in surveys)

    # > 1. lr x L2 heatmaps, one per weight decay value
    def draw_slices(axes, part, norm):
        best_wd = part.loc[part["val_loss_mean"].idxmin(), "wd"]
        for ax, wd in zip(axes, sorted(set(part["wd"]))):
            image = draw_search_heatmap(ax, part[part["wd"] == wd], "lr", "reg", norm=norm)
            ax.set_title(f"Weight decay {tick_label(wd)}" + (" (best)" if wd == best_wd else ""), fontsize=8, fontweight="bold" if wd == best_wd else "normal")
        for ax in axes[part["wd"].nunique():]:
            ax.axis("off")
        return image
    for survey in surveys:
        part = df[df["model"] == survey]
        norm, threshold = grid_norm(part)
        fig, axes = plt.subplots(1, wd_count, figsize=(2.6 * wd_count + 1, 2.8), layout="constrained", squeeze=False)
        image = draw_slices(axes[0], part, norm)
        fig.colorbar(image, ax=list(axes[0]), label="Mean validation loss", shrink=0.9)
        fig.suptitle(f"{stage.title()} search, {survey}: every config, sliced by weight decay{clip_note(threshold)}", fontsize=9)
        save_figure(fig, out / f"slices_{name}_{survey}.png")
    fig, axes = plt.subplots(len(surveys), wd_count, figsize=(2.4 * wd_count + 1, 2.3 * len(surveys)), layout="constrained", squeeze=False)
    for row, survey in zip(axes, surveys):
        part = df[df["model"] == survey]
        norm, _ = grid_norm(part)
        image = draw_slices(row, part, norm)
        fig.colorbar(image, ax=list(row), shrink=0.9).set_label(survey, fontsize=9, fontweight="bold")
    fig.suptitle(f"{stage.title()} search: lr x L2 for each weight decay\n(colour scaled per input, excludes collapsed runs, darker is lower, box is best in slice)", fontsize=10)
    save_figure(fig, out / f"slices_{name}_combined.png")

    # > 2. Parallel coordinates
    for survey in surveys:
        part = df[df["model"] == survey]
        norm, threshold = grid_norm(part)
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        draw_parallel(ax, part, norm, threshold)
        ax.set_title(f"{stage.title()} search, {survey}: every config (black is best, darker is lower)", fontsize=9)
        save_figure(fig, out / f"parallel_{name}_{survey}.png")
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3 * rows), squeeze=False)
    for ax, survey in zip(axes.flat, surveys):
        part = df[df["model"] == survey]
        draw_parallel(ax, part, *grid_norm(part))
        ax.set_title(survey, fontsize=9, fontweight="bold")
    for ax in list(axes.flat)[len(surveys):]:
        ax.axis("off")
    fig.suptitle(f"{stage.title()} search: parallel coordinates of every config (black is best, darker is lower, dashed grey collapsed)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out / f"parallel_{name}_combined.png")

    # > 3. 3D scatter of the grid
    for survey in surveys:
        part = df[df["model"] == survey]
        norm, threshold = grid_norm(part)
        fig = plt.figure(figsize=(5, 4.2))
        ax = fig.add_subplot(projection="3d")
        image = draw_scatter3d(ax, part, norm, threshold)
        fig.colorbar(image, ax=ax, label="Mean validation loss", shrink=0.6, pad=0.18)
        ax.set_title(f"{stage.title()} search, {survey} (ring is best, x collapsed)", fontsize=9)
        save_figure(fig, out / f"scatter3d_{name}_{survey}.png")
    fig = plt.figure(figsize=(4 * columns, 3.6 * rows))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.9, wspace=0.3, hspace=0.3)
    for k, survey in enumerate(surveys):
        part = df[df["model"] == survey]
        norm, threshold = grid_norm(part)
        ax = fig.add_subplot(rows, columns, k + 1, projection="3d")
        image = draw_scatter3d(ax, part, norm, threshold)
        fig.colorbar(image, ax=ax, shrink=0.5, pad=0.2).ax.tick_params(labelsize=6)
        ax.set_title(survey, fontsize=9, fontweight="bold")
    fig.suptitle(f"{stage.title()} search: every config in the grid (colour scaled per input, ring is best, x collapsed)", fontsize=10)
    save_figure(fig, out / f"scatter3d_{name}_combined.png", bbox_inches=fig.bbox_inches)

    # > 4. Loss surfaces over lr x L2, one per weight decay
    for survey in surveys:
        part = df[df["model"] == survey]
        norm, threshold = grid_norm(part)
        fig = plt.figure(figsize=(5, 4.2))
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.04, top=0.9)
        ax = fig.add_subplot(projection="3d")
        handles = draw_surface3d(ax, part, norm, threshold)
        ax.legend(handles=handles, loc="upper left", fontsize=7)
        ax.set_title(f"{stage.title()} search, {survey}" + (" (collapsed losses clipped)" if threshold is not None else ""), fontsize=9)
        save_figure(fig, out / f"surface_{name}_{survey}.png", bbox_inches=fig.bbox_inches)
    fig = plt.figure(figsize=(4 * columns, 3.6 * rows))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.9, wspace=0.3, hspace=0.3)
    for k, survey in enumerate(surveys):
        part = df[df["model"] == survey]
        ax = fig.add_subplot(rows, columns, k + 1, projection="3d")
        handles = draw_surface3d(ax, part, *grid_norm(part))
        ax.legend(handles=handles, loc="upper left", fontsize=6)
        ax.set_title(survey, fontsize=9, fontweight="bold")
    fig.suptitle(f"{stage.title()} search: loss surfaces over learning rate x L2, one per weight decay (collapsed losses clipped)", fontsize=10)
    save_figure(fig, out / f"surface_{name}_combined.png", bbox_inches=fig.bbox_inches)
# endregion

# region TRAINING
def training_summary(configs, histories, dirs):
    final = configs[configs["stage"] == "FINAL"]
    if final.empty:
        print("No FINAL training results found, skipping the training summary.")
        return
    save_csv(final, dirs, "training_final_models")

    for transfer, name in [(False, "scratch"), (True, "transfer")]:
        df = final[final["transfer_learning"] == transfer]
        if df.empty:
            continue
        df = df.assign(order=df["model"].map({s: i for i, s in enumerate(ALL_SURVEYS)})).sort_values(["base_model", "order"])
        table = pd.DataFrame({
            "Model": [strategy_label(b, m) for b, m in zip(df["base_model"], df["model"])],
            "lr": [sci(v) for v in df["lr"]], "L2": [sci(v) for v in df["reg"]], "wd": [sci(v) for v in df["wd"]],
            "Best epoch": [mean_std(m, s, 1) for m, s in zip(df["best_epoch_mean"], df["best_epoch_std"])],
            "Val. loss": [mean_std(m, s) for m, s in zip(df["val_loss_mean"], df["val_loss_std"])],
            "Val. acc.": [mean_std(m, s) for m, s in zip(df["val_accuracy_mean"], df["val_accuracy_std"])],
            "Gap": [mean_std(m, s) for m, s in zip(df["generalisation_gap_mean"], df["generalisation_gap_std"])],
            "Time/fold (min)": [mean_std(m, s, 1) for m, s in zip(df["training_time_min_mean"], df["training_time_min_std"])],
            "val_loss_numeric": df["val_loss_mean"].values,
        })
        if transfer:
            for group, members in CHANNEL_GROUPS.items():
                part = table[df["model"].isin(members).values]
                if not part.empty:
                    write_latex_table(dirs, f"training_transfer_{group}", part.reset_index(drop=True),
                        f"Transfer-learning training summary ({group}-channel inputs). Learning rate is the fine-tuning rate (a tenth of the base model's). Values are mean $\\pm$ std over folds at the early-stopping epoch; the gap is validation minus training loss.",
                        f"tab:training_transfer_{group}", bold_min=[("Val. loss", "val_loss_numeric")])
        else:
            write_latex_table(dirs, "training_scratch", table,
                "Training summary of the final models trained from scratch. Values are mean $\\pm$ std over folds at the early-stopping epoch; the gap is validation minus training loss.",
                "tab:training_scratch", bold_min=[("Val. loss", "val_loss_numeric")])

    plot_learning_curve_grid(final[~final["transfer_learning"]], histories, dirs)
    if SAVE_INDIVIDUAL_PLOTS:
        for folder in final["folder"]:
            plot_learning_curve(folder, histories[folder], dirs)
    plot_best_epochs(final, dirs)

def draw_mean_curve(ax, fold_histories, key, colour, label, band=True):
    '''
    Mean over folds: solid while every fold is still training, thin and dashed after the first fold stops early (the tail
    averages only the folds that ran longer, so it is not a like-for-like mean).
    '''
    mean, std, count = mean_curve(fold_histories, key)
    epochs = np.arange(1, len(mean) + 1)
    full = count == len(fold_histories)
    end = int(np.argmin(full)) if not full.all() else len(full)
    ax.plot(epochs[:end], mean[:end], color=colour, label=label)
    if band:
        ax.fill_between(epochs[:end], (mean - std)[:end], (mean + std)[:end], color=colour, alpha=0.15, linewidth=0)
    if end < len(mean):
        ax.plot(epochs[max(end - 1, 0):], mean[max(end - 1, 0):], color=colour, linewidth=1, linestyle="--", alpha=0.7)

def mean_curve(fold_histories, key):
    length = max(len(h["history"][key]) for h in fold_histories)
    curves = np.full((len(fold_histories), length), np.nan)
    for i, h in enumerate(fold_histories):
        curves[i, :len(h["history"][key])] = h["history"][key]
    return np.nanmean(curves, axis=0), np.nanstd(curves, axis=0), np.sum(np.isfinite(curves), axis=0)

def fill_learning_grid(axes, surveys, scratch, histories, metric, ylabel):
    for ax, survey in zip(axes.flat, surveys):
        folder = scratch[scratch["model"] == survey]["folder"].iloc[0]
        for split, key in [("train", metric), ("val", f"val_{metric}")]:
            draw_mean_curve(ax, histories[folder], key, SPLIT_COLOURS[split], "Training" if split == "train" else "Validation")
        ax.set_title(survey)
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes.flat[0].legend(loc="best")

def plot_learning_curve_grid(scratch, histories, dirs):
    '''
    Mean learning curves of the scratch models: one figure per channel group, plus one combined figure of all of them.
    '''
    panels = {group: [s for s in members if s in set(scratch["model"])] for group, members in CHANNEL_GROUPS.items()}
    every = [s for surveys in panels.values() for s in surveys]
    for metric, ylabel in [("loss", "Categorical cross-entropy"), ("accuracy", "Accuracy")]:
        for group, surveys in panels.items():
            if not surveys:
                continue
            fig, axes = panel_grid(len(surveys), 3, sharey=True)
            fill_learning_grid(axes, surveys, scratch, histories, metric, ylabel)
            fig.suptitle(f"Learning curves, {group}-channel models from scratch (mean $\\pm$ std over folds; dashed once folds stop early)", fontsize=10)
            fig.tight_layout()
            save_figure(fig, dirs["training"] / f"learning_curves_{metric}_{group}.png")
        if sum(len(s) > 0 for s in panels.values()) > 1:
            fig, axes = panel_grid(len(every), 3, height=2.0, sharey=True)
            fill_learning_grid(axes, every, scratch, histories, metric, ylabel)
            fig.suptitle(f"Learning curves of every model trained from scratch (mean $\\pm$ std over folds; dashed once folds stop early)", fontsize=10)
            fig.tight_layout()
            save_figure(fig, dirs["training"] / f"learning_curves_{metric}_combined.png")

def plot_learning_curve(folder, fold_histories, dirs):
    '''
    Loss and accuracy of every fold (thin) and their mean (thick) for one model. Transfer runs mark the end of the
    output-layer-only phase with a dashed line.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))
    for ax, metric, ylabel in [(axes[0], "loss", "Categorical cross-entropy"), (axes[1], "accuracy", "Accuracy")]:
        for split, key in [("train", metric), ("val", f"val_{metric}")]:
            for h in fold_histories:
                ax.plot(np.arange(1, len(h["history"][key]) + 1), h["history"][key], color=SPLIT_COLOURS[split], alpha=0.25, linewidth=0.8)
            draw_mean_curve(ax, fold_histories, key, SPLIT_COLOURS[split], "Training" if split == "train" else "Validation", band=False)
        if fold_histories and fold_histories[0]["head_epochs"]:
            ax.axvline(fold_histories[0]["head_epochs"] + 0.5, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="best")
    parameters = parse_folder(folder)
    fig.suptitle(f"{plain_label(parameters['base_model'], parameters['model'])}: lr={parameters['lr']:.2g}, L2={parameters['reg']:.2g}, wd={parameters['wd']:.2g}", fontsize=10)
    fig.tight_layout()
    save_figure(fig, dirs["learning_curves"] / f"{folder}.png")

def plot_best_epochs(final, dirs):
    scratch = final[~final["transfer_learning"]]
    surveys = [s for s in ALL_SURVEYS if s in set(scratch["model"])]
    if not surveys:
        return
    df = scratch.set_index("model").loc[surveys]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))
    group_colours = {"single": SERIES[0], "duo": SERIES[1], "trio": SERIES[2]}
    colours = [group_colours[channel_group(s)] for s in surveys]
    axes[0].bar(range(len(surveys)), df["best_epoch_mean"], yerr=df["best_epoch_std"], color=colours, edgecolor=SURFACE, linewidth=2, error_kw={"elinewidth": 1, "ecolor": INK_SECONDARY})
    axes[0].set_ylabel("Early-stopping epoch")
    axes[1].bar(range(len(surveys)), df["training_time_min_mean"], yerr=df["training_time_min_std"], color=colours, edgecolor=SURFACE, linewidth=2, error_kw={"elinewidth": 1, "ecolor": INK_SECONDARY})
    axes[1].set_ylabel("Training time per fold (min)")
    for ax in axes:
        ax.set_xticks(range(len(surveys)))
        ax.set_xticklabels(surveys, rotation=45, ha="right")
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f"{g.title()} channel") for g, c in group_colours.items()]
    axes[1].legend(handles=handles, loc="upper left")
    fig.suptitle("Final models from scratch: early stopping and training cost (mean $\\pm$ std over folds)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, dirs["training"] / "best_epochs_and_training_time.png")
# endregion

# region TESTING
def testing_summary(runs, dirs):
    if len(runs) == 0:
        print("No FINAL testing results found, skipping the testing summary.")
        return
    folds, ensemble = evaluate_test_runs(runs)
    ensemble["relation"] = [survey_relation(m, s, f, p) if kind == "ablation" else None
                            for m, s, f, p, kind in zip(ensemble["model"], ensemble["ablation_survey"], ensemble["fill"], ensemble["position"], ensemble["test_kind"])]
    save_csv(folds, dirs, "test_metrics_per_fold")
    save_csv(ensemble, dirs, "test_metrics_ensemble")
    scores = generalisation_scores(ensemble)
    save_csv(scores, dirs, "generalisation_scores")
    runs_by_key = {(r["base_model"], r["model"], r["dataset"]): r for r in runs}

    in_domain_table(ensemble, dirs)
    generalisation_tables(ensemble, scores, dirs)
    transfer_tables(ensemble, runs_by_key, dirs)
    per_class_table(ensemble, scores, dirs)

    for group in CHANNEL_GROUPS:
        plot_generalisation_matrix(ensemble, group, dirs)
        plot_transfer_matrices(ensemble, group, dirs)
        plot_leaderboard(scores, group, dirs)
    plot_generalisation_matrices_combined(ensemble, dirs)
    plot_transfer_matrices_combined(ensemble, dirs)
    plot_leaderboard_combined(scores, dirs)
    plot_confusion_grid(runs, dirs)
    plot_confusion_in_domain(runs, dirs)
    if SAVE_INDIVIDUAL_PLOTS:
        for run in runs:
            plot_confusion_matrix(run, dirs)
    plot_roc_best(scores, runs_by_key, dirs)
    plot_channel_order(ensemble, dirs)
    plot_fold_spread(folds, dirs)
    plot_target_comparison(ensemble, dirs)
    plot_survey_comparison(ensemble, dirs)
    plot_metric_heatmaps(ensemble, dirs)
    plot_test_set_heatmaps(ensemble, dirs)
    plot_test_set_metrics(ensemble, dirs)
    plot_base_vs_target(ensemble, dirs)

    ablation_summary(ensemble, dirs)
    print_headline(scores)

def in_domain_table(ensemble, dirs):
    df = ensemble[ensemble["in_domain"] & ~ensemble["transfer_learning"]].copy()
    if df.empty:
        return
    df = df.assign(order=df["model"].map({s: i for i, s in enumerate(ALL_SURVEYS)})).sort_values("order")
    table = pd.DataFrame({
        "Model": df["model"].values,
        "Accuracy": [mean_std(m, s) for m, s in zip(df["fold_accuracy_mean"], df["fold_accuracy_std"])],
        "Bal. acc.": [mean_std(m, s) for m, s in zip(df["fold_balanced_accuracy_mean"], df["fold_balanced_accuracy_std"])],
        "Macro F1": [mean_std(m, s) for m, s in zip(df["fold_macro_f1_mean"], df["fold_macro_f1_std"])],
        "MCC": [mean_std(m, s) for m, s in zip(df["fold_mcc_mean"], df["fold_mcc_std"])],
        "ROC AUC": [mean_std(m, s) for m, s in zip(df["fold_roc_auc_macro_mean"], df["fold_roc_auc_macro_std"])],
        "Ensemble F1 [95\\% CI]": [f"{f:.3f} [{lo:.3f}, {hi:.3f}]" for f, lo, hi in zip(df["macro_f1"], df["macro_f1_ci_low"], df["macro_f1_ci_high"])],
        "f1_numeric": df["macro_f1"].values,
    })
    write_latex_table(dirs, "test_in_domain_scratch", table,
        "In-domain test performance of the models trained from scratch (tested on the survey combination they were trained on). Columns show mean $\\pm$ std over the cross-validation fold models; the last column is the fold-ensemble macro F1 with a 95\\% bootstrap confidence interval.",
        "tab:test_in_domain_scratch", bold_max=[("Ensemble F1 [95\\% CI]", "f1_numeric")])

def generalisation_tables(ensemble, scores, dirs):
    # > Single-channel cross-survey matrix: the core "which model generalises across all three surveys" table
    single = ensemble[ensemble["group"] == "single"]
    if not single.empty:
        rows = []
        for (base_model, model), df in single.groupby(["base_model", "model"]):
            row = {"Model": strategy_label(base_model, model)}
            for dataset in SINGLE_CHANNEL:
                entry = df[df["dataset"] == dataset]
                row[dataset] = f"{entry['macro_f1'].iloc[0]:.3f}" if len(entry) else "--"
                row[f"{dataset}_numeric"] = entry["macro_f1"].iloc[0] if len(entry) else np.nan
            score = scores[(scores["base_model"] == base_model) & (scores["model"] == model)].iloc[0]
            row["Mean"] = f"{score['mean_macro_f1']:.3f}"
            row["Worst"] = f"{score['worst_macro_f1']:.3f}"
            row["mean_numeric"] = score["mean_macro_f1"]
            row["worst_numeric"] = score["worst_macro_f1"]
            row["order"] = (base_model != "NONE", SINGLE_CHANNEL.index(model), base_model)
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("order").drop(columns="order").reset_index(drop=True)
        write_latex_table(dirs, "test_generalisation_single", table,
            "Cross-survey generalisation of the single-channel models: fold-ensemble macro F1 on each survey's test set. $A\\rightarrow B$ is a model pre-trained on $A$ and fine-tuned on $B$. Bold marks the best per column.",
            "tab:test_generalisation_single",
            bold_max=[(d, f"{d}_numeric") for d in SINGLE_CHANNEL] + [("Mean", "mean_numeric"), ("Worst", "worst_numeric")])

    # > Leaderboard per group
    for group in CHANNEL_GROUPS:
        df = scores[scores["group"] == group].head(10)
        if df.empty:
            continue
        table = pd.DataFrame({
            "Rank": df["rank_in_group"].astype(str).values,
            "Model": [strategy_label(b, m) for b, m in zip(df["base_model"], df["model"])],
            "Mean F1": [f"{v:.3f}" for v in df["mean_macro_f1"]],
            "Worst F1 (dataset)": [f"{v:.3f} ({d})" for v, d in zip(df["worst_macro_f1"], df["worst_dataset"])],
            "In-domain F1": [f"{v:.3f}" if np.isfinite(v) else "--" for v in df["in_domain_macro_f1"]],
            "Out-of-domain F1": [f"{v:.3f}" if np.isfinite(v) else "--" for v in df["out_domain_macro_f1"]],
            "Retention": [f"{v:.3f}" if np.isfinite(v) else "--" for v in df["retention"]],
        })
        if group == "duo":
            # > Split the out-of-domain score: same surveys swapped vs a different survey in a channel
            table.insert(5, "Swapped order F1", [f"{v:.3f}" if np.isfinite(v) else "--" for v in df["other_order_macro_f1"]])
            table.insert(6, "Other surveys F1", [f"{v:.3f}" if np.isfinite(v) else "--" for v in df["other_survey_macro_f1"]])
        test_sets = {"single": "the FIRST, NVSS and LoTSS test sets",
                     "duo": "the six duo-channel inputs (the model's own surveys in either order, and the other survey pairs)",
                     "trio": "the six channel orders of the three surveys"}[group]
        note = {"single": "", "duo": " Swapped order is the model's own two surveys in the other channel order (channel-order robustness); other surveys replaces a survey (cross-survey generalisation).",
                "trio": " Every trio input holds all three surveys, so for trio models out-of-domain measures robustness to channel order, not cross-survey generalisation."}[group]
        write_latex_table(dirs, f"leaderboard_{group}", table,
            f"Generalisation leaderboard of the {group}-channel models: fold-ensemble macro F1 averaged over {test_sets}, the worst case, and the in-domain versus out-of-domain scores (retention = out-of-domain / in-domain).{note} Models are ranked on the test set here for reporting only; hyperparameters were chosen on validation loss.",
            f"tab:leaderboard_{group}", resize=group == "duo")

def transfer_tables(ensemble, runs_by_key, dirs):
    '''
    In-domain score of each transfer model against the scratch model on the same target, with an exact McNemar test.
    '''
    rows = []
    for _, tl in ensemble[ensemble["transfer_learning"] & ensemble["in_domain"]].iterrows():
        scratch = ensemble[(ensemble["base_model"] == "NONE") & (ensemble["model"] == tl["model"]) & (ensemble["dataset"] == tl["dataset"])]
        if scratch.empty:
            continue
        scratch = scratch.iloc[0]
        run_tl = runs_by_key[(tl["base_model"], tl["model"], tl["dataset"])]
        run_scratch = runs_by_key[("NONE", tl["model"], tl["dataset"])]
        only_tl, only_scratch, p = mcnemar(run_tl["y_true"], run_tl["ensemble"], run_scratch["ensemble"])
        rows.append({
            "group": tl["group"], "base_model": tl["base_model"], "target": tl["model"],
            "scratch_macro_f1": scratch["macro_f1"], "transfer_macro_f1": tl["macro_f1"],
            "delta_macro_f1": tl["macro_f1"] - scratch["macro_f1"],
            "scratch_accuracy": scratch["accuracy"], "transfer_accuracy": tl["accuracy"],
            "only_transfer_correct": only_tl, "only_scratch_correct": only_scratch, "mcnemar_p": p,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return
    # > Many transfer pairs are tested at once, so the p-values are Holm-adjusted (family-wise error 0.05 over every pair)
    df["mcnemar_p_holm"] = holm(df["mcnemar_p"].values)
    save_csv(df, dirs, "transfer_learning_gains")
    for group in CHANNEL_GROUPS:
        part = df[df["group"] == group]
        if part.empty:
            continue
        part = part.assign(order=part["target"].map({s: i for i, s in enumerate(ALL_SURVEYS)})).sort_values(["order", "base_model"])
        table = pd.DataFrame({
            "Target": part["target"].values, "Base": part["base_model"].values,
            "Scratch F1": [f"{v:.3f}" for v in part["scratch_macro_f1"]],
            "Transfer F1": [f"{v:.3f}" for v in part["transfer_macro_f1"]],
            "$\\Delta$F1": [f"{v:+.3f}" for v in part["delta_macro_f1"]],
            "McNemar $p$": [f"{p:.3f}" if p >= 0.001 else "$<$0.001" for p in part["mcnemar_p"]],
            "Holm $p$": [f"{p:.3f}" if p >= 0.001 else "$<$0.001" for p in part["mcnemar_p_holm"]],
            "delta_numeric": part["delta_macro_f1"].values,
        })
        write_latex_table(dirs, f"transfer_gains_{group}", table,
            f"Transfer learning versus training from scratch ({group}-channel inputs), in-domain fold-ensemble macro F1 on the target's test set. The exact McNemar test compares the two models' predictions on the same test images; Holm $p$ is adjusted for the {len(df)} transfer pairs tested.",
            f"tab:transfer_gains_{group}", bold_max=[("$\\Delta$F1", "delta_numeric")])
    plot_transfer_gains(df, dirs)

def per_class_table(ensemble, scores, dirs):
    best = scores[scores["group"] == "single"].head(1)
    if best.empty:
        return
    best = best.iloc[0]
    df = ensemble[(ensemble["base_model"] == best["base_model"]) & (ensemble["model"] == best["model"])]
    rows = []
    for _, row in df.iterrows():
        for name in CLASSES:
            rows.append({"Test set": row["dataset"], "Class": name,
                         "Precision": f"{row[f'precision_{name}']:.3f}", "Recall": f"{row[f'recall_{name}']:.3f}", "F1": f"{row[f'f1_{name}']:.3f}"})
    write_latex_table(dirs, "per_class_best_single", pd.DataFrame(rows),
        f"Per-class fold-ensemble test metrics of the best-generalising single-channel model ({strategy_label(best['base_model'], best['model'])}).",
        "tab:per_class_best_single", column_format="llccc")

def generalisation_matrix(ensemble, group):
    df = ensemble[(ensemble["group"] == group) & ~ensemble["transfer_learning"] & (ensemble["test_kind"] == "standard")]
    if df.empty:
        return None
    members = [s for s in CHANNEL_GROUPS[group] if s in set(df["model"]) or s in set(df["dataset"])]
    return df.pivot_table(index="model", columns="dataset", values="macro_f1").reindex(index=members, columns=members)

def draw_matrix(ax, matrix, cmap=SEQUENTIAL, norm=None, fmt="{:.3f}"):
    image = annotate_heatmap(ax, matrix.values, fmt=fmt, cmap=cmap, norm=norm)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    return image

def plot_generalisation_matrix(ensemble, group, dirs):
    matrix = generalisation_matrix(ensemble, group)
    if matrix is None:
        return
    size = 1.6 + 0.5 * len(matrix)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    image = draw_matrix(ax, matrix)
    ax.set_xlabel("Test set")
    ax.set_ylabel("Trained on")
    fig.colorbar(image, ax=ax, label="Macro F1 (fold ensemble)", shrink=0.8)
    ax.set_title(f"Cross-dataset generalisation, {group}-channel models from scratch")
    save_figure(fig, dirs["testing"] / f"generalisation_matrix_{group}.png")

def plot_generalisation_matrices_combined(ensemble, dirs):
    '''
    The three generalisation matrices side by side on one colour scale.
    '''
    matrices = {group: generalisation_matrix(ensemble, group) for group in CHANNEL_GROUPS}
    matrices = {group: matrix for group, matrix in matrices.items() if matrix is not None}
    if len(matrices) < 2:
        return
    values = np.concatenate([matrix.values.ravel() for matrix in matrices.values()])
    norm = Normalize(np.nanmin(values), np.nanmax(values))
    fig, axes = plt.subplots(1, len(matrices), figsize=(12, 4.2), layout="constrained", gridspec_kw={"width_ratios": [len(m) for m in matrices.values()]})
    for ax, (group, matrix) in zip(axes, matrices.items()):
        image = draw_matrix(ax, matrix, norm=norm)
        ax.set_title(f"{group.title()}-channel models")
        ax.set_xlabel("Test set")
    axes[0].set_ylabel("Trained on")
    fig.colorbar(image, ax=list(axes), label="Macro F1 (fold ensemble)", shrink=0.85)
    label_panels(axes)
    fig.suptitle("Cross-dataset generalisation of the models trained from scratch", fontsize=10)
    save_figure(fig, dirs["testing"] / "generalisation_matrix_combined.png")

def transfer_matrices(ensemble, group):
    '''
    Base x target matrices of in-domain macro F1 (diagonal = trained from scratch) and the gain over scratch.
    '''
    df = ensemble[(ensemble["group"] == group) & ensemble["in_domain"]]
    if df[df["transfer_learning"]].empty:
        return None
    members = CHANNEL_GROUPS[group]
    scores = np.full((len(members), len(members)), np.nan)
    gains = np.full_like(scores, np.nan)
    for i, base in enumerate(members):
        for j, target in enumerate(members):
            key = "NONE" if base == target else base
            entry = df[(df["base_model"] == key) & (df["model"] == target)]
            scratch = df[(df["base_model"] == "NONE") & (df["model"] == target)]
            if len(entry):
                scores[i, j] = entry["macro_f1"].iloc[0]
                if base != target and len(scratch):
                    gains[i, j] = entry["macro_f1"].iloc[0] - scratch["macro_f1"].iloc[0]
    return pd.DataFrame(scores, index=members, columns=members), pd.DataFrame(gains, index=members, columns=members)

def gain_norm(values):
    limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1
    return TwoSlopeNorm(0, -limit - 1e-9, limit + 1e-9)

def label_transfer_axes(ax):
    ax.set_xlabel("Target (fine-tuned on)")
    ax.set_ylabel("Base (pre-trained on)")

def plot_transfer_matrices(ensemble, group, dirs):
    matrices = transfer_matrices(ensemble, group)
    if matrices is None:
        return
    scores, gains = matrices
    size = 1.6 + 0.5 * len(scores)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 2, size))
    image = draw_matrix(axes[0], scores)
    fig.colorbar(image, ax=axes[0], label="Macro F1 on target test set", shrink=0.8)
    axes[0].set_title("Target performance (diagonal: from scratch)")
    image = draw_matrix(axes[1], gains, cmap=DIVERGING, norm=gain_norm(gains.values), fmt="{:+.3f}")
    fig.colorbar(image, ax=axes[1], label="$\\Delta$ macro F1 vs. scratch", shrink=0.8)
    axes[1].set_title("Gain from transfer learning")
    for ax in axes:
        label_transfer_axes(ax)
    fig.tight_layout()
    save_figure(fig, dirs["testing"] / f"transfer_matrix_{group}.png")

def plot_transfer_matrices_combined(ensemble, dirs):
    '''
    Every group's transfer matrices in one figure: performance on the top row, gain over scratch on the bottom row, one colour scale per row.
    '''
    matrices = {group: transfer_matrices(ensemble, group) for group in CHANNEL_GROUPS}
    matrices = {group: m for group, m in matrices.items() if m is not None}
    if len(matrices) < 2:
        return
    score_values = np.concatenate([s.values.ravel() for s, _ in matrices.values()])
    gain_values = np.concatenate([g.values.ravel() for _, g in matrices.values()])
    score_norm = Normalize(np.nanmin(score_values), np.nanmax(score_values))
    fig, axes = plt.subplots(2, len(matrices), figsize=(12, 8), layout="constrained", gridspec_kw={"width_ratios": [len(s) for s, _ in matrices.values()]})
    for k, (group, (scores, gains)) in enumerate(matrices.items()):
        top = draw_matrix(axes[0, k], scores, norm=score_norm)
        bottom = draw_matrix(axes[1, k], gains, cmap=DIVERGING, norm=gain_norm(gain_values), fmt="{:+.3f}")
        axes[0, k].set_title(f"{group.title()}: target macro F1")
        axes[1, k].set_title(f"{group.title()}: gain over scratch")
        for ax in axes[:, k]:
            label_transfer_axes(ax)
    fig.colorbar(top, ax=list(axes[0]), label="Macro F1 on target test set", shrink=0.85)
    fig.colorbar(bottom, ax=list(axes[1]), label="$\\Delta$ macro F1 vs. scratch", shrink=0.85)
    label_panels(axes.flat)
    fig.suptitle("Transfer learning: in-domain macro F1 (diagonal: from scratch) and gain over training from scratch", fontsize=10)
    save_figure(fig, dirs["testing"] / "transfer_matrix_combined.png")

def plot_transfer_gains(gains, dirs):
    df = gains.assign(order=gains["target"].map({s: i for i, s in enumerate(ALL_SURVEYS)})).sort_values(["order", "base_model"])
    labels = [f"{b}->{t}" for b, t in zip(df["base_model"], df["target"])]
    colours = [SERIES[0] if v >= 0 else SERIES[7] for v in df["delta_macro_f1"]]
    fig, ax = plt.subplots(figsize=(7.5, 0.18 * len(df) + 1))
    ax.barh(range(len(df)), df["delta_macro_f1"], color=colours, edgecolor=SURFACE, linewidth=1, height=0.8)
    for i, p in enumerate(df["mcnemar_p_holm"]):
        if p < 0.05:
            ax.text(df["delta_macro_f1"].iloc[i], i, " *" if df["delta_macro_f1"].iloc[i] >= 0 else "* ", va="center", ha="left" if df["delta_macro_f1"].iloc[i] >= 0 else "right", color=INK)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$ macro F1, transfer minus scratch (in-domain)   * McNemar $p<0.05$ after Holm correction")
    ax.grid(axis="y", visible=False)
    ax.set_title("Effect of transfer learning on every target")
    save_figure(fig, dirs["testing"] / "transfer_gains.png")

LEADERBOARD_HANDLES = [Line2D([], [], marker="o", linestyle="", color=SERIES[0], markersize=7, label="From scratch (mean)"),
                       Line2D([], [], marker="o", linestyle="", color=SERIES[1], markersize=7, label="Transfer learning (mean)"),
                       Line2D([], [], marker="|", linestyle="-", color=INK, markersize=10, markeredgewidth=2, linewidth=1, label="Worst test set")]

def draw_leaderboard(ax, df):
    '''
    Dot at the mean macro F1, tick at the worst test set, joined by a line. Dots, not bars, because the axis does not start at 0.
    '''
    y = np.arange(len(df))
    colours = [SERIES[1] if t else SERIES[0] for t in df["transfer_learning"]]
    ax.hlines(y, df["worst_macro_f1"], df["mean_macro_f1"], color=MUTED, linewidth=1)
    ax.scatter(df["worst_macro_f1"], y, marker="|", s=120, color=INK, linewidths=2, zorder=3)
    ax.scatter(df["mean_macro_f1"], y, s=50, color=colours, edgecolors=SURFACE, linewidths=1, zorder=4)
    for i, value in enumerate(df["mean_macro_f1"]):
        ax.text(value + 0.006, i, f"{value:.3f}", va="center", fontsize=7, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    low = max(0, min(df["worst_macro_f1"].min(), df["mean_macro_f1"].min()) - 0.05)
    ax.set_xlim(low, 1.0)
    ax.set_xlabel("Macro F1 (fold ensemble)")
    ax.grid(axis="y", visible=False)

def plot_leaderboard(scores, group, dirs):
    df = scores[scores["group"] == group].head(LEADERBOARD_SIZE)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(df) + 1))
    draw_leaderboard(ax, df)
    ax.legend(handles=LEADERBOARD_HANDLES, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 1.5 / (0.3 * len(df) + 1) * 0.1), ncol=3)
    ax.set_title(f"Generalisation leaderboard, {group}-channel models")
    save_figure(fig, dirs["testing"] / f"leaderboard_{group}.png")

def plot_leaderboard_combined(scores, dirs):
    groups = [g for g in CHANNEL_GROUPS if not scores.empty and (scores["group"] == g).any()]
    if len(groups) < 2:
        return
    rows = max(len(scores[scores["group"] == g].head(LEADERBOARD_SIZE)) for g in groups)
    fig, axes = plt.subplots(1, len(groups), figsize=(12, 0.3 * rows + 1.6), layout="constrained")
    for ax, group in zip(axes, groups):
        draw_leaderboard(ax, scores[scores["group"] == group].head(LEADERBOARD_SIZE))
        ax.set_title(f"{group.title()}-channel models")
    fig.legend(handles=LEADERBOARD_HANDLES, loc="outside lower center", ncol=3)
    label_panels(axes)
    fig.suptitle("Generalisation leaderboards (mean macro F1 over every test set of the channel group)", fontsize=10)
    save_figure(fig, dirs["testing"] / "leaderboard_combined.png")

def draw_confusion(ax, y_true, probabilities, title, show_labels=True):
    matrix = confusion_matrix(y_true, np.argmax(probabilities, axis=1), labels=range(len(CLASSES)))
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    ax.imshow(normalised, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for (i, j), value in np.ndenumerate(normalised):
        ax.text(j, i, f"{value:.2f}\n({matrix[i, j]})", ha="center", va="center", fontsize=6, color="#ffffff" if value > 0.6 else INK)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES if show_labels else [], fontsize=7)
    ax.set_yticklabels(CLASSES if show_labels else [], fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.grid(False)

def plot_confusion_matrix(run, dirs):
    fig, ax = plt.subplots(figsize=(3.2, 3))
    draw_confusion(ax, run["y_true"], run["ensemble"], f"{run['label']} on {run['dataset']}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    save_figure(fig, dirs["confusion_matrices"] / f"{run['folder']}.png")

def plot_confusion_grid(runs, dirs):
    '''
    Single-channel models from scratch (rows) on every survey's test set (columns), row-normalised.
    '''
    lookup = {(r["model"], r["dataset"]): r for r in runs if r["base_model"] == "NONE" and r["group"] == "single"}
    if not lookup:
        return
    fig, axes = plt.subplots(3, 3, figsize=(7, 6.6))
    for i, model in enumerate(SINGLE_CHANNEL):
        for j, dataset in enumerate(SINGLE_CHANNEL):
            ax = axes[i, j]
            if (model, dataset) not in lookup:
                ax.axis("off")
                continue
            run = lookup[(model, dataset)]
            draw_confusion(ax, run["y_true"], run["ensemble"], f"Trained {model}, tested {dataset}", show_labels=True)
            if j == 0:
                ax.set_ylabel("True")
            if i == 2:
                ax.set_xlabel("Predicted")
    fig.suptitle("Confusion matrices of the single-channel models from scratch (row-normalised, counts in brackets)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, dirs["testing"] / "confusion_grid_single.png")

def plot_confusion_in_domain(runs, dirs):
    '''
    Combined figure: the in-domain confusion matrix of every model trained from scratch.
    '''
    lookup = {r["model"]: r for r in runs if r["base_model"] == "NONE" and r["in_domain"]}
    models = [m for m in ALL_SURVEYS if m in lookup]
    if len(models) < 2:
        return
    fig, axes = panel_grid(len(models), 3, width=7, height=2.25)
    for k, (ax, model) in enumerate(zip(axes.flat, models)):
        run = lookup[model]
        draw_confusion(ax, run["y_true"], run["ensemble"], f"{model} (macro F1 {classification_metrics(run['y_true'], run['ensemble'])['macro_f1']:.3f})")
        if k % 3 == 0:
            ax.set_ylabel("True")
    for ax in axes[-1]:
        ax.set_xlabel("Predicted")
    fig.suptitle("In-domain confusion matrices of every model trained from scratch (row-normalised, counts in brackets)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, dirs["testing"] / "confusion_in_domain_combined.png")

def plot_roc_best(scores, runs_by_key, dirs):
    '''
    One-vs-rest ROC curves per class of the best-generalising single-channel model on each survey's test set.
    '''
    best = scores[scores["group"] == "single"].head(1)
    if best.empty:
        return
    best = best.iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7), sharey=True)
    for k, name in enumerate(CLASSES):
        ax = axes[k]
        for dataset in SINGLE_CHANNEL:
            run = runs_by_key.get((best["base_model"], best["model"], dataset))
            if run is None:
                continue
            truth = (run["y_true"] == k).astype(int)
            if truth.min() == truth.max():
                continue
            fpr, tpr, _ = roc_curve(truth, run["ensemble"][:, k])
            auc = roc_auc_score(truth, run["ensemble"][:, k])
            ax.plot(fpr, tpr, color=SURVEY_COLOURS[dataset], label=f"{dataset} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(f"{name} vs. rest")
        ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", fontsize=6)
    axes[0].set_ylabel("True positive rate")
    fig.suptitle(f"ROC curves of the best-generalising single-channel model ({best['label']})", fontsize=10)
    fig.tight_layout()
    save_figure(fig, dirs["testing"] / "roc_best_single.png")

def plot_channel_order(ensemble, dirs):
    '''
    In-domain macro F1 of each channel ordering of the same survey combination (models from scratch).
    '''
    df = ensemble[ensemble["in_domain"] & ~ensemble["transfer_learning"] & ensemble["group"].isin(["duo", "trio"])]
    if df.empty:
        return
    members = [s for s in DUO_CHANNEL + TRIO_CHANNEL if s in set(df["model"])]
    combinations = []
    for s in members:
        key = "".join(sorted(s))
        if key not in combinations:
            combinations.append(key)
    colours = {key: SERIES[i] for i, key in enumerate(combinations)}
    df = df.set_index("model").loc[members]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    # > Dots with error bars, not bars: the axis is zoomed in, so bar lengths would exaggerate the differences
    for i, model in enumerate(members):
        ax.errorbar(i, df.loc[model, "macro_f1"], yerr=df.loc[model, "fold_macro_f1_std"], fmt="o", color=colours["".join(sorted(model))], ecolor=INK_SECONDARY, elinewidth=1, capsize=0, markersize=7, markeredgecolor=SURFACE)
    ax.set_xticks(range(len(members)))
    ax.set_xticklabels(members)
    low = max(0, (df["macro_f1"] - df["fold_macro_f1_std"].fillna(0)).min() - 0.05)
    ax.set_ylim(low, 1)
    ax.set_ylabel("In-domain macro F1")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], marker="o", linestyle="", color=colours[k], label="+".join(SURVEY_NAMES[c] for c in k)) for k in combinations]
    ax.legend(handles=handles, loc="lower right", ncol=2)
    ax.set_title("Effect of channel order (dot: fold ensemble, error bar: std over fold models)")
    save_figure(fig, dirs["testing"] / "channel_order_effect.png")

def plot_fold_spread(folds, dirs):
    '''
    Spread of the individual fold models' macro F1 against the fold ensemble, for in-domain scratch models.
    '''
    df = folds[folds["in_domain"] & ~folds["transfer_learning"]]
    if df.empty:
        return
    members = [s for s in ALL_SURVEYS if s in set(df["model"])]
    group_colours = {"single": SERIES[0], "duo": SERIES[1], "trio": SERIES[2]}
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for i, model in enumerate(members):
        values = df[df["model"] == model]["macro_f1"].values
        ax.scatter(np.full(len(values), i), values, s=18, color=group_colours[channel_group(model)], alpha=0.8, linewidths=0)
    ax.set_xticks(range(len(members)))
    ax.set_xticklabels(members, rotation=45, ha="right")
    ax.set_ylabel("In-domain macro F1 per fold model")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=f"{g.title()} channel") for g, c in group_colours.items()]
    ax.legend(handles=handles, loc="lower right", ncol=3)
    ax.set_title("Variation between cross-validation fold models")
    save_figure(fig, dirs["testing"] / "fold_spread_in_domain.png")

def print_headline(scores):
    print("\n" + "=" * 80)
    print("HEADLINE RESULTS (fold-ensemble macro F1)")
    print("=" * 80)
    for group in CHANNEL_GROUPS:
        df = scores[scores["group"] == group]
        if df.empty:
            continue
        best = df.iloc[0]
        print(f"Best {group}-channel model: {best['label']} | mean {best['mean_macro_f1']:.3f} | worst {best['worst_macro_f1']:.3f} ({best['worst_dataset']}) | in-domain {best['in_domain_macro_f1']:.3f} | out-of-domain {best['out_domain_macro_f1']:.3f}")
    print("=" * 80 + "\n")
# endregion

# region SURVEY COMPARISON
def collapse_positions(rows, keys, columns):
    '''
    One row per keys group. A zero or noise input is tested with the survey in each channel in turn. When the model was trained
    on that survey, only the row with the survey in its own channel is used: the other rows put it in a channel where the model
    learnt a different survey, which tests channel placement, not the survey (see the ablation position charts for that).
    When the model never saw the survey, every channel is equally foreign, so the rows are averaged.
    '''
    averaged = rows.groupby(keys)[columns].mean()
    native = rows[rows["relation"] == "native"]
    if len(native):
        native = native.groupby(keys)[columns].mean()
        averaged.loc[native.index, columns] = native[columns]
    return averaged

def representative_rows(rows):
    '''
    The rows a bar stands for: the survey's own channel if the model was trained on it, otherwise every row.
    '''
    native = rows[rows["relation"] == "native"]
    return native if len(native) else rows

TEST_TYPES = ["STD"] + FILLS
TEST_COLOURS = {"STD": SERIES[5], **FILL_COLOURS}
TEST_NAMES = {"STD": "The survey's own test set (single-channel models)", "REP": "Survey repeated in every channel",
              "ZER": "Survey in one channel, zeros elsewhere", "NSE": "Survey in one channel, white noise elsewhere"}

def model_sort_key(base_model, model):
    '''
    Channel group, then scratch before transfer, then target and base in the canonical survey order.
    '''
    return (list(CHANNEL_GROUPS).index(channel_group(model)), base_model != "NONE", model_order(model), model_order(base_model), base_model)

def base_colour(base):
    '''
    Scratch is grey; a base keeps its survey colour, or for duo/trio bases the colour of its place in its channel group.
    '''
    if base == "NONE":
        return INK_SECONDARY
    if base in SURVEY_COLOURS:
        return SURVEY_COLOURS[base]
    members = CHANNEL_GROUPS.get(channel_group(base), [])
    return SERIES[members.index(base) % len(SERIES)] if base in members else MUTED

def draw_target_comparison(ax, df, target):
    '''
    Grouped bars: one group per test set of the target's channel group, one bar per model fine-tuned on target (scratch first,
    then by base), 95% bootstrap CI.
    '''
    bases = sorted(df["base_model"].unique(), key=lambda b: (b != "NONE", model_order(b), b))
    datasets = [d for d in CHANNEL_GROUPS[channel_group(target)] if d in set(df["dataset"])]
    width = 0.8 / len(bases)
    for k, base in enumerate(bases):
        part = df[df["base_model"] == base].set_index("dataset").reindex(datasets)
        x = np.arange(len(datasets)) + (k - (len(bases) - 1) / 2) * width
        values = part["macro_f1"].values
        errors = np.array([values - part["macro_f1_ci_low"].values, part["macro_f1_ci_high"].values - values])
        colour = base_colour(base)
        label = f"{target} from scratch" if base == "NONE" else f"{base}$\\rightarrow${target}"
        ax.bar(x, values, width, yerr=errors, color=colour, edgecolor=SURFACE, linewidth=1, label=label, error_kw={"elinewidth": 1, "ecolor": INK})
        for xi, value in zip(x, values):
            if np.isfinite(value):
                ax.text(xi, 0.02, f"{value:.3f}", rotation=90, ha="center", va="bottom", fontsize=6, color=SURFACE)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([f"{d}\n(in-domain)" if d == target else d for d in datasets])
    ax.set_xlabel("Test set")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)

def plot_target_comparison(ensemble, dirs):
    '''
    Per target (every channel group): the model trained from scratch against every model fine-tuned on that target, on each
    standard test set of its channel group (each survey for single-channel targets, each channel order for duo/trio targets).
    '''
    for group, members in CHANNEL_GROUPS.items():
        df = ensemble[(ensemble["group"] == group) & (ensemble["test_kind"] == "standard")]
        targets = [t for t in members if t in set(df["model"])]
        test_sets = "every survey's test set" if group == "single" else "every channel order of the test set"
        for target in targets:
            fig, ax = plt.subplots(figsize=(1.6 * len(members) + 1.5, 3))
            draw_target_comparison(ax, df[df["model"] == target], target)
            ax.set_ylabel("Macro F1 (fold ensemble, 95% CI)")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)
            ax.set_title(f"{target} from scratch versus every model fine-tuned on {target}")
            save_figure(fig, dirs["survey_comparison"] / f"target_{target}.png")
        if len(targets) < 2:
            continue
        columns = 3 if group == "single" else 2
        fig, axes = panel_grid(len(targets), columns, width=12, height=3.8, sharey=True, layout="constrained")
        for ax, target in zip(axes.flat, targets):
            draw_target_comparison(ax, df[df["model"] == target], target)
            ax.set_title(f"Fine-tuned on {target}")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=min(3, df[df["model"] == target]["base_model"].nunique()))
        for row in axes:
            row[0].set_ylabel("Macro F1 (fold ensemble, 95% CI)")
        label_panels(list(axes.flat)[:len(targets)])
        fig.suptitle(f"Training from scratch versus transfer learning, {group}-channel targets, on {test_sets}", fontsize=10)
        save_figure(fig, dirs["survey_comparison"] / f"target_comparison_{group}_combined.png")

def survey_comparison_frame(ensemble, survey):
    '''
    Every model given only this survey: single-channel models on its test set, duo and trio models on its single-survey inputs.
    '''
    single = ensemble[(ensemble["group"] == "single") & (ensemble["test_kind"] == "standard") & (ensemble["dataset"] == survey)].assign(test="STD")
    multi = ensemble[(ensemble["test_kind"] == "ablation") & (ensemble["ablation_survey"] == survey)]
    return pd.concat([single, multi.assign(test=multi["fill"])], ignore_index=True)

def draw_survey_comparison(ax, df):
    '''
    Horizontal bars, one block per model (by channel group, scratch models first) with one bar per test type. Zero and noise bars
    show the survey in its own channel when the model was trained on it (else the mean over channels), with each channel's score
    as a dot; a bar standing for one input carries its 95% bootstrap CI.
    RETURNS: the test types drawn, for the legend
    '''
    keys = sorted(set(zip(df["base_model"], df["model"])), key=lambda k: model_sort_key(*k))
    groups = list(dict.fromkeys(channel_group(m) for _, m in keys))
    y, ticks, labels, drawn = 0, [], [], set()
    for g, group in enumerate(groups):
        if len(groups) > 1:
            if g > 0:
                ax.axhline(y - 0.8, color=MUTED, linewidth=0.8)
            ax.text(0.005, y, f"{group.title()}-channel models", ha="left", va="center", fontsize=8, fontweight="bold", color=INK)
            y += 1.2
        for base_model, model in [k for k in keys if channel_group(k[1]) == group]:
            part = df[(df["base_model"] == base_model) & (df["model"] == model)]
            start = y
            for test in [t for t in TEST_TYPES if t in set(part["test"])]:
                rows = part[part["test"] == test]
                shown = representative_rows(rows)
                value = shown["macro_f1"].mean()
                ax.barh(y, value, height=0.85, color=TEST_COLOURS[test], edgecolor=SURFACE, linewidth=0.5)
                if len(shown) == 1:
                    row = shown.iloc[0]
                    ax.errorbar(value, y, xerr=[[value - row["macro_f1_ci_low"]], [row["macro_f1_ci_high"] - value]], color=INK_SECONDARY, elinewidth=1, capsize=0)
                if len(rows) > 1:
                    ax.scatter(rows["macro_f1"], np.full(len(rows), y), s=8, color=INK, linewidths=0, zorder=3)
                ax.text(1.01, y, f"{value:.3f}", ha="left", va="center", fontsize=6, color=INK_SECONDARY, clip_on=False)
                drawn.add(test)
                y += 1
            ticks.append((start + y - 1) / 2)
            labels.append(plain_label(base_model, model))
            y += 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(y - 0.3, -0.8) # | Inverted: first model at the top
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 (fold ensemble)")
    ax.grid(axis="y", visible=False)
    return [t for t in TEST_TYPES if t in drawn]

def plot_survey_figure(df, title, path):
    if df.empty:
        return
    units = df.groupby(["base_model", "model", "test"]).ngroups + 0.6 * df.groupby(["base_model", "model"]).ngroups + 1.2 * df["group"].nunique()
    fig, ax = plt.subplots(figsize=(7.5, 0.14 * units + 1.6), layout="constrained")
    tests = draw_survey_comparison(ax, df)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TEST_COLOURS[t], label=TEST_NAMES[t]) for t in tests]
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=1, label="95% bootstrap CI"))
    if {"ZER", "NSE"} & set(tests):
        handles.append(Line2D([], [], marker="o", linestyle="", color=INK, markersize=3, label="Score with the survey in each channel (bar: its own channel if trained on it, else the mean)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    ax.set_title(title)
    save_figure(fig, path)

def plot_survey_comparison(ensemble, dirs):
    '''
    Per survey: every model given only that survey, one chart per channel group and one master chart with every channel group.
    '''
    frames = []
    for survey in SINGLE_CHANNEL:
        df = survey_comparison_frame(ensemble, survey)
        if df.empty:
            continue
        frames.append(df.assign(survey_given=survey))
        for group in CHANNEL_GROUPS:
            where = f"the {survey} test set" if group == "single" else f"{survey}-only inputs"
            plot_survey_figure(df[df["group"] == group], f"{group.title()}-channel models on {where}", dirs["survey_comparison"] / f"{survey}_{group}.png")
        plot_survey_figure(df, f"Every model given only {survey} (single-channel: {survey} test set; duo/trio: {survey}-only inputs)",
                           dirs["survey_comparison"] / f"{survey}_all_channels.png")
    if frames:
        columns = ["survey_given", "group", "base_model", "model", "label", "transfer_learning", "dataset", "test", "position",
                   "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "accuracy", "mcc"]
        save_csv(pd.concat(frames, ignore_index=True)[columns], dirs, "survey_comparison")
# endregion

# region METRIC HEATMAPS
HEATMAP_METRICS = {"accuracy": "Accuracy", "balanced_accuracy": "Bal. acc.", "macro_precision": "Macro precision", "macro_recall": "Macro recall",
                   "macro_f1": "Macro F1", "weighted_f1": "Weighted F1", "mcc": "MCC", "cohen_kappa": "Cohen's $\\kappa$", "roc_auc_macro": "ROC AUC",
                   **{f"f1_{name}": f"F1 {name}" for name in CLASSES}}
TEST_SET_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "mcc", "cohen_kappa", "roc_auc_macro"]

def ordered_models(df):
    return sorted(set(zip(df["base_model"], df["model"])), key=lambda k: model_sort_key(*k))

def model_dividers(ax, keys):
    '''
    White lines between the scratch and transfer models, and between channel groups.
    '''
    for i in range(1, len(keys)):
        (base_a, model_a), (base_b, model_b) = keys[i - 1], keys[i]
        if channel_group(model_a) != channel_group(model_b) or (base_a == "NONE") != (base_b == "NONE"):
            ax.axhline(i - 0.5, color=SURFACE, linewidth=3)

def metric_matrix(ensemble, group=None, scratch_only=False):
    '''
    Rows: models (scratch first). Columns: every in-domain metric (each model on the test set it was trained on).
    '''
    df = ensemble[ensemble["in_domain"] & (ensemble["test_kind"] == "standard")]
    if group is not None:
        df = df[df["group"] == group]
    if scratch_only:
        df = df[~df["transfer_learning"]]
    keys = ordered_models(df)
    if not keys:
        return None, keys
    rows = df.set_index(["base_model", "model"]).loc[keys]
    return pd.DataFrame(rows[list(HEATMAP_METRICS)].values, index=[plain_label(*k) for k in keys], columns=list(HEATMAP_METRICS.values())), keys

def draw_metric_heatmap(ax, matrix):
    '''
    Each column is coloured on its own min-max scale (metrics have different ranges), annotated with the actual values.
    '''
    values = matrix.values.astype(float)
    low, high = matrix.min().values, matrix.max().values
    spread = high > low
    normed = (values - low) / np.where(spread, high - low, 1)
    normed[:, ~spread] = 0.5
    normed[~np.isfinite(values)] = np.nan
    ax.imshow(normed, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for (i, j), value in np.ndenumerate(values):
        if not np.isfinite(value):
            ax.text(j, i, "-", ha="center", va="center", color=MUTED, fontsize=7)
            continue
        red, green, blue, _ = SEQUENTIAL(normed[i, j])
        dark = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
        ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6, color="#ffffff" if dark else INK)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.grid(False)

def plot_metric_heatmaps(ensemble, dirs):
    '''
    Per channel group: every model (scratch and transfer) against every in-domain metric, plus every scratch model in one figure.
    '''
    matrix, _ = metric_matrix(ensemble)
    if matrix is None:
        return
    save_csv(matrix.rename_axis("model").reset_index(), dirs, "metrics_in_domain")
    for group in CHANNEL_GROUPS:
        matrix, keys = metric_matrix(ensemble, group)
        if matrix is None:
            continue
        fig, ax = plt.subplots(figsize=(0.62 * matrix.shape[1] + 1.8, 0.28 * len(matrix) + 1.8), layout="constrained")
        draw_metric_heatmap(ax, matrix)
        model_dividers(ax, keys)
        ax.set_title(f"In-domain test metrics, {group}-channel models (colour scaled per column)")
        save_figure(fig, dirs["metric_heatmaps"] / f"metrics_in_domain_{group}.png")
    matrix, keys = metric_matrix(ensemble, scratch_only=True)
    if len({channel_group(m) for _, m in keys}) < 2:
        return
    fig, ax = plt.subplots(figsize=(0.62 * matrix.shape[1] + 1.8, 0.28 * len(matrix) + 1.8), layout="constrained")
    draw_metric_heatmap(ax, matrix)
    model_dividers(ax, keys)
    ax.set_title("In-domain test metrics of every model trained from scratch (colour scaled per column)")
    save_figure(fig, dirs["metric_heatmaps"] / "metrics_in_domain_combined.png")

def test_set_matrix(ensemble, group, metric, scratch_only=False):
    '''
    Rows: models of the group (scratch first). Columns: the group's standard test sets.
    '''
    df = ensemble[(ensemble["group"] == group) & (ensemble["test_kind"] == "standard")]
    if scratch_only:
        df = df[~df["transfer_learning"]]
    keys = ordered_models(df)
    if not keys:
        return None, keys
    columns = [d for d in CHANNEL_GROUPS[group] if d in set(df["dataset"])]
    matrix = df.pivot_table(index=["base_model", "model"], columns="dataset", values=metric).reindex(index=pd.MultiIndex.from_tuples(keys), columns=columns)
    matrix.index = [plain_label(*k) for k in keys]
    return matrix, keys

def draw_test_set_heatmap(ax, matrix, keys, norm=None):
    image = draw_matrix(ax, matrix, norm=norm)
    columns = list(matrix.columns)
    for i, (_, model) in enumerate(keys):
        if model in columns:
            ax.add_patch(plt.Rectangle((columns.index(model) - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.2))
    model_dividers(ax, keys)
    ax.set_xlabel("Test set")
    return image

def plot_test_set_heatmaps(ensemble, dirs):
    '''
    Per metric and channel group: every model (scratch and transfer) on every test set of its group, plus the scratch models
    of every group side by side on one colour scale.
    '''
    for metric in TEST_SET_METRICS:
        name = HEATMAP_METRICS[metric]
        for group in CHANNEL_GROUPS:
            matrix, keys = test_set_matrix(ensemble, group, metric)
            if matrix is None:
                continue
            fig, ax = plt.subplots(figsize=(0.75 * matrix.shape[1] + 2.8, 0.28 * len(matrix) + 1.8), layout="constrained")
            image = draw_test_set_heatmap(ax, matrix, keys)
            ax.set_ylabel("Model")
            fig.colorbar(image, ax=ax, label=f"{name} (fold ensemble)", shrink=0.8)
            ax.set_title(f"{name}, {group}-channel models\n(box: in-domain)")
            save_figure(fig, dirs["metric_heatmaps"] / f"{metric}_{group}.png")
        results = {group: test_set_matrix(ensemble, group, metric, scratch_only=True) for group in CHANNEL_GROUPS}
        results = {group: r for group, r in results.items() if r[0] is not None}
        if len(results) < 2:
            continue
        values = np.concatenate([m.values.ravel() for m, _ in results.values()])
        norm = Normalize(np.nanmin(values), np.nanmax(values))
        fig, axes = plt.subplots(1, len(results), figsize=(12, 4.2), layout="constrained", gridspec_kw={"width_ratios": [m.shape[1] for m, _ in results.values()]})
        for ax, (group, (matrix, keys)) in zip(axes, results.items()):
            image = draw_test_set_heatmap(ax, matrix, keys, norm=norm)
            ax.set_title(f"{group.title()}-channel models", pad=24) # | Above the panel label, which the narrow single-channel title would overlap
        axes[0].set_ylabel("Trained on")
        fig.colorbar(image, ax=list(axes), label=f"{name} (fold ensemble)", shrink=0.85)
        label_panels(axes)
        fig.suptitle(f"{name} of the models trained from scratch on every test set of their channel group (box: in-domain)", fontsize=10)
        save_figure(fig, dirs["metric_heatmaps"] / f"{metric}_combined.png")
# endregion

# region TEST SET METRICS
TEST_SET_BAR_METRICS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "mcc", "cohen_kappa", "roc_auc_macro"]

# > Per survey, one set of charts per entry as (test inputs, channel group or None for every group): the survey's own test set
#   (single-channel models) with the survey repeated in every channel (duo and trio models), then the duo and the trio models
#   each on their own with the survey repeated, with zeros and with white noise in the other channels side by side
TEST_SET_CHARTS = {"repeated": (["STD", "REP"], None), "duo": (FILLS, "duo"), "trio": (FILLS, "trio")}
TEST_SET_TITLES = {"repeated": "the {s} test set (single-channel models) and {s} repeated in every channel (duo and trio models)",
                   "duo": "{s}-only inputs, duo-channel models (repeated, zeros and white noise)",
                   "trio": "{s}-only inputs, trio-channel models (repeated, zeros and white noise)"}

def test_set_frame(ensemble, survey, tests, group=None):
    '''
    Every model given only this survey in the chosen test inputs: "STD" is the survey's own test set (single-channel models),
    a fill is the survey's single-survey input (duo and trio models). Ordered by channel group, scratch first, then transfer by base.
    RETURNS:
    --------
        > bars: one row per model (zeros and noise: the survey in its own channel if the model was trained on it, else the mean over channels)
        > rows: every test row behind the bars (one per channel position for zeros and noise)
    '''
    single = ensemble[(ensemble["group"] == "single") & (ensemble["test_kind"] == "standard") & (ensemble["dataset"] == survey)].assign(test="STD")
    multi = ensemble[(ensemble["test_kind"] == "ablation") & (ensemble["ablation_survey"] == survey)]
    rows = pd.concat([single, multi.assign(test=multi["fill"])], ignore_index=True)
    rows = rows[rows["test"].isin(tests) & ((rows["group"] == group) if group else True)]
    keys = ordered_models(rows)
    if not keys:
        return rows, rows
    columns = [c for c in TEST_SET_BAR_METRICS + [f"fold_{m}_std" for m in TEST_SET_BAR_METRICS] if c in rows]
    bars = collapse_positions(rows, ["base_model", "model"], columns).loc[keys].reset_index()
    return bars, rows

MULTI_BASE_COLOURS = {"duo": SERIES[6], "trio": SERIES[4]}

def test_set_colour(base):
    '''
    Scratch grey, a single-channel base its survey colour, and every duo or trio base one colour per channel group (the tick
    labels name the base), so charts mixing every channel group keep a short legend.
    '''
    return MULTI_BASE_COLOURS.get(channel_group(base), base_colour(base))

def test_set_base_label(base):
    if base == "NONE":
        return "From scratch"
    return f"Pretrained on {base}" if channel_group(base) == "single" else f"Pretrained on a {channel_group(base)}-channel model"

def trained_with(model, survey):
    '''
    Outlined bars: the single-channel model trained on the survey, or the duo/trio models with the survey in one of their channels.
    '''
    return model == survey if channel_group(model) == "single" else SURVEY_LETTERS[survey] in model

def draw_test_set_metric(ax, bars, rows, survey, metric, tick_labels=True):
    '''
    One bar per model (coloured by base, grey from scratch), with the fold models' std as the error bar. Zeros and noise bars show
    the survey in its own channel when the model was trained on it (else the mean over channels), with each channel's score as a dot.
    Channel groups are split by a line.
    '''
    values = bars[metric].values.astype(float)
    x = np.arange(len(bars))
    keys = list(zip(bars["base_model"], bars["model"]))
    repeats = rows.groupby(["base_model", "model"]).size().max() > 1
    std_column = f"fold_{metric}_std"
    errors = bars[std_column].values if std_column in bars else None
    outlined = np.array([trained_with(m, survey) for m in bars["model"]])
    ax.bar(x, values, 0.75, yerr=errors, color=[test_set_colour(b) for b in bars["base_model"]], edgecolor=np.where(outlined, INK, SURFACE),
           linewidth=np.where(outlined, 1.4, 0.8), error_kw={"elinewidth": 1, "ecolor": INK_SECONDARY})
    if repeats:
        positions = {key: i for i, key in enumerate(keys)}
        ax.scatter([positions[k] for k in zip(rows["base_model"], rows["model"])], rows[metric], s=8, color=INK, linewidths=0, zorder=3)
    for xi, value in zip(x, values):
        if np.isfinite(value) and value > 0.15: # | Inside the bar, clear of the error bar
            ax.text(xi, 0.02, f"{value:.3f}", rotation=90, ha="center", va="bottom", fontsize=6, color=SURFACE)
        elif np.isfinite(value):
            ax.text(xi, max(value, 0) + 0.02, f"{value:.3f}", rotation=90, ha="center", va="bottom", fontsize=6, color=INK_SECONDARY)
    for i in range(1, len(keys)):
        if channel_group(keys[i - 1][1]) != channel_group(keys[i][1]):
            ax.axvline(i - 0.5, color=MUTED, linewidth=0.8, linestyle="--")
    shown = np.concatenate([values, rows[metric].values.astype(float)])
    low = np.nanmin(shown) if np.isfinite(shown).any() else 0
    ax.set_ylim(min(0, low - 0.05), 1.05) # | MCC and kappa can be negative
    if low < 0:
        ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([strategy_label(b, m) for b, m in keys] if tick_labels else [], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"{HEATMAP_METRICS[metric]}")
    ax.grid(axis="x", visible=False)

def test_set_handles(bars, rows, survey):
    bases = sorted(set(bars["base_model"]), key=lambda b: (b != "NONE", model_order(b)))
    labels = dict.fromkeys(test_set_base_label(b) for b in bases) # | One entry per duo/trio group, in first-seen order
    colours = {test_set_base_label(b): test_set_colour(b) for b in bases}
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[label], label=label) for label in labels]
    groups = set(bars["model"].map(channel_group))
    outline = ([f"Trained on {survey}"] if "single" in groups else []) + ([f"{survey} in one of the model's channels"] if groups - {"single"} else [])
    label = " / ".join(outline)
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=INK, linewidth=1.4, label=label))
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=1, label="Std over fold models"))
    if rows.groupby(["base_model", "model"]).size().max() > 1:
        handles.append(Line2D([], [], marker="o", linestyle="", color=INK, markersize=3, label=f"Score with {survey} in each channel"))
    return handles

def draw_fill_metric(ax, bars, rows, survey, metric, tick_labels=True):
    '''
    One block per model (scratch first, then transfer by base) with one bar per fill of the other channels, with the fold models' std
    as the error bar. Zeros and noise bars show the survey in its own channel when the model was trained on it (else the mean over
    channels), with each channel's score as a dot.
    '''
    keys = list(zip(bars["base_model"], bars["model"]))
    positions = {key: i for i, key in enumerate(keys)}
    tests = [t for t in FILLS if t in set(rows["test"])]
    width = 0.8 / len(tests)
    for k, test in enumerate(tests):
        part = rows[rows["test"] == test]
        means = collapse_positions(part, ["base_model", "model"], [metric] + [c for c in [f"fold_{metric}_std"] if c in part]).reindex(pd.MultiIndex.from_tuples(keys))
        values = means[metric].values.astype(float)
        x = np.arange(len(keys)) + (k - (len(tests) - 1) / 2) * width
        repeats = part.groupby(["base_model", "model"]).size().max() > 1
        errors = means[f"fold_{metric}_std"].values if f"fold_{metric}_std" in means else None
        outlined = np.array([trained_with(m, survey) for _, m in keys])
        ax.bar(x, values, width, yerr=errors, color=FILL_COLOURS[test], edgecolor=np.where(outlined, INK, SURFACE),
               linewidth=np.where(outlined, 1.2, 0.5), error_kw={"elinewidth": 0.8, "ecolor": INK_SECONDARY})
        if repeats:
            offsets = (k - (len(tests) - 1) / 2) * width
            ax.scatter([positions[key] + offsets for key in zip(part["base_model"], part["model"])], part[metric], s=6, color=INK, linewidths=0, zorder=3)
        for xi, value in zip(x, values):
            if np.isfinite(value) and value > 0.15: # | Inside the bar, clear of the error bar
                ax.text(xi, 0.02, f"{value:.3f}", rotation=90, ha="center", va="bottom", fontsize=5, color=SURFACE)
            elif np.isfinite(value):
                ax.text(xi, max(value, 0) + 0.02, f"{value:.3f}", rotation=90, ha="center", va="bottom", fontsize=5, color=INK_SECONDARY)
    for i in range(1, len(keys)):
        if (keys[i - 1][0] == "NONE") != (keys[i][0] == "NONE"):
            ax.axvline(i - 0.5, color=MUTED, linewidth=0.8, linestyle="--")
    shown = rows[metric].values.astype(float)
    low = np.nanmin(shown) if np.isfinite(shown).any() else 0
    ax.set_ylim(min(0, low - 0.05), 1.05) # | MCC and kappa can be negative
    if low < 0:
        ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([strategy_label(b, m) for b, m in keys] if tick_labels else [], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"{HEATMAP_METRICS[metric]}")
    ax.grid(axis="x", visible=False)

def fill_handles(rows, survey):
    tests = [t for t in FILLS if t in set(rows["test"])]
    names = {"REP": f"{survey} repeated in every channel", "ZER": f"{survey} in one channel, zeros in the others", "NSE": f"{survey} in one channel, white noise in the others"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=FILL_COLOURS[t], label=names[t]) for t in tests]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=INK, linewidth=1.2, label=f"{survey} in one of the model's channels"))
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=1, label="Std over fold models"))
    if {"ZER", "NSE"} & set(tests):
        handles.append(Line2D([], [], marker="o", linestyle="", color=INK, markersize=3, label=f"Score with {survey} in each channel (bar: its own channel if trained on it)"))
    return handles

def plot_test_set_metrics(ensemble, dirs):
    '''
    Per test set (FIRST, LoTSS, NVSS): the single-channel models on it with the duo and trio models given it repeated in every
    channel, and separately the duo and the trio models given it repeated, with zeros and with white noise in the other channels,
    against each metric. One PNG per metric and one grid of every metric, in one folder per survey.
    '''
    frames = []
    for survey in SINGLE_CHANNEL:
        folder = dirs["test_set_metrics"] / survey
        for name, (tests, group) in TEST_SET_CHARTS.items():
            bars, rows = test_set_frame(ensemble, survey, tests, group)
            if bars.empty:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            frames.append(rows)
            where = TEST_SET_TITLES[name].format(s=survey)
            metrics = [m for m in TEST_SET_BAR_METRICS if m in bars and bars[m].notna().any()]
            draw = draw_test_set_metric if group is None else draw_fill_metric
            handles = test_set_handles(bars, rows, survey) if group is None else fill_handles(rows, survey)
            per_model = 0.3 if group is None else 0.55 # | Inches per model on the x axis (three bars per model for duo/trio)
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(max(5, per_model * len(bars) + 1.5), 3.4), layout="constrained")
                draw(ax, bars, rows, survey, metric)
                ax.set_xlabel("Model")
                fig.legend(handles=handles, loc="outside lower center", ncol=min(4, len(handles)))
                ax.set_title(f"{HEATMAP_METRICS[metric]} on {where}", fontsize=9)
                save_figure(fig, folder / f"{survey}_{name}_{metric}.png")
            columns = 3
            fig, axes = panel_grid(len(metrics), columns, width=max(12, 3 * (0.75 * per_model * len(bars) + 0.8)), height=3.2, sharex=True, layout="constrained")
            for k, (ax, metric) in enumerate(zip(axes.flat, metrics)):
                draw(ax, bars, rows, survey, metric, tick_labels=k >= len(metrics) - columns) # | Model names on the bottom panel of each column
                ax.set_title(HEATMAP_METRICS[metric])
            label_panels(list(axes.flat)[:len(metrics)])
            fig.legend(handles=handles, loc="outside lower center", ncol=min(6, len(handles)))
            fig.suptitle(f"Every model on {where} (fold ensemble)", fontsize=10)
            save_figure(fig, folder / f"{survey}_{name}_all_metrics_combined.png")
    if frames:
        columns = ["dataset", "test", "group", "base_model", "model", "label", "transfer_learning", "in_domain", "position"] + TEST_SET_BAR_METRICS + ["log_loss"]
        save_csv(pd.concat(frames, ignore_index=True)[columns], dirs, "test_set_metrics")
# endregion

# region BASE VS TARGET
# > Every duo and trio transfer model on the two test sets it learnt from: its base model's dataset (what it was pretrained on)
#   and its own dataset (what it was transfer trained on), so any loss on the base task after fine-tuning shows next to the gain.
BASE_TARGET_COLOURS = {"base": SERIES[0], "target": SERIES[1]}

def base_target_frame(ensemble, group):
    '''
    One row per (transfer model, test set) for the duo or trio transfer models, tested on the base model's dataset ("base") and on
    the model's own dataset ("target"). Ordered by target, then base model.
    '''
    standard = ensemble[(ensemble["test_kind"] == "standard") & (ensemble["group"] == group) & ensemble["transfer_learning"]]
    base = standard[standard["dataset"] == standard["base_model"]].assign(tested_on="base")
    target = standard[standard["dataset"] == standard["model"]].assign(tested_on="target")
    rows = pd.concat([base, target], ignore_index=True)
    keys = ordered_models(rows)
    rows["order"] = rows.apply(lambda r: keys.index((r["base_model"], r["model"])), axis=1)
    return rows.sort_values(["order", "tested_on"]).drop(columns="order").reset_index(drop=True), keys

def draw_base_target(ax, rows, keys, metric, tick_labels=True):
    '''
    Two bars per model: tested on the base model's dataset and on the model's own dataset, with the fold models' std as the error
    bar. Models transfer trained on the same dataset are grouped between dashed lines.
    '''
    width = 0.4
    std_column = f"fold_{metric}_std"
    for k, side in enumerate(["base", "target"]):
        part = rows[rows["tested_on"] == side].set_index(["base_model", "model"]).reindex(pd.MultiIndex.from_tuples(keys))
        values = part[metric].values.astype(float)
        x = np.arange(len(keys)) + (k - 0.5) * width
        errors = part[std_column].values if std_column in part else None
        ax.bar(x, values, width, yerr=errors, color=BASE_TARGET_COLOURS[side], edgecolor=SURFACE, linewidth=0.5,
               error_kw={"elinewidth": 0.8, "ecolor": INK_SECONDARY})
        for xi, value in zip(x, values):
            if np.isfinite(value) and value > 0.15: # | Inside the bar, clear of the error bar
                ax.text(xi, 0.02, f"{value:.3f}", rotation=90, ha="center", va="bottom", fontsize=5, color=SURFACE)
            elif np.isfinite(value):
                ax.text(xi, max(value, 0) + 0.02, f"{value:.3f}", rotation=90, ha="center", va="bottom", fontsize=5, color=INK_SECONDARY)
    for i in range(1, len(keys)):
        if keys[i - 1][1] != keys[i][1]:
            ax.axvline(i - 0.5, color=MUTED, linewidth=0.8, linestyle="--")
    shown = rows[metric].values.astype(float)
    low = np.nanmin(shown) if np.isfinite(shown).any() else 0
    ax.set_ylim(min(0, low - 0.05), 1.05) # | MCC and kappa can be negative
    if low < 0:
        ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([strategy_label(b, m) for b, m in keys] if tick_labels else [], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"{HEATMAP_METRICS[metric]}")
    ax.grid(axis="x", visible=False)

BASE_TARGET_HANDLES = [plt.Rectangle((0, 0), 1, 1, color=BASE_TARGET_COLOURS["base"], label="Tested on the base model's dataset (pretraining)"),
                       plt.Rectangle((0, 0), 1, 1, color=BASE_TARGET_COLOURS["target"], label="Tested on the model's own dataset (transfer training)"),
                       Line2D([], [], color=INK_SECONDARY, linewidth=1, label="Std over fold models")]

def plot_base_vs_target(ensemble, dirs):
    '''
    Per channel group (duo, trio): every transfer model on its base model's test set and on its own test set, against each metric.
    One PNG per metric and one grid of every metric, in one folder per group, plus a CSV and a LaTeX table of the macro F1 and
    accuracy on both test sets and the change between them.
    '''
    frames = []
    for group in ["duo", "trio"]:
        rows, keys = base_target_frame(ensemble, group)
        if rows.empty:
            continue
        folder = dirs["base_vs_target"] / group
        folder.mkdir(parents=True, exist_ok=True)
        frames.append(rows)
        metrics = [m for m in TEST_SET_BAR_METRICS if m in rows and rows[m].notna().any()]
        where = f"{group}-channel transfer models on their base and transfer-training test sets"
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(keys) + 1.5), 3.4), layout="constrained")
            draw_base_target(ax, rows, keys, metric)
            ax.set_xlabel("Model (base $\\rightarrow$ transfer training)")
            fig.legend(handles=BASE_TARGET_HANDLES, loc="outside lower center", ncol=3)
            ax.set_title(f"{HEATMAP_METRICS[metric]} of the {where}", fontsize=9)
            save_figure(fig, folder / f"base_vs_target_{group}_{metric}.png")
        columns = 3
        fig, axes = panel_grid(len(metrics), columns, width=max(12, 3 * (0.35 * len(keys) + 0.8)), height=3.2, sharex=True, layout="constrained")
        for k, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            draw_base_target(ax, rows, keys, metric, tick_labels=k >= len(metrics) - columns) # | Model names on the bottom panel of each column
            ax.set_title(HEATMAP_METRICS[metric])
        label_panels(list(axes.flat)[:len(metrics)])
        fig.legend(handles=BASE_TARGET_HANDLES, loc="outside lower center", ncol=3)
        fig.suptitle(f"The {where} (fold ensemble)", fontsize=10)
        save_figure(fig, folder / f"base_vs_target_{group}_all_metrics_combined.png")

        wide = rows.pivot_table(index=["base_model", "model"], columns="tested_on", values=["macro_f1", "accuracy"]).reindex(pd.MultiIndex.from_tuples(keys))
        table = pd.DataFrame({
            "Model": [strategy_label(b, m) for b, m in keys],
            "Base F1": [f"{v:.3f}" for v in wide[("macro_f1", "base")]],
            "Transfer F1": [f"{v:.3f}" for v in wide[("macro_f1", "target")]],
            "$\\Delta$ F1": [f"{v:+.3f}" for v in wide[("macro_f1", "target")] - wide[("macro_f1", "base")]],
            "Base acc.": [f"{v:.3f}" for v in wide[("accuracy", "base")]],
            "Transfer acc.": [f"{v:.3f}" for v in wide[("accuracy", "target")]],
            "$\\Delta$ acc.": [f"{v:+.3f}" for v in wide[("accuracy", "target")] - wide[("accuracy", "base")]],
        })
        write_latex_table(dirs, f"base_vs_target_{group}", table,
                          f"Macro F1 and accuracy (fold ensemble) of every {group}-channel transfer model on its base model's test set "
                          "(pretraining) and on its own test set (transfer training). $\\Delta$ is transfer minus base.",
                          f"tab:base_vs_target_{group}", resize=True)
    if frames:
        columns = ["group", "base_model", "model", "label", "tested_on", "dataset", "in_domain"] + TEST_SET_BAR_METRICS + \
                  [f"fold_{m}_std" for m in TEST_SET_BAR_METRICS] + ["log_loss"]
        df = pd.concat(frames, ignore_index=True)
        save_csv(df[[c for c in columns if c in df]], dirs, "base_vs_target")
# endregion

# region ABLATION
def ablation_frame(ensemble):
    '''
    The single-survey (ablation) test rows of the duo and trio models, with the same model's in-domain macro F1, the retention
    (ablation / in-domain) and where the kept survey sits relative to the model's own channels (see survey_relation).
    '''
    abl = ensemble[ensemble["test_kind"] == "ablation"].copy()
    if abl.empty:
        return abl
    in_domain = {(b, m): f for b, m, f in ensemble[ensemble["in_domain"]][["base_model", "model", "macro_f1"]].itertuples(index=False)}
    abl["in_domain_macro_f1"] = [in_domain.get((b, m), np.nan) for b, m in zip(abl["base_model"], abl["model"])]
    abl["retention"] = abl["macro_f1"] / abl["in_domain_macro_f1"]
    abl["relation"] = [survey_relation(m, s, f, p) for m, s, f, p in zip(abl["model"], abl["ablation_survey"], abl["fill"], abl["position"])]
    abl["order"] = abl["model"].map(model_order)
    return abl.sort_values(["order", "base_model"]).reset_index(drop=True)

def ablation_summary(ensemble, dirs):
    abl = ablation_frame(ensemble)
    if abl.empty:
        print("No single-survey (ablation) tests found, skipping the ablation summary.")
        return
    save_csv(abl.drop(columns="order"), dirs, "ablation_all_tests")
    ablation_tables(abl, dirs)
    single_survey_comparison(ensemble, abl, dirs)

    for group in ["duo", "trio"]:
        for transfer in [False, True]:
            plot_ablation_heatmap(abl, group, transfer, dirs)
        plot_fill_comparison(abl, group, dirs)
        plot_ablation_position(abl, group, dirs)
        for fill in ["ZER", "NSE"]:
            plot_channel_reliance(abl, group, fill, dirs)
    plot_ablation_heatmap_combined(abl, dirs)
    plot_ablation_effects_combined(abl, dirs)
    plot_channel_reliance_combined(abl, dirs)

def ablation_tables(abl, dirs):
    '''
    One row per model: in-domain macro F1 and the macro F1 with only one survey left, per survey and fill. Zero and noise fills
    use the survey in its own channel when the model was trained on it, else the mean over channels. CSV for every model, LaTeX
    for the scratch models.
    '''
    rows = []
    for (base_model, model), df in abl.groupby(["base_model", "model"], sort=False):
        row = {"base_model": base_model, "model": model, "label": plain_label(base_model, model), "group": df["group"].iloc[0],
               "transfer_learning": base_model != "NONE", "in_domain_macro_f1": df["in_domain_macro_f1"].iloc[0]}
        for survey in SINGLE_CHANNEL:
            for fill in FILLS:
                part = representative_rows(df[(df["ablation_survey"] == survey) & (df["fill"] == fill)])
                row[f"{survey}_{fill}_macro_f1"] = part["macro_f1"].mean() if len(part) else np.nan
        row["mean_ablation_macro_f1"] = df["macro_f1"].mean()
        row["native_zero_macro_f1"] = df[(df["relation"] == "native") & (df["fill"] == "ZER")]["macro_f1"].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    save_csv(summary, dirs, "ablation_summary")

    for group in ["duo", "trio"]:
        df = summary[(summary["group"] == group) & ~summary["transfer_learning"]]
        if df.empty:
            continue
        table = {"Model": df["model"].values, "In-domain": [f"{v:.3f}" for v in df["in_domain_macro_f1"]]}
        bold = []
        for survey in SINGLE_CHANNEL:
            for fill in FILLS:
                column = f"{SURVEY_LETTERS[survey]} {['rep.', 'zero', 'noise'][FILLS.index(fill)]}"
                table[column] = [f"{v:.3f}" if np.isfinite(v) else "--" for v in df[f"{survey}_{fill}_macro_f1"]]
                table[f"{column}_numeric"] = df[f"{survey}_{fill}_macro_f1"].values
                bold.append((column, f"{column}_numeric"))
        write_latex_table(dirs, f"ablation_{group}", pd.DataFrame(table),
            f"Single-survey tests of the {group}-channel models trained from scratch: fold-ensemble macro F1 when only one survey is given. "
            "F, L and N are FIRST, LoTSS and NVSS. Rep.: the survey is repeated in every channel; zero / noise: the survey is placed in one channel and the others are "
            "zeros / white noise, with the survey in its own channel when the model was trained on it (otherwise averaged over the channels). Bold marks the best model per column. Transfer-learning models are in ablation\\_summary.csv.",
            f"tab:ablation_{group}", column_format="lc|" + "ccc|" * 2 + "ccc", bold_max=bold, resize=True)

def single_survey_comparison(ensemble, abl, dirs):
    '''
    Given only one survey, is a single-channel model better than a duo or trio model that was trained on more surveys?
    Compares the best single-channel model on that survey's test set with the best duo and trio model on any single-survey input of it.
    '''
    rows = []
    candidates = {"single": ensemble[(ensemble["group"] == "single") & (ensemble["test_kind"] == "standard")]}
    for survey in SINGLE_CHANNEL:
        row = {"survey": survey}
        for group, df in [("single", candidates["single"][candidates["single"]["dataset"] == survey]),
                          ("duo", abl[(abl["group"] == "duo") & (abl["ablation_survey"] == survey)]),
                          ("trio", abl[(abl["group"] == "trio") & (abl["ablation_survey"] == survey)])]:
            if df.empty:
                continue
            best = df.loc[df["macro_f1"].idxmax()]
            row.update({f"{group}_model": best["label"], f"{group}_dataset": best["dataset"], f"{group}_macro_f1": best["macro_f1"],
                        f"{group}_ci_low": best["macro_f1_ci_low"], f"{group}_ci_high": best["macro_f1_ci_high"]})
        scratch = candidates["single"][(candidates["single"]["model"] == survey) & ~candidates["single"]["transfer_learning"] & (candidates["single"]["dataset"] == survey)]
        row["single_scratch_macro_f1"] = scratch["macro_f1"].iloc[0] if len(scratch) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    save_csv(df, dirs, "single_survey_comparison")

    def cell(row, group):
        if not np.isfinite(row.get(f"{group}_macro_f1", np.nan)):
            return "--"
        model = row[f"{group}_model"].replace("->", "$\\rightarrow$")
        where = "" if group == "single" else f", {ablation_short(row[f'{group}_dataset'])}"
        return f"{row[f'{group}_macro_f1']:.3f} [{row[f'{group}_ci_low']:.3f}, {row[f'{group}_ci_high']:.3f}] ({model}{where})"
    table = pd.DataFrame({"Survey given": df["survey"],
                          "Single-channel model": [cell(r, "single") for _, r in df.iterrows()],
                          "Best duo model": [cell(r, "duo") for _, r in df.iterrows()],
                          "Best trio model": [cell(r, "trio") for _, r in df.iterrows()]})
    write_latex_table(dirs, "single_survey_comparison", table,
        "Best fold-ensemble macro F1 [95\\% bootstrap CI] when only one survey is available: the best single-channel model on that survey's test set, "
        "and the best duo and trio model on any single-survey input of it (model and input in brackets). The best is picked on the test set, so treat it as an upper bound.",
        "tab:single_survey_comparison", column_format="lccc", resize=True)
    plot_single_survey_comparison(df, dirs)

def plot_single_survey_comparison(df, dirs):
    fig, ax = plt.subplots(figsize=(7.5, 3))
    x = np.arange(len(df))
    width = 0.26
    names = {"single": "Single-channel model", "duo": "Best duo model, only this survey", "trio": "Best trio model, only this survey"}
    for k, group in enumerate(["single", "duo", "trio"]):
        if f"{group}_macro_f1" not in df:
            continue
        values = df[f"{group}_macro_f1"].values
        errors = np.array([values - df[f"{group}_ci_low"].values, df[f"{group}_ci_high"].values - values])
        bars = ax.bar(x + (k - 1) * width, values, width, yerr=errors, color=GROUP_COLOURS[group], edgecolor=SURFACE, linewidth=1, label=names[group], error_kw={"elinewidth": 1, "ecolor": INK_SECONDARY})
        for bar, model in zip(bars, df[f"{group}_model"]):
            if isinstance(model, str):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, model, rotation=90, ha="center", va="bottom", fontsize=6, color=SURFACE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Only {s} given" for s in df["survey"]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1 (fold ensemble, 95% CI)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_title("Single-survey input: single-channel model versus the best multi-channel model")
    save_figure(fig, dirs["ablation"] / "single_survey_comparison.png")

def ablation_matrix(abl, group, transfer):
    '''
    Rows: models. Columns: in-domain, then every single-survey input. Also returns the (row, column) cells where the kept survey is in its own channel.
    '''
    df = abl[(abl["group"] == group) & (abl["transfer_learning"] == transfer)]
    if df.empty:
        return None
    columns = [d for d in ablation_datasets(GROUP_CHANNELS[group]) if d in set(df["dataset"])]
    df = df.assign(row=df["label"])
    rows = list(dict.fromkeys(df["row"]))
    matrix = df.pivot_table(index="row", columns="dataset", values="macro_f1").reindex(index=rows, columns=columns)
    matrix.insert(0, "In-domain", df.groupby("row")["in_domain_macro_f1"].first().reindex(rows))
    native = set(zip(df[df["relation"] == "native"]["row"], df[df["relation"] == "native"]["dataset"]))
    return matrix, native

def draw_ablation_heatmap(ax, matrix, native, norm=None):
    image = annotate_heatmap(ax, matrix.values, fmt="{:.2f}", cmap=SEQUENTIAL, norm=norm)
    columns = list(matrix.columns)
    for i, row in enumerate(matrix.index):
        for j, column in enumerate(columns):
            if (row, column) in native:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.2))
    # > White gaps between the in-domain column and each survey's block of inputs
    for j in range(1, len(columns)):
        if j == 1 or describe_dataset(columns[j])["ablation_survey"] != describe_dataset(columns[j - 1])["ablation_survey"]:
            ax.axvline(j - 0.5, color=SURFACE, linewidth=3)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(["In-domain"] + [ablation_short(c) for c in columns[1:]], rotation=90)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    return image

def plot_ablation_heatmap(abl, group, transfer, dirs):
    result = ablation_matrix(abl, group, transfer)
    if result is None:
        return
    matrix, native = result
    fig, ax = plt.subplots(figsize=(0.36 * matrix.shape[1] + 2.2, 0.3 * matrix.shape[0] + 1.6))
    image = draw_ablation_heatmap(ax, matrix, native)
    fig.colorbar(image, ax=ax, label="Macro F1 (fold ensemble)", shrink=0.8, pad=0.02)
    kind = "transfer-learning" if transfer else "from-scratch"
    ax.set_title(f"Single-survey tests, {group}-channel {kind} models (box: survey in its own channel)")
    save_figure(fig, dirs["ablation"] / f"ablation_heatmap_{group}_{'transfer' if transfer else 'scratch'}.png")

def plot_ablation_heatmap_combined(abl, dirs):
    results = {group: ablation_matrix(abl, group, False) for group in ["duo", "trio"]}
    results = {group: r for group, r in results.items() if r is not None}
    if len(results) < 2:
        return
    values = np.concatenate([m.values.ravel() for m, _ in results.values()])
    norm = Normalize(np.nanmin(values), np.nanmax(values))
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.6), layout="constrained", gridspec_kw={"width_ratios": [m.shape[1] for m, _ in results.values()]})
    for ax, (group, (matrix, native)) in zip(axes, results.items()):
        image = draw_ablation_heatmap(ax, matrix, native, norm=norm)
        ax.set_title(f"{group.title()}-channel models")
    fig.colorbar(image, ax=list(axes), label="Macro F1 (fold ensemble)", shrink=0.85)
    label_panels(axes)
    fig.suptitle("Single-survey tests of the models trained from scratch (box: survey in the channel it was trained in)", fontsize=10)
    save_figure(fig, dirs["ablation"] / "ablation_heatmap_combined.png")

def model_level(abl, group, keys, known_only=False):
    '''
    Scratch models of a group: mean macro F1 per model and keys, then mean and std over models. With known_only, only surveys the
    model was trained on, with zero and noise inputs in the survey's own channel, so the fills are compared like for like.
    '''
    df = abl[(abl["group"] == group) & ~abl["transfer_learning"]]
    if known_only:
        df = df[df["relation"].isin(["native", "repeated"])]
    if df.empty:
        return None, None
    per_model = df.groupby(["model"] + keys)["macro_f1"].mean().reset_index()
    stats = per_model.groupby(keys)["macro_f1"].agg(["mean", "std", "count"])
    return stats, abl[(abl["group"] == group) & ~abl["transfer_learning"]].groupby("model")["in_domain_macro_f1"].first().mean()

def draw_fill_comparison(ax, abl, group):
    stats, in_domain = model_level(abl, group, ["ablation_survey", "fill"], known_only=True)
    if stats is None:
        return False
    x = np.arange(len(SINGLE_CHANNEL))
    width = 0.26
    for k, fill in enumerate(FILLS):
        means = [stats["mean"].get((s, fill), np.nan) for s in SINGLE_CHANNEL]
        stds = [stats["std"].get((s, fill), np.nan) for s in SINGLE_CHANNEL]
        ax.bar(x + (k - 1) * width, means, width, yerr=stds, color=FILL_COLOURS[fill], edgecolor=SURFACE, linewidth=1, label=FILL_NAMES[fill] + (" in every channel" if fill == "REP" else " in other channels"), error_kw={"elinewidth": 1, "ecolor": INK_SECONDARY})
    ax.axhline(in_domain, color=INK_SECONDARY, linestyle="--", linewidth=1, label="In-domain (all surveys)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Only {s}" for s in SINGLE_CHANNEL])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1 (mean over models)")
    ax.grid(axis="x", visible=False)
    return True

def plot_fill_comparison(abl, group, dirs):
    fig, ax = plt.subplots(figsize=(6, 3))
    if not draw_fill_comparison(ax, abl, group):
        plt.close(fig)
        return
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title(f"Single-survey inputs by fill, {group}-channel models from scratch trained on that survey\n(zeros/noise: survey in its own channel; error bar: std over models)")
    save_figure(fig, dirs["ablation"] / f"ablation_fill_{group}.png")

def draw_ablation_position(ax, abl, group):
    stats, in_domain = model_level(abl, group, ["relation", "fill"])
    if stats is None:
        return False
    relations = [r for r in RELATIONS if r in stats.index.get_level_values(0)]
    width = 0.26
    for i, relation in enumerate(relations):
        fills = [f for f in FILLS if (relation, f) in stats.index]
        for k, fill in enumerate(fills):
            ax.bar(i + (k - (len(fills) - 1) / 2) * width, stats.loc[(relation, fill), "mean"], width, yerr=stats.loc[(relation, fill), "std"],
                   color=FILL_COLOURS[fill], edgecolor=SURFACE, linewidth=1, error_kw={"elinewidth": 1, "ecolor": INK_SECONDARY})
    ax.axhline(in_domain, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(relations)))
    ax.set_xticklabels([RELATIONS[r] for r in relations])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1 (mean over models)")
    ax.grid(axis="x", visible=False)
    return True

FILL_HANDLES = [plt.Rectangle((0, 0), 1, 1, color=FILL_COLOURS[f], label=f"{FILL_NAMES[f]}" + (" (survey in every channel)" if f == "REP" else " in other channels")) for f in FILLS] + \
               [Line2D([], [], color=INK_SECONDARY, linestyle="--", linewidth=1, label="In-domain (all surveys)")]

def plot_ablation_position(abl, group, dirs):
    fig, ax = plt.subplots(figsize=(6, 3))
    if not draw_ablation_position(ax, abl, group):
        plt.close(fig)
        return
    ax.legend(handles=FILL_HANDLES, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_title(f"Where the kept survey sits, {group}-channel models from scratch")
    save_figure(fig, dirs["ablation"] / f"ablation_position_{group}.png")

def plot_ablation_effects_combined(abl, dirs):
    '''
    Combined figure: fill comparison (top) and channel-position effect (bottom) for the duo and trio models.
    '''
    groups = [g for g in ["duo", "trio"] if not abl[(abl["group"] == g) & ~abl["transfer_learning"]].empty]
    if len(groups) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True, layout="constrained")
    for k, group in enumerate(groups):
        draw_fill_comparison(axes[0, k], abl, group)
        axes[0, k].set_title(f"{group.title()} channel: fill of the other channels (survey in its own channel)")
        draw_ablation_position(axes[1, k], abl, group)
        axes[1, k].set_title(f"{group.title()} channel: where the kept survey sits")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    fig.legend(handles=FILL_HANDLES, loc="outside lower center", ncol=4)
    label_panels(axes.flat)
    fig.suptitle("Single-survey tests of the models trained from scratch (mean over models, error bar: std over models)", fontsize=10)
    save_figure(fig, dirs["ablation"] / "ablation_effects_combined.png")

def draw_channel_reliance(ax, abl, group, fill):
    '''
    Per scratch model, the macro F1 with only one of its surveys left in its own channel (the other channels filled), in channel order,
    against the in-domain score with every channel.
    '''
    df = abl[(abl["group"] == group) & ~abl["transfer_learning"] & (abl["relation"] == "native") & (abl["fill"] == fill)]
    models = [m for m in CHANNEL_GROUPS[group] if m in set(df["model"])]
    if not models:
        return False
    for i, model in enumerate(models):
        width = 0.8 / len(model)
        for k, letter in enumerate(model):
            entry = df[(df["model"] == model) & (df["ablation_survey"] == SURVEY_NAMES[letter])]
            if len(entry):
                ax.bar(i - 0.4 + width * (k + 0.5), entry["macro_f1"].iloc[0], width, color=SURVEY_COLOURS[SURVEY_NAMES[letter]], edgecolor=SURFACE, linewidth=1)
        in_domain = df[df["model"] == model]["in_domain_macro_f1"].iloc[0]
        ax.plot([i - 0.42, i + 0.42], [in_domain] * 2, color=INK, linewidth=1.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1 (fold ensemble)")
    ax.grid(axis="x", visible=False)
    return True

RELIANCE_HANDLES = [plt.Rectangle((0, 0), 1, 1, color=SURVEY_COLOURS[s], label=f"Only {s}") for s in SINGLE_CHANNEL] + \
                   [Line2D([], [], color=INK, linewidth=1.5, label="All channels (in-domain)")]

def plot_channel_reliance(abl, group, fill, dirs):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    if not draw_channel_reliance(ax, abl, group, fill):
        plt.close(fig)
        return
    ax.legend(handles=RELIANCE_HANDLES, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.set_title(f"Channel reliance, {group}-channel models (other channels: {FILL_NAMES[fill].lower()})")
    save_figure(fig, dirs["ablation"] / f"channel_reliance_{group}_{fill.lower()}.png")

def plot_channel_reliance_combined(abl, dirs):
    groups = [g for g in ["duo", "trio"] if not abl[(abl["group"] == g) & ~abl["transfer_learning"]].empty]
    if len(groups) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharey=True, layout="constrained")
    for i, fill in enumerate(["ZER", "NSE"]):
        for k, group in enumerate(groups):
            draw_channel_reliance(axes[i, k], abl, group, fill)
            axes[i, k].set_title(f"{group.title()} channel, other channels {FILL_NAMES[fill].lower()}")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    fig.legend(handles=RELIANCE_HANDLES, loc="outside lower center", ncol=4)
    label_panels(axes.flat)
    fig.suptitle("Channel reliance of the models trained from scratch: one survey kept in the channel it was trained in (bars in channel order)", fontsize=10)
    save_figure(fig, dirs["ablation"] / "channel_reliance_combined.png")
# endregion

# region MAIN
def main():
    set_style()
    dirs = output_dirs()
    print(f"Summarising {EXPERIMENT_DIRECTORY} into {SUMMARY_DIRECTORY}\n")

    dataset_summary(dirs)

    all_folds, all_histories = [], {}
    for stage in STAGES:
        folds, histories = load_training_runs(stage)
        print(f"{stage}: {folds['folder'].nunique() if not folds.empty else 0} trained configs found.")
        if not folds.empty:
            all_folds.append(folds)
            all_histories.update(histories)
    if all_folds:
        folds = pd.concat(all_folds, ignore_index=True)
        save_csv(folds, dirs, "training_all_folds")
        configs = aggregate_configs(folds)
        search_progression(configs, dirs)
        training_summary(configs, all_histories, dirs)
    else:
        print("No training results found.")

    runs = load_test_runs()
    print(f"FINAL: {len(runs)} test folders found.")
    testing_summary(runs, dirs)

    write_latex_index(dirs)
    print(f"\nDone. Tables: {dirs['latex']} | CSVs: {dirs['csv']} | Figures: {dirs['figures']}")

if __name__ == "__main__":
    log_path = SUMMARY_DIRECTORY / "results_summary_log.txt"
    SUMMARY_DIRECTORY.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, text): [(s.write(text), s.flush()) for s in self.streams]
        def flush(self): [s.flush() for s in self.streams]

    with open(log_path, "w", encoding="utf-8") as log:
        sys.stdout = Tee(sys.__stdout__, log)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
# endregion
