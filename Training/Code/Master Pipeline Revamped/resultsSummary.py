# region ABOUT
# ===================================================================================================
# > This script summarises everything the revamped master pipeline produces, for writing up in the thesis:
    # - SEARCH: the COARSE and FINE hyperparameter searches and the FINAL configs, so the progression from the
    #   coarse grid to the chosen hyperparameters can be shown (rankings, heatmaps, sensitivity and spread plots).
    # - TRAINING: learning curves, early-stopping epochs and training times of every FINAL model (scratch and transfer).
    # - TESTING: per-fold and fold-ensemble test metrics with bootstrap confidence intervals, cross-survey
    #   generalisation matrices, transfer-learning gains with McNemar tests, confusion matrices, ROC curves,
    #   channel-order effects and a generalisation leaderboard per channel group.
    # - ABLATIONS: how the duo- and trio-channel models cope when only one survey is available (the survey repeated in
    #   every channel, or with zeros / white noise in the other channels), which fill works best, whether the channel
    #   position matters, how much each model relies on each survey, and how they compare with single-channel models.
# > Where plots belong together, every separate PNG is also combined into one figure (files ending in _combined.png).
# > Every result is saved as a CSV, every table also as a LaTeX (booktabs) file, and every chart as a PNG, under
#   Results/Experiment_<n>/Summary/. Include the tables with \input{...}; they need \usepackage{booktabs}.
# > Hyperparameters are only ever selected on validation loss. Test metrics are reported, never used for selection.
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
    for sub in ["search", "training", "testing", "ablation", "learning_curves", "confusion_matrices"]:
        dirs[sub] = dirs["figures"] / sub
    dirs["search_heatmaps"] = dirs["search"] / "heatmaps"
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

def save_figure(fig, path):
    fig.savefig(path)
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
    '''
    if folds.empty:
        return folds
    keys = ["stage", "folder", "base_model", "model", "group", "transfer_learning", "lr", "reg", "wd"]
    metrics = ["val_loss", "val_accuracy", "train_loss", "train_accuracy", "generalisation_gap", "best_epoch", "epochs_run", "training_time_min"]
    grouped = folds.groupby(keys)
    configs = grouped[metrics].mean().add_suffix("_mean").join(grouped[metrics].std().add_suffix("_std")).reset_index()
    configs["n_folds"] = grouped.size().values
    configs = configs.sort_values(["stage", "base_model", "model", "val_loss_mean"]).reset_index(drop=True)
    configs["rank"] = configs.groupby(["stage", "base_model", "model"]).cumcount() + 1
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
        stats = folds.groupby("folder")[["accuracy", "balanced_accuracy", "macro_f1", "mcc", "roc_auc_macro"]].agg(["mean", "std"])
        stats.columns = [f"fold_{metric}_{stat}" for metric, stat in stats.columns]
        ensemble = ensemble.merge(stats.reset_index(), on="folder", how="left")
    return folds, ensemble

def generalisation_scores(ensemble):
    '''
    One row per trained model: mean and worst-case ensemble macro F1 over every standard test set in its channel group
    (real survey combinations, not ablations), the in-domain score, the out-of-domain mean and retention (out-of-domain / in-domain).
    '''
    rows = []
    for (base_model, model), df in ensemble[ensemble["test_kind"] == "standard"].groupby(["base_model", "model"]):
        in_domain = df[df["in_domain"]]
        out_domain = df[~df["in_domain"]]
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
    best = scratch[scratch["rank"] == 1].copy()
    save_csv(best, dirs, "search_best_per_stage")

    # > Progression table: chosen hyperparameters and validation loss at each stage
    rows = []
    for survey in ALL_SURVEYS:
        row = {"Survey": survey}
        numeric = {}
        for stage in STAGES:
            entry = best[(best["stage"] == stage) & (best["model"] == survey)]
            if entry.empty:
                row[f"{stage.title()} (lr, reg, wd)"] = "--"
                row[f"{stage.title()} val. loss"] = "--"
                numeric[stage] = np.nan
                continue
            entry = entry.iloc[0]
            row[f"{stage.title()} (lr, reg, wd)"] = f"{sci(entry['lr'])}, {sci(entry['reg'])}, {sci(entry['wd'])}"
            row[f"{stage.title()} val. loss"] = mean_std(entry["val_loss_mean"], entry["val_loss_std"])
            numeric[stage] = entry["val_loss_mean"]
        if all(not np.isfinite(v) for v in numeric.values()):
            continue
        rows.append(row)
    table = pd.DataFrame(rows)
    # > FINAL uses the FINE best, so its hyperparameter column is dropped from the LaTeX table
    latex = table.drop(columns=[c for c in table.columns if c.startswith("Final (")])
    write_latex_table(dirs, "search_progression", latex,
        "Hyperparameter search progression. For each input, the configuration with the lowest mean validation loss (mean $\\pm$ std over folds) in the coarse and fine searches, and the validation loss of the final model trained with the fine-search optimum. Surveys without a coarse entry took their fine-search centre from the coarse result of the same survey combination.",
        "tab:search_progression", column_format="l" + "cc" * 2 + "c")

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
    for stage in ["COARSE", "FINE"]:
        df = scratch[scratch["stage"] == stage]
        if not df.empty:
            plot_search_heatmaps(df, stage, dirs)
            plot_search_sensitivity(df, stage, dirs)

def draw_search_progression(ax, best):
    surveys = [s for s in ALL_SURVEYS if s in set(best["model"])]
    offsets = {"COARSE": -0.2, "FINE": 0.0, "FINAL": 0.2}
    for stage in STAGES:
        df = best[best["stage"] == stage].set_index("model")
        x = [i + offsets[stage] for i, s in enumerate(surveys) if s in df.index]
        y = [df.loc[s, "val_loss_mean"] for s in surveys if s in df.index]
        e = [df.loc[s, "val_loss_std"] for s in surveys if s in df.index]
        if x:
            ax.errorbar(x, y, yerr=e, fmt="o", color=STAGE_COLOURS[stage], ecolor=STAGE_COLOURS[stage], elinewidth=1, capsize=0, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, label=stage.title())
    for boundary in [len(SINGLE_CHANNEL) - 0.5, len(SINGLE_CHANNEL) + len(DUO_CHANNEL) - 0.5]:
        if boundary < len(surveys):
            ax.axvline(boundary, color=GRIDLINE, linewidth=1)
    ax.set_xticks(range(len(surveys)))
    ax.set_xticklabels(surveys, rotation=45, ha="right")
    ax.set_ylabel("Best mean validation loss")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))

def plot_search_progression(best, dirs):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    draw_search_progression(ax, best)
    ax.set_title("Validation loss of the best configuration at each search stage")
    save_figure(fig, dirs["search"] / "search_progression.png")

def draw_search_spread(ax, search):
    surveys = [s for s in ALL_SURVEYS if s in set(search["model"])]
    rng = np.random.default_rng(SEED)
    offsets = {"COARSE": -0.18, "FINE": 0.18}
    for stage in ["COARSE", "FINE"]:
        df = search[search["stage"] == stage]
        for i, survey in enumerate(surveys):
            values = df[df["model"] == survey]["val_loss_mean"].values
            if len(values) == 0:
                continue
            ax.scatter(i + offsets[stage] + rng.uniform(-0.08, 0.08, len(values)), values, s=10, color=STAGE_COLOURS[stage], alpha=0.55, linewidths=0)
            ax.plot([i + offsets[stage] - 0.12, i + offsets[stage] + 0.12], [values.min()] * 2, color=INK, linewidth=1.5)
    ax.set_xticks(range(len(surveys)))
    ax.set_xticklabels(surveys, rotation=45, ha="right")
    ax.set_ylabel("Mean validation loss")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], marker="o", linestyle="", color=STAGE_COLOURS[s], label=s.title()) for s in ["COARSE", "FINE"]]
    ax.legend(handles=handles, ncol=2, loc="upper left")

def plot_search_spread(scratch, dirs):
    '''
    Every config's validation loss per input and stage: shows how the fine search narrows onto the good region.
    '''
    search = scratch[scratch["stage"].isin(["COARSE", "FINE"])]
    if search.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    draw_search_spread(ax, search)
    ax.set_title("Validation loss of every searched configuration (bar marks the best)")
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
    axes[1].set_title("Every searched configuration (bar marks the best)")
    label_panels(axes)
    fig.tight_layout()
    save_figure(fig, dirs["search"] / "search_overview_combined.png")

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
            mean, std, _ = mean_curve(histories[folder], key)
            epochs = np.arange(1, len(mean) + 1)
            ax.plot(epochs, mean, color=SPLIT_COLOURS[split], label="Training" if split == "train" else "Validation")
            ax.fill_between(epochs, mean - std, mean + std, color=SPLIT_COLOURS[split], alpha=0.15, linewidth=0)
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
            fig.suptitle(f"Learning curves, {group}-channel models from scratch (mean $\\pm$ std over folds)", fontsize=10)
            fig.tight_layout()
            save_figure(fig, dirs["training"] / f"learning_curves_{metric}_{group}.png")
        if sum(len(s) > 0 for s in panels.values()) > 1:
            fig, axes = panel_grid(len(every), 3, height=2.0, sharey=True)
            fill_learning_grid(axes, every, scratch, histories, metric, ylabel)
            fig.suptitle(f"Learning curves of every model trained from scratch (mean $\\pm$ std over folds)", fontsize=10)
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
            mean, _, _ = mean_curve(fold_histories, key)
            ax.plot(np.arange(1, len(mean) + 1), mean, color=SPLIT_COLOURS[split], label="Training" if split == "train" else "Validation")
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
        test_sets = "the FIRST, NVSS and LoTSS test sets" if group == "single" else f"all {group}-channel orderings of the test set"
        write_latex_table(dirs, f"leaderboard_{group}", table,
            f"Generalisation leaderboard of the {group}-channel models: fold-ensemble macro F1 averaged over {test_sets}, the worst case, and the in-domain versus out-of-domain scores (retention = out-of-domain / in-domain).",
            f"tab:leaderboard_{group}")

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
            "delta_numeric": part["delta_macro_f1"].values,
        })
        write_latex_table(dirs, f"transfer_gains_{group}", table,
            f"Transfer learning versus training from scratch ({group}-channel inputs), in-domain fold-ensemble macro F1 on the target's test set. The exact McNemar test compares the two models' predictions on the same test images.",
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
    for i, p in enumerate(df["mcnemar_p"]):
        if p < 0.05:
            ax.text(df["delta_macro_f1"].iloc[i], i, " *" if df["delta_macro_f1"].iloc[i] >= 0 else "* ", va="center", ha="left" if df["delta_macro_f1"].iloc[i] >= 0 else "right", color=INK)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$ macro F1, transfer minus scratch (in-domain)   * McNemar $p<0.05$")
    ax.grid(axis="y", visible=False)
    ax.set_title("Effect of transfer learning on every target")
    save_figure(fig, dirs["testing"] / "transfer_gains.png")

LEADERBOARD_HANDLES = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0], label="From scratch (mean)"),
                       plt.Rectangle((0, 0), 1, 1, color=SERIES[1], label="Transfer learning (mean)"),
                       Line2D([], [], marker="|", linestyle="", color=INK, markersize=10, markeredgewidth=2, label="Worst test set")]

def draw_leaderboard(ax, df):
    y = np.arange(len(df))
    colours = [SERIES[1] if t else SERIES[0] for t in df["transfer_learning"]]
    ax.barh(y, df["mean_macro_f1"], color=colours, edgecolor=SURFACE, linewidth=2, height=0.7)
    ax.scatter(df["worst_macro_f1"], y, marker="|", s=120, color=INK, linewidths=2, zorder=3)
    for i, value in enumerate(df["mean_macro_f1"]):
        ax.text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=7, color=INK_SECONDARY)
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
    ax.bar(range(len(members)), df["macro_f1"], yerr=df["fold_macro_f1_std"], color=[colours["".join(sorted(s))] for s in members], edgecolor=SURFACE, linewidth=2, error_kw={"elinewidth": 1, "ecolor": INK_SECONDARY})
    ax.set_xticks(range(len(members)))
    ax.set_xticklabels(members)
    low = max(0, df["macro_f1"].min() - 0.1)
    ax.set_ylim(low, 1)
    ax.set_ylabel("In-domain macro F1")
    ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[k], label="+".join(SURVEY_NAMES[c] for c in k)) for k in combinations]
    ax.legend(handles=handles, loc="lower right", ncol=2)
    ax.set_title("Effect of channel order (ensemble, error bar = fold std)")
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
    One row per model: in-domain macro F1 and the macro F1 with only one survey left, per survey and fill (zero and
    noise fills averaged over the survey's channel positions). CSV for every model, LaTeX for the scratch models.
    '''
    rows = []
    for (base_model, model), df in abl.groupby(["base_model", "model"], sort=False):
        row = {"base_model": base_model, "model": model, "label": plain_label(base_model, model), "group": df["group"].iloc[0],
               "transfer_learning": base_model != "NONE", "in_domain_macro_f1": df["in_domain_macro_f1"].iloc[0]}
        for survey in SINGLE_CHANNEL:
            for fill in FILLS:
                part = df[(df["ablation_survey"] == survey) & (df["fill"] == fill)]
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
            "zeros / white noise, averaged over the channel the survey is placed in. Bold marks the best model per column. Transfer-learning models are in ablation\\_summary.csv.",
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

def model_level(abl, group, keys):
    '''
    Scratch models of a group: mean macro F1 per model and keys (averaging positions), then mean and std over models.
    '''
    df = abl[(abl["group"] == group) & ~abl["transfer_learning"]]
    if df.empty:
        return None, None
    per_model = df.groupby(["model"] + keys)["macro_f1"].mean().reset_index()
    stats = per_model.groupby(keys)["macro_f1"].agg(["mean", "std", "count"])
    return stats, df.groupby("model")["in_domain_macro_f1"].first().mean()

def draw_fill_comparison(ax, abl, group):
    stats, in_domain = model_level(abl, group, ["ablation_survey", "fill"])
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
    ax.set_title(f"Single-survey inputs by fill, {group}-channel models from scratch (error bar: std over models)")
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
        axes[0, k].set_title(f"{group.title()} channel: fill of the other channels")
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
