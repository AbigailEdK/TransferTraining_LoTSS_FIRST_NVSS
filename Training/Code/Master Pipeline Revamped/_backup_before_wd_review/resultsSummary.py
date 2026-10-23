import os
import re
import itertools
import sys
from contextlib import redirect_stdout
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


##############################################################################
# GLOBAL SETTINGS
##############################################################################

CLASSES = ["FRI", "FRII", "COMPACT"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

RUN_DIRECTORIES = [
    "/home/abigaildeklerk/Desktop/DeKlerk_Models/Training/Results/Experiment_1",
    "/home/abigaildeklerk/Desktop/DeKlerk_Models/Training/Results/Experiment_2",
    "/home/abigaildeklerk/Desktop/DeKlerk_Models/Training/Results/Experiment_3",
    "/home/abigaildeklerk/Desktop/DeKlerk_Models/Training/Results/Experiment_4",
    "/home/abigaildeklerk/Desktop/DeKlerk_Models/Training/Results/Experiment_5",
]

RESULTS_DIR = "/home/abigaildeklerk/Desktop/DeKlerk_Models/Results"

# COMPOSITE_WEIGHTS = {
#     "macro_f1": 0.40,
#     "balanced_accuracy": 0.30,
#     "mcc": 0.20,
#     "specificity": 0.10,
# }

COMPOSITE_WEIGHTS = {
    "macro_f1": 0.40,
    "balanced_accuracy": 0.30,
    "mcc": 0.20,
    "f1_COMPACT": 0.10,
}

# Below this in-domain accuracy, retention ratios are considered unreliable
# (a tiny denominator can make retention look artificially large or unstable).
MIN_RELIABLE_IN_DOMAIN_ACCURACY = 0.10

# Number of top-ranked strategies to include in pairwise significance testing.
SIGNIFICANCE_TOP_N = 5

# Number of bootstrap resamples used for the mean-difference confidence intervals.
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_RANDOM_SEED = 42

# Number of bins used for Expected Calibration Error.
ECE_BINS = 10

# --- Final model score ----------------------------------------------------
# For each model (one strategy in one experiment), average the five fold
# composite scores within each test set, then combine the three test-set
# means using these fixed weights.
FINAL_MODEL_DATASET_WEIGHTS = {
    "FIRST": 0.45,
    "LOFAR": 0.45,
    "NVSS": 0.10,
}

# Whether the descriptive generalisation score excludes each strategy's target
# dataset. The final model score always uses all three weighted test datasets.
EXCLUDE_TARGET_FROM_GENERALISATION_SCORE = False

# Datasets excluded from all experiment evaluation and published summaries.
# Set to set() to include them again.
EXCLUDED_DATASETS = {"RADCAT"}

# Metric averaged per dataset before weighting.
GENERALISATION_SCORE_METRIC = "composite_score"

PROBABILITY_FILES = [
    "y_prob_test_fold_{fold}.npy",
    "y_proba_test_fold_{fold}.npy",
    "y_pred_prob_test_fold_{fold}.npy",
    "y_pred_proba_test_fold_{fold}.npy",
    "y_pred_test_prob_fold_{fold}.npy",
    "y_pred_test_fold_{fold}_prob.npy",
]

FOLD_METRICS = [
    "accuracy",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "mcc",
    "kappa",
    "balanced_accuracy",
    "specificity",
    "log_loss",
    "roc_auc_macro_ovr",
    "brier_score",
    "ece",
    "precision_FRI",
    "recall_FRI",
    "f1_FRI",
    "precision_FRII",
    "recall_FRII",
    "f1_FRII",
    "precision_COMPACT",
    "recall_COMPACT",
    "f1_COMPACT",
    "composite_score",
]


##############################################################################
# HELPERS
##############################################################################

def ensure_output_dirs(root_dir):
    output_map = {
        "confusion_matrices": "confusion_matrices",
        "best_fold_confusion_matrices": "best_fold_confusion_matrices",
        "heatmaps": "heatmaps",
        "roc_curves": "roc_curves",
        "precision_recall_curves": "precision_recall_curves",
        "ranking_plots": "ranking_plots",
        "publication_tables": "publication_tables",
        "transfer_matrices": "transfer_matrices",
        "significance_tests": "significance_tests",
        "calibration": "calibration",
        "weighted_generalisation": "weighted_generalisation",
        "cross_dataset_consistency": "cross_dataset_consistency",
        "metric_csvs": "metric_csvs",
    }

    os.makedirs(root_dir, exist_ok=True)

    resolved = {}
    for key, relative_dir in output_map.items():
        full = os.path.join(root_dir, relative_dir)
        os.makedirs(full, exist_ok=True)
        resolved[key] = full
    return resolved


def parse_folder(folder_name):
    pattern = (
        r"base(.*?)_"
        r"target(.*?)_"
        r"dataset(.*?)_"
        r"([0-9eE\.-]+)l_"
        r"base([0-9eE\.-]+)r_"
        r"([0-9eE\.-]+)r_"
        r"ceiling(\d+)e"
    )
    match = re.search(pattern, folder_name)
    if match is None:
        return None

    base_dataset = match.group(1)
    target_dataset = match.group(2)
    test_dataset = match.group(3)

    strategy_id = (
        f"base={base_dataset}|target={target_dataset}|lr={match.group(4)}|"
        f"base_reg={match.group(5)}|transfer_reg={match.group(6)}|ceiling={match.group(7)}"
    )
    strategy_label = (
        f"{base_dataset}->{target_dataset} "
        f"(lr={match.group(4)}, br={match.group(5)}, tr={match.group(6)}, c={match.group(7)})"
    )

    return {
        "base_dataset": base_dataset,
        "target_dataset": target_dataset,
        "test_dataset": test_dataset,
        "learning_rate": float(match.group(4)),
        "base_regularisation": float(match.group(5)),
        "transfer_regularisation": float(match.group(6)),
        "ceiling_epochs": int(match.group(7)),
        "transfer_learning": base_dataset != "NONE",
        "training_category": "transfer_learning" if base_dataset != "NONE" else "scratch",
        "strategy_id": strategy_id,
        "strategy_label": strategy_label,
    }


def to_int_labels(y):
    y = np.asarray(y)
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
    if y.dtype.kind in {"U", "S", "O"}:
        y = np.array([CLASS_TO_IDX[str(v)] for v in y])
    return y.astype(int)


def maybe_probabilities(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return None
    if arr.shape[1] != len(CLASSES):
        return None
    if not np.isfinite(arr).all():
        return None
    if np.any(arr < 0):
        return None

    # Reject one-hot encoded class labels because these are not true probabilities.
    unique_values = np.unique(arr)
    row_sums = np.sum(arr, axis=1)
    if np.all(np.isin(unique_values, [0, 1])) and np.allclose(row_sums, 1.0, atol=1e-6):
        return None

    return normalise_probabilities(arr.astype(float))


def normalise_probabilities(y_prob):
    row_sums = np.sum(y_prob, axis=1, keepdims=True)
    valid = row_sums.squeeze() > 0
    if not np.any(valid):
        return None

    normalised = y_prob.copy()
    normalised[valid] = normalised[valid] / row_sums[valid]

    # Remove rows that cannot be normalized safely.
    if not np.isfinite(normalised).all():
        return None

    return normalised


def find_probability_array(test_folder, fold):
    for pattern in PROBABILITY_FILES:
        candidate = os.path.join(test_folder, pattern.format(fold=fold))
        if os.path.exists(candidate):
            try:
                prob = np.load(candidate)
            except Exception:
                continue
            prob = maybe_probabilities(prob)
            if prob is not None:
                return prob
    return None


def compute_specificity(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    class_specificity = []
    for cm in mcm:
        tn, fp, fn, tp = cm.ravel()
        denom = tn + fp
        class_specificity.append((tn / denom) if denom > 0 else np.nan)
    return np.nanmean(class_specificity)


def compute_per_class_confusion_stats(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    stats = {}
    for class_idx, class_name in enumerate(CLASSES):
        tn, fp, fn, tp = mcm[class_idx].ravel()
        denom = tn + fp
        specificity = (tn / denom) if denom > 0 else np.nan
        stats[f"tp_{class_name}"] = int(tp)
        stats[f"fp_{class_name}"] = int(fp)
        stats[f"fn_{class_name}"] = int(fn)
        stats[f"tn_{class_name}"] = int(tn)
        stats[f"specificity_{class_name}"] = float(specificity)
    return stats


def compute_brier_score(y_true, y_prob):
    """Multiclass Brier score: mean squared error between one-hot labels and
    predicted probabilities. Lower is better (0 = perfect). Unlike accuracy,
    this is sensitive to *how confident* a wrong or right prediction was,
    which tends to degrade under dataset shift even when raw accuracy holds up."""
    if y_prob is None:
        return np.nan
    n_classes = len(CLASSES)
    one_hot = np.zeros((len(y_true), n_classes))
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def compute_ece(y_true, y_prob, n_bins=ECE_BINS):
    """Expected Calibration Error using the top predicted class's confidence.
    Bins predictions by confidence and compares average confidence to
    observed accuracy within each bin. Lower is better (0 = perfectly
    calibrated)."""
    if y_prob is None:
        return np.nan

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)

        bin_count = np.sum(in_bin)
        if bin_count == 0:
            continue

        bin_confidence = np.mean(confidences[in_bin])
        bin_accuracy = np.mean(correct[in_bin])
        ece += (bin_count / n) * abs(bin_confidence - bin_accuracy)

    return float(ece)


def compute_composite_score(metric_row):
    score = 0.0
    for metric_name, weight in COMPOSITE_WEIGHTS.items():
        value = metric_row.get(metric_name, np.nan)
        if pd.isna(value):
            return np.nan
        score += weight * value
    return score


def safe_probability_metrics(y_true, y_prob):
    if y_prob is None:
        return np.nan, np.nan

    try:
        ll = log_loss(y_true, y_prob, labels=np.arange(len(CLASSES)))
    except Exception:
        ll = np.nan

    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = np.nan

    return ll, roc_auc


def find_training_folder_name(test_folder_name, target_dataset):
    return re.sub(r"_dataset[^_]+_", f"_dataset{target_dataset}_", test_folder_name, count=1)


def resolve_model_path(run_dir, training_folder_name, fold):
    training_root = os.path.join(run_dir, "Training", training_folder_name)
    if not os.path.isdir(training_root):
        return "", ""

    default_model = os.path.join(training_root, f"model_fold_{fold}.keras")
    if os.path.exists(default_model):
        return os.path.basename(default_model), default_model

    pattern = re.compile(rf"model_fold_{fold}\.")
    for file_name in sorted(os.listdir(training_root)):
        if pattern.search(file_name):
            full = os.path.join(training_root, file_name)
            return file_name, full
    return "", ""


def order_datasets(values):
    preferred = [
        "FIRST", "LOFAR", "NVSS", "RADCAT",
        "FNSE", "FREP", "FZER", "LNSE", "LREP", "LZER", "NNSE", "NREP", "NZER",
    ]
    known = [x for x in preferred if x in values]
    unknown = sorted([x for x in values if x not in preferred])
    return known + unknown


def ci95(series):
    series = pd.Series(series).dropna()
    n = len(series)
    if n <= 1:
        return np.nan
    return 1.96 * series.std(ddof=1) / np.sqrt(n)


def holm_bonferroni_adjust(p_values):
    """Holm-Bonferroni step-down correction for a list/array of p-values.
    Returns adjusted p-values in the same order as the input."""
    p_values = np.array([1.0 if pd.isna(p) else p for p in p_values], dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank_idx, idx in enumerate(order):
        p_adj = min(1.0, (m - rank_idx) * p_values[idx])
        running_max = max(running_max, p_adj)
        adjusted[idx] = running_max
    return adjusted


def export_table(df, directory, base_name):
    csv_path = os.path.join(directory, f"{base_name}.csv")
    md_path = os.path.join(directory, f"{base_name}.md")
    tex_path = os.path.join(directory, f"{base_name}.tex")

    df.to_csv(csv_path, index=False)

    try:
        markdown = df.to_markdown(index=False)
    except Exception:
        # Fallback markdown table writer when optional tabulate is unavailable.
        columns = [str(c) for c in df.columns]
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---" for _ in columns]) + " |"
        body = []
        for _, row in df.iterrows():
            body.append("| " + " | ".join([str(v) for v in row.values]) + " |")
        markdown = "\n".join([header, separator] + body)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    # LaTeX export for thesis-ready table inclusion.
    try:
        latex_table = df.to_latex(index=False, escape=True)
    except Exception:
        latex_table = ""

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)


class TeeOutput:
    """Write script output to the terminal and a persistent text file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


##############################################################################
# STAGE 1: LOAD ALL EXPERIMENT RESULTS (INDEX)
##############################################################################

def load_experiment_index(run_directories):
    rows = []
    for experiment_number, run_dir in enumerate(run_directories, start=1):
        testing_dir = os.path.join(run_dir, "Testing")
        if not os.path.isdir(testing_dir):
            continue

        for folder_name in sorted(os.listdir(testing_dir)):
            folder_path = os.path.join(testing_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            metadata = parse_folder(folder_name)
            if metadata is None:
                continue
            if any(
                metadata[field] in EXCLUDED_DATASETS
                for field in ("base_dataset", "target_dataset", "test_dataset")
            ):
                continue

            y_true_path = os.path.join(folder_path, "y_test.npy")
            if not os.path.exists(y_true_path):
                continue

            training_folder = find_training_folder_name(folder_name, metadata["target_dataset"])

            for fold in range(1, 6):
                y_pred_path = os.path.join(folder_path, f"y_pred_test_fold_{fold}.npy")
                if not os.path.exists(y_pred_path):
                    continue

                rows.append({
                    "experiment_number": experiment_number,
                    "run_dir": run_dir,
                    "testing_folder_name": folder_name,
                    "testing_folder_path": folder_path,
                    "training_folder_name": training_folder,
                    "fold": fold,
                    "y_true_path": y_true_path,
                    "y_pred_path": y_pred_path,
                    **metadata,
                })

    return pd.DataFrame(rows)


##############################################################################
# STAGE 2: EVALUATE EVERY FOLD INDIVIDUALLY
##############################################################################

def evaluate_fold(index_row):
    y_true = to_int_labels(np.load(index_row["y_true_path"]))
    raw_pred = np.load(index_row["y_pred_path"])

    y_prob = maybe_probabilities(raw_pred)
    if y_prob is None:
        y_prob = find_probability_array(index_row["testing_folder_path"], int(index_row["fold"]))

    if y_prob is not None:
        y_prob = normalise_probabilities(y_prob)

    y_pred = np.argmax(y_prob, axis=1) if y_prob is not None else to_int_labels(raw_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch for {index_row['testing_folder_name']} fold {index_row['fold']}: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    class_precision = precision_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(len(CLASSES)),
        zero_division=0,
    )
    class_recall = recall_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(len(CLASSES)),
        zero_division=0,
    )
    class_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(len(CLASSES)),
        zero_division=0,
    )
    class_specificity_stats = compute_per_class_confusion_stats(y_true, y_pred)

    ll, roc_auc = safe_probability_metrics(y_true, y_prob)
    brier = compute_brier_score(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)

    # Sample sizes / class support, so metric reliability can be judged
    # alongside the metric itself (an F1 of 0.9 on 20 samples means
    # something very different than on 2000 samples).
    class_support = np.bincount(y_true, minlength=len(CLASSES))
    model_filename, model_path = resolve_model_path(
        index_row["run_dir"],
        index_row["training_folder_name"],
        int(index_row["fold"]),
    )

    result = {
        "experiment_number": int(index_row["experiment_number"]),
        "base_dataset": index_row["base_dataset"],
        "target_dataset": index_row["target_dataset"],
        "test_dataset": index_row["test_dataset"],
        "fold": int(index_row["fold"]),
        "strategy_id": index_row["strategy_id"],
        "strategy_label": index_row["strategy_label"],
        "training_category": index_row["training_category"],
        "transfer_learning": bool(index_row["transfer_learning"]),
        "learning_rate": index_row["learning_rate"],
        "base_regularisation": index_row["base_regularisation"],
        "transfer_regularisation": index_row["transfer_regularisation"],
        "ceiling_epochs": int(index_row["ceiling_epochs"]),
        "training_folder_name": index_row["training_folder_name"],
        "testing_folder_name": index_row["testing_folder_name"],
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "specificity": compute_specificity(y_true, y_pred),
        "log_loss": ll,
        "roc_auc_macro_ovr": roc_auc,
        "brier_score": brier,
        "ece": ece,
        "precision_FRI": class_precision[0],
        "recall_FRI": class_recall[0],
        "f1_FRI": class_f1[0],
        "precision_FRII": class_precision[1],
        "recall_FRII": class_recall[1],
        "f1_FRII": class_f1[1],
        "precision_COMPACT": class_precision[2],
        "recall_COMPACT": class_recall[2],
        "f1_COMPACT": class_f1[2],
        "has_probabilities": y_prob is not None,
        "model_filename": model_filename,
        "model_path": model_path,
        "n_samples": int(len(y_true)),
        "support_FRI": int(class_support[0]),
        "support_FRII": int(class_support[1]),
        "support_COMPACT": int(class_support[2]),
    }
    result.update(class_specificity_stats)
    result["composite_score"] = compute_composite_score(result)

    curve_payload = {
        "metadata": {
            "experiment_number": int(index_row["experiment_number"]),
            "strategy_id": index_row["strategy_id"],
            "strategy_label": index_row["strategy_label"],
            "test_dataset": index_row["test_dataset"],
            "fold": int(index_row["fold"]),
            "base_dataset": index_row["base_dataset"],
            "target_dataset": index_row["target_dataset"],
        },
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES))),
    }

    return result, curve_payload


def evaluate_all_folds(index_df):
    fold_rows = []
    curve_rows = []
    failures = []

    for _, row in index_df.iterrows():
        try:
            fold_result, curve_payload = evaluate_fold(row)
            fold_rows.append(fold_result)
            curve_rows.append(curve_payload)
        except Exception as exc:
            failures.append(f"{row['testing_folder_name']} fold {row['fold']}: {exc}")

    return pd.DataFrame(fold_rows), curve_rows, failures


##############################################################################
# STAGE 3: SUMMARISE EACH MODEL BY AVERAGING ACROSS FOLDS
##############################################################################

def summarise_training_strategies(fold_df):
    group_cols = [
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "training_category",
        "transfer_learning",
    ]

    summary = fold_df.groupby(group_cols)[FOLD_METRICS].agg(["mean", "std", "median", "min", "max"])
    summary.columns = ["_".join(c).strip() for c in summary.columns]
    summary = summary.reset_index()

    run_level = (
        fold_df
        .groupby(group_cols + ["experiment_number"])[FOLD_METRICS]
        .mean()
        .reset_index()
    )

    ci_df = run_level.groupby(group_cols)[FOLD_METRICS].agg(ci95).reset_index()
    ci_df = ci_df.rename(columns={metric: f"{metric}_ci95" for metric in FOLD_METRICS})

    summary = summary.merge(ci_df, on=group_cols, how="left")
    return summary, run_level


def compute_experiment_score_deviations(fold_df):
    """Compare every fold metric between experiments for like-for-like models.

    Each row represents one strategy, test dataset, and metric. The overall
    fold statistics show variation across every observed fold, while the
    experiment-level range is the largest difference between experiment means.
    """
    group_cols = [
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "test_dataset",
        "training_category",
    ]

    long_scores = fold_df.melt(
        id_vars=group_cols + ["experiment_number", "fold"],
        value_vars=FOLD_METRICS,
        var_name="metric",
        value_name="score",
    ).dropna(subset=["score"])
    if long_scores.empty:
        return pd.DataFrame()

    comparison_cols = group_cols + ["metric"]
    overall = (
        long_scores.groupby(comparison_cols)["score"]
        .agg(
            total_folds="count",
            overall_fold_mean="mean",
            overall_fold_std="std",
            overall_fold_min="min",
            overall_fold_max="max",
        )
        .reset_index()
    )
    overall["overall_fold_range"] = (
        overall["overall_fold_max"] - overall["overall_fold_min"]
    )

    per_experiment = (
        long_scores.groupby(comparison_cols + ["experiment_number"])["score"]
        .agg(experiment_fold_count="count", experiment_fold_mean="mean", experiment_fold_std="std")
        .reset_index()
    )
    experiment_stats = (
        per_experiment.groupby(comparison_cols)["experiment_fold_mean"]
        .agg(
            experiments_observed="count",
            experiment_mean_std="std",
            experiment_mean_min="min",
            experiment_mean_max="max",
        )
        .reset_index()
    )
    experiment_stats["experiment_mean_range"] = (
        experiment_stats["experiment_mean_max"] - experiment_stats["experiment_mean_min"]
    )

    highest = per_experiment.loc[
        per_experiment.groupby(comparison_cols)["experiment_fold_mean"].idxmax(),
        comparison_cols + ["experiment_number"],
    ].rename(columns={"experiment_number": "highest_mean_experiment"})
    lowest = per_experiment.loc[
        per_experiment.groupby(comparison_cols)["experiment_fold_mean"].idxmin(),
        comparison_cols + ["experiment_number"],
    ].rename(columns={"experiment_number": "lowest_mean_experiment"})

    per_experiment_wide = per_experiment.pivot(
        index=comparison_cols,
        columns="experiment_number",
        values=["experiment_fold_count", "experiment_fold_mean", "experiment_fold_std"],
    )
    per_experiment_wide.columns = [
        f"experiment_{experiment_number}_{stat.removeprefix('experiment_')}"
        for stat, experiment_number in per_experiment_wide.columns
    ]
    per_experiment_wide = per_experiment_wide.reset_index()

    return (
        overall.merge(experiment_stats, on=comparison_cols, how="left")
        .merge(highest, on=comparison_cols, how="left")
        .merge(lowest, on=comparison_cols, how="left")
        .merge(per_experiment_wide, on=comparison_cols, how="left")
        .sort_values(["metric", "experiment_mean_range"], ascending=[True, False])
        .reset_index(drop=True)
    )


def rank_by_generalisation(generalisation_summary):
    """Rank strategies by how well they generalise, as distinct from raw
    blended performance. Primary sort key is out-of-domain composite score
    (performance on data the model was NOT trained/targeted on); ties are
    broken by worst-case dataset performance, since a strategy that does
    only moderately well everywhere is a better generalist than one that is
    excellent on two datasets and poor on the third."""
    if generalisation_summary.empty:
        return generalisation_summary.copy()

    ranked = generalisation_summary.copy()
    ranked = ranked.sort_values(
        by=["out_domain_composite_mean", "worst_case_dataset_composite_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "generalisation_rank", np.arange(1, len(ranked) + 1))

    keep_cols = [
        "generalisation_rank",
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "training_category",
        "out_domain_composite_mean",
        "out_domain_accuracy_mean",
        "generalisation_gap_mean",
        "retention_mean",
        "retention_low_reliability_runs",
        "worst_case_dataset_accuracy",
        "worst_case_dataset_composite_score",
        "cross_dataset_accuracy_cv",
        "best_performing_dataset",
        "worst_performing_dataset",
    ]
    return ranked[[c for c in keep_cols if c in ranked.columns]]


def build_transfer_matrix(fold_df, metric="composite_score"):
    """Full strategy x test-dataset matrix (rather than the coarser
    target-vs-rest split used elsewhere), so every train->test combination
    is visible directly, e.g. trained-on-A tested-on-B, trained-on-A
    tested-on-C, etc."""
    pivot = (
        fold_df.groupby(["strategy_label", "test_dataset"])[metric]
        .mean()
        .reset_index()
        .pivot(index="strategy_label", columns="test_dataset", values=metric)
    )
    n_pivot = (
        fold_df.groupby(["strategy_label", "test_dataset"])["n_samples"]
        .sum()
        .reset_index()
        .pivot(index="strategy_label", columns="test_dataset", values="n_samples")
    )
    return pivot, n_pivot


def build_sample_size_summary(fold_df):
    """Sample sizes and per-class support for every strategy/test-dataset
    combination, so metric reliability can be judged alongside the metric
    itself rather than assumed."""
    group_cols = ["strategy_id", "strategy_label", "test_dataset"]
    support_cols = [f"support_{c}" for c in CLASSES]
    summary = (
        fold_df.groupby(group_cols)[["n_samples"] + support_cols]
        .sum()
        .reset_index()
    )
    n_folds = fold_df.groupby(group_cols).size().reset_index(name="n_folds_observed")
    summary = summary.merge(n_folds, on=group_cols, how="left")
    return summary


##############################################################################
# WEIGHTED PER-DATASET GENERALISATION SCORE
##############################################################################

def resolve_dataset_weights(observed_datasets, weights_dict):
    """Turn a (possibly partial, possibly None) weights dict into weights
    for exactly the datasets observed for a given strategy, renormalised to
    sum to 1. Missing entries share the remaining weight equally."""
    if not observed_datasets:
        return {}

    if weights_dict is None:
        equal = 1.0 / len(observed_datasets)
        return {d: equal for d in observed_datasets}

    resolved = {}
    unspecified = []
    for d in observed_datasets:
        if d in weights_dict:
            resolved[d] = weights_dict[d]
        else:
            unspecified.append(d)

    specified_total = sum(resolved.values())
    if unspecified:
        remaining = max(0.0, 1.0 - specified_total)
        share = remaining / len(unspecified)
        for d in unspecified:
            resolved[d] = share

    total = sum(resolved.values())
    if total <= 0:
        equal = 1.0 / len(observed_datasets)
        return {d: equal for d in observed_datasets}
    return {d: v / total for d, v in resolved.items()}


def compute_weighted_generalisation_score(
    fold_df,
    weights_dict,
    exclude_target,
    metric,
    scheme_label,
):
    """Weighted average of a per-dataset metric, one row per strategy. This
    is the 'assign each dataset a composite score, then weight the datasets'
    scoring approach: average `metric` within each test dataset, then
    combine those per-dataset averages using `weights_dict`."""
    strategy_cols = [
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "training_category",
    ]

    records = []
    for keys, grp in fold_df.groupby(strategy_cols):
        key_map = dict(zip(strategy_cols, keys))
        target = key_map["target_dataset"]

        working = grp[grp["test_dataset"] != target].copy() if exclude_target else grp.copy()
        if working.empty:
            continue

        per_dataset_mean = working.groupby("test_dataset")[metric].mean()
        observed = per_dataset_mean.index.tolist()
        weights = resolve_dataset_weights(observed, weights_dict)
        weighted_score = sum(per_dataset_mean[d] * weights[d] for d in observed)

        record = {
            **key_map,
            "weighting_scheme": scheme_label,
            "target_excluded": exclude_target,
            "weighted_generalisation_score": weighted_score,
            "n_datasets_scored": len(observed),
        }
        for d in observed:
            record[f"weight_{d}"] = weights[d]
            record[f"{metric}_{d}"] = per_dataset_mean[d]
        records.append(record)

    return pd.DataFrame(records)


def compute_final_model_scores(fold_df):
    """Calculate the required three-test-set weighted score for each model.

    A model is one strategy in one experiment. Its score for each test set is
    the mean of exactly five fold composite scores. The final score is only
    reported when FIRST, LOFAR, and NVSS each have all five folds available.
    """
    model_cols = [
        "experiment_number",
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "training_category",
    ]
    component_cols = ["macro_f1", "balanced_accuracy", "mcc", "f1_COMPACT"]
    required_cols = model_cols + ["test_dataset", "fold"] + component_cols
    calculation_steps = fold_df[required_cols].copy()
    calculation_steps["macro_f1_weight"] = COMPOSITE_WEIGHTS["macro_f1"]
    calculation_steps["balanced_accuracy_weight"] = COMPOSITE_WEIGHTS["balanced_accuracy"]
    calculation_steps["mcc_weight"] = COMPOSITE_WEIGHTS["mcc"]
    calculation_steps["compact_f1_weight"] = COMPOSITE_WEIGHTS["f1_COMPACT"]
    calculation_steps["macro_f1_contribution"] = (
        calculation_steps["macro_f1"] * calculation_steps["macro_f1_weight"]
    )
    calculation_steps["balanced_accuracy_contribution"] = (
        calculation_steps["balanced_accuracy"] * calculation_steps["balanced_accuracy_weight"]
    )
    calculation_steps["mcc_contribution"] = (
        calculation_steps["mcc"] * calculation_steps["mcc_weight"]
    )
    calculation_steps["compact_f1_contribution"] = (
        calculation_steps["f1_COMPACT"] * calculation_steps["compact_f1_weight"]
    )
    calculation_steps["fold_composite_score"] = calculation_steps[
        [
            "macro_f1_contribution",
            "balanced_accuracy_contribution",
            "mcc_contribution",
            "compact_f1_contribution",
        ]
    ].sum(axis=1)

    per_test_set = (
        calculation_steps.groupby(model_cols + ["test_dataset"])["fold_composite_score"]
        .agg(fold_score_sum="sum", folds_observed="count", test_set_score="mean")
        .reset_index()
    )

    final_records = []
    for keys, model_scores in per_test_set.groupby(model_cols):
        record = dict(zip(model_cols, keys))
        complete = True
        final_score = 0.0

        for dataset, weight in FINAL_MODEL_DATASET_WEIGHTS.items():
            dataset_scores = model_scores[model_scores["test_dataset"] == dataset]
            record[f"{dataset}_weight"] = weight
            if len(dataset_scores) != 1:
                record[f"{dataset}_fold_score_sum"] = np.nan
                record[f"{dataset}_folds_observed"] = 0
                record[f"{dataset}_test_set_score"] = np.nan
                record[f"{dataset}_weighted_contribution"] = np.nan
                complete = False
                continue

            dataset_score = dataset_scores.iloc[0]
            test_set_score = dataset_score["test_set_score"]
            folds_observed = int(dataset_score["folds_observed"])
            contribution = test_set_score * weight
            record[f"{dataset}_fold_score_sum"] = dataset_score["fold_score_sum"]
            record[f"{dataset}_folds_observed"] = folds_observed
            record[f"{dataset}_test_set_score"] = test_set_score
            record[f"{dataset}_weighted_contribution"] = contribution
            final_score += contribution
            complete = complete and folds_observed == 5

        record["has_all_test_sets_with_five_folds"] = complete
        record["final_model_score"] = final_score if complete else np.nan
        final_records.append(record)

    return calculation_steps, pd.DataFrame(final_records)


def compute_final_model_score_deviations(final_model_scores):
    """Compare each strategy's completed final model scores across experiments."""
    score_cols = [
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "training_category",
    ]
    complete_scores = final_model_scores[
        final_model_scores["has_all_test_sets_with_five_folds"]
    ].copy()
    if complete_scores.empty:
        return pd.DataFrame()

    summary = (
        complete_scores.groupby(score_cols)["final_model_score"]
        .agg(
            experiments_observed="count",
            final_model_score_mean="mean",
            final_model_score_std="std",
            final_model_score_min="min",
            final_model_score_max="max",
        )
        .reset_index()
    )
    summary["final_model_score_deviation_range"] = (
        summary["final_model_score_max"] - summary["final_model_score_min"]
    )

    per_experiment = complete_scores.pivot(
        index=score_cols,
        columns="experiment_number",
        values="final_model_score",
    )
    per_experiment.columns = [
        f"experiment_{experiment_number}_final_model_score"
        for experiment_number in per_experiment.columns
    ]
    return (
        summary.merge(per_experiment.reset_index(), on=score_cols, how="left")
        .sort_values("final_model_score_deviation_range", ascending=False)
        .reset_index(drop=True)
    )


def rank_final_models(final_model_scores):
    """Rank strategies by the configured final weighted score."""
    if final_model_scores.empty:
        return pd.DataFrame()

    score_cols = [
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "training_category",
    ]
    complete_scores = final_model_scores[
        final_model_scores["has_all_test_sets_with_five_folds"]
    ].dropna(subset=["final_model_score"])
    if complete_scores.empty:
        return pd.DataFrame()

    ranked = (
        complete_scores.groupby(score_cols)["final_model_score"]
        .mean()
        .reset_index()
        .sort_values("final_model_score", ascending=False)
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def compute_all_experiment_metric_averages(fold_df):
    """Average every metric by folds within experiments, then across experiments.

    The second average gives every experiment equal weight, regardless of how
    many fold rows it contributes. Existing score columns and rankings are not
    modified; these are reporting-only calculations.
    """
    model_cols = [
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "training_category",
    ]
    long_metrics = fold_df.melt(
        id_vars=model_cols + ["experiment_number", "test_dataset", "fold"],
        value_vars=FOLD_METRICS,
        var_name="metric",
        value_name="fold_metric_value",
    ).dropna(subset=["fold_metric_value"])
    if long_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()

    experiment_group_cols = model_cols + ["experiment_number", "test_dataset", "metric"]
    per_experiment = (
        long_metrics.groupby(experiment_group_cols)["fold_metric_value"]
        .agg(
            fold_metric_sum="sum",
            folds_observed="count",
            experiment_metric_mean="mean",
            experiment_metric_std="std",
            fold_metric_min="min",
            fold_metric_max="max",
        )
        .reset_index()
    )
    per_experiment["fold_metric_range"] = (
        per_experiment["fold_metric_max"] - per_experiment["fold_metric_min"]
    )

    summary_group_cols = model_cols + ["test_dataset", "metric"]
    summary = (
        per_experiment.groupby(summary_group_cols)["experiment_metric_mean"]
        .agg(
            experiments_observed="count",
            experiment_mean_sum="sum",
            average_across_experiments="mean",
            experiment_mean_std="std",
            experiment_mean_min="min",
            experiment_mean_max="max",
        )
        .reset_index()
    )
    summary["experiment_mean_range"] = (
        summary["experiment_mean_max"] - summary["experiment_mean_min"]
    )

    per_experiment_wide = per_experiment.pivot(
        index=summary_group_cols,
        columns="experiment_number",
        values=[
            "fold_metric_sum",
            "folds_observed",
            "experiment_metric_mean",
            "experiment_metric_std",
        ],
    )
    per_experiment_wide.columns = [
        f"experiment_{experiment_number}_{stat}"
        for stat, experiment_number in per_experiment_wide.columns
    ]
    summary = summary.merge(per_experiment_wide.reset_index(), on=summary_group_cols, how="left")
    summary = summary.sort_values(
        ["metric", "strategy_id", "test_dataset"]
    ).reset_index(drop=True)
    return per_experiment, summary


def append_all_experiment_average_rows(fold_df, all_experiment_metric_averages):
    """Append one all-experiment average row per strategy and test dataset."""
    if all_experiment_metric_averages.empty:
        return fold_df.copy()

    model_cols = [
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "training_category",
    ]
    group_cols = model_cols + ["test_dataset"]
    average_rows = all_experiment_metric_averages.pivot(
        index=group_cols,
        columns="metric",
        values="average_across_experiments",
    ).reset_index()
    average_rows["experiment_number"] = "ALL_EXPERIMENTS"
    average_rows["fold"] = "AVERAGE"

    for column in fold_df.columns:
        if column not in average_rows:
            average_rows[column] = np.nan
    return pd.concat(
        [fold_df, average_rows[fold_df.columns]],
        ignore_index=True,
    )


##############################################################################
# CROSS-DATASET PER-OBJECT CONSISTENCY (same sources, different resolution)
##############################################################################

def _group_curve_rows_by_model(curve_rows):
    """Groups curve rows by (strategy, experiment, fold) -- i.e. by a single
    trained model instance -- with one entry per test dataset it was
    evaluated on. Since the three test sets are the same sources at
    different resolution/frequency, row order within a group is the
    per-source pairing key."""
    groups = defaultdict(dict)
    labels = {}
    for item in curve_rows:
        md = item["metadata"]
        key = (md["strategy_id"], md["experiment_number"], md["fold"])
        groups[key][md["test_dataset"]] = (item["y_true"], item["y_pred"])
        labels[md["strategy_id"]] = md["strategy_label"]
    return groups, labels


def compute_cross_dataset_consistency(curve_rows):
    """For each trained model instance, checks whether the SAME source gets
    classified consistently as resolution/frequency changes. This catches a
    failure mode that dataset-level averages miss: a model that scores well
    on every dataset in aggregate but flips its prediction on different
    individual sources at each resolution isn't really generalising, it's
    averaging out different errors."""
    groups, labels = _group_curve_rows_by_model(curve_rows)

    fold_rows = []
    for (strategy_id, exp, fold), datasets in groups.items():
        dataset_names = sorted(datasets.keys())
        if len(dataset_names) < 2:
            continue

        lengths = {d: len(datasets[d][0]) for d in dataset_names}
        if len(set(lengths.values())) != 1:
            # Row counts don't match across datasets for this model instance,
            # so index-based pairing isn't safe here -- skip rather than
            # silently mis-pairing sources.
            continue
        n = lengths[dataset_names[0]]

        true_matrix = np.stack([datasets[d][0] for d in dataset_names], axis=1)
        pred_matrix = np.stack([datasets[d][1] for d in dataset_names], axis=1)

        # Sanity check: since these are the same sources, the true label
        # should be identical across datasets for a given row. If it isn't,
        # the alignment assumption is violated for that source.
        true_consistent = np.all(true_matrix == true_matrix[:, [0]], axis=1)

        prediction_stable = np.all(pred_matrix == pred_matrix[:, [0]], axis=1)
        correct_matrix = (pred_matrix == true_matrix)
        robust_correct = np.all(correct_matrix, axis=1)

        fold_rows.append({
            "strategy_id": strategy_id,
            "strategy_label": labels[strategy_id],
            "experiment_number": exp,
            "fold": fold,
            "datasets_compared": ",".join(dataset_names),
            "n_datasets_compared": len(dataset_names),
            "n_objects": n,
            "n_label_alignment_mismatches": int(n - np.sum(true_consistent)),
            "prediction_stability_rate": float(np.mean(prediction_stable)),
            "robust_correct_rate": float(np.mean(robust_correct)),
        })

    fold_level = pd.DataFrame(fold_rows)
    if fold_level.empty:
        return pd.DataFrame(), fold_level

    strategy_level = (
        fold_level.groupby(["strategy_id", "strategy_label"])
        .agg(
            prediction_stability_rate_mean=("prediction_stability_rate", "mean"),
            prediction_stability_rate_std=("prediction_stability_rate", "std"),
            robust_correct_rate_mean=("robust_correct_rate", "mean"),
            robust_correct_rate_std=("robust_correct_rate", "std"),
            n_model_instances_compared=("prediction_stability_rate", "count"),
            total_label_alignment_mismatches=("n_label_alignment_mismatches", "sum"),
        )
        .reset_index()
    )
    return strategy_level, fold_level


def compute_cross_dataset_mcnemar(curve_rows):
    """Paired McNemar's test, per strategy, between each pair of test
    datasets: for the same sources, is the model significantly more likely
    to get it right on dataset A and wrong on B than vice versa? This has
    far more statistical power than an experiment-level test since every
    source is a paired observation, not just every experiment. Counts are
    pooled across all folds/experiments for a strategy (repeated
    measurements of the same sources across trained model instances), so
    treat this as a diagnostic rather than a fully independent-samples test."""
    groups, labels = _group_curve_rows_by_model(curve_rows)

    pair_counts = defaultdict(lambda: {"b": 0, "c": 0, "n_groups": 0})
    for (strategy_id, exp, fold), datasets in groups.items():
        dataset_names = sorted(datasets.keys())
        for d_a, d_b in itertools.combinations(dataset_names, 2):
            yt_a, yp_a = datasets[d_a]
            yt_b, yp_b = datasets[d_b]
            if len(yt_a) != len(yt_b):
                continue

            correct_a = (yp_a == yt_a)
            correct_b = (yp_b == yt_b)
            b = int(np.sum(correct_a & ~correct_b))
            c = int(np.sum(~correct_a & correct_b))

            key = (strategy_id, d_a, d_b)
            pair_counts[key]["b"] += b
            pair_counts[key]["c"] += c
            pair_counts[key]["n_groups"] += 1

    rows = []
    for (strategy_id, d_a, d_b), counts in pair_counts.items():
        b, c = counts["b"], counts["c"]
        n = b + c
        if n == 0:
            p_value = 1.0
        else:
            k = min(b, c)
            p_value = scipy_stats.binomtest(k, n, p=0.5, alternative="two-sided").pvalue

        if b > c:
            favours = d_a
        elif c > b:
            favours = d_b
        else:
            favours = "no difference"

        rows.append({
            "strategy_id": strategy_id,
            "strategy_label": labels[strategy_id],
            "dataset_a": d_a,
            "dataset_b": d_b,
            "n_paired_model_instances": counts["n_groups"],
            "discordant_a_correct_b_wrong": b,
            "discordant_a_wrong_b_correct": c,
            "mcnemar_p_raw": p_value,
            "favours": favours,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["mcnemar_p_holm_adjusted"] = holm_bonferroni_adjust(result["mcnemar_p_raw"].to_numpy())
    result["significant_at_0.05"] = result["mcnemar_p_holm_adjusted"] < 0.05
    return result


def compute_pairwise_significance(
    fold_df,
    leaderboard,
    top_n=SIGNIFICANCE_TOP_N,
    metric="composite_score",
):
    """Paired significance testing between the top-N ranked strategies, so
    'model A is best' claims can be backed by a statistical test rather than
    a raw mean comparison. Pairing is done at the experiment level (each
    experiment run treated as one paired observation), using a Wilcoxon
    signed-rank test plus a bootstrap confidence interval on the mean
    difference. P-values are corrected for multiple comparisons using the
    Holm-Bonferroni method."""
    if leaderboard.empty:
        return pd.DataFrame()

    top_strategies = leaderboard.head(top_n)["strategy_id"].tolist()
    if len(top_strategies) < 2:
        return pd.DataFrame()

    run_level = (
        fold_df.groupby(["strategy_id", "strategy_label", "experiment_number"])[metric]
        .mean()
        .reset_index()
    )

    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    rows = []

    for strat_a, strat_b in itertools.combinations(top_strategies, 2):
        series_a = run_level.loc[run_level["strategy_id"] == strat_a].set_index("experiment_number")[metric]
        series_b = run_level.loc[run_level["strategy_id"] == strat_b].set_index("experiment_number")[metric]

        paired = pd.concat([series_a, series_b], axis=1, keys=["a", "b"]).dropna()
        if len(paired) < 2:
            continue

        diffs = (paired["a"] - paired["b"]).to_numpy()
        mean_diff = float(np.mean(diffs))

        try:
            if np.allclose(diffs, 0):
                wilcoxon_stat, wilcoxon_p = np.nan, 1.0
            else:
                wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(diffs)
        except ValueError:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan

        # Bootstrap CI on the mean paired difference.
        boot_means = np.empty(BOOTSTRAP_ITERATIONS)
        n = len(diffs)
        for i in range(BOOTSTRAP_ITERATIONS):
            sample = rng.choice(diffs, size=n, replace=True)
            boot_means[i] = sample.mean()
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        label_a = run_level.loc[run_level["strategy_id"] == strat_a, "strategy_label"].iloc[0]
        label_b = run_level.loc[run_level["strategy_id"] == strat_b, "strategy_label"].iloc[0]

        rows.append({
            "strategy_a": label_a,
            "strategy_b": label_b,
            "metric": metric,
            "n_paired_experiments": n,
            "mean_difference_a_minus_b": mean_diff,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p_raw": wilcoxon_p,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Holm-Bonferroni correction across all pairwise tests in this table.
    result["wilcoxon_p_holm_adjusted"] = holm_bonferroni_adjust(result["wilcoxon_p_raw"].to_numpy())
    result["significant_at_0.05"] = result["wilcoxon_p_holm_adjusted"] < 0.05
    return result


##############################################################################
# STAGE 5: SELECT THE HARDWARE CANDIDATE FROM THE TOP FINAL MODEL
##############################################################################

def rank_folds_for_best_final_model(fold_df, final_model_scores):
    completed_models = final_model_scores[
        final_model_scores["has_all_test_sets_with_five_folds"]
    ].dropna(subset=["final_model_score"])
    if completed_models.empty:
        raise ValueError("No model has all three required test sets with five folds.")

    top_model = completed_models.sort_values("final_model_score", ascending=False).iloc[0]
    subset = fold_df[
        (fold_df["strategy_id"] == top_model["strategy_id"])
        & (fold_df["experiment_number"] == top_model["experiment_number"])
    ].copy()

    fold_group_cols = [
        "experiment_number",
        "fold",
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "training_folder_name",
        "model_filename",
        "model_path",
    ]

    fold_rank = subset.groupby(fold_group_cols)[FOLD_METRICS].mean().reset_index()
    per_dataset_fold_scores = (
        subset.groupby(fold_group_cols + ["test_dataset"])["composite_score"]
        .mean()
        .unstack("test_dataset")
    )
    for dataset, weight in FINAL_MODEL_DATASET_WEIGHTS.items():
        fold_rank = fold_rank.merge(
            per_dataset_fold_scores[[dataset]].rename(
                columns={dataset: f"{dataset}_fold_composite_score"}
            ).reset_index(),
            on=fold_group_cols,
            how="left",
        )
        fold_rank[f"{dataset}_weighted_contribution"] = (
            fold_rank[f"{dataset}_fold_composite_score"] * weight
        )

    required_score_cols = [
        f"{dataset}_fold_composite_score" for dataset in FINAL_MODEL_DATASET_WEIGHTS
    ]
    fold_rank = fold_rank.dropna(subset=required_score_cols)
    fold_rank["final_weighted_fold_score"] = sum(
        fold_rank[f"{dataset}_weighted_contribution"]
        for dataset in FINAL_MODEL_DATASET_WEIGHTS
    )
    fold_rank["final_model_score"] = top_model["final_model_score"]
    fold_rank = fold_rank.sort_values("final_weighted_fold_score", ascending=False).reset_index(drop=True)
    fold_rank.insert(0, "rank", np.arange(1, len(fold_rank) + 1))

    candidate = fold_rank.iloc[0].to_dict()
    candidate["selection_basis"] = "highest five-fold final weighted model score"
    candidate_per_dataset = subset[
        (subset["experiment_number"] == candidate["experiment_number"])
        & (subset["fold"] == candidate["fold"])
    ].copy()

    return fold_rank, candidate, candidate_per_dataset


##############################################################################
# STAGE 6: GENERATE ALL VISUALISATIONS
##############################################################################

def plot_matrix(cm, title, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_confusion_matrix_outputs(curve_rows, output_dirs, strategy_order, dataset_order, hardware_candidate):
    grouped = defaultdict(list)
    for item in curve_rows:
        md = item["metadata"]
        key = (md["strategy_id"], md["strategy_label"], md["test_dataset"])
        grouped[key].append(item["confusion_matrix"])

    for strategy_id in strategy_order:
        for dataset in dataset_order:
            matching_keys = [k for k in grouped if k[0] == strategy_id and k[2] == dataset]
            if not matching_keys:
                continue
            key = matching_keys[0]
            avg_cm = np.mean(np.stack(grouped[key], axis=0), axis=0)
            safe_label = re.sub(r"[^A-Za-z0-9_\-]", "_", key[1])
            plot_matrix(
                avg_cm,
                f"Average confusion matrix\n{key[1]} | Test: {dataset}",
                os.path.join(output_dirs["confusion_matrices"], f"avg_cm_{safe_label}_{dataset}.png"),
            )

    for item in curve_rows:
        md = item["metadata"]
        if (
            md["strategy_id"] == hardware_candidate["strategy_id"]
            and md["experiment_number"] == hardware_candidate["experiment_number"]
            and md["fold"] == hardware_candidate["fold"]
        ):
            plot_matrix(
                item["confusion_matrix"],
                (
                    "Selected hardware candidate confusion matrix\n"
                    f"Exp {md['experiment_number']} Fold {md['fold']} | Test: {md['test_dataset']}"
                ),
                os.path.join(
                    output_dirs["best_fold_confusion_matrices"],
                    f"hardware_candidate_cm_exp{md['experiment_number']}_fold{md['fold']}_{md['test_dataset']}.png",
                ),
            )


def generate_heatmaps(fold_df, output_dirs, strategy_order, dataset_order, selected_strategy_id):
    heatmap_metrics = ["composite_score", "macro_f1", "balanced_accuracy", "mcc"]

    for metric in heatmap_metrics:
        pivot = (
            fold_df.groupby(["strategy_id", "strategy_label", "test_dataset"])[metric]
            .mean()
            .reset_index()
            .pivot(index="strategy_label", columns="test_dataset", values=metric)
        )

        ordered_labels = [
            fold_df.loc[fold_df["strategy_id"] == sid, "strategy_label"].iloc[0]
            for sid in strategy_order
            if sid in fold_df["strategy_id"].values
        ]
        pivot = pivot.reindex(index=ordered_labels, columns=dataset_order)

        fig_height = max(5, 0.45 * len(pivot.index))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": metric},
            ax=ax,
        )

        selected_rows = [
            i for i, lbl in enumerate(pivot.index)
            if lbl == fold_df.loc[fold_df["strategy_id"] == selected_strategy_id, "strategy_label"].iloc[0]
        ]
        if selected_rows:
            y = selected_rows[0]
            ax.add_patch(
                Rectangle((0, y), len(pivot.columns), 1, fill=False, edgecolor="red", linewidth=3)
            )

        ax.set_title(f"Model ranking heatmap: {metric}")
        ax.set_xlabel("Test dataset")
        ax.set_ylabel("Training strategy")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dirs["heatmaps"], f"heatmap_{metric}.png"), dpi=300)
        plt.close(fig)


def generate_model_ranking_plot(leaderboard, output_dirs, selected_strategy_id):
    fig_height = max(5, len(leaderboard) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    colors = ["#1f77b4" for _ in range(len(leaderboard))]
    selected_idx = leaderboard.index[leaderboard["strategy_id"] == selected_strategy_id]
    if len(selected_idx) > 0:
        colors[int(selected_idx[0])] = "#d62728"

    ax.barh(leaderboard["strategy_label"], leaderboard["final_model_score"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Final weighted score")
    ax.set_ylabel("Training strategy")
    ax.set_title("Final weighted training strategy ranking (higher is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dirs["ranking_plots"], "training_strategy_ranking.png"), dpi=300)
    plt.close(fig)


def generate_fold_ranking_plot(fold_rank, output_dirs):
    display_labels = [f"Exp{int(r.experiment_number)}-Fold{int(r.fold)}" for r in fold_rank.itertuples()]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4" for _ in range(len(fold_rank))]
    if colors:
        colors[0] = "#d62728"

    score_col = (
        "final_weighted_fold_score"
        if "final_weighted_fold_score" in fold_rank.columns
        else "composite_score"
    )
    ax.bar(display_labels, fold_rank[score_col], color=colors)
    ax.set_xlabel("Fold instance")
    ax.set_ylabel("Weighted testing score")
    ax.set_title("Fold ranking for selected final model")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dirs["ranking_plots"], "selected_strategy_fold_ranking.png"), dpi=300)
    plt.close(fig)


def generate_roc_pr_outputs(curve_rows, output_dirs, hardware_candidate, selected_strategy_id):
    rows_with_prob = [r for r in curve_rows if r["y_prob"] is not None]
    if not rows_with_prob:
        note = (
            "No probability outputs were found. "
            "ROC, PR, log-loss and probability-based plots are generated only when per-sample probabilities are available."
        )
        for key in ["roc_curves", "precision_recall_curves"]:
            with open(os.path.join(output_dirs[key], "README.txt"), "w", encoding="utf-8") as f:
                f.write(note)
        return

    strategy_prob_rows = [r for r in rows_with_prob if r["metadata"]["strategy_id"] == selected_strategy_id]
    if not strategy_prob_rows:
        return

    mean_fpr = np.linspace(0.0, 1.0, 200)
    mean_tprs = []

    for row in strategy_prob_rows:
        md = row["metadata"]
        y_true = row["y_true"]
        y_prob = row["y_prob"]
        y_true_bin = label_binarize(y_true, classes=np.arange(len(CLASSES)))

        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

        per_class_tpr = []

        for i, class_name in enumerate(CLASSES):
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc_i = auc(fpr_i, tpr_i)
            ax_roc.plot(fpr_i, tpr_i, label=f"{class_name} (AUC={roc_auc_i:.3f})")

            precision_i, recall_i, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            ap_i = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            ax_pr.plot(recall_i, precision_i, label=f"{class_name} (AP={ap_i:.3f})")

            per_class_tpr.append(np.interp(mean_fpr, fpr_i, tpr_i))

        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        mean_tpr = np.mean(np.vstack(per_class_tpr), axis=0)
        mean_tprs.append(mean_tpr)

        ax_roc.plot(mean_fpr, mean_tpr, linestyle="--", linewidth=2, color="black", label=f"Macro mean (AUC={macro_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], linestyle=":", color="grey")
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(
            f"ROC | Exp {md['experiment_number']} Fold {md['fold']} | Test {md['test_dataset']}"
        )
        ax_roc.legend(loc="lower right", fontsize=8)
        fig_roc.tight_layout()
        fig_roc.savefig(
            os.path.join(
                output_dirs["roc_curves"],
                f"roc_exp{md['experiment_number']}_fold{md['fold']}_{md['test_dataset']}.png",
            ),
            dpi=300,
        )
        plt.close(fig_roc)

        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(
            f"Precision-Recall | Exp {md['experiment_number']} Fold {md['fold']} | Test {md['test_dataset']}"
        )
        ax_pr.legend(loc="lower left", fontsize=8)
        fig_pr.tight_layout()
        fig_pr.savefig(
            os.path.join(
                output_dirs["precision_recall_curves"],
                f"pr_exp{md['experiment_number']}_fold{md['fold']}_{md['test_dataset']}.png",
            ),
            dpi=300,
        )
        plt.close(fig_pr)

    if mean_tprs:
        fig, ax = plt.subplots(figsize=(7, 6))
        mean_tpr = np.mean(np.vstack(mean_tprs), axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color="black", linewidth=2, label=f"Mean ROC (AUC={mean_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Mean ROC curve for selected training strategy")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dirs["roc_curves"], "mean_roc_selected_strategy.png"), dpi=300)
        plt.close(fig)

    for row in strategy_prob_rows:
        md = row["metadata"]
        if (
            md["experiment_number"] == hardware_candidate["experiment_number"]
            and md["fold"] == hardware_candidate["fold"]
        ):
            y_true = row["y_true"]
            y_prob = row["y_prob"]
            y_true_bin = label_binarize(y_true, classes=np.arange(len(CLASSES)))

            fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
            fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

            for i, class_name in enumerate(CLASSES):
                fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                precision_i, recall_i, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                ax_roc.plot(fpr_i, tpr_i, label=class_name)
                ax_pr.plot(recall_i, precision_i, label=class_name)

            ax_roc.plot([0, 1], [0, 1], linestyle=":", color="grey")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Selected hardware model ROC")
            ax_roc.legend(loc="lower right")
            fig_roc.tight_layout()
            fig_roc.savefig(
                os.path.join(output_dirs["roc_curves"], "hardware_candidate_roc.png"), dpi=300
            )
            plt.close(fig_roc)

            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Selected hardware model Precision-Recall")
            ax_pr.legend(loc="lower left")
            fig_pr.tight_layout()
            fig_pr.savefig(
                os.path.join(output_dirs["precision_recall_curves"], "hardware_candidate_pr.png"), dpi=300
            )
            plt.close(fig_pr)
            break


##############################################################################
# STAGE 7: EXPORT ALL RESULTS
##############################################################################

def compute_generalisation_summary(fold_df):
    strategy_cols = [
        "strategy_id",
        "strategy_label",
        "base_dataset",
        "target_dataset",
        "training_category",
    ]

    class_f1_cols = [f"f1_{c}" for c in CLASSES]
    run_metrics = ["accuracy", "composite_score"] + class_f1_cols

    run_level = (
        fold_df
        .groupby(strategy_cols + ["experiment_number", "test_dataset"])[run_metrics]
        .mean()
        .reset_index()
    )

    records = []
    for keys, grp in run_level.groupby(strategy_cols):
        key_map = dict(zip(strategy_cols, keys))
        target = key_map["target_dataset"]

        run_stats = []
        for exp, exp_grp in grp.groupby("experiment_number"):
            in_rows = exp_grp.loc[exp_grp["test_dataset"] == target]
            out_rows = exp_grp.loc[exp_grp["test_dataset"] != target]

            if in_rows.empty:
                continue

            in_acc = in_rows["accuracy"].mean()
            out_acc = out_rows["accuracy"].mean() if not out_rows.empty else np.nan
            in_composite = in_rows["composite_score"].mean()
            out_composite = out_rows["composite_score"].mean() if not out_rows.empty else np.nan

            gap = in_acc - out_acc if not pd.isna(out_acc) else np.nan

            # A retention ratio computed against a near-zero in-domain
            # accuracy is numerically unstable and easy to over-interpret,
            # so it is only reported when the in-domain accuracy clears a
            # minimum reliability threshold.
            low_reliability = in_acc < MIN_RELIABLE_IN_DOMAIN_ACCURACY
            retention = (
                (out_acc / in_acc)
                if (not pd.isna(out_acc) and not low_reliability)
                else np.nan
            )

            record = {
                "experiment_number": exp,
                "in_domain_accuracy": in_acc,
                "out_domain_accuracy": out_acc,
                "generalisation_gap": gap,
                "retention": retention,
                "low_in_domain_reliability": low_reliability,
                "in_domain_composite": in_composite,
                "out_domain_composite": out_composite,
            }

            # Per-class generalisation: does this strategy's transfer hold up
            # uniformly across FRI/FRII/COMPACT, or does one class (typically
            # the minority COMPACT class) quietly collapse out-of-domain?
            for cls, col in zip(CLASSES, class_f1_cols):
                in_val = in_rows[col].mean()
                out_val = out_rows[col].mean() if not out_rows.empty else np.nan
                record[f"in_domain_f1_{cls}"] = in_val
                record[f"out_domain_f1_{cls}"] = out_val
                record[f"gap_f1_{cls}"] = (
                    in_val - out_val if not pd.isna(out_val) else np.nan
                )

            run_stats.append(record)

        if not run_stats:
            continue

        run_df = pd.DataFrame(run_stats)
        dataset_means = grp.groupby("test_dataset")["accuracy"].mean().sort_values(ascending=False)

        # Consistency across the three datasets directly (rather than the
        # coarser target-vs-rest split above): the worst-case dataset and
        # the coefficient of variation are what actually support a claim of
        # "generalises well everywhere" rather than "does well on average".
        per_dataset_accuracy = grp.groupby("test_dataset")["accuracy"].mean()
        per_dataset_composite = grp.groupby("test_dataset")["composite_score"].mean()

        acc_mean = per_dataset_accuracy.mean()
        acc_std = per_dataset_accuracy.std(ddof=1) if len(per_dataset_accuracy) > 1 else np.nan
        acc_cv = (acc_std / acc_mean) if (acc_mean and not pd.isna(acc_std)) else np.nan

        record_out = {
            **key_map,
            "in_domain_accuracy_mean": run_df["in_domain_accuracy"].mean(),
            "in_domain_accuracy_std": run_df["in_domain_accuracy"].std(ddof=1),
            "in_domain_accuracy_ci95": ci95(run_df["in_domain_accuracy"]),
            "out_domain_accuracy_mean": run_df["out_domain_accuracy"].mean(),
            "out_domain_accuracy_std": run_df["out_domain_accuracy"].std(ddof=1),
            "out_domain_accuracy_ci95": ci95(run_df["out_domain_accuracy"]),
            "generalisation_gap_mean": run_df["generalisation_gap"].mean(),
            "generalisation_gap_std": run_df["generalisation_gap"].std(ddof=1),
            "generalisation_gap_ci95": ci95(run_df["generalisation_gap"]),
            "retention_mean": run_df["retention"].mean(),
            "retention_std": run_df["retention"].std(ddof=1),
            "retention_ci95": ci95(run_df["retention"]),
            "retention_low_reliability_runs": int(run_df["low_in_domain_reliability"].sum()),
            "in_domain_composite_mean": run_df["in_domain_composite"].mean(),
            "out_domain_composite_mean": run_df["out_domain_composite"].mean(),
            "best_performing_dataset": dataset_means.index[0] if len(dataset_means) > 0 else "",
            "worst_performing_dataset": dataset_means.index[-1] if len(dataset_means) > 0 else "",
            "worst_case_dataset_accuracy": per_dataset_accuracy.min(),
            "worst_case_dataset_composite_score": per_dataset_composite.min(),
            "cross_dataset_accuracy_std": acc_std,
            "cross_dataset_accuracy_cv": acc_cv,
            "n_test_datasets_observed": int(len(per_dataset_accuracy)),
        }

        for cls in CLASSES:
            record_out[f"gap_f1_{cls}_mean"] = run_df[f"gap_f1_{cls}"].mean()
            record_out[f"gap_f1_{cls}_std"] = run_df[f"gap_f1_{cls}"].std(ddof=1)

        records.append(record_out)

    return pd.DataFrame(records)


def build_hardware_candidate_summary(candidate, candidate_per_dataset):
    summary_row = dict(candidate)
    summary_row["test_dataset"] = "ALL"
    return pd.DataFrame([summary_row]), candidate_per_dataset.copy()


def publish_results(
    fold_df,
    strategy_summary,
    scratch_summary,
    transfer_summary,
    generalisation_summary,
    generalisation_leaderboard,
    leaderboard,
    fold_rank,
    hardware_summary,
    hardware_per_dataset,
    sample_size_summary,
    experiment_score_deviations,
    final_model_calculation_steps,
    final_model_scores,
    final_model_score_deviations,
    per_experiment_metric_averages,
    all_experiment_metric_averages,
    significance_composite,
    significance_generalisation,
    weighted_generalisation_scores,
    consistency_strategy,
    consistency_per_fold,
    mcnemar_results,
    output_dirs,
):
    metric_csv_dir = output_dirs["metric_csvs"]
    per_fold_output = append_all_experiment_average_rows(
        fold_df,
        all_experiment_metric_averages,
    )
    per_fold_output.to_csv(
        os.path.join(metric_csv_dir, "per_fold_metrics.csv"),
        index=False,
    )
    strategy_summary.to_csv(
        os.path.join(metric_csv_dir, "training_strategy_summary.csv"),
        index=False,
    )
    experiment_score_deviations.to_csv(
        os.path.join(metric_csv_dir, "fold_score_deviation_between_experiments.csv"),
        index=False,
    )
    final_model_calculation_steps.to_csv(
        os.path.join(metric_csv_dir, "fold_composite_score_calculation_steps.csv"),
        index=False,
    )
    final_model_scores.to_csv(
        os.path.join(metric_csv_dir, "final_weighted_model_scores.csv"),
        index=False,
    )
    final_model_score_deviations.to_csv(
        os.path.join(metric_csv_dir, "final_model_score_deviation_between_experiments.csv"),
        index=False,
    )
    per_experiment_metric_averages.to_csv(
        os.path.join(metric_csv_dir, "per_experiment_metric_averages.csv"),
        index=False,
    )
    all_experiment_metric_averages.to_csv(
        os.path.join(metric_csv_dir, "all_experiment_metric_averages.csv"),
        index=False,
    )
    scratch_summary.to_csv(
        os.path.join(metric_csv_dir, "scratch_training_summary.csv"),
        index=False,
    )
    transfer_summary.to_csv(
        os.path.join(metric_csv_dir, "transfer_learning_summary.csv"),
        index=False,
    )
    generalisation_summary.to_csv(
        os.path.join(metric_csv_dir, "generalisation_summary.csv"),
        index=False,
    )
    generalisation_leaderboard.to_csv(
        os.path.join(metric_csv_dir, "generalisation_leaderboard.csv"),
        index=False,
    )
    leaderboard.to_csv(os.path.join(metric_csv_dir, "overall_leaderboard.csv"), index=False)
    fold_rank.to_csv(
        os.path.join(metric_csv_dir, "best_strategy_fold_ranking.csv"),
        index=False,
    )
    hardware_summary.to_csv(
        os.path.join(metric_csv_dir, "hardware_candidate_summary.csv"),
        index=False,
    )
    hardware_per_dataset.to_csv(
        os.path.join(metric_csv_dir, "hardware_candidate_per_dataset.csv"),
        index=False,
    )
    sample_size_summary.to_csv(
        os.path.join(metric_csv_dir, "sample_size_and_class_support.csv"),
        index=False,
    )
    significance_composite.to_csv(
        os.path.join(metric_csv_dir, "pairwise_significance_composite_score.csv"),
        index=False,
    )
    significance_generalisation.to_csv(
        os.path.join(
            metric_csv_dir,
            "pairwise_significance_out_of_domain_composite_score.csv",
        ),
        index=False,
    )

    # Full strategy x test-dataset transfer matrices (composite score and
    # accuracy), plus the underlying sample counts for each cell.
    for metric in ["composite_score", "accuracy"]:
        matrix, n_matrix = build_transfer_matrix(fold_df, metric=metric)
        matrix.to_csv(os.path.join(metric_csv_dir, f"transfer_matrix_{metric}.csv"))
        n_matrix.to_csv(os.path.join(metric_csv_dir, f"transfer_matrix_{metric}_n_samples.csv"))

    # Weighted per-dataset generalisation score using the authoritative final
    # dataset weights only.
    for obsolete_name in [
        "weighting_scheme_comparison.csv",
        "weighted_score_equal_weights.csv",
        "weighted_score_configured_weights.csv",
    ]:
        obsolete_path = os.path.join(metric_csv_dir, obsolete_name)
        if os.path.exists(obsolete_path):
            os.remove(obsolete_path)
    weighted_generalisation_scores.to_csv(
        os.path.join(metric_csv_dir, "weighted_score_final_dataset_weights.csv"),
        index=False,
    )

    # Per-object (same-source, different resolution/frequency) consistency.
    consistency_strategy.to_csv(
        os.path.join(metric_csv_dir, "consistency_by_strategy.csv"),
        index=False,
    )
    consistency_per_fold.to_csv(
        os.path.join(metric_csv_dir, "consistency_by_model_instance.csv"),
        index=False,
    )
    mcnemar_results.to_csv(
        os.path.join(metric_csv_dir, "mcnemar_pairwise_dataset_tests.csv"),
        index=False,
    )

    export_table(scratch_summary, output_dirs["publication_tables"], "table_scratch_model_summary")
    export_table(transfer_summary, output_dirs["publication_tables"], "table_transfer_learning_summary")
    export_table(generalisation_summary, output_dirs["publication_tables"], "table_generalisation_summary")
    export_table(
        generalisation_leaderboard,
        output_dirs["publication_tables"],
        "table_generalisation_leaderboard",
    )
    export_table(hardware_summary, output_dirs["publication_tables"], "table_hardware_candidate_summary")
    export_table(leaderboard, output_dirs["publication_tables"], "table_overall_leaderboard")
    export_table(
        significance_composite,
        output_dirs["publication_tables"],
        "table_pairwise_significance_composite_score",
    )
    export_table(
        significance_generalisation,
        output_dirs["publication_tables"],
        "table_pairwise_significance_out_of_domain_composite_score",
    )
    export_table(sample_size_summary, output_dirs["publication_tables"], "table_sample_size_and_class_support")
    export_table(
        weighted_generalisation_scores,
        output_dirs["publication_tables"],
        "table_weighted_generalisation_score",
    )
    export_table(
        consistency_strategy,
        output_dirs["publication_tables"],
        "table_cross_dataset_consistency",
    )
    export_table(
        mcnemar_results,
        output_dirs["publication_tables"],
        "table_mcnemar_pairwise_dataset_tests",
    )


def print_console_summary(
    leaderboard,
    scratch_summary,
    transfer_summary,
    generalisation_summary,
    generalisation_leaderboard,
    significance_composite,
    final_model_scores,
    final_model_score_deviations,
    consistency_strategy,
    mcnemar_results,
    candidate,
    weighted_generalisation_scores,
):
    best_scratch = None
    if not scratch_summary.empty:
        best_scratch = scratch_summary.sort_values("composite_score_mean", ascending=False).iloc[0]

    best_transfer = None
    if not transfer_summary.empty:
        best_transfer = transfer_summary.sort_values("composite_score_mean", ascending=False).iloc[0]

    best_generalisation = None
    if not generalisation_leaderboard.empty:
        best_generalisation = generalisation_leaderboard.iloc[0]

    best_overall = leaderboard.iloc[0] if not leaderboard.empty else None

    print("\n================ STRUCTURED RESULTS SUMMARY ================")

    complete_final_scores = final_model_scores[
        final_model_scores["has_all_test_sets_with_five_folds"]
    ].dropna(subset=["final_model_score"])
    if not complete_final_scores.empty:
        top_final_model = complete_final_scores.sort_values(
            "final_model_score", ascending=False
        ).iloc[0]
        print("\nTop model by final weighted testing score")
        print(f"  Strategy: {top_final_model['strategy_label']}")
        print(f"  Experiment: {int(top_final_model['experiment_number'])}")
        for dataset in FINAL_MODEL_DATASET_WEIGHTS:
            print(
                f"  {dataset} five-fold mean: {top_final_model[f'{dataset}_test_set_score']:.4f} "
                f"x {top_final_model[f'{dataset}_weight']:.2f} "
                f"= {top_final_model[f'{dataset}_weighted_contribution']:.4f}"
            )
        print(f"  Final weighted testing score: {top_final_model['final_model_score']:.4f}")
    else:
        print("\nFinal weighted testing score unavailable: each test set requires five folds.")

    if not final_model_score_deviations.empty:
        largest_deviation = final_model_score_deviations.iloc[0]
        print("\nLargest final-score deviation between experiments")
        print(f"  Strategy: {largest_deviation['strategy_label']}")
        experiment_score_columns = sorted(
            [
                column
                for column in final_model_score_deviations.columns
                if column.startswith("experiment_")
                and column.endswith("_final_model_score")
            ],
            key=lambda column: int(column.split("_")[1]),
        )
        for column in experiment_score_columns:
            experiment_number = int(column.split("_")[1])
            experiment_score = largest_deviation[column]
            if not pd.isna(experiment_score):
                print(f"  Experiment {experiment_number} final score: {experiment_score:.4f}")
        print(
            "  Final-score range across experiments: "
            f"{largest_deviation['final_model_score_deviation_range']:.4f}"
        )

    if best_scratch is not None:
        print("\nBest scratch model")
        print(f"  Strategy: {best_scratch['strategy_label']}")
        print(f"  Composite score: {best_scratch['composite_score_mean']:.4f}")

    if best_transfer is not None:
        print("\nBest transfer-learning model")
        print(f"  Strategy: {best_transfer['strategy_label']}")
        print(f"  Composite score: {best_transfer['composite_score_mean']:.4f}")

    if best_overall is not None:
        print("\nTop-ranked strategy (final weighted score)")
        print(f"  Strategy: {best_overall['strategy_label']}")
        print(f"  Final weighted score: {best_overall['final_model_score']:.4f}")

    if best_generalisation is not None:
        print("\nBest-generalising strategy (ranked on out-of-domain performance)")
        print(f"  Strategy: {best_generalisation['strategy_label']}")
        print(f"  Out-of-domain composite score: {best_generalisation['out_domain_composite_mean']:.4f}")
        print(f"  Worst-case dataset composite score: {best_generalisation['worst_case_dataset_composite_score']:.4f}")
        print(f"  Cross-dataset accuracy CV: {best_generalisation['cross_dataset_accuracy_cv']:.4f}")
        if not pd.isna(best_generalisation.get("retention_mean", np.nan)):
            print(f"  Retention (out/in accuracy): {best_generalisation['retention_mean']:.4f}")
        if best_overall is not None and best_generalisation["strategy_label"] != best_overall["strategy_label"]:
            print(
                "  NOTE: this is a descriptive out-of-domain diagnostic; the "
                "authoritative overall ranking is the final weighted score above."
            )

    if not weighted_generalisation_scores.empty:
        top_weighted = weighted_generalisation_scores.sort_values(
            "weighted_generalisation_score", ascending=False
        ).iloc[0]
        print("\nBest strategy by final dataset-weighted generalisation score")
        print(f"  Strategy: {top_weighted['strategy_label']}")
        print(f"  Weighted score: {top_weighted['weighted_generalisation_score']:.4f}")

    if not consistency_strategy.empty:
        top_by_stability = consistency_strategy.sort_values(
            "robust_correct_rate_mean", ascending=False
        ).iloc[0]
        print("\nMost consistent strategy across resolution/frequency (same sources, paired)")
        print(f"  Strategy: {top_by_stability['strategy_label']}")
        print(f"  Prediction stability rate (same predicted class at every resolution): {top_by_stability['prediction_stability_rate_mean']:.4f}")
        print(f"  Robust correct rate (correct at every resolution): {top_by_stability['robust_correct_rate_mean']:.4f}")
        if top_by_stability["total_label_alignment_mismatches"] > 0:
            print(
                f"  WARNING: {int(top_by_stability['total_label_alignment_mismatches'])} label "
                "alignment mismatches detected -- verify row order is truly aligned across "
                "test datasets for this strategy before trusting the pairing above."
            )

    if not mcnemar_results.empty:
        sig_pairs = mcnemar_results[mcnemar_results["significant_at_0.05"]]
        print(f"\nMcNemar pairwise dataset tests: {len(sig_pairs)}/{len(mcnemar_results)} pairs show a significant difference")
        print("  Full pairwise results: metric_csvs/mcnemar_pairwise_dataset_tests.csv")

    if not significance_composite.empty:
        top_row = significance_composite.iloc[0]
        sig_flag = "significant" if top_row.get("significant_at_0.05", False) else "not significant"
        print("\nSignificance check (top-ranked strategies, Holm-adjusted Wilcoxon test)")
        print(f"  {top_row['strategy_a']} vs {top_row['strategy_b']}: difference is {sig_flag}")
        print(f"  Holm-adjusted p-value: {top_row['wilcoxon_p_holm_adjusted']:.4f}")
        print("  Full pairwise results: metric_csvs/pairwise_significance_composite_score.csv")

    print("\nSelected hardware candidate")
    print(f"  Strategy: {candidate['strategy_label']}")
    print(f"  Experiment: {int(candidate['experiment_number'])}")
    print(f"  Fold number: {int(candidate['fold'])}")
    print(f"  Model filename: {candidate.get('model_filename', '')}")
    print(f"  Model path: {candidate.get('model_path', '')}")
    print(f"  Selection basis: {candidate['selection_basis']}")
    print(f"  Final five-fold weighted model score: {candidate['final_model_score']:.4f}")
    print(f"  Selected fold weighted testing score: {candidate['final_weighted_fold_score']:.4f}")

    print("\n============================================================")
    print("Analysis complete.")


##############################################################################
# MAIN PIPELINE
##############################################################################

def main():
    output_dirs = ensure_output_dirs(RESULTS_DIR)

    print("Stage 1/9: Loading all experiment results...")
    index_df = load_experiment_index(RUN_DIRECTORIES)
    if index_df.empty:
        print("No valid fold predictions were found.")
        return

    print("Stage 2/9: Evaluating every fold individually...")
    fold_df, curve_rows, failures = evaluate_all_folds(index_df)
    if fold_df.empty:
        print("No folds could be evaluated.")
        return

    if failures:
        failure_log = os.path.join(RESULTS_DIR, "evaluation_failures.txt")
        with open(failure_log, "w", encoding="utf-8") as f:
            f.write("\n".join(failures))

    print("Stage 3/9: Summarising each model by averaging across folds...")
    strategy_summary, _ = summarise_training_strategies(fold_df)
    scratch_summary = strategy_summary[strategy_summary["training_category"] == "scratch"].copy()
    transfer_summary = strategy_summary[strategy_summary["training_category"] == "transfer_learning"].copy()
    final_model_calculation_steps, final_model_scores = compute_final_model_scores(fold_df)
    final_model_score_deviations = compute_final_model_score_deviations(final_model_scores)
    per_experiment_metric_averages, all_experiment_metric_averages = (
        compute_all_experiment_metric_averages(fold_df)
    )

    print("Stage 4/9: Ranking all training strategies...")
    generalisation_summary = compute_generalisation_summary(fold_df)
    leaderboard = rank_final_models(final_model_scores)
    if leaderboard.empty:
        print("No strategies could be ranked.")
        return

    # The overall leaderboard uses only the configured final dataset weights.
    # Generalisation remains a descriptive fold-mean report.
    generalisation_leaderboard = rank_by_generalisation(generalisation_summary)
    best_strategy_id = leaderboard.iloc[0]["strategy_id"]
    best_generalising_strategy_id = (
        generalisation_leaderboard.iloc[0]["strategy_id"]
        if not generalisation_leaderboard.empty
        else best_strategy_id
    )

    print("Stage 5/9: Selecting the hardware candidate from the top final model...")
    fold_rank, candidate, candidate_per_dataset = rank_folds_for_best_final_model(
        fold_df, final_model_scores
    )
    hardware_summary, hardware_per_dataset = build_hardware_candidate_summary(candidate, candidate_per_dataset)

    strategy_order = leaderboard["strategy_id"].tolist()
    dataset_order = order_datasets(fold_df["test_dataset"].unique().tolist())

    print("Stage 6/9: Generating visualisations...")
    generate_confusion_matrix_outputs(curve_rows, output_dirs, strategy_order, dataset_order, candidate)
    generate_heatmaps(fold_df, output_dirs, strategy_order, dataset_order, best_strategy_id)
    generate_model_ranking_plot(leaderboard, output_dirs, best_strategy_id)
    generate_fold_ranking_plot(fold_rank, output_dirs)
    generate_roc_pr_outputs(curve_rows, output_dirs, candidate, best_strategy_id)

    print("Stage 7/9: Sample sizes and significance testing...")
    sample_size_summary = build_sample_size_summary(fold_df)
    experiment_score_deviations = compute_experiment_score_deviations(fold_df)

    # Significance test #1: fold-level composite score for the top-N strategies
    # from the final weighted leaderboard.
    significance_composite = compute_pairwise_significance(
        fold_df, leaderboard, metric="composite_score"
    )

    # Significance test #2: same idea, but restricted to out-of-domain rows
    # only and using the generalisation leaderboard's ranking, since that is
    # the comparison that actually matters for a generalisation claim.
    out_of_domain_fold_df = fold_df[fold_df["test_dataset"] != fold_df["target_dataset"]].copy()
    generalisation_leaderboard_for_sig = generalisation_leaderboard.rename(
        columns={"generalisation_rank": "rank"}
    )
    if not out_of_domain_fold_df.empty and not generalisation_leaderboard_for_sig.empty:
        significance_generalisation = compute_pairwise_significance(
            out_of_domain_fold_df, generalisation_leaderboard_for_sig, metric="composite_score"
        )
    else:
        significance_generalisation = pd.DataFrame()

    print("Stage 8/9: Weighted per-dataset score and cross-dataset consistency...")
    # Average the metric within each test dataset, then apply the configured
    # final dataset weights. No equal-weight sensitivity calculation is run.
    weighted_generalisation_scores = compute_weighted_generalisation_score(
        fold_df,
        FINAL_MODEL_DATASET_WEIGHTS,
        EXCLUDE_TARGET_FROM_GENERALISATION_SCORE,
        GENERALISATION_SCORE_METRIC,
        "final_dataset_weights",
    )

    # Per-object pairing across the three resolution/frequency variants
    # (same underlying sources), plus paired McNemar tests between dataset
    # pairs -- a stronger and more specific generalisation signal than
    # dataset-level averages alone.
    consistency_strategy, consistency_per_fold = compute_cross_dataset_consistency(curve_rows)
    mcnemar_results = compute_cross_dataset_mcnemar(curve_rows)

    print("Stage 9/9: Exporting all results...")
    publish_results(
        fold_df,
        strategy_summary,
        scratch_summary,
        transfer_summary,
        generalisation_summary,
        generalisation_leaderboard,
        leaderboard,
        fold_rank,
        hardware_summary,
        hardware_per_dataset,
        sample_size_summary,
        experiment_score_deviations,
        final_model_calculation_steps,
        final_model_scores,
        final_model_score_deviations,
        per_experiment_metric_averages,
        all_experiment_metric_averages,
        significance_composite,
        significance_generalisation,
        weighted_generalisation_scores,
        consistency_strategy,
        consistency_per_fold,
        mcnemar_results,
        output_dirs,
    )

    print_console_summary(
        leaderboard,
        scratch_summary,
        transfer_summary,
        generalisation_summary,
        generalisation_leaderboard,
        significance_composite,
        final_model_scores,
        final_model_score_deviations,
        consistency_strategy,
        mcnemar_results,
        candidate,
        weighted_generalisation_scores,
    )

    print(f"\nBest strategy by final weighted score:       {best_strategy_id}")
    print(f"Best strategy by out-of-domain generalisation: {best_generalising_strategy_id}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_log_path = os.path.join(RESULTS_DIR, "results_summary_terminal_output.txt")
    with open(output_log_path, "w", encoding="utf-8") as output_log:
        with redirect_stdout(TeeOutput(sys.stdout, output_log)):
            main()