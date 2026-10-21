"""Summarise training metrics and learning curves for Experiment_0."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import (
	accuracy_score,
	auc,
	f1_score,
	matthews_corrcoef,
	precision_recall_fscore_support,
	precision_score,
	roc_auc_score,
	roc_curve,
)


# Default title/legend sizes (12 / 10 pt) doubled.
plt.rcParams.update({
	"axes.titlesize": 24,
	"figure.titlesize": 24,
	"legend.fontsize": 20,
	"axes.labelsize": 20,
	"xtick.labelsize": 20,
	"ytick.labelsize": 20,
})

PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_DIR / "Training" / "Results" / "Experiment_0"
TRAINING_DIR = EXPERIMENT_DIR / "Training"
OUTPUT_DIR = EXPERIMENT_DIR / "EvaluationSummary"
CSV_DIR = OUTPUT_DIR / "CSVs"
ROC_DIR = OUTPUT_DIR / "ROC_Curves"
BUBBLE_DIR = OUTPUT_DIR / "Metric_Bubble_Plots"
BUBBLE_BEST_DIR = OUTPUT_DIR / "Metric_Bubble_Plots_Best_Highlighted"
HEATMAP_DIR = OUTPUT_DIR / "Metric_Heatmaps"
FOLD_CURVE_DIR = OUTPUT_DIR / "Training_Curves_Per_Fold"
ACCURACY_DIR = OUTPUT_DIR / "Accuracy_Curves"
TESTING_DIR = EXPERIMENT_DIR / "Testing"

CLASSES = ["FRI", "FRII", "COMPACT"]
DATASET_WEIGHTS = {"FIRST": 0.45, "LoTSS": 0.45, "NVSS": 0.10}
# Result folders on disk are still named LOFAR.
DATASET_LABELS = {"LOFAR": "LoTSS"}
HEATMAP_METRICS = [
	("accuracy", "Accuracy"),
	("precision", "Precision (macro)"),
	("roc_auc", "ROC AUC (macro OvR)"),
	("f1", "F1 (macro)"),
	("mcc", "MCC"),
]
TESTING_FOLDER_PATTERN = re.compile(r"base(.*?)_target(.*?)_dataset(.*?)_")


def load_history(history_path):
	"""Load one saved Keras History dictionary from a NumPy object file."""
	saved_history = np.load(history_path, allow_pickle=True)
	if isinstance(saved_history, np.ndarray) and saved_history.shape == ():
		saved_history = saved_history.item()
	if not isinstance(saved_history, dict):
		raise ValueError(f"Expected a metric dictionary in {history_path}")

	history = {}
	for metric, values in saved_history.items():
		history[str(metric)] = np.asarray(values, dtype=float).reshape(-1)
	if not history or max(map(len, history.values())) == 0:
		raise ValueError(f"No epoch metrics found in {history_path}")
	return history


def load_training_runs():
	"""Read all saved fold histories from Experiment_0."""
	runs = []
	if not TRAINING_DIR.is_dir():
		return runs

	for model_dir in sorted(path for path in TRAINING_DIR.iterdir() if path.is_dir()):
		for history_path in sorted(model_dir.glob("loss_fold_*.npy")):
			match = re.fullmatch(r"loss_fold_(\d+)", history_path.stem)
			if match is None:
				continue

			fold = int(match.group(1))
			try:
				history = load_history(history_path)
			except (OSError, ValueError, TypeError) as error:
				print(f"Skipping {history_path}: {error}")
				continue

			time_path = model_dir / f"training_time_fold_{fold}.npy"
			training_time = None
			if time_path.exists():
				try:
					training_time = float(np.asarray(np.load(time_path)).item())
				except (OSError, ValueError, TypeError):
					print(f"Could not read training time from {time_path}")

			runs.append({
				"model": model_dir.name,
				"fold": fold,
				"history": history,
				"training_time_seconds": training_time,
			})
	return runs


def save_figure(fig, path, dpi=180):
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=dpi)


def write_csv(path, rows, fieldnames):
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as output_file:
		writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)


def build_epoch_rows(runs):
	metric_names = sorted({metric for run in runs for metric in run["history"]})
	rows = []
	for run in runs:
		history = run["history"]
		epoch_count = max(map(len, history.values()))
		for epoch_index in range(epoch_count):
			row = {
				"experiment": "Experiment_0",
				"model": run["model"],
				"fold": run["fold"],
				"epoch": epoch_index + 1,
			}
			for metric in metric_names:
				values = history.get(metric, [])
				row[metric] = float(values[epoch_index]) if epoch_index < len(values) else ""
			rows.append(row)
	return metric_names, rows


def build_fold_summary_rows(runs, metric_names):
	rows = []
	for run in runs:
		history = run["history"]
		epoch_count = max(map(len, history.values()))
		row = {
			"experiment": "Experiment_0",
			"model": run["model"],
			"fold": run["fold"],
			"epochs_trained": epoch_count,
			"training_time_seconds": (
				run["training_time_seconds"]
				if run["training_time_seconds"] is not None else ""
			),
		}
		for metric in metric_names:
			values = history.get(metric, [])
			row[f"final_{metric}"] = float(values[-1]) if len(values) else ""

		if "val_loss" in history and len(history["val_loss"]):
			best_epoch_index = int(np.nanargmin(history["val_loss"]))
		elif "val_accuracy" in history and len(history["val_accuracy"]):
			best_epoch_index = int(np.nanargmax(history["val_accuracy"]))
		else:
			best_epoch_index = epoch_count - 1

		row["best_epoch"] = best_epoch_index + 1
		for metric in ("accuracy", "val_accuracy"):
			values = history.get(metric, [])
			row[f"{metric}_at_best_epoch"] = (
				float(values[best_epoch_index]) if best_epoch_index < len(values) else ""
			)
		rows.append(row)
	return rows


def build_model_summary_rows(fold_rows, metric_names):
	rows = []
	models = sorted({row["model"] for row in fold_rows})
	for model in models:
		model_folds = [row for row in fold_rows if row["model"] == model]
		row = {
			"experiment": "Experiment_0",
			"model": model,
			"folds": len(model_folds),
		}
		training_times = np.asarray(
			[float(fold["training_time_seconds"]) for fold in model_folds
			 if fold["training_time_seconds"] != ""],
			dtype=float,
		)
		row["training_time_seconds_mean"] = (
			float(np.mean(training_times)) if len(training_times) else ""
		)
		row["training_time_seconds_std"] = (
			float(np.std(training_times, ddof=1)) if len(training_times) > 1 else ""
		)
		for metric in metric_names:
			values = np.asarray(
				[float(fold[f"final_{metric}"]) for fold in model_folds
				 if fold[f"final_{metric}"] != ""],
				dtype=float,
			)
			row[f"final_{metric}_mean"] = float(np.mean(values)) if len(values) else ""
			row[f"final_{metric}_std"] = (
				float(np.std(values, ddof=1)) if len(values) > 1 else ""
			)
		rows.append(row)
	return rows


def training_run_title(folder_name):
	"""Format a training folder name as 'BASE->FINE_TUNED: TEST'."""
	match = TESTING_FOLDER_PATTERN.match(folder_name)
	if match is None:
		return folder_name
	base, target, test_dataset = (DATASET_LABELS.get(name, name) for name in match.groups())
	return f"{base}->{target}: {test_dataset}"


def plot_accuracy_curves(runs):
	models = sorted({run["model"] for run in runs})
	for model in models:
		model_runs = [run for run in runs if run["model"] == model]
		if not any(
			"accuracy" in run["history"] or "val_accuracy" in run["history"]
			for run in model_runs
		):
			continue

		fig, axis = plt.subplots(figsize=(9, 5.5))
		colors = {"accuracy": "#176B5B", "val_accuracy": "#D15B35"}
		labels = {"accuracy": "Training accuracy", "val_accuracy": "Validation accuracy"}

		for metric in ("accuracy", "val_accuracy"):
			curves = [
				np.asarray(run["history"][metric], dtype=float)
				for run in model_runs
				if metric in run["history"] and len(run["history"][metric])
			]
			if not curves:
				continue

			max_epochs = max(map(len, curves))
			values = np.full((len(curves), max_epochs), np.nan)
			for index, curve in enumerate(curves):
				values[index, :len(curve)] = curve
			epoch_numbers = np.arange(1, max_epochs + 1)
			mean_values = np.nanmean(values, axis=0)
			std_values = np.nanstd(values, axis=0)
			axis.plot(epoch_numbers, mean_values, color=colors[metric], label=labels[metric])
			axis.fill_between(
				epoch_numbers,
				mean_values - std_values,
				mean_values + std_values,
				color=colors[metric],
				alpha=0.16,
				linewidth=0,
			)

		axis.set_title(training_run_title(model))
		axis.set_xlabel("Epoch")
		axis.set_ylabel("Accuracy")
		axis.set_ylim(0, 1)
		axis.grid(axis="y", alpha=0.25)
		axis.legend(frameon=False)
		fig.tight_layout()
		safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
		save_figure(fig, ACCURACY_DIR / f"{safe_name}_accuracy.png")
		plt.close(fig)


def plot_fold_curves(runs):
	panels = [
		("accuracy", "Training accuracy"),
		("val_accuracy", "Validation accuracy"),
		("loss", "Training loss"),
		("val_loss", "Validation loss"),
	]
	for model in sorted({run["model"] for run in runs}):
		model_runs = sorted(
			(run for run in runs if run["model"] == model), key=lambda run: run["fold"]
		)
		available = [
			panel for panel in panels
			if any(len(run["history"].get(panel[0], [])) for run in model_runs)
		]
		if not available:
			continue

		fig, axes = plt.subplots(
			2, 2, figsize=(13, 9), sharex=True, squeeze=False
		)
		flat_axes = axes.ravel()
		colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(model_runs), 2)))
		for axis, (metric, title) in zip(flat_axes, panels):
			for color, run in zip(colors, model_runs):
				values = run["history"].get(metric, [])
				if len(values):
					axis.plot(
						np.arange(1, len(values) + 1), values,
						color=color, label=f"Fold {run['fold']}",
					)
			axis.set_title(title)
			axis.set_ylabel("Accuracy" if "accuracy" in metric else "Loss")
			axis.grid(alpha=0.25)
			if "accuracy" in metric:
				axis.set_ylim(0, 1)
		for axis in flat_axes[2:]:
			axis.set_xlabel("Epoch")
		handles, labels = flat_axes[0].get_legend_handles_labels()
		if not handles:
			handles, labels = flat_axes[1].get_legend_handles_labels()
		fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
		fig.suptitle(training_run_title(model))
		fig.tight_layout(rect=(0, 0.04, 1, 0.95))
		safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
		save_figure(fig, FOLD_CURVE_DIR / f"{safe_name}_folds.png")
		plt.close(fig)


def to_int_labels(y):
	y = np.asarray(y)
	if y.ndim > 1:
		y = np.argmax(y, axis=1)
	if y.dtype.kind in {"U", "S", "O"}:
		y = np.array([CLASSES.index(str(value)) for value in y])
	return y.astype(int)


def compute_fold_test_metrics(y_true, y_prob):
	y_pred = np.argmax(y_prob, axis=1)
	try:
		roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
	except ValueError:
		roc_auc = np.nan
	return {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
		"roc_auc": roc_auc,
		"f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
		"mcc": matthews_corrcoef(y_true, y_pred),
	}


def load_test_metrics():
	"""Compute per-fold test metrics for every model and test dataset."""
	rows = []
	if not TESTING_DIR.is_dir():
		return rows

	for folder in sorted(path for path in TESTING_DIR.iterdir() if path.is_dir()):
		match = TESTING_FOLDER_PATTERN.match(folder.name)
		y_true_path = folder / "y_test.npy"
		if match is None or not y_true_path.exists():
			continue

		base, target, test_dataset = (DATASET_LABELS.get(name, name) for name in match.groups())
		y_true = to_int_labels(np.load(y_true_path, allow_pickle=True))
		for fold in range(1, 6):
			pred_path = folder / f"y_pred_test_fold_{fold}.npy"
			if not pred_path.exists():
				continue
			y_prob = np.asarray(np.load(pred_path), dtype=float)
			if y_prob.ndim != 2 or y_prob.shape != (len(y_true), len(CLASSES)):
				print(f"Skipping {pred_path}: expected probabilities of shape ({len(y_true)}, {len(CLASSES)})")
				continue
			rows.append({
				"experiment": "Experiment_0",
				"model": f"{base}->{target}",
				"test_dataset": test_dataset,
				"fold": fold,
				**compute_fold_test_metrics(y_true, y_prob),
			})
	return rows


def _heatmap_axes(panel_count, column_count, model_count):
	"""Figure with one heatmap axis per panel (2 columns up to 4 panels, else 3); spare axes are hidden."""
	ncols = 2 if panel_count <= 4 else 3
	nrows = int(np.ceil(panel_count / ncols))
	fig, axes = plt.subplots(
		nrows, ncols,
		figsize=(ncols / 2 * (5 + 2.2 * column_count), nrows / 2 * (2.5 + 0.55 * model_count * 2)),
		squeeze=False,
	)
	flat_axes = list(axes.ravel())
	for axis in flat_axes[panel_count:]:
		axis.set_visible(False)
	return fig, flat_axes[:panel_count]


def plot_metric_heatmaps(test_rows):
	models = sorted({row["model"] for row in test_rows})
	datasets = sorted({row["test_dataset"] for row in test_rows})
	mean_rows = []
	grids = {}
	for metric, _ in HEATMAP_METRICS:
		grid = np.full((len(models), len(datasets)), np.nan)
		for i, model in enumerate(models):
			for j, dataset in enumerate(datasets):
				values = [
					row[metric] for row in test_rows
					if row["model"] == model and row["test_dataset"] == dataset
				]
				if values and not np.all(np.isnan(values)):
					grid[i, j] = np.nanmean(values)
		grids[metric] = grid

	# A model missing any dataset gets no average/weighted score.
	weights = np.array([DATASET_WEIGHTS.get(dataset, np.nan) for dataset in datasets])
	for metric, _ in HEATMAP_METRICS:
		per_dataset = grids[metric]
		grids[metric] = np.hstack([
			per_dataset,
			per_dataset.mean(axis=1)[:, None],
			(per_dataset @ weights)[:, None],
		])
	columns = datasets + ["Average", "Weighted"]

	for i, model in enumerate(models):
		for j, dataset in enumerate(columns):
			folds = [
				row for row in test_rows
				if row["model"] == model and row["test_dataset"] == dataset
			]
			mean_row = {"experiment": "Experiment_0", "model": model, "test_dataset": dataset, "folds": len(folds)}
			for metric, _ in HEATMAP_METRICS:
				mean_row[metric] = grids[metric][i, j]
			mean_rows.append(mean_row)
	write_csv(
		CSV_DIR / "test_metrics_fold_mean.csv",
		mean_rows,
		["experiment", "model", "test_dataset", "folds"] + [metric for metric, _ in HEATMAP_METRICS],
	)

	fig, axes = _heatmap_axes(len(HEATMAP_METRICS), len(columns), len(models))
	for axis, (metric, title) in zip(axes, HEATMAP_METRICS):
		grid = grids[metric]
		image = axis.imshow(np.ma.masked_invalid(grid), cmap="viridis", vmin=0, vmax=1, aspect="auto")
		axis.set_xticks(range(len(columns)), columns, fontsize=10)
		axis.set_yticks(range(len(models)), models, fontsize=10)
		axis.set_xlabel("Test dataset", fontsize=10)
		axis.set_title(title)
		for i in range(len(models)):
			for j in range(len(columns)):
				if not np.isnan(grid[i, j]):
					axis.text(
						j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
						color="white" if grid[i, j] < 0.6 else "black", fontsize=14,
					)
		axis.axvline(len(datasets) - 0.5, color="white", linewidth=2)
		for column_index, color, inset in ((-2, "blue", 0.08), (-1, "red", 0.0)):
			scores = grid[:, column_index]
			if np.all(np.isnan(scores)):
				continue
			best = int(np.nanargmax(scores))
			axis.add_patch(Rectangle(
				(-0.5 + inset, best - 0.5 + inset), len(columns) - 2 * inset, 1 - 2 * inset,
				fill=False, edgecolor=color, linewidth=3, clip_on=False,
			))
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	fig.suptitle("Test metrics")
	fig.tight_layout(rect=(0, 0, 1, 0.95))
	save_figure(fig, HEATMAP_DIR / "test_metric_heatmaps.png")
	plt.close(fig)


def plot_plain_metric_heatmaps(test_rows):
	"""Heatmaps of fold-mean test metrics per dataset only: no Average/Weighted scores, no highlighting."""
	models = sorted({row["model"] for row in test_rows})
	datasets = sorted({row["test_dataset"] for row in test_rows})
	fig, axes = _heatmap_axes(len(HEATMAP_METRICS), len(datasets), len(models))
	for axis, (metric, title) in zip(axes, HEATMAP_METRICS):
		grid = np.full((len(models), len(datasets)), np.nan)
		for i, model in enumerate(models):
			for j, dataset in enumerate(datasets):
				values = [
					row[metric] for row in test_rows
					if row["model"] == model and row["test_dataset"] == dataset
				]
				if values and not np.all(np.isnan(values)):
					grid[i, j] = np.nanmean(values)
		image = axis.imshow(np.ma.masked_invalid(grid), cmap="viridis", vmin=0, vmax=1, aspect="auto")
		axis.set_xticks(range(len(datasets)), datasets, fontsize=10)
		axis.set_yticks(range(len(models)), models, fontsize=10)
		axis.set_xlabel("Test dataset", fontsize=10)
		axis.set_title(title)
		for i in range(len(models)):
			for j in range(len(datasets)):
				if not np.isnan(grid[i, j]):
					axis.text(
						j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
						color="white" if grid[i, j] < 0.6 else "black", fontsize=14,
					)
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	fig.suptitle("Test metrics")
	fig.tight_layout(rect=(0, 0, 1, 0.95))
	save_figure(fig, HEATMAP_DIR / "test_metric_heatmaps_plain.png")
	plt.close(fig)


# Per-class heatmap panels: (CSV key, panel title, index into load_per_class_metrics' metric axis).
# Per-class accuracy is the fraction of that class's sources classified correctly (i.e. class recall).
PER_CLASS_HEATMAP_METRICS = [
	("accuracy", "Accuracy (class recall)", 0),
	("precision", "Precision", 1),
	("roc_auc", "ROC AUC (OvR)", 3),
	("f1", "F1", 2),
	("mcc", "MCC (OvR)", 4),
]


def _draw_heatmap_panels(panels, models, columns, dataset_count, suptitle, path, highlight):
	"""Draw a figure of heatmaps; panels is a list of (title, grid[model, column])."""
	fig, axes = _heatmap_axes(len(panels), len(columns), len(models))
	for axis, (title, grid) in zip(axes, panels):
		image = axis.imshow(np.ma.masked_invalid(grid), cmap="viridis", vmin=0, vmax=1, aspect="auto")
		axis.set_xticks(range(len(columns)), columns, fontsize=10)
		axis.set_yticks(range(len(models)), models, fontsize=10)
		axis.set_xlabel("Test dataset", fontsize=10)
		axis.set_title(title)
		for i in range(len(models)):
			for j in range(len(columns)):
				if not np.isnan(grid[i, j]):
					axis.text(
						j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
						color="white" if grid[i, j] < 0.6 else "black", fontsize=14,
					)
		if highlight:
			axis.axvline(dataset_count - 0.5, color="white", linewidth=2)
			for column_index, color, inset in ((-2, "blue", 0.08), (-1, "red", 0.0)):
				scores = grid[:, column_index]
				if np.all(np.isnan(scores)):
					continue
				best = int(np.nanargmax(scores))
				axis.add_patch(Rectangle(
					(-0.5 + inset, best - 0.5 + inset), len(columns) - 2 * inset, 1 - 2 * inset,
					fill=False, edgecolor=color, linewidth=3, clip_on=False,
				))
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	fig.suptitle(suptitle)
	fig.tight_layout(rect=(0, 0, 1, 0.95))
	save_figure(fig, path)
	plt.close(fig)


def plot_per_class_metric_heatmaps(test_rows):
	"""Per-class versions of the test metric heatmaps: one highlighted and one plain figure per class."""
	per_class = load_per_class_metrics()
	if not per_class:
		return
	models = sorted({row["model"] for row in test_rows})
	datasets = sorted({row["test_dataset"] for row in test_rows})
	weights = np.array([DATASET_WEIGHTS.get(dataset, np.nan) for dataset in datasets])
	columns = datasets + ["Average", "Weighted"]

	csv_rows = []
	for class_index, class_name in enumerate(CLASSES):
		class_label = BUBBLE_CLASS_LABELS.get(class_name, class_name)
		grids = {}
		for metric, _, metric_index in PER_CLASS_HEATMAP_METRICS:
			grid = np.full((len(models), len(datasets)), np.nan)
			for i, model in enumerate(models):
				for j, dataset in enumerate(datasets):
					values = per_class.get((model, dataset))
					if values is None:
						continue
					values = values[:, class_index, metric_index]
					if not np.all(np.isnan(values)):
						grid[i, j] = np.nanmean(values)
			# As for the macro heatmaps, a model missing any dataset gets no average/weighted score.
			grids[metric] = np.hstack([grid, grid.mean(axis=1)[:, None], (grid @ weights)[:, None]])

		for i, model in enumerate(models):
			for j, column in enumerate(columns):
				row = {"experiment": "Experiment_0", "model": model, "test_dataset": column, "class": class_name}
				for metric, _, _ in PER_CLASS_HEATMAP_METRICS:
					row[metric] = grids[metric][i, j]
				csv_rows.append(row)

		safe_class = re.sub(r"[^A-Za-z0-9._-]+", "_", class_name)
		_draw_heatmap_panels(
			[(title, grids[metric]) for metric, title, _ in PER_CLASS_HEATMAP_METRICS],
			models, columns, len(datasets), f"Test metrics - {class_label}",
			HEATMAP_DIR / f"test_metric_heatmaps_{safe_class}.png", highlight=True,
		)
		_draw_heatmap_panels(
			[(title, grids[metric][:, :len(datasets)]) for metric, title, _ in PER_CLASS_HEATMAP_METRICS],
			models, datasets, len(datasets), f"Test metrics - {class_label}",
			HEATMAP_DIR / f"test_metric_heatmaps_{safe_class}_plain.png", highlight=False,
		)

	write_csv(
		CSV_DIR / "test_metrics_per_class_fold_mean.csv",
		csv_rows,
		["experiment", "model", "test_dataset", "class"] + [m for m, _, _ in PER_CLASS_HEATMAP_METRICS],
	)


def macro_roc_for_fold(y_true, y_prob, fpr_grid):
	"""Per-class and macro-average (interpolated) ROC curves on a common FPR grid."""
	class_tprs = []
	for index in range(len(CLASSES)):
		positives = y_true == index
		if positives.all() or not positives.any():
			class_tprs.append(None)
			continue
		fpr, tpr, _ = roc_curve(positives, y_prob[:, index])
		class_tprs.append(np.interp(fpr_grid, fpr, tpr))
	valid = [tpr for tpr in class_tprs if tpr is not None]
	macro = np.mean(valid, axis=0) if valid else None
	return class_tprs, macro


def plot_roc_curves():
	"""One figure per model: a panel per test dataset with class and macro ROC curves."""
	if not TESTING_DIR.is_dir():
		return

	fpr_grid = np.linspace(0, 1, 201)
	data = {}
	for folder in sorted(path for path in TESTING_DIR.iterdir() if path.is_dir()):
		match = TESTING_FOLDER_PATTERN.match(folder.name)
		y_true_path = folder / "y_test.npy"
		if match is None or not y_true_path.exists():
			continue
		base, target, test_dataset = (DATASET_LABELS.get(name, name) for name in match.groups())
		y_true = to_int_labels(np.load(y_true_path, allow_pickle=True))
		fold_results = []
		for fold in range(1, 6):
			pred_path = folder / f"y_pred_test_fold_{fold}.npy"
			if not pred_path.exists():
				continue
			y_prob = np.asarray(np.load(pred_path), dtype=float)
			if y_prob.shape != (len(y_true), len(CLASSES)):
				continue
			fold_results.append(macro_roc_for_fold(y_true, y_prob, fpr_grid))
		if fold_results:
			data.setdefault(f"{base}->{target}", {})[test_dataset] = fold_results

	class_colors = ["#176B5B", "#D15B35", "#3B6FB6"]
	for model, per_dataset in data.items():
		datasets = sorted(per_dataset)
		fig, axes = plt.subplots(
			1, len(datasets), figsize=(5.2 * len(datasets), 5), squeeze=False
		)
		for axis, dataset in zip(axes.ravel(), datasets):
			fold_results = per_dataset[dataset]
			for index, class_name in enumerate(CLASSES):
				curves = [r[0][index] for r in fold_results if r[0][index] is not None]
				if not curves:
					continue
				mean_tpr = np.mean(curves, axis=0)
				axis.plot(
					fpr_grid, mean_tpr, color=class_colors[index], linewidth=1.5,
					label=f"{class_name} (AUC = {auc(fpr_grid, mean_tpr):.3f})",
				)
				if len(curves) > 1:
					std = np.std(curves, axis=0)
					axis.fill_between(
						fpr_grid, np.clip(mean_tpr - std, 0, 1), np.clip(mean_tpr + std, 0, 1),
						color=class_colors[index], alpha=0.12, linewidth=0,
					)
			macros = [r[1] for r in fold_results if r[1] is not None]
			if macros:
				mean_macro = np.mean(macros, axis=0)
				axis.plot(
					fpr_grid, mean_macro, color="black", linewidth=2.5, linestyle="--",
					label=f"Macro average (AUC = {auc(fpr_grid, mean_macro):.3f})",
				)
				if len(macros) > 1:
					std = np.std(macros, axis=0)
					axis.fill_between(
						fpr_grid, np.clip(mean_macro - std, 0, 1), np.clip(mean_macro + std, 0, 1),
						color="black", alpha=0.12, linewidth=0,
					)
			axis.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Chance")
			axis.set_xlim(0, 1)
			axis.set_ylim(0, 1.02)
			axis.set_xlabel("False positive rate", fontsize=16)
			axis.set_ylabel("True positive rate", fontsize=16)
			axis.tick_params(labelsize=16)
			axis.set_title(f"Test: {dataset}")
			axis.grid(alpha=0.25)
			axis.legend(loc="lower right", frameon=False, fontsize=8)
		fig.suptitle(f"{model} - ROC curves")
		fig.tight_layout(rect=(0, 0, 1, 0.94))
		safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
		save_figure(fig, ROC_DIR / f"{safe_name}_roc.png")
		plt.close(fig)


SILVER = (0.95, 0.95, 0.95)  # matches \definecolor{Silver}{rgb}{0.87,0.87,0.87}
BUBBLE_METRICS = [
	("recall", "Recall", SILVER),
	("precision", "Precision", "white"),
	("f1", "F1", SILVER),
	("auc", "AUC", "white"),
]
BUBBLE_CLASS_LABELS = {"FRI": "FR I", "FRII": "FR II", "COMPACT": "Compact"}


def load_per_class_metrics():
	"""Return {(model, test_dataset): array[fold, class, metric]}.

	Metrics are in BUBBLE_METRICS order (recall, precision, F1, AUC), followed by one-vs-rest MCC at index 4."""
	results = {}
	if not TESTING_DIR.is_dir():
		return results
	for folder in sorted(path for path in TESTING_DIR.iterdir() if path.is_dir()):
		match = TESTING_FOLDER_PATTERN.match(folder.name)
		y_true_path = folder / "y_test.npy"
		if match is None or not y_true_path.exists():
			continue
		base, target, test_dataset = (DATASET_LABELS.get(name, name) for name in match.groups())
		y_true = to_int_labels(np.load(y_true_path, allow_pickle=True))
		folds = []
		for fold in range(1, 6):
			pred_path = folder / f"y_pred_test_fold_{fold}.npy"
			if not pred_path.exists():
				continue
			y_prob = np.asarray(np.load(pred_path), dtype=float)
			if y_prob.shape != (len(y_true), len(CLASSES)):
				continue
			y_pred = np.argmax(y_prob, axis=1)
			labels = list(range(len(CLASSES)))
			precision, recall, f1, _ = precision_recall_fscore_support(
				y_true, y_pred, labels=labels, zero_division=0
			)
			# Precision is undefined (0/0) for a class that is never predicted.
			precision = np.where(np.bincount(y_pred, minlength=len(labels)) == 0, np.nan, precision)
			aucs, mccs = [], []
			for index in labels:
				positives = y_true == index
				defined = positives.any() and not positives.all()
				aucs.append(roc_auc_score(positives, y_prob[:, index]) if defined else np.nan)
				mccs.append(matthews_corrcoef(positives, y_pred == index) if defined else np.nan)
			folds.append(np.stack([recall, precision, f1, np.array(aucs), np.array(mccs)], axis=1))
		if folds:
			results[(f"{base}->{target}", test_dataset)] = np.stack(folds)
	return results


def best_heatmap_models(test_rows, metric="f1"):
	"""Models with the best Weighted and Average score in the heatmaps (same rules as plot_metric_heatmaps)."""
	models = sorted({row["model"] for row in test_rows})
	datasets = sorted({row["test_dataset"] for row in test_rows})
	grid = np.full((len(models), len(datasets)), np.nan)
	for i, model in enumerate(models):
		for j, dataset in enumerate(datasets):
			values = [
				row[metric] for row in test_rows
				if row["model"] == model and row["test_dataset"] == dataset
			]
			if values and not np.all(np.isnan(values)):
				grid[i, j] = np.nanmean(values)
	weights = np.array([DATASET_WEIGHTS.get(dataset, np.nan) for dataset in datasets])
	best = {}
	for key, scores in (("weighted", grid @ weights), ("average", grid.mean(axis=1))):
		if not np.all(np.isnan(scores)):
			best[key] = models[int(np.nanargmax(scores))]
	return best


def plot_bubble_summaries(best_models=None, output_dir=BUBBLE_DIR):
	"""Bubble plots per test dataset; optionally ring the best Weighted (red) and Average (blue) heatmap models."""
	best_models = best_models or {}
	ring_styles = {
		"weighted": ("red", 150, "Best model (heatmap, weighted)"),
	}
	metrics = load_per_class_metrics()
	if not metrics:
		return
	test_datasets = sorted({dataset for _, dataset in metrics})
	palette = plt.cm.tab10.colors[1:]

	for test_dataset in test_datasets:
		# From-scratch (NONE->) models are listed first, followed by the transfer models.
		transfer = {
			model: values for (model, dataset), values in sorted(
				metrics.items(), key=lambda item: (not item[0][0].startswith("NONE->"), item[0])
			)
			if dataset == test_dataset
		}
		if not transfer:
			continue
		model_names = list(transfer)
		colors = {model: palette[i % len(palette)] for i, model in enumerate(model_names)}

		fig, axes = plt.subplots(
			len(CLASSES), len(BUBBLE_METRICS),
			figsize=(3.2 * len(BUBBLE_METRICS) + 1, 2.6 * len(CLASSES) + 1.6),
			squeeze=False,
		)
		fig.subplots_adjust(left=0.18, right=0.98, top=0.9, bottom=0.27, wspace=0.3, hspace=0.25)
		for row, class_name in enumerate(CLASSES):
			for col, (_, metric_title, background) in enumerate(BUBBLE_METRICS):
				axis = axes[row, col]
				axis.set_facecolor(background)
				axis.set_xlim(0, 1)
				axis.set_ylim(len(model_names) + 0.4, -0.9)
				axis.set_yticks([])
				axis.set_xticks([0, 0.5, 1.0])
				axis.tick_params(labelsize=18)
				axis_width_pts = axis.get_position().width * fig.get_figwidth() * 72
				for model in model_names:
					if not model.startswith("NONE->"):
						continue
					value = np.nanmean(transfer[model][:, row, col])
					if not np.isnan(value):
						axis.axvline(
							value, color=colors[model], zorder=1, linewidth=2,
							linestyle="--" if model == f"NONE->{test_dataset}" else ":",
						)
				for index, model in enumerate(model_names):
					values = transfer[model][:, row, col]
					mean, std = np.nanmean(values), np.nanstd(values)
					if np.isnan(mean):
						continue
					diameter_pts = max(2 * std * axis_width_pts, 0)
					if diameter_pts > 0:
						axis.scatter(
							mean, index, s=diameter_pts ** 2, color=colors[model],
							alpha=0.3, edgecolors=colors[model], linewidths=1, zorder=2,
						)
					axis.scatter(mean, index, s=22, color=colors[model], edgecolors="black",
								 linewidths=0.4, zorder=3)
					for key, (ring_color, ring_size, _) in ring_styles.items():
						if best_models.get(key) == model:
							axis.scatter(mean, index, s=ring_size, facecolors="none", edgecolors=ring_color,
										 linewidths=2.2, zorder=4)
				axis.plot([0.05, 0.15], [-0.6, -0.6], color="black", linewidth=1.2)
				axis.text(0.1, -0.35, "0.1", ha="center", va="center", fontsize=7)
				if row == 0:
					axis.set_title(metric_title, fontsize=22)
				if col == 0:
					axis.set_ylabel(BUBBLE_CLASS_LABELS.get(class_name, class_name),
									rotation=0, labelpad=60, va="center", fontsize=18)

		handles = [
			plt.Line2D([], [], marker="o", linestyle="", color=colors[model], label=model, markersize=8)
			for model in model_names
		]
		for model in model_names:
			if model.startswith("NONE->"):
				own = model == f"NONE->{test_dataset}"
				handles.append(plt.Line2D(
					[], [], color=colors[model], linestyle="--" if own else ":", linewidth=2,
					label=f"{model} mean",
				))
		for key, (ring_color, _, ring_label) in ring_styles.items():
			if best_models.get(key) in model_names:
				handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
										  markeredgecolor=ring_color, markeredgewidth=2, markersize=10,
										  label=f"{ring_label}: {best_models[key]}"))
		fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, fontsize=18)
		fig.suptitle(f"Metric evaluation (test input: {test_dataset})", fontsize=24)
		save_figure(fig, output_dir / f"bubble_metrics_test_{test_dataset}.png")
		plt.close(fig)


BAR_DIR = OUTPUT_DIR / "Metric_Bar_Graphs"

# Tang et al. results (scores on a 0-1 scale). Per-class values were only published for the
# NONE->X models (Tang Xavier, Table 3.3-3.5); transfer models use Tang Method 0 (accuracy/AUC only).
# Compact was not evaluated, and their AUC is a single overall value.
def _tang(accuracy, auc, recall=(None, None), precision=(None, None), f1=(None, None)):
	return {
		"accuracy": accuracy,
		"auc": auc,
		"classes": {
			"FRI": {"recall": recall[0], "precision": precision[0], "f1": f1[0]},
			"FRII": {"recall": recall[1], "precision": precision[1], "f1": f1[1]},
		},
	}


TANG_RESULTS = {
	("NONE->FIRST", "FIRST"): _tang(0.891, 0.94, (0.85, 0.94), (0.95, 0.83), (0.90, 0.88)),
	("NONE->FIRST", "NVSS"): _tang(0.485, 0.54, (0.49, 0.40), (0.92, 0.05), (0.64, 0.09)),
	("NONE->NVSS", "FIRST"): _tang(0.719, 0.78, (0.74, 0.70), (0.67, 0.77), (0.70, 0.73)),
	("NONE->NVSS", "NVSS"): _tang(0.730, 0.80, (0.67, 0.87), (0.92, 0.54), (0.77, 0.67)),
	("FIRST->NVSS", "FIRST"): _tang(0.784, 0.86),
	("FIRST->NVSS", "NVSS"): _tang(0.730, 0.81, (0.66, 0.89), (0.93, 0.53), (0.78, 0.66)),
	("NVSS->FIRST", "FIRST"): _tang(0.874, 0.94, (0.82, 0.95), (0.95, 0.79), (0.89, 0.86)),
	("NVSS->FIRST", "NVSS"): _tang(0.503, 0.59),
}

# Standard deviations of the Tang et al. per-class metrics (Method 0 transfer models), estimated from
# the bubble radii in Tang Fig 3.9 / 3.10 (radius = std; 0.1 scale bar = 0.1 on the axis). Accurate to ~0.01;
# the smallest bubbles sit inside the marker and are given as ~0.01-0.02. AUC is not shown per class.
TANG_STD = {
	("FIRST->NVSS", "NVSS"): {
		"FRI": {"recall": 0.01, "precision": 0.01, "f1": 0.01},
		"FRII": {"recall": 0.035, "precision": 0.03, "f1": 0.045},
	},
	("NVSS->FIRST", "FIRST"): {
		"FRI": {"recall": 0.02, "precision": 0.02, "f1": 0.04},
		"FRII": {"recall": 0.04, "precision": 0.04, "f1": 0.06},
	},
}

# (test dataset, models to compare, include Tang et al.). LoTSS was not tested by Tang et al.
BAR_GRAPH_SETS = [
	("NVSS", ["NONE->FIRST", "NVSS->FIRST", "NONE->NVSS", "FIRST->NVSS"], True),
	("FIRST", ["NONE->FIRST", "NVSS->FIRST", "NONE->NVSS", "FIRST->NVSS"], True),
	("LoTSS", ["NONE->LoTSS", "FIRST->LoTSS", "NVSS->LoTSS"], False),
]


TANG_BUBBLE_MODELS = ["NONE->FIRST", "NVSS->FIRST", "NONE->NVSS", "FIRST->NVSS"]


def plot_tang_bubble_summaries(output_dir=BUBBLE_DIR):
	"""Bubble plots per test dataset comparing this work (mean +/- std bubbles) with Tang et al. (diamonds)."""
	metrics = load_per_class_metrics()
	palette = plt.cm.tab10.colors[1:]
	colors = {model: palette[i] for i, model in enumerate(TANG_BUBBLE_MODELS)}
	for test_dataset in ("FIRST", "NVSS"):
		models = [m for m in TANG_BUBBLE_MODELS if (m, test_dataset) in metrics]
		if not models:
			continue
		n = len(models)
		fig, axes = plt.subplots(
			len(CLASSES), len(BUBBLE_METRICS),
			figsize=(3.2 * len(BUBBLE_METRICS) + 1, 2.6 * len(CLASSES) + 1.6), squeeze=False,
		)
		fig.subplots_adjust(left=0.18, right=0.98, top=0.9, bottom=0.27, wspace=0.3, hspace=0.25)
		for row, class_name in enumerate(CLASSES):
			for col, (metric_key, metric_title, background) in enumerate(BUBBLE_METRICS):
				axis = axes[row, col]
				axis.set_facecolor(background)
				axis.set_xlim(0, 1)
				axis.set_ylim(2 * n + 0.4, -0.9)
				axis.set_yticks([])
				axis.set_xticks([0, 0.5, 1.0])
				axis.tick_params(labelsize=18)
				for model in models:
					if not model.startswith("NONE->"):
						continue
					own_mean = np.nanmean(metrics[(model, test_dataset)][:, row, col])
					scores = TANG_RESULTS.get((model, test_dataset))
					tang_value = scores["classes"].get(class_name, {}).get(metric_key) if scores else None
					for line_value in (own_mean, tang_value):
						if line_value is not None and not np.isnan(line_value):
							axis.axvline(line_value, color=colors[model], linestyle=":", linewidth=0.8, zorder=1)
				axis_width_pts = axis.get_position().width * fig.get_figwidth() * 72
				for index, model in enumerate(models):
					values = metrics[(model, test_dataset)][:, row, col]
					mean, std = np.nanmean(values), np.nanstd(values)
					if not np.isnan(mean):
						diameter_pts = max(2 * std * axis_width_pts, 0)
						if diameter_pts > 0:
							axis.scatter(mean, index, s=diameter_pts ** 2, color=colors[model], alpha=0.3,
										 edgecolors=colors[model], linewidths=1, zorder=2)
						axis.scatter(mean, index, s=22, color=colors[model], edgecolors="black",
									 linewidths=0.4, zorder=3)
					scores = TANG_RESULTS.get((model, test_dataset))
					tang = scores["classes"].get(class_name, {}).get(metric_key) if scores else None
					if tang is not None:
						tang_std = TANG_STD.get((model, test_dataset), {}).get(class_name, {}).get(metric_key)
						if tang_std:
							axis.scatter(tang, n + index, s=(2 * tang_std * axis_width_pts) ** 2, color=colors[model],
										 alpha=0.3, edgecolors=colors[model], linewidths=1, zorder=2)
						axis.scatter(tang, n + index, s=40, marker="D", color=colors[model],
									 edgecolors="black", linewidths=0.6, zorder=3)
				axis.plot([0.05, 0.15], [-0.6, -0.6], color="black", linewidth=1.2)
				axis.text(0.1, -0.35, "0.1", ha="center", va="center", fontsize=7)
				if row == 0:
					axis.set_title(metric_title, fontsize=22)
				if col == 0:
					axis.set_ylabel(BUBBLE_CLASS_LABELS.get(class_name, class_name),
									rotation=0, labelpad=60, va="center", fontsize=18)
		handles = [
			plt.Line2D([], [], marker="o", linestyle="", color=colors[m], label=f"{m} (this work)", markersize=8)
			for m in models
		] + [
			plt.Line2D([], [], marker="D", linestyle="", color=colors[m], markeredgecolor="black",
					   label=f"{m} (Tang et al.)", markersize=7)
			for m in models
		]
		fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, fontsize=16)
		fig.suptitle(f"This work vs Tang et al. (test input: {test_dataset})", fontsize=24)
		save_figure(fig, output_dir / f"bubble_metrics_tang_comparison_{test_dataset}.png")
		plt.close(fig)


LOTSS_FIRST_BAR_SETS = [
	("FIRST", ["NONE->FIRST", "LoTSS->FIRST", "NONE->LoTSS", "FIRST->LoTSS"], False),
	("LoTSS", ["NONE->FIRST", "LoTSS->FIRST", "NONE->LoTSS", "FIRST->LoTSS"], False),
]
LOTSS_NVSS_BAR_SETS = [
	("NVSS", ["NONE->NVSS", "LoTSS->NVSS", "NONE->LoTSS", "NVSS->LoTSS"], False),
	("LoTSS", ["NONE->NVSS", "LoTSS->NVSS", "NONE->LoTSS", "NVSS->LoTSS"], False),
]
FIRST_NVSS_BAR_SETS = [
	("FIRST", ["NONE->FIRST", "NVSS->FIRST", "NONE->NVSS", "FIRST->NVSS"], False),
	("NVSS", ["NONE->FIRST", "NVSS->FIRST", "NONE->NVSS", "FIRST->NVSS"], False),
]
# (graph sets, combined single-figure file name)
PAIRED_BAR_GRAPHS = [
	(LOTSS_FIRST_BAR_SETS, "bar_metrics_lotss_first_combined.png"),
	(LOTSS_NVSS_BAR_SETS, "bar_metrics_lotss_nvss_combined.png"),
	(FIRST_NVSS_BAR_SETS, "bar_metrics_first_nvss_combined.png"),
]


def plot_paired_bar_graphs(test_rows):
	for graph_sets, combined_name in PAIRED_BAR_GRAPHS:
		plot_bar_graphs(test_rows, graph_sets, combined_name=combined_name)


def plot_bar_graphs(test_rows, graph_sets=BAR_GRAPH_SETS, file_prefix="bar_metrics_test_", combined_name=None):
	"""Grouped bar graphs of all metrics, models interleaved per metric, optionally with Tang et al.

	With combined_name, all test sets are stacked as separate panels of one figure sharing a legend."""
	metrics = load_per_class_metrics()
	if not metrics:
		return

	# Each group: (label, own getter, Tang getter, section id for separators, Tang value inferred from figures)
	groups = []
	for class_index, class_name in enumerate(CLASSES):
		for metric_index, (metric_key, metric_title, _) in enumerate(BUBBLE_METRICS):
			groups.append((
				f"{BUBBLE_CLASS_LABELS.get(class_name, class_name)}\n{metric_title}",
				lambda model, dataset, c=class_index, m=metric_index: metrics[(model, dataset)][:, c, m],
				lambda scores, c=class_name, k=metric_key: scores["classes"].get(c, {}).get(k),
				class_index,
				True,
			))
	for key, paper_key, title in (("accuracy", "accuracy", "Accuracy"), ("roc_auc", "auc", "AUC\n(macro)")):
		groups.append((
			f"Overall\n{title}",
			lambda model, dataset, k=key: np.array([
				row[k] for row in test_rows
				if row["model"] == model and row["test_dataset"] == dataset
			], dtype=float),
			lambda scores, k=paper_key: scores.get(k),
			len(CLASSES),
			False,
		))

	graph_sets = [
		(dataset, [m for m in models if (m, dataset) in metrics], with_paper)
		for dataset, models, with_paper in graph_sets
	]
	graph_sets = [entry for entry in graph_sets if entry[1]]
	if not graph_sets:
		return

	fig_width = 1.45 * len(groups) + 2
	combined_axes = None
	if combined_name:
		fig, combined_axes = plt.subplots(
			len(graph_sets), 1, figsize=(fig_width, 6 * len(graph_sets) + 3), squeeze=False
		)
	combined_handles = None

	for set_index, (test_dataset, models, with_paper) in enumerate(graph_sets):
		series = [(model, "own") for model in models]
		if with_paper:
			series += [(model, "paper") for model in models]

		colors = plt.cm.tab10.colors
		bar_width = 0.8 / len(series)
		if combined_axes is not None:
			axis = combined_axes[set_index, 0]
		else:
			fig, axis = plt.subplots(figsize=(fig_width, 6.5))
		for series_index, (model, kind) in enumerate(series):
			color = colors[models.index(model) % len(colors)]
			means, errors = [], []
			for _, own_getter, paper_getter, _, _ in groups:
				if kind == "own":
					values = own_getter(model, test_dataset)
					means.append(np.nanmean(values) if len(values) else np.nan)
					errors.append(np.nanstd(values) if len(values) else 0)
				else:
					scores = TANG_RESULTS.get((model, test_dataset))
					score = paper_getter(scores) if scores else None
					means.append(np.nan if score is None else score)
					errors.append(0)
			positions = np.arange(len(groups)) - 0.4 + bar_width * (series_index + 0.5)
			if kind == "own":
				axis.bar(
					positions, means, bar_width, yerr=errors, capsize=2, color=color,
					edgecolor="black", linewidth=0.5,
				)
			else:
				# Tang et al. accuracy/AUC come from tables; per-class values are read off figures.
				for position, mean, group in zip(positions, means, groups):
					axis.bar(
						position, mean, bar_width, color=color, alpha=0.55,
						hatch="//" if group[4] else None, edgecolor="black", linewidth=0.5,
					)

		axis.set_xticks(np.arange(len(groups)), [group[0] for group in groups], fontsize=18)
		axis.set_ylim(0, 1.05)
		axis.set_ylabel("Score", fontsize=18)
		axis.grid(axis="y", alpha=0.25)
		for index in range(1, len(groups)):
			if groups[index][3] != groups[index - 1][3]:
				axis.axvline(index - 0.5, color="grey", linewidth=0.8)
		handles = []
		for model, kind in series:
			color = colors[models.index(model) % len(colors)]
			if kind == "own":
				handles.append(Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=model))
			else:
				handles.append(Patch(
					facecolor=color, alpha=0.55, edgecolor="black", linewidth=0.5,
					label=f"{model} (Tang et al.)",
				))
				handles.append(Patch(
					facecolor=color, alpha=0.55, edgecolor="black", linewidth=0.5, hatch="//",
					label=f"{model} (Tang et al., inferred from figures)",
				))
		legend_columns = 3 if with_paper else len(series)
		if combined_axes is not None:
			combined_handles = (handles, legend_columns)
			axis.set_title(f"All metrics - test input: {test_dataset}")
			continue
		axis.legend(handles=handles, frameon=False, ncol=legend_columns,
					loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=14 if with_paper else 18)
		axis.set_title(f"All metrics - test input: {test_dataset}")
		fig.tight_layout()
		BAR_DIR.mkdir(parents=True, exist_ok=True)
		fig.savefig(BAR_DIR / f"{file_prefix}{test_dataset}.png", dpi=180, bbox_inches="tight")
		plt.close(fig)

	if combined_axes is not None:
		handles, legend_columns = combined_handles
		fig.legend(handles=handles, frameon=False, ncol=legend_columns, loc="lower center", fontsize=18)
		fig.tight_layout(rect=(0, 0.1, 1, 1))
		save_figure(fig, BAR_DIR / combined_name)
		plt.close(fig)


LATEX_DIR = OUTPUT_DIR / "LaTeX_Tables"
LATEX_DATASETS = ["FIRST", "NVSS", "LoTSS"]
LATEX_ACRONYMS = {"FIRST": r"\ac{first}", "NVSS": r"\ac{nvss}", "LoTSS": r"\ac{lotss}"}


def _latex_cell(values):
	"""Stack the FRI / FRII / Compact values under one another inside a tblr cell."""
	return "{" + " \\\\ ".join("--" if np.isnan(v) else f"{v:.3f}" for v in values) + "}"


def build_latex_results_table(test_rows, per_class, bases, caption, label, filename):
	"""Write a tblr table; accuracy and overall AUC are single values, class metrics are stacked."""
	acc = {}
	overall_auc = {}
	for row in test_rows:
		acc.setdefault((row["model"], row["test_dataset"]), []).append(row["accuracy"])
		overall_auc.setdefault((row["model"], row["test_dataset"]), []).append(row["roc_auc"])

	groups = [(base, target) for base in bases for target in LATEX_DATASETS if base != target]
	n = len(LATEX_DATASETS)
	body = []
	for base, target in groups:
		model = f"{base}->{target}"
		if base == "NONE":
			label_text = LATEX_ACRONYMS[target]
		else:
			label_text = f"{LATEX_ACRONYMS[base]}$\\rightarrow${LATEX_ACRONYMS[target]}"
		for k, test_set in enumerate(LATEX_DATASETS):
			prefix = f"\\begin{{sideways}}{label_text}\\end{{sideways}}\n    & " if k == 0 else "    & "
			cells = [LATEX_ACRONYMS[test_set].ljust(10)]
			accuracies = acc.get((model, test_set))
			cells.append("--" if not accuracies else f"{np.nanmean(accuracies):.3f}")
			auc_values = overall_auc.get((model, test_set))
			cells.append("--" if not auc_values else f"{np.nanmean(auc_values):.3f}")
			metrics = per_class.get((model, test_set))
			if metrics is None:
				cells += ["--"] * 4
			else:
				mean = np.nanmean(metrics, axis=0)  # [class, metric]: recall, precision, f1, auc
				for metric_index in (3, 0, 1, 2):
					cells.append(_latex_cell(mean[:, metric_index]))
			body.append(prefix + " & ".join(cells) + " \\\\")
	body[-1] = body[-1][:-3]

	starts = [2 + n * g for g in range(len(groups))]
	last_row = 1 + n * len(groups)
	spec = []
	for row in range(3, last_row + 1, 2):
		if row not in starts:
			spec.append(f"row{{{row}}} = {{Silver}}")
	for g, s in enumerate(starts):
		spec.append(f"cell{{{s}}}{{1}} = {{r={n}}}{{{'Silver' if g % 2 == 0 else ''}}}")
		if g % 2 == 1:
			spec += [f"cell{{{s}}}{{{c}}} = {{Silver}}" for c in range(2, 9)]
	outer = ",".join(["1-2"] + [str(s) for s in starts[1:]] + [str(last_row + 1)])
	inner = ",".join(f"{s + 1}-{s + 2}" for s in starts)
	spec = ",\n  ".join(spec)
	text = (
		"\\definecolor{Silver}{rgb}{0.87,0.87,0.87}\n"
		"\\begin{table}\n"
		f"\\caption{{{caption}}}\n"
		"\\centering\n"
		"\\begin{tblr}{\n"
		"   cells = {c},\n"
		f"  {spec},\n"
		"  vlines,\n"
		f"  hline{{{outer}}} = {{-}}{{}},\n"
		f"  hline{{{inner}}} = {{2-8}}{{}},\n"
		"}\n"
		"Model & Test Set & Accuracy & Overall AUC & AUC & Recall & Precision & F1 \\\\\n"
		+ "\n".join(body)
		+ "\n\\end{tblr}\n"
		f"\\label{{{label}}}\n"
		"\\end{table}\n"
	)
	LATEX_DIR.mkdir(parents=True, exist_ok=True)
	(LATEX_DIR / filename).write_text(text, encoding="utf-8")


WINNER_COLUMNS = 14  # 3 classes x (recall, precision, F1, AUC) + overall accuracy + overall AUC


def _winner_marks(vectors):
	"""Mark the best model per column with X (T on a tie, N/A where a value is missing)."""
	marks = [[""] * WINNER_COLUMNS for _ in vectors]
	for column in range(WINNER_COLUMNS):
		values = [vector[column] for vector in vectors]
		valid = [k for k, value in enumerate(values) if not np.isnan(value)]
		for k in range(len(values)):
			if k not in valid:
				marks[k][column] = "N/A"
		if len(valid) < 2:
			continue
		best = max(values[k] for k in valid)
		winners = [k for k in valid if values[k] == best]
		for k in winners:
			marks[k][column] = "T" if len(winners) > 1 else "X"
	return marks


def build_winner_table(test_rows, per_class, graph_sets, filename):
	"""Write a tblr table of every model on every test set in graph_sets.

	Within each test set the models are grouped by the dataset they were (fine-)tuned on; the models in a
	group are compared with each other, the better one per metric is marked with X, and Score counts the
	X marks. The in-domain group (fine-tuned on the test set itself) is listed first."""
	overall = {}
	for row in test_rows:
		overall.setdefault((row["model"], row["test_dataset"]), []).append((row["accuracy"], row["roc_auc"]))

	def score_vector(model, test_set):
		"""Class metrics (recall, precision, F1, AUC per class) followed by overall accuracy and AUC."""
		metrics, totals = per_class.get((model, test_set)), overall.get((model, test_set))
		if metrics is None or not totals:
			return [np.nan] * WINNER_COLUMNS
		mean = np.nanmean(metrics[:, :, :4], axis=0)  # recall, precision, F1, AUC per class
		return [round(float(v), 3) for v in mean.reshape(-1)] + [
			round(float(v), 3) for v in np.nanmean(totals, axis=0)
		]

	body = []
	set_starts, group_starts, model_rows = [], [], []
	spec = []
	row_number = 3  # rows 1-2 are the header
	for test_set, models, _ in graph_sets:
		by_target = {}
		for model in models:
			by_target.setdefault(model.split("->")[1], []).append(model)
		ordered_groups = sorted(by_target.items(), key=lambda item: item[0] != test_set)

		set_start = row_number
		set_starts.append(set_start)
		for _, group in ordered_groups:
			group_starts.append(row_number)
			marks = _winner_marks([score_vector(model, test_set) for model in group])
			for k, model in enumerate(group):
				if k > 0:
					model_rows.append(row_number)
				base, target = model.split("->")
				base_text = "NONE" if base == "NONE" else LATEX_ACRONYMS[base]
				cells = [
					LATEX_ACRONYMS[test_set] if row_number == set_start else "",
					f"{base_text}$\\rightarrow${LATEX_ACRONYMS[target]}",
				]
				cells += marks[k] + [str(marks[k].count("X"))]
				body.append(" & ".join(cells) + " \\\\")
				row_number += 1
		spec.append(f"cell{{{set_start}}}{{1}} = {{r={row_number - set_start}}}{{}}")
	body[-1] = body[-1][:-3]
	end_row = row_number

	full_lines = ",".join(str(r) for r in [1] + set_starts + [end_row])
	group_lines = [r for r in group_starts if r not in set_starts]
	hlines = [f"  hline{{{full_lines}}} = {{-}}{{}},", "  hline{2} = {3-14}{},"]
	if group_lines:
		hlines.append(f"  hline{{{','.join(map(str, group_lines))}}} = {{2-17}}{{}},")
	if model_rows:
		hlines.append(f"  hline{{{','.join(map(str, model_rows))}}} = {{2-17}}{{dashed}},")

	text = (
		"% \\usepackage{tabularray}\n"
		"\\begin{table}\n"
		"\\centering\n"
		"\\begin{tblr}{\n"
		"  cells = {c},\n"
		"  cell{1}{1} = {r=2}{},\n"
		"  cell{1}{2} = {r=2}{},\n"
		"  cell{1}{3} = {c=4}{c},\n"
		"  cell{1}{7} = {c=4}{c},\n"
		"  cell{1}{11} = {c=4}{c},\n"
		"  cell{1}{15} = {r=2}{c},\n"
		"  cell{1}{16} = {r=2}{c},\n"
		"  cell{1}{17} = {r=2}{c},\n"
		+ "".join(f"  {entry},\n" for entry in spec)
		+ "  vlines,\n"
		+ "\n".join(hlines) + "\n"
		"}\n"
		"Test Set & Model & FRI &  &  &  & FRII &  &  &  & Compact &  &  &  & Overall Acc & Overall AUC & Score \\\\\n"
		" &  & Rec & Prec & F1 & AUC & Rec & Prec & F1 & AUC & Rec & Prec & F1 & AUC &  &  &  \\\\\n"
		+ "\n".join(body)
		+ "\n\\end{tblr}\n"
		"\\end{table}\n"
	)
	LATEX_DIR.mkdir(parents=True, exist_ok=True)
	(LATEX_DIR / filename).write_text(text, encoding="utf-8")


# Every model on every test set: within a test set, each fine-tune group holds NONE->X and both A->X models.
ALL_WINNER_MODELS = [
	f"{base}->{target}" for target in LATEX_DATASETS
	for base in ["NONE"] + LATEX_DATASETS if base != target
]
ALL_WINNER_SETS = [(test_set, ALL_WINNER_MODELS, False) for test_set in LATEX_DATASETS]


def write_winner_tables(test_rows):
	per_class = load_per_class_metrics()
	for graph_sets, combined_name in PAIRED_BAR_GRAPHS:
		name = combined_name.replace("bar_metrics_", "").replace("_combined.png", "")
		build_winner_table(test_rows, per_class, graph_sets, f"winner_table_{name}.tex")
	build_winner_table(test_rows, per_class, ALL_WINNER_SETS, "winner_table_all.tex")


def write_roc_auc_table(filename="roc_macro_auc_table.tex"):
	"""Macro AUC of the mean macro-average ROC curve (the value in the ROC plot legends) for every model and test set."""
	if not TESTING_DIR.is_dir():
		return
	fpr_grid = np.linspace(0, 1, 201)
	values = {}
	for folder in sorted(path for path in TESTING_DIR.iterdir() if path.is_dir()):
		match = TESTING_FOLDER_PATTERN.match(folder.name)
		y_true_path = folder / "y_test.npy"
		if match is None or not y_true_path.exists():
			continue
		base, target, test_dataset = (DATASET_LABELS.get(name, name) for name in match.groups())
		y_true = to_int_labels(np.load(y_true_path, allow_pickle=True))
		macros = []
		for fold in range(1, 6):
			pred_path = folder / f"y_pred_test_fold_{fold}.npy"
			if not pred_path.exists():
				continue
			y_prob = np.asarray(np.load(pred_path), dtype=float)
			if y_prob.shape != (len(y_true), len(CLASSES)):
				continue
			macro = macro_roc_for_fold(y_true, y_prob, fpr_grid)[1]
			if macro is not None:
				macros.append(macro)
		if macros:
			values[(f"{base}->{target}", test_dataset)] = auc(fpr_grid, np.mean(macros, axis=0))

	models = [f"NONE->{t}" for t in LATEX_DATASETS] + [
		f"{b}->{t}" for b in LATEX_DATASETS for t in LATEX_DATASETS if b != t
	]
	models = [m for m in models if any((m, d) in values for d in LATEX_DATASETS)]
	if not models:
		return

	def model_label(model):
		base, target = model.split("->")
		if base == "NONE":
			return f"NONE$\\rightarrow${LATEX_ACRONYMS[target]}"
		return f"{LATEX_ACRONYMS[base]}$\\rightarrow${LATEX_ACRONYMS[target]}"

	body = []
	for model in models:
		cells = [model_label(model)] + [
			f"{values[(model, d)]:.3f}" if (model, d) in values else "--" for d in LATEX_DATASETS
		]
		body.append(" & ".join(cells) + " \\\\")
	body[-1] = body[-1][:-3]
	shading = ",\n  ".join(f"row{{{row}}} = {{Silver}}" for row in range(3, len(models) + 2, 2))
	text = (
		"\\definecolor{Silver}{rgb}{0.87,0.87,0.87}\n"
		"\\begin{table}\n"
		"\\caption{Macro-average AUC of the ROC curves for every model (rows; NONE$\\rightarrow$X "
		"trained from scratch, A$\\rightarrow$B transfer-learned) on each test set (columns).}\n"
		"\\centering\n"
		"\\begin{tblr}{\n"
		"  cells = {c},\n"
		+ (f"  {shading},\n" if shading else "")
		+ "  vlines,\n"
		f"  hline{{1-2,{len(models) + 2}}} = {{-}}{{}},\n"
		"}\n"
		"Model & " + " & ".join(LATEX_ACRONYMS[d] for d in LATEX_DATASETS) + " \\\\\n"
		+ "\n".join(body)
		+ "\n\\end{tblr}\n"
		"\\label{tab:rocauc}\n"
		"\\end{table}\n"
	)
	LATEX_DIR.mkdir(parents=True, exist_ok=True)
	(LATEX_DIR / filename).write_text(text, encoding="utf-8")


def write_top_models(test_rows, metric="f1", top_n=3, filename="top_models_per_test_set.tex"):
	"""Rank every model on each test set by the fold-mean of `metric`; write the top_n as CSV, LaTeX and console output."""
	metric_titles = {
		"accuracy": "Accuracy", "precision": "Precision (macro)", "roc_auc": "ROC AUC (macro)", "f1": "F1 (macro)",
		"mcc": "MCC",
	}
	metric_keys = list(metric_titles)
	summary = {}
	for row in test_rows:
		summary.setdefault((row["test_dataset"], row["model"]), []).append([row[k] for k in metric_keys])

	csv_rows, body, set_starts = [], [], []
	row_number = 2
	test_sets = [d for d in LATEX_DATASETS if any(key[0] == d for key in summary)]
	for test_set in test_sets:
		ranked = []
		for (dataset, model), values in summary.items():
			if dataset != test_set:
				continue
			values = np.asarray(values, dtype=float)
			means = dict(zip(metric_keys, np.nanmean(values, axis=0)))
			stds = dict(zip(metric_keys, np.nanstd(values, axis=0)))
			if not np.isnan(means[metric]):
				ranked.append((model, means, stds))
		ranked.sort(key=lambda item: item[1][metric], reverse=True)
		ranked = ranked[:top_n]
		if not ranked:
			continue

		print(f"Top {len(ranked)} models on {test_set} (by {metric_titles[metric]}):")
		set_starts.append(row_number)
		for rank, (model, means, stds) in enumerate(ranked, start=1):
			print(f"  {rank}. {model:<14} " + "  ".join(
				f"{k}={means[k]:.3f}+/-{stds[k]:.3f}" for k in metric_keys
			))
			csv_row = {"test_dataset": test_set, "rank": rank, "model": model, "ranked_by": metric}
			for k in metric_keys:
				csv_row[f"{k}_mean"], csv_row[f"{k}_std"] = means[k], stds[k]
			csv_rows.append(csv_row)

			base, target = model.split("->")
			base_text = "NONE" if base == "NONE" else LATEX_ACRONYMS.get(base, base)
			cells = [
				LATEX_ACRONYMS.get(test_set, test_set) if rank == 1 else "",
				str(rank),
				f"{base_text}$\\rightarrow${LATEX_ACRONYMS.get(target, target)}",
			]
			for k in metric_keys:
				value = f"{means[k]:.3f} $\\pm$ {stds[k]:.3f}"
				cells.append(f"\\textbf{{{value}}}" if k == metric else value)
			body.append(" & ".join(cells) + " \\\\")
			row_number += 1
	if not body:
		return

	write_csv(
		CSV_DIR / "top_models_per_test_set.csv", csv_rows,
		["test_dataset", "rank", "model", "ranked_by"] + [f"{k}_{s}" for k in metric_keys for s in ("mean", "std")],
	)
	body[-1] = body[-1][:-3]
	spans = [
		f"  cell{{{start}}}{{1}} = {{r={end - start}}}{{}},"
		for start, end in zip(set_starts, set_starts[1:] + [row_number])
	]
	text = (
		"% \\usepackage{tabularray}\n"
		"\\begin{table}\n"
		f"\\caption{{Top {top_n} models on each test set, ranked by fold-mean {metric_titles[metric]} "
		"(bold). Values are mean $\\pm$ standard deviation over the folds.}\n"
		"\\centering\n"
		"\\begin{tblr}{\n"
		"  cells = {c},\n"
		+ "\n".join(spans) + "\n"
		"  vlines,\n"
		f"  hline{{1,2,{','.join(map(str, set_starts[1:] + [row_number]))}}} = {{-}}{{}},\n"
		"}\n"
		"Test Set & Rank & Model & " + " & ".join(metric_titles[k] for k in metric_keys) + " \\\\\n"
		+ "\n".join(body)
		+ "\n\\end{tblr}\n"
		"\\label{tab:topmodels}\n"
		"\\end{table}\n"
	)
	LATEX_DIR.mkdir(parents=True, exist_ok=True)
	(LATEX_DIR / filename).write_text(text, encoding="utf-8")


def write_latex_tables(test_rows):
	write_winner_tables(test_rows)
	write_top_models(test_rows)
	write_roc_auc_table()
	per_class = load_per_class_metrics()
	scratch_caption = (
		"Results of models trained from scratch, evaluated on all three test sets. Accuracy and "
		"overall AUC are single values; class-specific metrics indicate results for \\ac{fri}, "
		"\\ac{frii} and Compact classes respectively."
	)
	transfer_caption = (
		"Results of transfer-learned models (rows labelled source$\\rightarrow$target), evaluated "
		"on all three test sets. Accuracy and overall AUC are single values; class-specific metrics "
		"indicate results for \\ac{fri}, \\ac{frii} and Compact classes respectively."
	)
	build_latex_results_table(
		test_rows, per_class, ["NONE"], scratch_caption, "tab:scratchresults", "scratch_results_table.tex"
	)
	build_latex_results_table(
		test_rows, per_class, LATEX_DATASETS, transfer_caption, "tab:transferresults",
		"transfer_results_table.tex",
	)
	print(f"LaTeX tables saved to {LATEX_DIR}")


BEST_MODEL_DIR = OUTPUT_DIR / "Best_Model"
# Validated categorical palette (blue, orange, aqua): passes colour-blind separation checks.
BEST_CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
BEST_COMPARE_COLORS = {"selected": "#2a78d6", "baseline": "#eb6834"}
BEST_MACRO_METRICS = [
	("accuracy", "Accuracy"), ("precision", "Precision\n(macro)"), ("roc_auc", "ROC AUC\n(macro)"),
	("f1", "F1\n(macro)"), ("mcc", "MCC"),
]
BEST_CLASS_METRICS = [("Recall", 0), ("Precision", 1), ("F1", 2), ("AUC", 3), ("MCC", 4)]


def load_model_predictions(model):
	"""Return {test_dataset: (y_true, [y_prob per fold])} for one model."""
	predictions = {}
	if not TESTING_DIR.is_dir():
		return predictions
	for folder in sorted(path for path in TESTING_DIR.iterdir() if path.is_dir()):
		match = TESTING_FOLDER_PATTERN.match(folder.name)
		y_true_path = folder / "y_test.npy"
		if match is None or not y_true_path.exists():
			continue
		base, target, test_dataset = (DATASET_LABELS.get(name, name) for name in match.groups())
		if f"{base}->{target}" != model:
			continue
		y_true = to_int_labels(np.load(y_true_path, allow_pickle=True))
		folds = []
		for fold in range(1, 6):
			pred_path = folder / f"y_pred_test_fold_{fold}.npy"
			if pred_path.exists():
				y_prob = np.asarray(np.load(pred_path), dtype=float)
				if y_prob.shape == (len(y_true), len(CLASSES)):
					folds.append(y_prob)
		if folds:
			predictions[test_dataset] = (y_true, folds)
	return predictions


def _fold_stats(test_rows, model, test_set, metric):
	values = np.array([
		row[metric] for row in test_rows if row["model"] == model and row["test_dataset"] == test_set
	], dtype=float)
	if not len(values) or np.all(np.isnan(values)):
		return np.nan, np.nan
	return float(np.nanmean(values)), float(np.nanstd(values))


def plot_best_model_summary(test_rows, best_model, ranking_metric="f1"):
	"""Final summary tables and plots for the selected model (best weighted macro F1 by default)."""
	if not best_model:
		return
	BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
	per_class = load_per_class_metrics()
	predictions = load_model_predictions(best_model)
	test_sets = [d for d in LATEX_DATASETS if (best_model, d) in per_class]
	if not test_sets:
		return
	metric_keys = [key for key, _ in BEST_MACRO_METRICS]
	all_models = sorted({row["model"] for row in test_rows})

	# --- Macro summary per test set, with rank among all models and the Average/Weighted aggregates.
	summary_rows = []
	for test_set in test_sets:
		row = {"model": best_model, "test_dataset": test_set}
		for key in metric_keys:
			row[f"{key}_mean"], row[f"{key}_std"] = _fold_stats(test_rows, best_model, test_set, key)
		scores = {m: _fold_stats(test_rows, m, test_set, ranking_metric)[0] for m in all_models}
		ordered = sorted((m for m in scores if not np.isnan(scores[m])), key=lambda m: -scores[m])
		row["rank"] = ordered.index(best_model) + 1 if best_model in ordered else ""
		row["models_ranked"] = len(ordered)
		summary_rows.append(row)
	weights = np.array([DATASET_WEIGHTS.get(d, np.nan) for d in test_sets])
	per_set_rows = list(summary_rows)
	for label, combine in (("Average", np.mean), ("Weighted", lambda v: float(v @ weights))):
		row = {"model": best_model, "test_dataset": label}
		for key in metric_keys:
			means = np.array([r[f"{key}_mean"] for r in per_set_rows], dtype=float)
			row[f"{key}_mean"] = combine(means) if len(test_sets) == len(LATEX_DATASETS) else np.nan
			row[f"{key}_std"] = ""
		summary_rows.append(row)
	write_csv(
		BEST_MODEL_DIR / "best_model_summary.csv", summary_rows,
		["model", "test_dataset", "rank", "models_ranked"]
		+ [f"{k}_{s}" for k in metric_keys for s in ("mean", "std")],
	)

	# --- Per-class summary.
	class_rows = []
	for test_set in test_sets:
		values = per_class[(best_model, test_set)]
		for class_index, class_name in enumerate(CLASSES):
			row = {"model": best_model, "test_dataset": test_set, "class": class_name}
			for name, metric_index in BEST_CLASS_METRICS:
				column = values[:, class_index, metric_index]
				all_nan = np.all(np.isnan(column))
				row[f"{name.lower()}_mean"] = np.nan if all_nan else float(np.nanmean(column))
				row[f"{name.lower()}_std"] = np.nan if all_nan else float(np.nanstd(column))
			class_rows.append(row)
	write_csv(
		BEST_MODEL_DIR / "best_model_per_class.csv", class_rows,
		["model", "test_dataset", "class"]
		+ [f"{name.lower()}_{s}" for name, _ in BEST_CLASS_METRICS for s in ("mean", "std")],
	)

	# --- LaTeX summary table.
	base, target = best_model.split("->")
	model_tex = (
		f"{'NONE' if base == 'NONE' else LATEX_ACRONYMS.get(base, base)}"
		f"$\\rightarrow${LATEX_ACRONYMS.get(target, target)}"
	)
	body = []
	for row in summary_rows:
		label = LATEX_ACRONYMS.get(row["test_dataset"], row["test_dataset"])
		cells = [label]
		for key in metric_keys:
			mean, std = row[f"{key}_mean"], row[f"{key}_std"]
			if np.isnan(mean):
				cells.append("--")
			elif std == "":
				cells.append(f"{mean:.3f}")
			else:
				cells.append(f"{mean:.3f} $\\pm$ {std:.3f}")
		cells.append(f"{row['rank']}/{row['models_ranked']}" if row.get("rank") else "--")
		body.append(" & ".join(cells) + " \\\\")
	body[-1] = body[-1][:-3]
	text = (
		"% \\usepackage{tabularray}\n"
		"\\begin{table}\n"
		f"\\caption{{Final results of the selected model ({model_tex}), chosen by the highest "
		"dataset-weighted macro F1. Values are fold mean $\\pm$ standard deviation; Average and "
		"Weighted combine the three test sets. Rank is the model's position by macro F1 among all "
		"models on that test set.}\n"
		"\\centering\n"
		"\\begin{tblr}{\n"
		"  cells = {c},\n"
		"  vlines,\n"
		f"  hline{{1,2,{len(test_sets) + 2},{len(summary_rows) + 2}}} = {{-}}{{}},\n"
		"}\n"
		"Test Set & Accuracy & Precision (macro) & ROC AUC (macro) & F1 (macro) & MCC & Rank \\\\\n"
		+ "\n".join(body)
		+ "\n\\end{tblr}\n"
		"\\label{tab:bestmodel}\n"
		"\\end{table}\n"
	)
	(BEST_MODEL_DIR / "best_model_summary.tex").write_text(text, encoding="utf-8")

	class_labels = [BUBBLE_CLASS_LABELS.get(c, c) for c in CLASSES]
	n = len(test_sets)

	# --- Confusion matrices: row-normalised per fold, then averaged over folds.
	if predictions:
		fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 7), squeeze=False)
		for axis, test_set in zip(axes[0], test_sets):
			if test_set not in predictions:
				axis.set_visible(False)
				continue
			y_true, folds = predictions[test_set]
			matrices, counts = [], []
			for y_prob in folds:
				y_pred = np.argmax(y_prob, axis=1)
				matrix = np.zeros((len(CLASSES), len(CLASSES)))
				np.add.at(matrix, (y_true, y_pred), 1)
				counts.append(matrix)
				totals = matrix.sum(axis=1, keepdims=True)
				matrices.append(np.divide(matrix, totals, out=np.full_like(matrix, np.nan), where=totals > 0))
			fraction = np.nanmean(matrices, axis=0)
			mean_counts = np.mean(counts, axis=0)
			axis.imshow(np.ma.masked_invalid(fraction), cmap="Blues", vmin=0, vmax=1)
			for i in range(len(CLASSES)):
				for j in range(len(CLASSES)):
					if np.isnan(fraction[i, j]):
						continue
					axis.text(
						j, i, f"{100 * fraction[i, j]:.1f}%\n(n={mean_counts[i, j]:.0f})",
						ha="center", va="center", fontsize=14,
						color="white" if fraction[i, j] > 0.55 else "#1a1a19",
					)
			axis.set_xticks(range(len(CLASSES)), class_labels, fontsize=14)
			axis.set_yticks(range(len(CLASSES)), class_labels, fontsize=14)
			axis.set_xlabel("Predicted class", fontsize=14)
			axis.set_ylabel("True class", fontsize=14)
			axis.set_title(f"Test: {test_set}", fontsize=18)
		fig.suptitle(f"{best_model} - confusion matrices (fold mean, row-normalised)", fontsize=20)
		fig.tight_layout(rect=(0, 0.02, 1, 0.93))
		save_figure(fig, BEST_MODEL_DIR / "best_model_confusion_matrices.png")
		plt.close(fig)

	# --- Per-class metrics: one panel per test set, classes side by side for each metric.
	fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.8), sharey=True, squeeze=False)
	bar_width = 0.8 / len(CLASSES)
	low = 0.0
	for axis, test_set in zip(axes[0], test_sets):
		values = per_class[(best_model, test_set)]
		for class_index, (class_label, color) in enumerate(zip(class_labels, BEST_CLASS_COLORS)):
			means, stds = [], []
			for _, metric_index in BEST_CLASS_METRICS:
				column = values[:, class_index, metric_index]
				all_nan = np.all(np.isnan(column))
				means.append(np.nan if all_nan else np.nanmean(column))
				stds.append(0 if all_nan else np.nanstd(column))
			low = min(low, np.nanmin(np.array(means) - np.array(stds)))
			positions = np.arange(len(BEST_CLASS_METRICS)) - 0.4 + bar_width * (class_index + 0.5)
			axis.bar(
				positions, means, bar_width, yerr=stds, capsize=3, color=color,
				edgecolor="white", linewidth=1, label=class_label,
				error_kw={"elinewidth": 1, "ecolor": "#3d3d3a"},
			)
		axis.set_xticks(range(len(BEST_CLASS_METRICS)), [name for name, _ in BEST_CLASS_METRICS], fontsize=14)
		axis.tick_params(axis="y", labelsize=14)
		axis.set_title(f"Test: {test_set}", fontsize=18)
		axis.grid(axis="y", alpha=0.25)
		axis.set_axisbelow(True)
		axis.axhline(0, color="#3d3d3a", linewidth=0.8)
	axes[0, 0].set_ylabel("Score (fold mean ± std)", fontsize=14)
	axes[0, 0].set_ylim(min(0, low - 0.05), 1.05)
	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=len(CLASSES), frameon=False, fontsize=14)
	fig.suptitle(f"{best_model} - per-class metrics", fontsize=20)
	fig.tight_layout(rect=(0, 0.08, 1, 0.93))
	save_figure(fig, BEST_MODEL_DIR / "best_model_per_class_metrics.png")
	plt.close(fig)

	# --- Selected model vs the from-scratch model trained on each test survey (in-domain baseline).
	fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.8), sharey=True, squeeze=False)
	low = 0.0
	for axis, test_set in zip(axes[0], test_sets):
		baseline = f"NONE->{test_set}"
		series = [(best_model, "selected", f"{best_model} (selected)")]
		if baseline != best_model:
			series.append((baseline, "baseline", f"NONE->test survey (in-domain baseline)"))
		width = 0.8 / len(series)
		for index, (model, role, label) in enumerate(series):
			stats = [_fold_stats(test_rows, model, test_set, key) for key in metric_keys]
			means = np.array([s[0] for s in stats])
			stds = np.nan_to_num(np.array([s[1] for s in stats]))
			low = min(low, np.nanmin(means - stds)) if not np.all(np.isnan(means)) else low
			positions = np.arange(len(metric_keys)) - 0.4 + width * (index + 0.5)
			axis.bar(
				positions, means, width, yerr=stds, capsize=3, color=BEST_COMPARE_COLORS[role],
				edgecolor="white", linewidth=1, label=label,
				error_kw={"elinewidth": 1, "ecolor": "#3d3d3a"},
			)
			for position, mean in zip(positions, means):
				if not np.isnan(mean) and mean > 0.12:
					axis.text(position, 0.03, f"{mean:.2f}", ha="center", va="bottom",
							  fontsize=11, color="white", fontweight="bold")
		axis.set_xticks(range(len(metric_keys)), [title for _, title in BEST_MACRO_METRICS], fontsize=13)
		axis.tick_params(axis="y", labelsize=14)
		axis.set_title(f"Test: {test_set}" + (" (selected model is the baseline)" if baseline == best_model else ""),
					   fontsize=16)
		axis.grid(axis="y", alpha=0.25)
		axis.set_axisbelow(True)
		axis.axhline(0, color="#3d3d3a", linewidth=0.8)
	axes[0, 0].set_ylabel("Score (fold mean ± std)", fontsize=14)
	axes[0, 0].set_ylim(min(0, low - 0.05), 1.12)
	legend_entries = {}
	for axis in axes[0]:
		for handle, label in zip(*axis.get_legend_handles_labels()):
			legend_entries.setdefault(label, handle)
	fig.legend(legend_entries.values(), legend_entries.keys(), loc="lower center",
			   ncol=len(legend_entries), frameon=False, fontsize=14)
	fig.suptitle(f"{best_model} vs from-scratch baseline per test survey", fontsize=20)
	fig.tight_layout(rect=(0, 0.08, 1, 0.93))
	save_figure(fig, BEST_MODEL_DIR / "best_model_vs_scratch_baseline.png")
	plt.close(fig)

	# --- Copy the model's ROC figure so everything for the selected model sits in one folder.
	roc_path = ROC_DIR / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', best_model)}_roc.png"
	if roc_path.exists():
		(BEST_MODEL_DIR / "best_model_roc.png").write_bytes(roc_path.read_bytes())

	# --- Console summary.
	print(f"\nSelected model: {best_model} (highest weighted macro F1)")
	for row in summary_rows:
		rank = f"  rank {row['rank']}/{row['models_ranked']}" if row.get("rank") else ""
		print(f"  {row['test_dataset']:<9}" + "  ".join(
			f"{key}={row[f'{key}_mean']:.3f}" for key in metric_keys
		) + rank)
	print(f"Best-model summaries saved to {BEST_MODEL_DIR}")


def main():
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	plot_bubble_summaries()
	plot_roc_curves()
	test_rows = load_test_metrics()
	plot_bar_graphs(test_rows)
	plot_tang_bubble_summaries()
	plot_paired_bar_graphs(test_rows)
	if test_rows:
		write_csv(
			CSV_DIR / "test_metrics_per_fold.csv",
			test_rows,
			["experiment", "model", "test_dataset", "fold"] + [metric for metric, _ in HEATMAP_METRICS],
		)
		plot_metric_heatmaps(test_rows)
		plot_plain_metric_heatmaps(test_rows)
		plot_per_class_metric_heatmaps(test_rows)
		write_latex_tables(test_rows)
		best_models = best_heatmap_models(test_rows)
		print(f"Best heatmap models (F1): {best_models}")
		plot_bubble_summaries(best_models, BUBBLE_BEST_DIR)
		plot_best_model_summary(test_rows, best_models.get("weighted"))
		print(f"Test metric heatmaps saved to {OUTPUT_DIR}")
	else:
		print(f"No test predictions found under {TESTING_DIR}")

	runs = load_training_runs()
	if not runs:
		print(f"No saved fold histories found under {TRAINING_DIR}")
		print("Run the training pipeline first; its loss_fold_<fold>.npy files are the input.")
		return

	metric_names, epoch_rows = build_epoch_rows(runs)
	fold_rows = build_fold_summary_rows(runs, metric_names)
	model_rows = build_model_summary_rows(fold_rows, metric_names)

	base_fields = ["experiment", "model", "fold"]
	write_csv(
		CSV_DIR / "epoch_metrics.csv",
		epoch_rows,
		base_fields + ["epoch"] + metric_names,
	)
	fold_fields = [
		"experiment", "model", "fold", "epochs_trained", "best_epoch",
		"training_time_seconds", "accuracy_at_best_epoch", "val_accuracy_at_best_epoch",
	] + [f"final_{metric}" for metric in metric_names]
	write_csv(CSV_DIR / "fold_summary.csv", fold_rows, fold_fields)
	model_fields = ["experiment", "model", "folds", "training_time_seconds_mean", "training_time_seconds_std"]
	for metric in metric_names:
		model_fields.extend([f"final_{metric}_mean", f"final_{metric}_std"])
	write_csv(CSV_DIR / "model_summary.csv", model_rows, model_fields)
	# plot_accuracy_curves(runs)
	plot_fold_curves(runs)

	print(f"Summarised {len(runs)} fold histories across {len(model_rows)} models.")
	print(f"Reports and accuracy plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
	main()