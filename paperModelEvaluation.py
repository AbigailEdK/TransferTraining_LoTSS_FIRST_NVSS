"""Summarise training metrics and learning curves for Experiment_0."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import (
	accuracy_score,
	auc,
	f1_score,
	precision_recall_fscore_support,
	precision_score,
	roc_auc_score,
	roc_curve,
)


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

		axis.set_title(f"{model}\nExperiment_0")
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
		fig.suptitle(f"{model}\nExperiment_0 - performance per fold")
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

	fig, axes = plt.subplots(
		2, 2, figsize=(5 + 2.2 * len(columns), 2.5 + 0.55 * len(models) * 2), squeeze=False
	)
	for axis, (metric, title) in zip(axes.ravel(), HEATMAP_METRICS):
		grid = grids[metric]
		image = axis.imshow(np.ma.masked_invalid(grid), cmap="viridis", vmin=0, vmax=1, aspect="auto")
		axis.set_xticks(range(len(columns)), columns)
		axis.set_yticks(range(len(models)), models)
		axis.set_xlabel("Test dataset")
		axis.set_title(title)
		for i in range(len(models)):
			for j in range(len(columns)):
				if not np.isnan(grid[i, j]):
					axis.text(
						j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
						color="white" if grid[i, j] < 0.6 else "black", fontsize=8,
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
	fig.suptitle(
		"Experiment_0 - test metrics averaged across folds\n"
		"Weighted = 0.45 FIRST + 0.45 LoTSS + 0.10 NVSS (red); "
		"Average = equal weights (blue)"
	)
	fig.tight_layout(rect=(0, 0, 1, 0.95))
	save_figure(fig, HEATMAP_DIR / "test_metric_heatmaps.png")
	plt.close(fig)


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
			axis.set_xlabel("False positive rate")
			axis.set_ylabel("True positive rate")
			axis.set_title(f"Test: {dataset}")
			axis.grid(alpha=0.25)
			axis.legend(loc="lower right", frameon=False, fontsize=8)
		fig.suptitle(f"{model} - ROC curves (mean ± std over folds)")
		fig.tight_layout(rect=(0, 0, 1, 0.94))
		safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
		save_figure(fig, ROC_DIR / f"{safe_name}_roc.png")
		plt.close(fig)


BUBBLE_METRICS = [
	("recall", "Recall", "#fdf0ee"),
	("precision", "Precision", "#f8f1f8"),
	("f1", "F1", "#f0f0fd"),
	("auc", "AUC", "#eef8f0"),
]
BUBBLE_CLASS_LABELS = {"FRI": "FR I", "FRII": "FR II", "COMPACT": "Compact"}


def load_per_class_metrics():
	"""Return {(model, test_dataset): array[fold, class, metric]} with metrics in BUBBLE_METRICS order."""
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
			aucs = []
			for index in labels:
				positives = y_true == index
				aucs.append(
					roc_auc_score(positives, y_prob[:, index])
					if positives.any() and not positives.all() else np.nan
				)
			folds.append(np.stack([recall, precision, f1, np.array(aucs)], axis=1))
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
		baseline = metrics.get((f"NONE->{test_dataset}", test_dataset))
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
		fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.17, wspace=0.12, hspace=0.25)
		for row, class_name in enumerate(CLASSES):
			for col, (_, metric_title, background) in enumerate(BUBBLE_METRICS):
				axis = axes[row, col]
				axis.set_facecolor(background)
				axis.set_xlim(0, 1)
				axis.set_ylim(len(model_names) + 0.4, -0.9)
				axis.set_yticks([])
				axis.set_xticks(np.arange(0, 1.01, 0.25))
				axis.tick_params(labelsize=8)
				axis_width_pts = axis.get_position().width * fig.get_figwidth() * 72
				if baseline is not None:
					value = np.nanmean(baseline[:, row, col])
					if not np.isnan(value):
						axis.axvline(value, color="#1f77b4", linestyle=":", linewidth=2)
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
					axis.set_title(metric_title, fontsize=11)
				if col == 0:
					axis.set_ylabel(BUBBLE_CLASS_LABELS.get(class_name, class_name),
									rotation=0, labelpad=26, va="center", fontsize=11)

		handles = [
			plt.Line2D([], [], marker="o", linestyle="", color=colors[model], label=model, markersize=8)
			for model in model_names
		]
		handles.append(plt.Line2D([], [], color="#1f77b4", linestyle=":", linewidth=2,
								  label=f"NONE->{test_dataset} (baseline)"))
		for key, (ring_color, _, ring_label) in ring_styles.items():
			if best_models.get(key) in model_names:
				handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
										  markeredgecolor=ring_color, markeredgewidth=2, markersize=10,
										  label=f"{ring_label}: {best_models[key]}"))
		fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=True, fontsize=9)
		fig.suptitle(
			f"Metric evaluation (test input: {test_dataset})\n"
			"Bubble centre = fold mean, bubble diameter = 2 x std across folds (bar = 0.1)",
			fontsize=12,
		)
		save_figure(fig, output_dir / f"bubble_metrics_test_{test_dataset}.png")
		plt.close(fig)


BAR_DIR = OUTPUT_DIR / "Metric_Bar_Graphs"

# Tang et al. results (scores on a 0-1 scale). Per-class values were only published for the
# NONE->X models (Tang Xavier, Table 3.3-3.5); transfer models use Tang Method 0 (accuracy/AUC only).`n# Compact was not evaluated, and their AUC is a single overall value.
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

# (test dataset, models to compare, include Tang et al.). LoTSS was not tested by Tang et al.
BAR_GRAPH_SETS = [
	("NVSS", ["NONE->NVSS", "FIRST->NVSS"], True),
	("FIRST", ["NONE->FIRST", "NVSS->FIRST"], True),
	("LoTSS", ["NONE->LoTSS", "FIRST->LoTSS", "NVSS->LoTSS"], False),
]


def plot_bar_graphs(test_rows):
	"""Grouped bar graphs of all metrics, models interleaved per metric, optionally with Tang et al."""
	metrics = load_per_class_metrics()
	if not metrics:
		return

	# Each group: (label, own fold values getter, Tang et al. value getter, section id for separators)
	groups = []
	for class_index, class_name in enumerate(CLASSES):
		for metric_index, (metric_key, metric_title, _) in enumerate(BUBBLE_METRICS):
			groups.append((
				f"{BUBBLE_CLASS_LABELS.get(class_name, class_name)}\n{metric_title}",
				lambda model, dataset, c=class_index, m=metric_index: metrics[(model, dataset)][:, c, m],
				lambda scores, c=class_name, k=metric_key: scores["classes"].get(c, {}).get(k),
				class_index,
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
		))

	for test_dataset, models, with_paper in BAR_GRAPH_SETS:
		models = [model for model in models if (model, test_dataset) in metrics]
		if not models:
			continue

		series = [(model, "own") for model in models]
		if with_paper:
			series += [(model, "paper") for model in models]

		colors = plt.cm.tab10.colors
		bar_width = 0.8 / len(series)
		fig, axis = plt.subplots(figsize=(1.45 * len(groups) + 2, 6.5))
		for series_index, (model, kind) in enumerate(series):
			color = colors[models.index(model) % len(colors)]
			means, errors = [], []
			for _, own_getter, paper_getter, _ in groups:
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
			label = model if kind == "own" else f"{model} (Tang et al.)"
			axis.bar(
				positions, means, bar_width, yerr=errors, capsize=2, color=color,
				alpha=1.0 if kind == "own" else 0.55, hatch=None if kind == "own" else "//",
				edgecolor="black", linewidth=0.5, label=label,
			)

		axis.set_xticks(np.arange(len(groups)), [group[0] for group in groups], fontsize=8)
		axis.set_ylim(0, 1.05)
		axis.set_ylabel("Score")
		axis.grid(axis="y", alpha=0.25)
		for index in range(1, len(groups)):
			if groups[index][3] != groups[index - 1][3]:
				axis.axvline(index - 0.5, color="grey", linewidth=0.8)
		axis.legend(frameon=False, ncol=len(series), loc="upper center",
					bbox_to_anchor=(0.5, -0.14), fontsize=9)
		axis.set_title(
			f"All metrics - test input: {test_dataset}"
			+ (" (own results vs Tang et al.)" if with_paper else "")
			+ "\nBars = fold mean, error bars = std across folds"
			+ ("; Tang et al. report FR I / FR II only" if with_paper else "")
		)
		fig.tight_layout()
		save_figure(fig, BAR_DIR / f"bar_metrics_test_{test_dataset}.png")
		plt.close(fig)


def main():
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	plot_bubble_summaries()
	plot_roc_curves()
	test_rows = load_test_metrics()
	plot_bar_graphs(test_rows)
	if test_rows:
		write_csv(
			CSV_DIR / "test_metrics_per_fold.csv",
			test_rows,
			["experiment", "model", "test_dataset", "fold"] + [metric for metric, _ in HEATMAP_METRICS],
		)
		plot_metric_heatmaps(test_rows)
		best_models = best_heatmap_models(test_rows)
		print(f"Best heatmap models (F1): {best_models}")
		plot_bubble_summaries(best_models, BUBBLE_BEST_DIR)
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
