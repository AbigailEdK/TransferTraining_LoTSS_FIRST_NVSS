"""Replace the visual Yes / No morphology decisions with an explicit, reproducible threshold rule.

A separate rule is fitted for each survey comparison (reviewComparisons.py: FIRST vs LoTSS from
morphologyReview.py, FIRST vs NVSS from morphologyReviewNVSS.py); by default every comparison with
saved answers, or the ones named on the command line, e.g.
	python morphologyAnalyticalClassification.py FIRST_vs_NVSS
Below, "A" and "B" are the two compared surveys.

The manual review is used only to calibrate the rule. Every source is then re-labelled purely
from its A and B image measurements (morphologyReviewStats.py), with no catalogue information,
and all statistics and paper figures are recalculated with the new labels.

The rule
--------
A source is Yes when every measurement in the rule passes its threshold, e.g.

	Yes  if  FIRST peak S/N >= a  AND  FIRST-LoTSS 3-sigma mask overlap >= b

Candidate measurements (candidate_features) are the image quantities that separated the manual
groups. The per-image mean / median / noise columns are left out because the images are
min-max scaled, which makes them restatements of the peak S/N. The survey that is not part of
the comparison (NVSS for FIRST vs LoTSS, LoTSS for FIRST vs NVSS) is left out, because the review
only compared A with B.

Every rule with one or two candidate measurements is tried, with thresholds at the percentiles
of each measurement. Rules are scored by balanced accuracy against the manual labels (the
average of the Yes and No recall, so the 15% No class counts as much as the 85% Yes class).
The form of the rule (which measurements) is chosen by 5-fold stratified cross-validation.
A one-measurement rule is preferred unless a pair improves the cross-validated score by more
than SIMPLER_RULE_TOLERANCE. Its thresholds are then refitted on all sources.

Outputs, in MorphologyReviewStats/<comparison>/Analytical/ (summary.txt names the comparison):
	analytical_labels.csv      every source: manual and analytical label, agreement, rule measurements
	disagreements.csv          sources where the rule and the manual label differ, worth re-inspecting
	numeric_comparison.csv     Yes vs No statistics for every image measurement, using the analytical labels
	categorical_<col>.csv      post-hoc breakdown by catalogue / label (not used by the rule)
	summary.txt                the rule, how well it agrees with the manual review, and the recalculated statistics
	rule_plane.pdf/png         the rule drawn over every source, coloured by the manual label
	confusion_matrix.pdf/png   manual vs analytical labels
	PaperFigures/              figures 1-5 and the LaTeX table, recalculated with the analytical labels

Because the rule is fitted to separate the groups, the measurements it uses separate the
analytical groups perfectly by construction (|delta| = 1 for those rows); the other measurements
show how the two groups differ beyond the rule itself.
"""

import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import morphologyReviewFigures as figures
import morphologyReviewStats as stats
from reviewComparisons import selected_comparisons


def candidate_features(comparison):
	"""column -> label used in the rule text and plots."""
	first, second = comparison.surveys
	pair = comparison.name
	return {
		f"{first}_peak_snr": f"{first} peak S/N",
		f"{second}_peak_snr": f"{second} peak S/N",
		f"{first}_n_pix_5sigma": rf"{first} pixels $>5\sigma$",
		f"{first}_largest_island_extent_px": f"{first} largest component extent (px)",
		f"{first}_centroid_offset_px": f"{first} emission offset from centre (px)",
		f"{pair}_pearson_r": f"{first}-{second} pixel correlation r",
		f"{pair}_mask_iou": rf"{first}-{second} $3\sigma$ mask overlap (IoU)",
		f"{pair}_log_size_ratio": rf"$\log_{{10}}$ {first}/{second} emission size",
	}


MAX_FEATURES = 2
THRESHOLD_PERCENTILES = np.linspace(1, 99, 99)
CV_FOLDS = 5
SIMPLER_RULE_TOLERANCE = 0.01
RANDOM_SEED = 0


# ---------- rule search ----------

def feature_direction(values, is_yes):
	""">= when Yes sources tend to have larger values, otherwise <=."""
	return ">=" if np.nanmedian(values[is_yes]) >= np.nanmedian(values[~is_yes]) else "<="


def passes(values, direction, thresholds):
	"""Boolean matrix (thresholds x sources); NaN never passes, so a missing measurement means No."""
	values = values[None, :]
	thresholds = np.asarray(thresholds)[:, None]
	with np.errstate(invalid="ignore"):
		return values >= thresholds if direction == ">=" else values <= thresholds


def candidate_thresholds(values):
	return np.unique(np.nanpercentile(values, THRESHOLD_PERCENTILES))


def balanced_accuracy(predicted_yes, is_yes):
	"""Works on a single prediction vector or on a stack of them (last axis = sources)."""
	yes_recall = (predicted_yes & is_yes).sum(axis=-1) / is_yes.sum()
	no_recall = (~predicted_yes & ~is_yes).sum(axis=-1) / (~is_yes).sum()
	return (yes_recall + no_recall) / 2


def fit_rule(table, features, is_yes):
	"""Best thresholds for an AND rule over `features`; returns (rule, training score)."""
	directions = [feature_direction(table[f].to_numpy(float), is_yes) for f in features]
	grids = [candidate_thresholds(table[f].to_numpy(float)) for f in features]
	masks = [passes(table[f].to_numpy(float), d, g).astype(np.int32) for f, d, g in zip(features, directions, grids)]
	y = is_yes.astype(np.int32)
	n_yes, n_no = y.sum(), len(y) - y.sum()

	if len(features) == 1:
		true_yes = masks[0] @ y
		predicted_yes = masks[0].sum(axis=1)
		score = (true_yes / n_yes + (n_no - (predicted_yes - true_yes)) / n_no) / 2
		best = np.unravel_index(np.argmax(score), score.shape)
	else:
		# counts for every threshold pair at once: (a & b & yes) and (a & b)
		true_yes = (masks[0] * y) @ masks[1].T
		predicted_yes = masks[0] @ masks[1].T
		score = (true_yes / n_yes + (n_no - (predicted_yes - true_yes)) / n_no) / 2
		best = np.unravel_index(np.argmax(score), score.shape)

	rule = [(f, d, float(g[i])) for f, d, g, i in zip(features, directions, grids, best)]
	return rule, float(score[best])


def apply_rule(table, rule):
	predicted = np.ones(len(table), dtype=bool)
	for feature, direction, threshold in rule:
		predicted &= passes(table[feature].to_numpy(float), direction, [threshold])[0]
	return predicted


def stratified_folds(is_yes, k, rng):
	fold = np.empty(len(is_yes), dtype=int)
	for group in (True, False):
		members = rng.permutation(np.flatnonzero(is_yes == group))
		fold[members] = np.arange(len(members)) % k
	return fold


def select_rule(table, is_yes, candidates):
	"""Cross-validate every one- and two-measurement rule form; return the chosen form and all scores."""
	features = [f for f in candidates if f in table]
	forms = [combo for n in range(1, MAX_FEATURES + 1) for combo in combinations(features, n)]
	fold = stratified_folds(is_yes, CV_FOLDS, np.random.default_rng(RANDOM_SEED))

	results = []
	for form in forms:
		scores = []
		for k in range(CV_FOLDS):
			train, test = fold != k, fold == k
			rule, _ = fit_rule(table[train], form, is_yes[train])
			scores.append(balanced_accuracy(apply_rule(table[test], rule), is_yes[test]))
		results.append({"features": form, "n_features": len(form),
			"cv_balanced_accuracy": np.mean(scores), "cv_std": np.std(scores)})
	results = pd.DataFrame(results).sort_values("cv_balanced_accuracy", ascending=False)

	best_overall = results["cv_balanced_accuracy"].max()
	for n in range(1, MAX_FEATURES + 1):
		subset = results[results["n_features"] == n]
		if subset["cv_balanced_accuracy"].max() >= best_overall - SIMPLER_RULE_TOLERANCE:
			chosen = subset.iloc[0]
			break
	return chosen, results


# ---------- agreement ----------

def agreement_summary(manual_yes, analytical_yes):
	n = len(manual_yes)
	matrix = pd.DataFrame(
		[[np.sum(manual_yes & analytical_yes), np.sum(manual_yes & ~analytical_yes)],
		 [np.sum(~manual_yes & analytical_yes), np.sum(~manual_yes & ~analytical_yes)]],
		index=["Manual Yes", "Manual No"], columns=["Analytical Yes", "Analytical No"],
	)
	observed = np.mean(manual_yes == analytical_yes)
	expected = np.mean(manual_yes) * np.mean(analytical_yes) + np.mean(~manual_yes) * np.mean(~analytical_yes)
	kappa = (observed - expected) / (1 - expected) if expected < 1 else np.nan
	return matrix, {
		"agreement": observed,
		"cohens_kappa": kappa,
		"balanced_accuracy": balanced_accuracy(analytical_yes, manual_yes),
		"yes_recall": np.sum(manual_yes & analytical_yes) / manual_yes.sum(),
		"no_recall": np.sum(~manual_yes & ~analytical_yes) / (~manual_yes).sum(),
		"n": n,
	}


def rule_text(rule, labels=None):
	"""Plain column names, or the LaTeX labels in `labels` when given."""
	latex = labels is not None
	parts = []
	for feature, direction, threshold in rule:
		name = labels[feature] if latex else feature
		symbol = {">=": r"$\geq$", "<=": r"$\leq$"}[direction] if latex else direction
		parts.append(f"{name} {symbol} {threshold:.3g}")
	return " AND ".join(parts)


# ---------- figures specific to the classification ----------

def is_log_feature(values):
	values = values[np.isfinite(values)]
	return values.min() > 0 and np.percentile(values, 99) / np.percentile(values, 1) > 100


def figure_rule_plane(table, rule, manual_yes, labels, output_dir):
	"""The rule drawn over every source; colour = manual label, so disagreements are visible."""
	fig, axis = plt.subplots(figsize=(figures.SINGLE_COLUMN, figures.SINGLE_COLUMN))
	manual = np.where(manual_yes, "Yes", "No")
	if len(rule) == 1:
		feature, direction, threshold = rule[0]
		values = table[feature].to_numpy(float)
		finite = values[np.isfinite(values)]
		log = is_log_feature(values)
		# counts that start at 0 (e.g. pixels above 5 sigma) span decades too: linear up to 1, log above
		symlog = not log and finite.min() >= 0 and finite.max() > 100
		if log:
			bins = np.geomspace(finite.min(), finite.max(), 40)
		elif symlog:
			bins = np.concatenate([[0, 0.5], np.geomspace(1, finite.max() * 1.01, 39)])
		else:
			bins = np.linspace(finite.min(), finite.max(), 40)
		for decision in ("Yes", "No"):
			subset = values[(manual == decision) & np.isfinite(values)]
			axis.hist(subset, bins=bins, weights=np.full(len(subset), 1 / len(subset)), histtype="step",
				color=figures.COLOURS[decision], linestyle=figures.LINESTYLES[decision], linewidth=1.1)
		if log:
			axis.set_xscale("log")
		elif symlog:
			axis.set_xscale("symlog", linthresh=1)
		axis.axvline(threshold, color="black", linestyle="--", linewidth=0.8)
		axis.text(threshold, axis.get_ylim()[1], f" {threshold:.3g}", ha="left", va="top", fontsize="small")
		axis.set_xlabel(labels[feature])
		axis.set_ylabel("Fraction of manual group")
		handles = figures.decision_legend(table.assign(decision=manual))
	else:
		(fx, dx, tx), (fy, dy, ty) = rule
		x, y = table[fx].to_numpy(float), table[fy].to_numpy(float)
		for decision, size, alpha in (("Yes", 5, 0.4), ("No", 9, 0.8)):
			chosen = manual == decision
			axis.scatter(x[chosen], y[chosen], s=size, alpha=alpha, marker=figures.MARKERS[decision],
				color=figures.COLOURS[decision], linewidths=0, rasterized=True)
		if is_log_feature(x):
			axis.set_xscale("log")
		if is_log_feature(y):
			axis.set_yscale("log")
		x_limits, y_limits = axis.get_xlim(), axis.get_ylim()
		# shade the region the rule calls Yes
		x_span = (tx, x_limits[1]) if dx == ">=" else (x_limits[0], tx)
		y_span = (ty, y_limits[1]) if dy == ">=" else (y_limits[0], ty)
		axis.fill_between(x_span, *y_span, color=figures.GRID, alpha=0.7, zorder=0, linewidth=0)
		axis.axvline(tx, color="black", linestyle="--", linewidth=0.8)
		axis.axhline(ty, color="black", linestyle="--", linewidth=0.8)
		axis.set_xlim(x_limits)
		axis.set_ylim(y_limits)
		axis.text(np.sqrt(x_span[0] * x_span[1]) if axis.get_xscale() == "log" else np.mean(x_span),
			y_span[1], "Analytical Yes", ha="center", va="top", fontsize="small", color=figures.MUTED)
		axis.set_xlabel(labels[fx])
		axis.set_ylabel(labels[fy])
		handles = figures.decision_legend(table.assign(decision=manual), lines=False)
	for handle in handles:
		handle.set_label("Manual " + handle.get_label())
	axis.legend(handles=handles, loc="center right")
	save(fig, "rule_plane", output_dir)


def figure_confusion(matrix, output_dir):
	"""Row-normalised, drawn like the confusion matrices in paperModelEvaluation.py."""
	fig, axis = plt.subplots(figsize=(figures.SINGLE_COLUMN, figures.SINGLE_COLUMN * 0.85))
	share = matrix.div(matrix.sum(axis=1), axis=0).to_numpy()
	axis.imshow(share, cmap="Blues", vmin=0, vmax=1)
	for (row, column), count in np.ndenumerate(matrix.to_numpy()):
		axis.text(column, row, f"{count}\n({share[row, column]:.1%})", ha="center", va="center",
			color="white" if share[row, column] > 0.55 else "#1a1a19")
	axis.set_xticks([0, 1], ["Yes", "No"])
	axis.set_yticks([0, 1], ["Yes", "No"])
	axis.set_xlabel("Analytical label")
	axis.set_ylabel("Manual label")
	save(fig, "confusion_matrix", output_dir)


def save(fig, name, output_dir):
	for extension, dpi in (("pdf", None), ("png", 300)):
		fig.savefig(output_dir / f"{name}.{extension}", dpi=dpi)
	plt.close(fig)
	print(f"  {name}")


# ---------- main ----------

def run(comparison):
	output_dir = comparison.output_dir / "Analytical"
	table = figures.load_features(comparison)
	output_dir.mkdir(parents=True, exist_ok=True)
	candidates = candidate_features(comparison)
	manual_yes = (table["decision"] == "Yes").to_numpy()

	print(f"[{comparison.label}] searching one- and two-measurement rules with {CV_FOLDS}-fold cross-validation ...")
	chosen, cv_results = select_rule(table, manual_yes, candidates)
	rule, training_score = fit_rule(table, list(chosen["features"]), manual_yes)
	analytical_yes = apply_rule(table, rule)
	matrix, agreement = agreement_summary(manual_yes, analytical_yes)
	cv_results.assign(features=cv_results["features"].map(" AND ".join)).to_csv(
		output_dir / "rule_search_cv.csv", index=False)

	# labels for every source
	labels = table[["decision"] + [f for f, _, _ in rule]].rename(columns={"decision": "manual"})
	labels.insert(1, "analytical", np.where(analytical_yes, "Yes", "No"))
	labels.insert(2, "agrees", labels["manual"] == labels["analytical"])
	labels.to_csv(output_dir / "analytical_labels.csv")
	labels[~labels["agrees"]].to_csv(output_dir / "disagreements.csv")

	# from here on, "decision" is the analytical label, so every statistic and figure uses it
	analytical = table.assign(decision=labels["analytical"])
	numeric = stats.compare_numeric(analytical)
	numeric.to_csv(output_dir / "numeric_comparison.csv", index=False)

	n_yes, n_no = analytical_yes.sum(), (~analytical_yes).sum()
	lines = [
		f"COMPARISON: {comparison.label}  (rule calibrated on {comparison.progress_file.name})",
		"",
		"ANALYTICAL RULE (image measurements only, no catalogue information)",
		f"  Yes  if  {rule_text(rule)}",
		f"  chosen by {CV_FOLDS}-fold cross-validated balanced accuracy: "
		f"{chosen['cv_balanced_accuracy']:.3f} +/- {chosen['cv_std']:.3f}",
		f"  balanced accuracy of the refitted rule on all sources: {training_score:.3f}",
		"\n  Best cross-validated rule of each size:",
	]
	for n in range(1, MAX_FEATURES + 1):
		best = cv_results[cv_results["n_features"] == n].iloc[0]
		lines.append(f"    {n} measurement(s): {' AND '.join(best['features'])}  "
			f"({best['cv_balanced_accuracy']:.3f} +/- {best['cv_std']:.3f})")

	lines += [
		"\nAGREEMENT WITH THE MANUAL REVIEW",
		f"  {agreement['agreement']:.1%} of {agreement['n']} sources agree   Cohen's kappa = {agreement['cohens_kappa']:.3f}",
		f"  manual Yes kept as Yes: {agreement['yes_recall']:.1%}   manual No kept as No: {agreement['no_recall']:.1%}",
		"",
		matrix.to_string(),
		f"\n  {(~labels['agrees']).sum()} disagreements written to disagreements.csv",
		f"\nRECALCULATED STATISTICS (analytical labels)   Yes: {n_yes}   No: {n_no}   No-rate: {n_no / len(table):.1%}",
		"  Measurements used by the rule separate the groups perfectly by construction.",
		f"{'feature':<44}{'median Yes':>12}{'median No':>12}{'delta':>8}{'AUC':>6}{'p (FDR)':>10}",
	]
	for _, row in numeric.iterrows():
		lines.append(
			f"  {row['feature']:<42}{row['median_yes']:>12.4g}{row['median_no']:>12.4g}"
			f"{row['cliffs_delta_no_vs_yes']:>+8.2f}{row['auc_single_feature']:>6.2f}{row['mannwhitney_p_fdr']:>10.1e}"
		)

	lines.append("\nPOST-HOC BREAKDOWN (catalogue information was not used by the rule)")
	for column in ("Catalog", "Type"):
		counts, p, _ = stats.compare_categorical(analytical, column)
		counts.to_csv(output_dir / f"categorical_{column}.csv")
		lines.append(f"\n  {column}: chi-squared p = {p:.2e}")
		for name, row in counts.iterrows():
			lines.append(f"    {str(name):<14} {int(row['No']):>4} / {int(row['total']):<5} No  ({row['no_rate']:.1%})")

	report = "\n".join(lines)
	print(report)
	(output_dir / "summary.txt").write_text(report, encoding="utf-8")

	print(f"\n[{comparison.label}] writing figures to {output_dir}")
	figures.set_style()
	figure_rule_plane(table, rule, manual_yes, candidates, output_dir)
	figure_confusion(matrix, output_dir)

	figures.FIGURE_DIR = output_dir / "PaperFigures"
	figures.FIGURE_DIR.mkdir(exist_ok=True)
	figures.figure_examples(analytical, comparison, group_prefix="Analytical")
	figures.figure_distributions(analytical, comparison)
	figures.figure_snr_plane(analytical, comparison)
	figures.figure_catalogues(analytical)
	effects = figures.effect_size_table(analytical, comparison)
	figures.figure_effect_sizes(effects, comparison)
	figures.write_latex_table(effects, analytical, comparison,
		how_labelled=f"satisfy the analytical criterion ({rule_text(rule, candidates)})",
		label=f"tab:morphology_analytical_{comparison.name}")
	print()


def main(names=()):
	for comparison in selected_comparisons(names):
		# every cross-validation fold needs at least one source of each label
		if stats.enough_answers(comparison, minimum=CV_FOLDS):
			run(comparison)


if __name__ == "__main__":
	main(sys.argv[1:])
