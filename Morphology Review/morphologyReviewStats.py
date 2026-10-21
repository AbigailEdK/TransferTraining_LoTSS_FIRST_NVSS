"""Compare the sources marked Yes and No in the morphology review (morphologyReview.py).

For every reviewed source the script reads its FIRST, LoTSS and NVSS images from
Data/FITS_BY_CHANNEL/<id>_<survey>.fits and measures:

	* pixel statistics: mean, median, standard deviation, fraction of blank (0) and saturated (1) pixels
	* background and noise: the median and standard deviation of the 3-sigma clipped image (the same
	  clipping used in Training/Code/Sigma Clipping), plus a MAD-based noise estimate
	* source emission: peak signal-to-noise, number / fraction of pixels above 3 and 5 sigma,
	  number of separate emission islands, size and extent of the largest island,
	  offset of the emission centroid from the image centre
	* agreement between surveys: pixel Pearson correlation and overlap (IoU) of the 3-sigma masks,
	  and the ratio of emission sizes

These are combined with the catalogue information (source catalogue, label, original label,
whether the label was changed, how much the votes in 'Divergence' agreed, RA/Dec,
and whether the source is in the Experiment_0 test or train/val set).

NOTE: the images are min-max scaled to 0-1 per image, so the pixel values and noise
are relative to each image's own peak, not in Jy/beam. A "noise" of 0.05 means the background
scatter is 5% of the brightest pixel, i.e. a dynamic range of about 20.

Skipped and unanswered sources are left out. Results are written to Results/MorphologyReviewStats/:
	source_features.csv     one row per reviewed source with every measurement
	numeric_comparison.csv  Yes vs No for each numeric measurement (medians, Mann-Whitney U,
	                        KS test, Cliff's delta, single-feature AUC, FDR-corrected p-values)
	categorical_<col>.csv   counts and No-rate per category, with a chi-squared test
	summary.txt             the printed report
	*.png                   plots
"""

import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu, pearsonr, sigmaclip

ROOT = Path(__file__).resolve().parent
CHANNEL_DIR = ROOT / "Data" / "FITS_BY_CHANNEL"
FITS_DIR = ROOT / "Data" / "RADCAT_F"
CATALOGUE = ROOT / "Data" / "RADCAT.csv"
PROGRESS_FILE = ROOT / "morphology_review_progress.json"
SPLIT_DIR = ROOT / "Training" / "Results" / "Experiment_0" / "Data"
OUTPUT_DIR = ROOT / "Results" / "MorphologyReviewStats"

SURVEYS = ("FIRST", "LoTSS", "NVSS")
SURVEY_PAIRS = (("FIRST", "LoTSS"), ("FIRST", "NVSS"), ("LoTSS", "NVSS"))
DECISIONS = {"yes": "Yes", "no": "No"}
CLIP_SIGMA = 3.0
MIN_ISLAND_PIXELS = 5  # islands smaller than this are treated as noise spikes
CATEGORICAL_COLUMNS = ("Catalog", "Type", "Original", "relabelled", "split", "vote_unanimous")

# Yes / No colours (categorical slots 1 and 2 of the dataviz reference palette)
COLOURS = {"Yes": "#2a78d6", "No": "#eb6834"}
TEXT_MUTED = "#6b6a65"


# ---------- loading ----------

def load_decisions():
	state = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
	return {int(k): DECISIONS[v] for k, v in state["decisions"].items() if v in DECISIONS}


def load_split():
	"""source id -> 'test' / 'train_val' from the Experiment_0 split (augmented copies share the id)."""
	split = {}
	for name, label in (("names_train_val.npy", "train_val"), ("names_test.npy", "test")):
		path = SPLIT_DIR / name
		if path.exists():
			for source_id in np.unique(np.load(path, allow_pickle=True)):
				split[int(source_id)] = label
	return split


def load_image(survey, source_id):
	"""2D float array, or None when the file is missing, empty or not 2D."""
	for path in (CHANNEL_DIR / f"{source_id}_{survey}.fits", FITS_DIR / survey / f"{source_id}.fits"):
		if path.exists() and path.stat().st_size > 0:
			try:
				with fits.open(path) as hdul:
					data = np.squeeze(np.array(hdul[0].data, dtype=float))
			except Exception:
				continue
			if data.ndim == 2:
				return np.nan_to_num(data)
	return None


# ---------- per-image measurements ----------

def image_features(data):
	"""Pixel, noise and emission measurements for one image, plus its 3-sigma emission mask."""
	flat = data.ravel()
	clipped, _, _ = sigmaclip(flat, low=CLIP_SIGMA, high=CLIP_SIGMA)
	background = float(np.median(clipped))
	noise = float(np.std(clipped))
	mad_noise = float(1.4826 * np.median(np.abs(flat - np.median(flat))))
	peak = float(flat.max())

	features = {
		"mean": float(flat.mean()),
		"median": float(np.median(flat)),
		"std": float(flat.std()),
		"blank_frac": float(np.mean(flat == 0)),
		"saturated_frac": float(np.mean(flat >= 1)),
		"background": background,
		"noise": noise,
		"noise_mad": mad_noise,
		"peak_snr": (peak - background) / noise if noise > 0 else np.nan,
	}

	if noise > 0:
		mask3 = data > background + 3 * noise
		mask5 = data > background + 5 * noise
	else:
		mask3 = mask5 = np.zeros_like(data, dtype=bool)
	features["n_pix_3sigma"] = int(mask3.sum())
	features["n_pix_5sigma"] = int(mask5.sum())
	features["frac_pix_3sigma"] = float(mask3.mean())

	labels, n_islands = ndimage.label(mask3)
	sizes = ndimage.sum(mask3, labels, range(1, n_islands + 1)) if n_islands else np.array([])
	keep = np.flatnonzero(sizes >= MIN_ISLAND_PIXELS) + 1
	features["n_islands"] = int(len(keep))
	features["largest_island_pix"] = int(sizes.max()) if len(sizes) else 0
	features["largest_island_frac_of_emission"] = (
		float(sizes.max() / mask3.sum()) if mask3.sum() else np.nan
	)

	# extent of the largest island: 4 * sqrt(largest eigenvalue of its pixel covariance) ~ full length
	if len(sizes):
		ys, xs = np.nonzero(labels == int(np.argmax(sizes)) + 1)
		if len(xs) > 2:
			eigenvalues = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys])))
			features["largest_island_extent_px"] = float(4 * np.sqrt(max(eigenvalues.max(), 0)))
			features["largest_island_axis_ratio"] = (
				float(np.sqrt(eigenvalues.min() / eigenvalues.max())) if eigenvalues.max() > 0 else np.nan
			)
		else:
			features["largest_island_extent_px"] = float(len(xs))
			features["largest_island_axis_ratio"] = np.nan
	else:
		features["largest_island_extent_px"] = 0.0
		features["largest_island_axis_ratio"] = np.nan

	# flux-weighted centroid of the emission, and how far it is from the image centre
	if mask3.any():
		weights = np.where(mask3, data - background, 0)
		cy, cx = ndimage.center_of_mass(weights)
		centre_y, centre_x = (np.array(data.shape) - 1) / 2
		features["centroid_offset_px"] = float(np.hypot(cx - centre_x, cy - centre_y))
		features["emission_flux"] = float(weights.sum())  # relative to the image peak
	else:
		features["centroid_offset_px"] = np.nan
		features["emission_flux"] = 0.0

	return features, mask3


def pair_features(data_a, data_b, mask_a, mask_b):
	features = {}
	if data_a.shape != data_b.shape:
		return {"pearson_r": np.nan, "mask_iou": np.nan}
	a, b = data_a.ravel(), data_b.ravel()
	features["pearson_r"] = float(pearsonr(a, b)[0]) if a.std() > 0 and b.std() > 0 else np.nan
	union = np.logical_or(mask_a, mask_b).sum()
	features["mask_iou"] = float(np.logical_and(mask_a, mask_b).sum() / union) if union else np.nan
	return features


def source_features(source_id):
	row = {}
	images, masks = {}, {}
	for survey in SURVEYS:
		data = load_image(survey, source_id)
		row[f"{survey}_available"] = data is not None
		if data is None:
			continue
		features, mask = image_features(data)
		images[survey], masks[survey] = data, mask
		row.update({f"{survey}_{key}": value for key, value in features.items()})

	for first, second in SURVEY_PAIRS:
		if first in images and second in images:
			prefix = f"{first}_vs_{second}"
			features = pair_features(images[first], images[second], masks[first], masks[second])
			row.update({f"{prefix}_{key}": value for key, value in features.items()})
			n_a, n_b = row[f"{first}_n_pix_3sigma"], row[f"{second}_n_pix_3sigma"]
			row[f"{prefix}_log_size_ratio"] = np.log10((n_a + 1) / (n_b + 1))
			snr_a, snr_b = row[f"{first}_peak_snr"], row[f"{second}_peak_snr"]
			row[f"{prefix}_log_snr_ratio"] = (
				np.log10(snr_a / snr_b) if snr_a > 0 and snr_b > 0 else np.nan
			)
	return row


# ---------- catalogue information ----------

def parse_votes(text):
	try:
		votes = ast.literal_eval(text)
		return [str(v) for v in votes] if isinstance(votes, (list, tuple)) else [str(votes)]
	except (ValueError, SyntaxError):
		return []


def catalogue_features(catalogue, split):
	table = catalogue[["Catalog", "Type", "Original", "RA/deg", "DEC/deg"]].copy()
	table["relabelled"] = np.where(table["Type"] != table["Original"], "changed", "unchanged")
	votes = catalogue["Divergence"].map(parse_votes)
	table["n_votes"] = votes.map(len)
	table["n_distinct_vote_labels"] = votes.map(lambda v: len(set(v)))
	table["vote_agreement"] = [
		v.count(label) / len(v) if v else np.nan for v, label in zip(votes, catalogue["Type"])
	]
	table["vote_unanimous"] = np.where(table["n_distinct_vote_labels"] <= 1, "unanimous", "split vote")
	table["split"] = [split.get(int(i), "not used") for i in table.index]
	return table


# ---------- statistics ----------

def benjamini_hochberg(p_values):
	p = np.asarray(p_values, dtype=float)
	result = np.full_like(p, np.nan)
	valid = ~np.isnan(p)
	if not valid.any():
		return result
	order = np.argsort(p[valid])
	ranked = p[valid][order] * valid.sum() / np.arange(1, valid.sum() + 1)
	ranked = np.minimum.accumulate(ranked[::-1])[::-1]
	adjusted = np.empty_like(ranked)
	adjusted[order] = np.minimum(ranked, 1)
	result[valid] = adjusted
	return result


def compare_numeric(table):
	rows = []
	numeric = table.select_dtypes(include=[np.number]).columns
	for column in numeric:
		yes = table.loc[table["decision"] == "Yes", column].dropna().astype(float)
		no = table.loc[table["decision"] == "No", column].dropna().astype(float)
		yes, no = yes[np.isfinite(yes)], no[np.isfinite(no)]
		if len(yes) < 3 or len(no) < 3 or (yes.nunique() <= 1 and no.nunique() <= 1):
			continue
		u, p_mw = mannwhitneyu(no, yes, alternative="two-sided")
		auc = u / (len(no) * len(yes))  # P(No value > Yes value), ties count half
		rows.append({
			"feature": column,
			"n_yes": len(yes), "n_no": len(no),
			"median_yes": yes.median(), "median_no": no.median(),
			"mean_yes": yes.mean(), "mean_no": no.mean(),
			"std_yes": yes.std(), "std_no": no.std(),
			"iqr_yes": yes.quantile(0.75) - yes.quantile(0.25),
			"iqr_no": no.quantile(0.75) - no.quantile(0.25),
			"cliffs_delta_no_vs_yes": 2 * auc - 1,
			"auc_single_feature": max(auc, 1 - auc),
			"mannwhitney_p": p_mw,
			"ks_p": ks_2samp(no, yes).pvalue,
		})
	result = pd.DataFrame(rows)
	result["mannwhitney_p_fdr"] = benjamini_hochberg(result["mannwhitney_p"])
	result["ks_p_fdr"] = benjamini_hochberg(result["ks_p"])
	result["abs_delta"] = result["cliffs_delta_no_vs_yes"].abs()
	return result.sort_values("abs_delta", ascending=False).drop(columns="abs_delta")


def compare_categorical(table, column):
	counts = pd.crosstab(table[column], table["decision"])
	for decision in DECISIONS.values():
		if decision not in counts:
			counts[decision] = 0
	counts = counts[["Yes", "No"]]
	counts["total"] = counts.sum(axis=1)
	counts["no_rate"] = counts["No"] / counts["total"]
	counts = counts.sort_values("total", ascending=False)
	observed = counts[["Yes", "No"]]
	observed = observed[observed.sum(axis=1) > 0]
	if len(observed) > 1 and (observed.sum(axis=0) > 0).all():
		_, p, _, expected = chi2_contingency(observed)
		low_expected = bool((expected < 5).any())
	else:
		p, low_expected = np.nan, False
	return counts, p, low_expected


# ---------- plots ----------

def style_axis(axis):
	for side in ("top", "right"):
		axis.spines[side].set_visible(False)
	axis.spines["left"].set_color(TEXT_MUTED)
	axis.spines["bottom"].set_color(TEXT_MUTED)
	axis.tick_params(colors=TEXT_MUTED, labelcolor="black")
	axis.grid(axis="x", color="#e5e4df", linewidth=0.8)
	axis.set_axisbelow(True)


def plot_no_rate(counts, column, overall_rate, path):
	counts = counts.sort_values("no_rate")
	fig, axis = plt.subplots(figsize=(7, 0.45 * len(counts) + 1.5))
	axis.barh(counts.index.astype(str), counts["no_rate"] * 100, color=COLOURS["No"], height=0.6)
	axis.axvline(overall_rate * 100, color=TEXT_MUTED, linestyle="--", linewidth=1)
	axis.text(overall_rate * 100, len(counts) - 0.4, f" overall {overall_rate:.0%}", color=TEXT_MUTED, fontsize=9)
	for position, (rate, no, total) in enumerate(zip(counts["no_rate"], counts["No"], counts["total"])):
		axis.text(rate * 100 + 0.5, position, f"{no}/{total}", va="center", fontsize=9)
	axis.set_xlabel("Sources marked No (%)")
	axis.set_title(f"No-rate by {column}", loc="left")
	style_axis(axis)
	fig.tight_layout()
	fig.savefig(path, dpi=150)
	plt.close(fig)


def plot_distributions(table, features, path):
	columns = 4
	rows = int(np.ceil(len(features) / columns))
	fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
	for axis, (feature, delta, p) in zip(axes.flat, features):
		values = table[feature].replace([np.inf, -np.inf], np.nan)
		title = feature
		# long-tailed non-negative measurements (counts, SNR) are unreadable on a linear axis
		if values.min() >= 0 and np.nanpercentile(values, 99) > 20 * max(np.nanmedian(values), 1e-9):
			values = np.log10(1 + values)
			title = f"log10(1 + {feature})"
		low, high = np.nanpercentile(values, [1, 99])
		if high <= low:
			high = low + 1
		bins = np.linspace(low, high, 31)
		for decision in ("Yes", "No"):
			subset = values[table["decision"] == decision].dropna().clip(low, high)
			axis.hist(subset, bins=bins, density=True, histtype="step", linewidth=2,
				color=COLOURS[decision], label=f"{decision} (n={len(subset)})")
		axis.set_title(title, fontsize=10, loc="left")
		axis.text(0.98, 0.95, f"δ={delta:+.2f}\np={p:.1e}", transform=axis.transAxes,
			ha="right", va="top", fontsize=8, color=TEXT_MUTED)
		axis.set_yticks([])
		style_axis(axis)
		axis.grid(False)
	for axis in list(axes.flat)[len(features):]:
		axis.axis("off")
	handles, labels = axes.flat[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper right", frameon=False)
	fig.suptitle("Measurements that differ most between Yes and No (δ = Cliff's delta, No vs Yes)", x=0.01, ha="left")
	fig.tight_layout(rect=(0, 0, 1, 0.96))
	fig.savefig(path, dpi=150)
	plt.close(fig)


def plot_examples(table, path, per_group=6):
	"""A few randomly chosen Yes and No sources, each shown as FIRST above LoTSS."""
	fig, axes = plt.subplots(2 * 2, per_group, figsize=(2 * per_group, 8.5))
	rng = np.random.default_rng(0)
	for group, decision in enumerate(("Yes", "No")):
		ids = table.index[table["decision"] == decision].to_numpy()
		chosen = rng.choice(ids, size=min(per_group, len(ids)), replace=False)
		for column, source_id in enumerate(chosen):
			for offset, survey in enumerate(("FIRST", "LoTSS")):
				axis = axes[2 * group + offset, column]
				data = load_image(survey, source_id)
				if data is not None:
					axis.imshow(data, origin="lower", cmap="inferno")
				axis.set_xticks([])
				axis.set_yticks([])
				if offset == 0:
					axis.set_title(f"{decision}: {source_id}", fontsize=9, color=COLOURS[decision])
				if column == 0:
					axis.set_ylabel(survey)
	fig.suptitle("Random examples (top two rows: Yes, bottom two rows: No)", x=0.01, ha="left")
	fig.tight_layout(rect=(0, 0, 1, 0.96))
	fig.savefig(path, dpi=120)
	plt.close(fig)


# ---------- main ----------

def main():
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	catalogue = pd.read_csv(CATALOGUE, index_col=0)
	decisions = load_decisions()
	reviewed_ids = [i for i in catalogue.index if int(i) in decisions]
	print(f"Measuring {len(reviewed_ids)} reviewed sources ...")

	measurements = pd.DataFrame(
		[source_features(int(i)) for i in reviewed_ids], index=pd.Index(reviewed_ids, name="id")
	)
	table = catalogue_features(catalogue.loc[reviewed_ids], load_split()).join(measurements)
	table.insert(0, "decision", [decisions[int(i)] for i in table.index])
	table.to_csv(OUTPUT_DIR / "source_features.csv")

	lines = []
	n_yes, n_no = (table["decision"] == "Yes").sum(), (table["decision"] == "No").sum()
	overall_rate = n_no / len(table)
	lines.append(f"Reviewed sources: {len(table)}   Yes: {n_yes}   No: {n_no}   No-rate: {overall_rate:.1%}")
	skipped = sum(1 for v in json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))["decisions"].values() if v == "skip")
	lines.append(f"Skipped (excluded): {skipped}   Not yet reviewed: {len(catalogue) - len(table) - skipped}")
	for survey in SURVEYS:
		missing = (~table[f"{survey}_available"]).sum()
		if missing:
			lines.append(f"  {survey} image missing for {missing} sources")

	lines.append("\n=== Catalogue information (No-rate per category) ===")
	for column in CATEGORICAL_COLUMNS:
		counts, p, low_expected = compare_categorical(table, column)
		counts.to_csv(OUTPUT_DIR / f"categorical_{column}.csv")
		warning = "  (some expected counts < 5, treat p with care)" if low_expected else ""
		lines.append(f"\n{column}: chi-squared p = {p:.2e}{warning}")
		for name, row in counts.iterrows():
			lines.append(f"  {str(name):<14} {int(row['No']):>4} / {int(row['total']):<5} No  ({row['no_rate']:.1%})")
		if counts.shape[0] > 1:
			plot_no_rate(counts, column, overall_rate, OUTPUT_DIR / f"no_rate_by_{column}.png")

	comparison = compare_numeric(table)
	comparison.to_csv(OUTPUT_DIR / "numeric_comparison.csv", index=False)
	lines.append("\n=== Numeric measurements, sorted by effect size ===")
	lines.append("Cliff's delta > 0 means No sources tend to have larger values. |delta| ~0.15 small, ~0.33 medium, ~0.47 large.")
	lines.append("AUC = how well that one measurement alone separates Yes from No (0.5 = not at all, 1 = perfectly).")
	lines.append(f"{'feature':<44}{'median Yes':>12}{'median No':>12}{'delta':>8}{'AUC':>6}{'p (FDR)':>10}")
	for _, row in comparison.iterrows():
		lines.append(
			f"{row['feature']:<44}{row['median_yes']:>12.4g}{row['median_no']:>12.4g}"
			f"{row['cliffs_delta_no_vs_yes']:>+8.2f}{row['auc_single_feature']:>6.2f}{row['mannwhitney_p_fdr']:>10.1e}"
		)

	top = comparison.head(16)
	plot_distributions(table, list(zip(top["feature"], top["cliffs_delta_no_vs_yes"], top["mannwhitney_p_fdr"])),
		OUTPUT_DIR / "top_feature_distributions.png")
	plot_examples(table, OUTPUT_DIR / "example_images.png")

	report = "\n".join(lines)
	print(report)
	(OUTPUT_DIR / "summary.txt").write_text(report, encoding="utf-8")
	print(f"\nResults written to {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
