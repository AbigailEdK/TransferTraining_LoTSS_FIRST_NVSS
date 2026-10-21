"""Publication figures comparing the sources marked Yes and No in the morphology review.

Reads Results/MorphologyReviewStats/source_features.csv (made by morphologyReviewStats.py, which is
run first if the file is missing) and writes vector PDF and 300 dpi PNG figures, plus a LaTeX table,
to Results/MorphologyReviewStats/PaperFigures/:

	fig1_example_cutouts      representative Yes / No sources (FIRST above LoTSS, 3-sigma contours)
	fig2_distributions        the six measurements that best explain the difference
	fig3_snr_plane            FIRST vs LoTSS peak S/N for every source
	fig4_catalogue_breakdown  share marked No per source catalogue and per label
	fig5_effect_sizes         Cliff's delta with bootstrap 95% intervals for every key measurement
	table_key_measurements.tex

Widths follow a two-column journal layout (single column 3.35 in, full width 7.0 in). Fonts, colours
and colour maps match paperModelEvaluation.py: Yes / No use its selected / baseline blue and orange.
Yes and No also differ in line style and marker, so the figures still read in greyscale.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, sigmaclip

import morphologyReviewStats as stats

FEATURES_CSV = stats.OUTPUT_DIR / "source_features.csv"
FIGURE_DIR = stats.OUTPUT_DIR / "PaperFigures"

SINGLE_COLUMN = 3.35  # inches
FULL_WIDTH = 7.0
PIXEL_ARCSEC = 1.5  # CDELT of the 128 x 128 cutouts
BOOTSTRAP_SAMPLES = 1000

COLOURS = stats.COLOURS
LINESTYLES = {"Yes": "-", "No": "--"}
MARKERS = {"Yes": "o", "No": "^"}
# neutral colours used by paperModelEvaluation.py: grey reference lines, Silver bands
GRID = "#dedede"
MUTED = "grey"

# column -> (axis label, multiplier to convert units); the order is the order in fig 5 and the table
KEY_MEASUREMENTS = {
	"FIRST_peak_snr": ("FIRST peak S/N", 1),
	"FIRST_noise": (r"FIRST noise ($\sigma$ / peak)", 1),
	"FIRST_n_pix_3sigma": (r"FIRST pixels $>3\sigma$", 1),
	"FIRST_largest_island_extent_px": ("FIRST largest component extent (arcsec)", PIXEL_ARCSEC),
	"FIRST_centroid_offset_px": ("FIRST emission offset from centre (arcsec)", PIXEL_ARCSEC),
	"LoTSS_peak_snr": ("LoTSS peak S/N", 1),
	"LoTSS_noise": (r"LoTSS noise ($\sigma$ / peak)", 1),
	"LoTSS_n_pix_3sigma": (r"LoTSS pixels $>3\sigma$", 1),
	"LoTSS_largest_island_extent_px": ("LoTSS largest component extent (arcsec)", PIXEL_ARCSEC),
	"LoTSS_centroid_offset_px": ("LoTSS emission offset from centre (arcsec)", PIXEL_ARCSEC),
	"NVSS_peak_snr": ("NVSS peak S/N", 1),
	"NVSS_n_pix_3sigma": (r"NVSS pixels $>3\sigma$", 1),
	"FIRST_vs_LoTSS_pearson_r": ("FIRST-LoTSS pixel correlation $r$", 1),
	"FIRST_vs_LoTSS_mask_iou": (r"FIRST-LoTSS $3\sigma$ mask overlap (IoU)", 1),
	"FIRST_vs_LoTSS_log_size_ratio": (r"$\log_{10}$ FIRST/LoTSS emission size", 1),
}
GROUPS = (
	("FIRST", "FIRST_"), ("LoTSS", "LoTSS_"), ("NVSS", "NVSS_"), ("FIRST vs LoTSS", "FIRST_vs_LoTSS_"),
)


def set_style():
	"""Match paperModelEvaluation.py: matplotlib's default font and ticks, with enlarged text.

	paperModelEvaluation.py uses 20 pt labels / 24 pt titles on panels about twice the print width used here,
	so the sizes are halved to give the same text size relative to the figure once both are placed in the paper.
	"""
	plt.rcParams.update({
		"font.size": 10,
		"axes.titlesize": 12,
		"figure.titlesize": 12,
		"legend.fontsize": 10,
		"axes.labelsize": 10,
		"xtick.labelsize": 10,
		"ytick.labelsize": 10,
		"legend.frameon": False,
		"savefig.bbox": "tight",
		"pdf.fonttype": 42,  # embed TrueType so the PDF text stays editable / searchable
	})


def save(fig, name):
	for extension, dpi in (("pdf", None), ("png", 300)):
		fig.savefig(FIGURE_DIR / f"{name}.{extension}", dpi=dpi)
	plt.close(fig)
	print(f"  {name}")


def panel_label(axis, letter):
	# above the axes, so it never sits on top of a tall histogram bin
	axis.set_title(f"({letter})", loc="left", fontweight="bold")


def decision_legend(table, lines=True):
	"""Legend handles for Yes / No: line style for histograms, or markers only for scatter plots."""
	return [
		Line2D([], [], color=COLOURS[d], linestyle=LINESTYLES[d] if lines else "none",
			marker=None if lines else MARKERS[d], markersize=4,
			label=f"{d} ($n={(table['decision'] == d).sum()}$)")
		for d in ("Yes", "No")
	]


def cliffs_delta(no, yes):
	u = mannwhitneyu(no, yes, alternative="two-sided").statistic
	return 2 * u / (len(no) * len(yes)) - 1


def bootstrap_delta(no, yes, rng):
	samples = [
		cliffs_delta(rng.choice(no, len(no)), rng.choice(yes, len(yes))) for _ in range(BOOTSTRAP_SAMPLES)
	]
	return np.percentile(samples, [2.5, 97.5])


def group_values(table, column, scale=1):
	values = table[column].replace([np.inf, -np.inf], np.nan) * scale
	return {d: values[table["decision"] == d].dropna().to_numpy() for d in ("Yes", "No")}


# ---------- figure 1: example cutouts ----------

def pick_examples(table, per_group=3):
	"""Sources nearest the 25th / 50th / 75th percentile of FIRST peak S/N within each group."""
	chosen = {}
	for decision in ("Yes", "No"):
		subset = table.loc[table["decision"] == decision, "FIRST_peak_snr"].dropna()
		targets = np.percentile(subset, np.linspace(25, 75, per_group))
		ids = []
		for target in targets:
			remaining = subset.drop(ids)
			ids.append((remaining - target).abs().idxmin())
		chosen[decision] = ids
	return chosen


def draw_cutout(axis, data):
	clipped, _, _ = sigmaclip(data.ravel(), low=stats.CLIP_SIGMA, high=stats.CLIP_SIGMA)
	background, noise = np.median(clipped), np.std(clipped)
	axis.imshow(data, origin="lower", cmap="viridis", vmin=background, vmax=data.max(), interpolation="nearest")
	if noise > 0 and data.max() > background + 3 * noise:
		axis.contour(data, levels=[background + 3 * noise], colors="white", linewidths=0.5)
	axis.set_xticks([])
	axis.set_yticks([])
	return (data.max() - background) / noise if noise > 0 else np.nan


def figure_examples(table, group_prefix="Marked"):
	examples = pick_examples(table)
	columns = [(d, i) for d in ("Yes", "No") for i in examples[d]]
	fig, axes = plt.subplots(2, len(columns), figsize=(FULL_WIDTH, FULL_WIDTH / len(columns) * 2 + 0.45),
		gridspec_kw={"wspace": 0.04, "hspace": 0.04})
	for column, (decision, source_id) in enumerate(columns):
		for row, survey in enumerate(("FIRST", "LoTSS")):
			axis = axes[row, column]
			data = stats.load_image(survey, int(source_id))
			if data is None:
				axis.axis("off")
				continue
			snr = draw_cutout(axis, data)
			axis.text(0.04, 0.04, f"S/N {snr:.0f}", transform=axis.transAxes, color="white", fontsize="small")
			for spine in axis.spines.values():
				spine.set_color(COLOURS[decision])
				spine.set_linewidth(1.2)
			if column == 0:
				axis.set_ylabel(survey)
		catalogue = table.at[source_id, "Catalog"]
		label = table.at[source_id, "Type"]
		axes[0, column].set_title(f"{source_id}\n{catalogue}, {label}", fontsize="small")

	# 1 arcmin scale bar on the first panel
	bar = 60 / PIXEL_ARCSEC
	axes[0, 0].plot([8, 8 + bar], [116, 116], color="white", linewidth=1.2)
	axes[0, 0].text(8 + bar / 2, 106, "1'", color="white", ha="center", va="top", fontsize="small")

	half = len(columns) // 2
	for decision, start in (("Yes", 0), ("No", half)):
		left = axes[0, start].get_position().x0
		right = axes[0, start + half - 1].get_position().x1
		fig.text((left + right) / 2, axes[0, 0].get_position().y1 + 0.13, f"{group_prefix} {decision}",
			ha="center", va="bottom", color=COLOURS[decision], fontweight="bold")
	save(fig, "fig1_example_cutouts")


# ---------- figure 2: key distributions ----------

def histogram_panel(axis, table, column, label, scale=1, log=False):
	groups = group_values(table, column, scale)
	everything = np.concatenate(list(groups.values()))
	if log:
		everything = everything[everything > 0]
		low, high = np.percentile(everything, [0.5, 99.5])
		bins = np.geomspace(low, high, 36)
		axis.set_xscale("log")
	else:
		low, high = np.percentile(everything, [0.5, 99.5])
		bins = np.linspace(low, high, 36)
	for decision in ("Yes", "No"):
		values = np.clip(groups[decision], bins[0], bins[-1])
		weights = np.full(len(values), 1 / len(values))  # fraction of the group, so the groups compare
		axis.hist(values, bins=bins, weights=weights, histtype="step", linewidth=1.1,
			color=COLOURS[decision], linestyle=LINESTYLES[decision])
		axis.axvline(np.median(groups[decision]), color=COLOURS[decision], linestyle=":", linewidth=0.9)
	delta = cliffs_delta(groups["No"], groups["Yes"])
	axis.text(0.97, 0.96, rf"$\delta = {delta:+.2f}$", transform=axis.transAxes, ha="right", va="top")
	axis.set_xlabel(label)
	axis.set_ylim(0, axis.get_ylim()[1] * 1.12)  # headroom for the delta label


def figure_distributions(table):
	panels = (
		("FIRST_peak_snr", True),
		("LoTSS_peak_snr", True),
		("FIRST_vs_LoTSS_pearson_r", False),
		("FIRST_vs_LoTSS_mask_iou", False),
		("FIRST_largest_island_extent_px", False),
		("FIRST_centroid_offset_px", False),
	)
	fig, axes = plt.subplots(3, 2, figsize=(FULL_WIDTH, 7.5))
	for axis, letter, (column, log) in zip(axes.flat, "abcdef", panels):
		label, scale = KEY_MEASUREMENTS[column]
		histogram_panel(axis, table, column, label, scale, log)
		panel_label(axis, letter)
	for axis in axes[:, 0]:
		axis.set_ylabel("Fraction of group")
	handles = decision_legend(table)
	handles.append(Line2D([], [], color=MUTED, linestyle=":", label="Group median"))
	fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
	fig.tight_layout(rect=(0, 0, 1, 0.97))
	save(fig, "fig2_distributions")


# ---------- figure 3: S/N plane ----------

def figure_snr_plane(table):
	fig, axis = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN))
	for decision, size, alpha in (("Yes", 5, 0.45), ("No", 9, 0.8)):
		subset = table[table["decision"] == decision]
		axis.scatter(subset["FIRST_peak_snr"], subset["LoTSS_peak_snr"], s=size, alpha=alpha,
			marker=MARKERS[decision], color=COLOURS[decision], linewidths=0, rasterized=True)
	axis.set_xscale("log")
	axis.set_yscale("log")
	axis.axvline(5, color="black", linestyle="--", linewidth=0.7)
	axis.text(5.3, axis.get_ylim()[1] * 0.75, r"$5\sigma$", va="top")
	axis.set_xlabel("FIRST peak S/N")
	axis.set_ylabel("LoTSS peak S/N")
	axis.legend(handles=decision_legend(table, lines=False), loc="lower right")
	save(fig, "fig3_snr_plane")


# ---------- figure 4: catalogue breakdown ----------

def stacked_share(axis, table, column, order=None):
	counts = pd.crosstab(table[column], table["decision"]).reindex(columns=["Yes", "No"], fill_value=0)
	counts["total"] = counts.sum(axis=1)
	counts = counts.loc[order] if order is not None else counts.sort_values("total")
	share = counts[["Yes", "No"]].div(counts["total"], axis=0) * 100
	positions = np.arange(len(counts))
	axis.barh(positions, share["Yes"], color=COLOURS["Yes"], height=0.7, edgecolor="white", linewidth=0.5)
	axis.barh(positions, share["No"], left=share["Yes"], color=COLOURS["No"], height=0.7,
		edgecolor="white", linewidth=0.5, hatch="////")
	axis.set_yticks(positions, counts.index)
	axis.set_xlim(0, 100)
	axis.set_xlabel("Percentage of sources")
	axis.tick_params(axis="y", length=0, right=False)
	for position, (no, total) in enumerate(zip(counts["No"], counts["total"])):
		axis.text(101, position, f"{no}/{total}", va="center", fontsize="small", clip_on=False)


def figure_catalogues(table):
	fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0), gridspec_kw={"width_ratios": (1.6, 1), "wspace": 0.6})
	stacked_share(left, table, "Catalog")
	stacked_share(right, table, "Type", order=["COMPACT", "FRII", "FRI"])
	left.set_title("(a) Catalogue", loc="left", fontweight="bold")
	right.set_title("(b) Label", loc="left", fontweight="bold")
	for axis in (left, right):
		axis.text(1.0, 1.02, "No / total", transform=axis.transAxes, fontsize="small", ha="left")
	handles = [
		plt.Rectangle((0, 0), 1, 1, color=COLOURS["Yes"], label="Yes"),
		plt.Rectangle((0, 0), 1, 1, facecolor=COLOURS["No"], hatch="////", edgecolor="white", label="No"),
	]
	fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.97))
	save(fig, "fig4_catalogue_breakdown")


# ---------- figure 5 + table: effect sizes ----------

def effect_size_table(table):
	rng = np.random.default_rng(0)
	comparison = stats.compare_numeric(table).set_index("feature")
	rows = []
	for column, (label, scale) in KEY_MEASUREMENTS.items():
		groups = group_values(table, column, scale)
		low, high = bootstrap_delta(groups["No"], groups["Yes"], rng)
		rows.append({
			"feature": column, "label": label,
			"median_yes": np.median(groups["Yes"]), "median_no": np.median(groups["No"]),
			"q1_yes": np.percentile(groups["Yes"], 25), "q3_yes": np.percentile(groups["Yes"], 75),
			"q1_no": np.percentile(groups["No"], 25), "q3_no": np.percentile(groups["No"], 75),
			"delta": comparison.at[column, "cliffs_delta_no_vs_yes"], "delta_low": low, "delta_high": high,
			"auc": comparison.at[column, "auc_single_feature"],
			"p_fdr": comparison.at[column, "mannwhitney_p_fdr"],
		})
	return pd.DataFrame(rows)


def survey_group(feature):
	# longest prefix wins so FIRST_vs_LoTSS_* is not filed under FIRST
	return max((g for g in GROUPS if feature.startswith(g[1])), key=lambda g: len(g[1]))[0]


def figure_effect_sizes(effects):
	effects = effects.copy()
	effects["group"] = effects["feature"].map(survey_group)
	fig, axis = plt.subplots(figsize=(SINGLE_COLUMN, 0.2 * len(effects) + 1.1))
	# shaded band = negligible effect (|delta| < 0.147); the large-effect boundary (0.474) is dotted
	axis.axvspan(-0.147, 0.147, color=GRID, alpha=0.6, linewidth=0, zorder=0)
	for sign in (-1, 1):
		axis.axvline(sign * 0.474, color=MUTED, linestyle=":", linewidth=0.6, zorder=0)
	axis.axvline(0, color="black", linewidth=0.6)

	positions, labels, y = [], [], 0
	previous_group = None
	for _, row in effects.iterrows():
		if row["group"] != previous_group:
			if previous_group is not None:
				y += 0.6
				axis.axhline(y - 0.8, color=GRID, linewidth=0.6)
			previous_group = row["group"]
		positions.append(y)
		labels.append(row["label"].replace(f"{row['group']} ", "") if row["group"] != "FIRST vs LoTSS" else row["label"])
		colour = COLOURS["No"] if row["delta"] > 0 else COLOURS["Yes"]
		axis.errorbar(row["delta"], y, xerr=[[row["delta"] - row["delta_low"]], [row["delta_high"] - row["delta"]]],
			fmt="o", markersize=3.5, color=colour, elinewidth=0.9, capsize=1.5)
		y += 1

	axis.set_yticks(positions, labels)
	axis.tick_params(axis="y", length=0, right=False)
	axis.set_ylim(y - 0.4, -1.25)
	axis.set_xlim(-1.05, 1.05)
	axis.set_xlabel(r"Cliff's $\delta$ (No vs Yes)")
	# group names in the left margin, beside their first row
	for group, _ in GROUPS:
		rows = [p for p, g in zip(positions, effects["group"]) if g == group]
		if rows:
			axis.text(-1.02, rows[0] - 0.55, group, fontsize="small", fontweight="bold", va="bottom")
	axis.text(1.0, 1.01, r"No larger $\rightarrow$", transform=axis.transAxes, ha="right", va="bottom",
		fontsize="small", color=COLOURS["No"])
	axis.text(0.0, 1.01, r"$\leftarrow$ Yes larger", transform=axis.transAxes, ha="left", va="bottom",
		fontsize="small", color=COLOURS["Yes"])
	save(fig, "fig5_effect_sizes")


def format_number(value):
	if abs(value) >= 10:
		text = f"{value:.0f}"
	elif abs(value) >= 1:
		text = f"{value:.1f}"
	else:
		text = f"{value:.2g}"
	return text.replace("-", "$-$")  # typeset minus sign


def format_p(p):
	if p < 1e-3:
		mantissa, exponent = f"{p:.1e}".split("e")
		return rf"${mantissa}\times10^{{{int(exponent)}}}$"
	return f"{p:.3f}"


def write_latex_table(effects, table, how_labelled="were judged to show the same morphology", label="tab:morphology_review"):
	n_yes, n_no = (table["decision"] == "Yes").sum(), (table["decision"] == "No").sum()
	lines = [
		r"\begin{table*}",
		r"\centering",
		rf"\caption{{Image measurements for sources whose FIRST and LoTSS images {how_labelled} "
		rf"(Yes, $n={n_yes}$) and those that did not (No, $n={n_no}$). Values are medians with the "
		r"interquartile range in brackets. Noise is the standard deviation of the $3\sigma$-clipped image as a "
		r"fraction of the image peak. $\delta$ is Cliff's delta (positive when No values are larger) with a "
		r"bootstrap 95\% interval; AUC is the area under the ROC curve when the measurement alone separates the "
		r"groups; $p$ is the Mann--Whitney $U$ test, Benjamini--Hochberg corrected.}",
		rf"\label{{{label}}}",
		r"\begin{tabular}{lccccc}",
		r"\hline",
		r"Measurement & Yes & No & $\delta$ & AUC & $p$ \\",
		r"\hline",
	]
	for _, row in effects.iterrows():
		label = row["label"].replace("%", r"\%")
		yes = f"{format_number(row['median_yes'])} [{format_number(row['q1_yes'])}, {format_number(row['q3_yes'])}]"
		no = f"{format_number(row['median_no'])} [{format_number(row['q1_no'])}, {format_number(row['q3_no'])}]"
		delta = f"${row['delta']:+.2f}$ $[{row['delta_low']:+.2f}, {row['delta_high']:+.2f}]$"
		lines.append(f"{label} & {yes} & {no} & {delta} & {row['auc']:.2f} & {format_p(row['p_fdr'])} \\\\")
	lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
	(FIGURE_DIR / "table_key_measurements.tex").write_text("\n".join(lines), encoding="utf-8")
	print("  table_key_measurements.tex")


def main():
	if not FEATURES_CSV.exists():
		print("source_features.csv not found, running morphologyReviewStats.py first ...")
		stats.main()
	FIGURE_DIR.mkdir(parents=True, exist_ok=True)
	table = pd.read_csv(FEATURES_CSV, index_col=0)
	set_style()
	print(f"Writing figures to {FIGURE_DIR}")
	figure_examples(table)
	figure_distributions(table)
	figure_snr_plane(table)
	figure_catalogues(table)
	effects = effect_size_table(table)
	figure_effect_sizes(effects)
	write_latex_table(effects, table)

	no = table[table["decision"] == "No"]
	yes = table[table["decision"] == "Yes"]
	print("\nNumbers for the caption / text:")
	print(f"  No sources with FIRST peak S/N < 5: {(no['FIRST_peak_snr'] < 5).mean():.1%}"
		f"   (Yes: {(yes['FIRST_peak_snr'] < 5).mean():.1%})")
	print(f"  No sources from LoMorph: {(no['Catalog'] == 'LoMorph').sum()} of {len(no)}")


if __name__ == "__main__":
	main()
