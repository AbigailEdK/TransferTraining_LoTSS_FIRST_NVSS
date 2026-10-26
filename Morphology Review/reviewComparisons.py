"""The survey pairs compared in the morphology review, and where each one's answers and results are kept.

Shared by the review programs (morphologyReview.py, morphologyReviewNVSS.py) and the evaluation scripts
(morphologyReviewStats.py, morphologyReviewFigures.py, morphologyAnalyticalClassification.py).
Kept free of Tk / matplotlib imports so the headless evaluation scripts can import it.
"""

from dataclasses import dataclass
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parent
ROOT = REVIEW_DIR.parent
DATA_DIR = ROOT / "Data"
CATALOGUE = DATA_DIR / "RADCAT.csv"
CHANNEL_DIR = DATA_DIR / "FITS_BY_CHANNEL"
FITS_DIR = DATA_DIR / "RADCAT_F"
STATS_DIR = REVIEW_DIR / "MorphologyReviewStats"

ALL_SURVEYS = ("FIRST", "LoTSS", "NVSS")


@dataclass(frozen=True)
class Comparison:
	name: str  # also the column prefix of the pair measurements in morphologyReviewStats.py
	surveys: tuple
	progress_file: Path
	results_file: Path

	@property
	def label(self):
		return " vs ".join(self.surveys)

	@property
	def output_dir(self):
		return STATS_DIR / self.name

	@property
	def other_surveys(self):
		return tuple(s for s in ALL_SURVEYS if s not in self.surveys)


COMPARISONS = {comparison.name: comparison for comparison in (
	Comparison(
		"FIRST_vs_LoTSS", ("FIRST", "LoTSS"),
		REVIEW_DIR / "morphology_review_progress.json",
		REVIEW_DIR / "morphology_review_results.csv",
	),
	Comparison(
		"FIRST_vs_NVSS", ("FIRST", "NVSS"),
		REVIEW_DIR / "morphology_review_FIRST_vs_NVSS_progress.json",
		REVIEW_DIR / "morphology_review_FIRST_vs_NVSS_results.csv",
	),
)}


def selected_comparisons(names):
	"""The comparisons named on the command line, or else every comparison that has saved review answers."""
	unknown = [name for name in names if name not in COMPARISONS]
	if unknown:
		raise SystemExit(f"Unknown comparison(s): {', '.join(unknown)}. Choose from: {', '.join(COMPARISONS)}")
	if names:
		return [COMPARISONS[name] for name in names]
	return [comparison for comparison in COMPARISONS.values() if comparison.progress_file.exists()]
