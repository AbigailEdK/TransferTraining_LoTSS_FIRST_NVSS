"""Review whether the FIRST and NVSS images of each source show the same morphology.

The same review window as morphologyReview.py (FIRST / LoTSS), with the NVSS image in place of LoTSS.
Each source in Data/RADCAT.csv is shown with its FIRST and NVSS image next to its current label.
Answer Yes / No / Skip; progress is saved after every answer and when the window is closed, so
running the script again continues where you left off.

Answers are kept separately from the FIRST / LoTSS review, in
	morphology_review_FIRST_vs_NVSS_progress.json
	morphology_review_FIRST_vs_NVSS_results.csv
and are picked up by morphologyReviewStats.py, morphologyReviewFigures.py and
morphologyAnalyticalClassification.py, which write them to MorphologyReviewStats/FIRST_vs_NVSS/.

Keys: Left / Right = back / forward, Y = Yes, N = No, S = Skip.
"""

import morphologyReview
from reviewComparisons import COMPARISONS

if __name__ == "__main__":
	morphologyReview.main(COMPARISONS["FIRST_vs_NVSS"])
