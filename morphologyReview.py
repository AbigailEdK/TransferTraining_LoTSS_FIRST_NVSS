"""Review whether the FIRST and LoTSS images of each source show the same morphology.

Each source in Data/RADCAT.csv is shown with its FIRST and LoTSS image next to its current label. Images
are read from Data/FITS_BY_CHANNEL/<id>_<survey>.fits, falling back to Data/RADCAT_F/<survey>/<id>.fits.
Answer Yes / No / Skip; progress is saved after every answer and when the
window is closed, so running the script again continues where you left off.

Keys: Left / Right = back / forward, Y = Yes, N = No, S = Skip.
"""

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
FITS_DIR = ROOT / "Data" / "RADCAT_F"
CHANNEL_DIR = ROOT / "Data" / "FITS_BY_CHANNEL"
CATALOGUE = ROOT / "Data" / "RADCAT.csv"
PROGRESS_FILE = ROOT / "morphology_review_progress.json"
RESULTS_CSV = ROOT / "morphology_review_results.csv"

SURVEYS = ("FIRST", "LoTSS")
# decision code -> text shown to the user / written to the results CSV
DECISION_NAMES = {"yes": "Yes", "no": "No", "skip": "Skipped"}
PILES = {"Remaining": None, "Yes": "yes", "No": "no", "Skipped": "skip"}


def find_image_path(survey, source_id):
	"""First non-empty FITS file for this source, looking in FITS_BY_CHANNEL and then RADCAT_F."""
	candidates = (
		CHANNEL_DIR / f"{source_id}_{survey}.fits",
		FITS_DIR / survey / f"{source_id}.fits",
	)
	existing = [path for path in candidates if path.exists()]
	for path in existing:
		if path.stat().st_size > 0:
			return path, None
	return None, "(empty file)" if existing else "(file not found)"


def load_image(survey, source_id):
	"""Return (2D array, None) or (None, reason the image cannot be shown)."""
	path, problem = find_image_path(survey, source_id)
	if path is None:
		return None, f"Image not available\n{problem}"
	try:
		with fits.open(path) as hdul:
			data = np.squeeze(np.array(hdul[0].data, dtype=float))
	except Exception as error:
		return None, f"Image not available\n(could not read file: {error})"
	if data.ndim != 2:
		return None, "Image not available\n(no 2D image data)"
	return np.nan_to_num(data), None


class ReviewApp:
	def __init__(self, window, catalogue, progress_file=PROGRESS_FILE, results_file=RESULTS_CSV):
		self.window = window
		self.catalogue = catalogue
		self.progress_file = Path(progress_file)
		self.results_file = Path(results_file)
		self.all_ids = [int(source_id) for source_id in catalogue.index]
		self.decisions = {}
		self.mode = "all"
		self.queue = list(self.all_ids)
		self.index = 0
		self.pile_ids = []

		current_id = self._load_progress()
		if current_id in self.queue:
			self.index = self.queue.index(current_id)
		else:
			self.index = self._first_undecided_index()

		self._build_ui()
		self.window.protocol("WM_DELETE_WINDOW", self.close)
		self.refresh()

	@property
	def current_id(self):
		return self.queue[self.index]

	# ---------- persistence ----------

	def _load_progress(self):
		if not self.progress_file.exists():
			return None
		try:
			state = json.loads(self.progress_file.read_text(encoding="utf-8"))
			valid = set(self.all_ids)
			self.decisions = {
				int(source_id): decision for source_id, decision in state["decisions"].items()
				if int(source_id) in valid and decision in DECISION_NAMES
			}
			if state.get("mode") == "skipped":
				queue = [source_id for source_id in state.get("queue", []) if source_id in valid]
				if queue:
					self.mode, self.queue = "skipped", queue
			return state.get("current_id")
		except (ValueError, KeyError, TypeError, AttributeError):
			backup = self.progress_file.with_suffix(".json.bak")
			os.replace(self.progress_file, backup)
			self.decisions = {}
			messagebox.showwarning(
				"Progress file unreadable",
				f"The saved progress could not be read, so a fresh session was started.\n"
				f"The old file was kept as {backup.name}.",
			)
			return None

	def save(self):
		state = {
			"decisions": {str(source_id): decision for source_id, decision in self.decisions.items()},
			"mode": self.mode,
			"queue": self.queue if self.mode == "skipped" else [],
			"current_id": self.current_id,
		}
		try:
			temp_file = self.progress_file.with_suffix(".json.tmp")
			temp_file.write_text(json.dumps(state), encoding="utf-8")
			os.replace(temp_file, self.progress_file)
		except OSError as error:
			messagebox.showerror("Could not save progress", str(error))
			return
		self._write_results()

	def _write_results(self):
		table = self.catalogue[["Type", "Original", "Catalog"]].copy()
		table.insert(0, "decision", [
			DECISION_NAMES.get(self.decisions.get(int(source_id)), "") for source_id in table.index
		])
		try:
			table.to_csv(self.results_file, index_label="id")
		except OSError:
			# Typically the CSV is open in Excel; the JSON progress file is still saved.
			self.status.config(text="Could not update the results CSV (is it open in another program?)")

	def close(self):
		self.save()
		self.window.destroy()

	# ---------- UI ----------

	def _build_ui(self):
		self.window.title("FIRST / LoTSS morphology review")
		self.window.geometry("1250x720")
		self.window.columnconfigure(0, weight=1)
		self.window.rowconfigure(1, weight=1)

		header = ttk.Frame(self.window, padding=8)
		header.grid(row=0, column=0, columnspan=2, sticky="ew")
		self.title_label = ttk.Label(header, font=("Segoe UI", 16, "bold"))
		self.title_label.pack(anchor="w")
		self.detail_label = ttk.Label(header, font=("Segoe UI", 11))
		self.detail_label.pack(anchor="w")
		self.question_label = ttk.Label(
			header, font=("Segoe UI", 12), text="Do the FIRST and LoTSS images show the same morphology?"
		)
		self.question_label.pack(anchor="w", pady=(4, 0))

		self.figure = Figure(figsize=(8.5, 4.3))
		self.axes = [self.figure.add_subplot(1, 2, k + 1) for k in range(len(SURVEYS))]
		self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
		self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=8)

		side = ttk.Frame(self.window, padding=8)
		side.grid(row=1, column=1, rowspan=2, sticky="ns")
		self.pile_var = tk.StringVar(value="Remaining")
		self.pile_buttons = {}
		for name in PILES:
			button = ttk.Radiobutton(
				side, variable=self.pile_var, value=name, command=self._refresh_pile_list
			)
			button.pack(anchor="w")
			self.pile_buttons[name] = button
		list_frame = ttk.Frame(side)
		list_frame.pack(fill="both", expand=True, pady=(6, 0))
		scrollbar = ttk.Scrollbar(list_frame)
		scrollbar.pack(side="right", fill="y")
		self.pile_list = tk.Listbox(
			list_frame, width=26, exportselection=False, yscrollcommand=scrollbar.set
		)
		self.pile_list.pack(side="left", fill="both", expand=True)
		scrollbar.config(command=self.pile_list.yview)
		self.pile_list.bind("<Double-Button-1>", self._jump_to_selected)
		ttk.Label(side, text="Double-click a source to open it").pack(anchor="w")
		self.mode_button = ttk.Button(side, command=self.toggle_mode)
		self.mode_button.pack(fill="x", pady=(8, 0))

		controls = ttk.Frame(self.window, padding=8)
		controls.grid(row=2, column=0, sticky="ew")
		for column in range(5):
			controls.columnconfigure(column, weight=1)
		self.back_button = ttk.Button(controls, text="\u25C0  Back", command=self.back)
		self.yes_button = ttk.Button(controls, text="Yes", command=lambda: self.decide("yes"))
		self.no_button = ttk.Button(controls, text="No", command=lambda: self.decide("no"))
		self.skip_button = ttk.Button(controls, text="Skip", command=lambda: self.decide("skip"))
		self.forward_button = ttk.Button(controls, text="Forward  \u25B6", command=self.forward)
		for column, button in enumerate((
			self.back_button, self.yes_button, self.no_button, self.skip_button, self.forward_button
		)):
			button.grid(row=0, column=column, sticky="ew", padx=4, ipady=8)

		self.status = ttk.Label(self.window, padding=(8, 0, 8, 8))
		self.status.grid(row=3, column=0, columnspan=2, sticky="ew")

		self.window.bind("<Left>", lambda event: self.back())
		self.window.bind("<Right>", lambda event: self.forward())
		for key, decision in (("y", "yes"), ("n", "no"), ("s", "skip")):
			self.window.bind(f"<{key}>", lambda event, d=decision: self.decide(d))

	def _draw_image(self, axis, survey, source_id):
		axis.clear()
		axis.set_title(survey, fontsize=14)
		axis.set_xticks([])
		axis.set_yticks([])
		data, problem = load_image(survey, source_id)
		if data is None:
			axis.text(0.5, 0.5, problem, ha="center", va="center", transform=axis.transAxes, color="grey")
			return
		low, high = float(data.min()), float(data.max())
		axis.imshow(data, origin="lower", cmap="inferno", vmin=low, vmax=high if high > low else low + 1)

	def refresh(self):
		source_id = self.current_id
		row = self.catalogue.loc[source_id]
		decision = self.decisions.get(source_id)
		queue_name = "Skipped review" if self.mode == "skipped" else "Full queue"
		self.title_label.config(text=f"Source {source_id}  -  current label: {row['Type']}")
		self.detail_label.config(
			text=f"{queue_name}: {self.index + 1} of {len(self.queue)}    |    "
			f"Catalogue: {row['Catalog']}    |    Original label: {row['Original']}    |    "
			f"Your answer: {DECISION_NAMES.get(decision, 'not answered yet')}"
		)
		for axis, survey in zip(self.axes, SURVEYS):
			self._draw_image(axis, survey, source_id)
		self.figure.tight_layout()
		self.canvas.draw_idle()

		self.back_button.state(["!disabled"] if self.index > 0 else ["disabled"])
		self.forward_button.state(["!disabled"] if self.index < len(self.queue) - 1 else ["disabled"])
		skipped_count = self._count("skip")
		for name, code in PILES.items():
			count = self._count(code)
			self.pile_buttons[name].config(text=f"{name} ({count})")
		if self.mode == "skipped":
			self.mode_button.config(text="Back to full queue", state="normal")
		else:
			self.mode_button.config(
				text=f"Review skipped ({skipped_count})", state="normal" if skipped_count else "disabled"
			)
		self.status.config(text=f"Progress saved to {self.progress_file.name}")
		self._refresh_pile_list()

	def _count(self, code):
		if code is None:
			return len(self.all_ids) - len(self.decisions)
		return sum(1 for value in self.decisions.values() if value == code)

	def _refresh_pile_list(self):
		code = PILES[self.pile_var.get()]
		if code is None:
			self.pile_ids = [i for i in self.all_ids if i not in self.decisions]
		else:
			self.pile_ids = [i for i in self.all_ids if self.decisions.get(i) == code]
		self.pile_list.delete(0, "end")
		for source_id in self.pile_ids:
			self.pile_list.insert("end", f"{source_id}   {self.catalogue.at[source_id, 'Type']}")
		if self.current_id in self.pile_ids:
			position = self.pile_ids.index(self.current_id)
			self.pile_list.selection_set(position)
			self.pile_list.see(position)

	# ---------- actions ----------

	def _first_undecided_index(self):
		return next((k for k, source_id in enumerate(self.queue) if source_id not in self.decisions), 0)

	def back(self):
		if self.index > 0:
			self.index -= 1
			self.save()
			self.refresh()

	def forward(self):
		if self.index < len(self.queue) - 1:
			self.index += 1
			self.save()
			self.refresh()

	def decide(self, decision):
		# Decisions live in one dict, so changing an answer moves the source out of its old pile.
		self.decisions[self.current_id] = decision
		at_end = self.index == len(self.queue) - 1
		if not at_end:
			self.index += 1
		self.save()
		self.refresh()
		if at_end:
			self._finish_queue()

	def _finish_queue(self):
		unanswered = [source_id for source_id in self.queue if source_id not in self.decisions]
		if unanswered:
			self.index = self.queue.index(unanswered[0])
			self.save()
			self.refresh()
			messagebox.showinfo(
				"Unanswered sources", f"{len(unanswered)} source(s) were left unanswered. Going to the first one."
			)
			return
		if self._count("skip"):
			if messagebox.askyesno(
				"Queue finished",
				f"{self._count('skip')} source(s) are still skipped. Review them now?",
			):
				self.start_skipped_review()
		else:
			messagebox.showinfo("All done", "Every source has been answered. Nothing is left skipped.")

	def start_skipped_review(self):
		self.mode = "skipped"
		self.queue = [i for i in self.all_ids if self.decisions.get(i) == "skip"]
		self.index = 0
		self.save()
		self.refresh()

	def back_to_full_queue(self):
		current_id = self.current_id
		self.mode = "all"
		self.queue = list(self.all_ids)
		undecided = self._first_undecided_index()
		self.index = undecided if self.queue[undecided] not in self.decisions else self.queue.index(current_id)
		self.save()
		self.refresh()

	def toggle_mode(self):
		if self.mode == "skipped":
			self.back_to_full_queue()
		else:
			self.start_skipped_review()

	def _jump_to_selected(self, event=None):
		selection = self.pile_list.curselection()
		if not selection:
			return
		source_id = self.pile_ids[selection[0]]
		if source_id not in self.queue:
			self.mode = "all"
			self.queue = list(self.all_ids)
		self.index = self.queue.index(source_id)
		self.save()
		self.refresh()


def main():
	if not CATALOGUE.exists():
		raise SystemExit(f"Catalogue not found: {CATALOGUE}")
	catalogue = pd.read_csv(CATALOGUE, index_col=0)
	window = tk.Tk()
	ReviewApp(window, catalogue)
	window.mainloop()


if __name__ == "__main__":
	main()
