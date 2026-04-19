# Quiz data — notebooks and outputs

This package is a **self-contained** submission: raw quiz exports, three Jupyter notebooks (all logic inlined in the notebooks), and generated figures and tables. No extra source tree is required to reproduce the analysis.

## What to submit

Include **only** these items (plus this file):

| Item | Notes |
|------|--------|
| **`README.md`** | This file (project overview and how to run the notebooks). |
| **`data/`** | Moodle CSV exports for the three quizzes (see paths below). |
| **`notebooks/`** | `EDA.ipynb`, `task01.ipynb`, `task02.ipynb` only. **Do not** include `notebooks/_exec_check/` (that folder is for local test runs only). |
| **`outputs/`** | Saved figures and CSVs produced by the notebooks (`outputs/eda/` for EDA, plus PNGs in `outputs/` for hypothesis tasks). |

Suggested layout:

```text
README.md
data/
  quiz1/…
  quiz2/…
  quiz3/…
notebooks/
  EDA.ipynb
  task01.ipynb
  task02.ipynb
outputs/
  eda/…
  h1_*.png … h5_*.png
  h6_*.png … h10_*.png
```

## Prerequisites

Use Python **3.11+** with:

- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`
- `jupyter` and `ipykernel` (to open and run the `.ipynb` files)

Install with pip, for example:

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels jupyter ipykernel
```

## Data paths

The notebooks expect these CSV paths **relative to the folder that contains `data`, `notebooks`, and `outputs`** (the “project root”). If you open a notebook from inside `notebooks/`, the first code cell still switches the working directory to that root so paths resolve correctly.

| Quiz | Path |
|------|------|
| Quiz 1 | `data/quiz1/quiz1_marks.csv` |
| Quiz 2 | `data/quiz2/quiz2_marks.csv` |
| Quiz 3 | `data/quiz3/quiz3_marks.csv` |

If your files live elsewhere, change the path variables in each notebook’s setup cell.

## How to run

1. Open Jupyter (or VS Code / Cursor) using the environment where the packages above are installed.
2. Open a notebook under `notebooks/`.
3. **Run all cells from top to bottom.**

**Suggested order** for a full refresh: `EDA.ipynb` → `task01.ipynb` → `task02.ipynb`.

---

## The three notebooks

### `notebooks/EDA.ipynb` — Exploratory Data Analysis

- Loads the three Moodle exports, keeps finished attempts, parses grades and durations, and builds per-attempt and per-student summaries.
- Covers data quality, cohort overlap across quizzes, re-attempt behaviour, grade and time distributions, grade vs time, first-to-last attempt change, per-question difficulty, attempt timelines, and best score vs number of attempts.
- **Writes to:** **`outputs/eda/`** (CSV tables and PNG figures such as `eda_data_quality.csv`, `eda_grade_histograms.png`, `eda_time_boxplot_by_quiz.png`, and others referenced in the notebook).

### `notebooks/task01.ipynb` — Task 1 (hypotheses H1–H5)

- Tests instructor hypotheses: time vs score (H1), question difficulty (H2), improvement patterns by performance tier (H3), question marks and total time vs tier (H4), and an “optimal time window” view (H5).
- **Writes to:** **`outputs/`** as PNG files `h1_*.png` through `h5_*.png` (H4 produces several files, e.g. `h4_difficulty_perf_heatmap.png`, `h4_grade_time_scatter_by_tier.png`).

### `notebooks/task02.ipynb` — Task 2 (hypotheses H6–H10)

- **H6:** Consecutive retake pairs — Δtime vs Δscore (scatter + LOWESS), then Δtime by improved vs not (boxplot). Two figures, same hypothesis.
- **H7:** Knowledge and efficiency across attempts (mean score and mean time by attempt, plus per-student scatter).
- **H8:** Slow vs fast first-attempt “failers” and recovery (best minus first grade).
- **H9:** Paired improvement from attempt 1 to attempt 2 (Wilcoxon + slope chart).
- **H10:** Cross-quiz consistency of best grades for students who appear in multiple quizzes.

| ID | Output figures (under `outputs/`) |
|----|-----------------------------------|
| H6 | `h6_retake_delta_scatter.png`, `h6_retake_delta_box.png` |
| H7 | `h7_knowledge_efficiency_across_attempts.png` |
| H8 | `h8_slow_fast_failers_recovery.png` |
| H9 | `h9_reattempt_improvement.png` |
| H10 | `h10_cross_quiz_consistency.png` |

---

## Re-running from the command line

From the project root (the directory that contains `data`, `notebooks`, and `outputs`), after installing `nbconvert`:

```bash
jupyter nbconvert --execute --inplace notebooks/EDA.ipynb
jupyter nbconvert --execute --inplace notebooks/task01.ipynb
jupyter nbconvert --execute --inplace notebooks/task02.ipynb
```

Use a generous timeout if needed, for example `--ExecutePreprocessor.timeout=900`.
