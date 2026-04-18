# CS3751 Data Visualization — Class Project

## About this project

This is a **group** data visualization project for **CS3751**. You analyze Moodle-style exports of **three quizzes**: each row is one quiz attempt with timestamps, **time taken**, total **grade out of 10**, and **per-question marks**. The same student can appear **multiple times** (retakes).

The work is to explore hypotheses with **well-designed visualizations** (Task 1: five hypotheses given in the brief; Task 2: five hypotheses you define). For each figure you explain whether the evidence **supports or rejects** the hypothesis and **justify marks and channels** as required by the course.

**Authoritative brief:** `docs/description.pdf` (sections, grading, submission format).

**Data in this repo:** `data/quiz1/`, `data/quiz2/`, `data/quiz3/` — CSV files with attempt-level and question-level scores.

---

## Prerequisites

- **Python 3.10, 3.11, 3.12, or 3.13** (3.10+). Check with `python --version` or `python3 --version`.
- **pip** (usually bundled with Python).

---

## Setup (multiple environments)

Clone or copy the project folder, then open a terminal **in the repository root** (the directory that contains `requirements.txt`).

### A. Virtual environment — macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Deactivate later with `deactivate`.

### B. Virtual environment — Windows (Command Prompt or PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py` is not installed, try `python -m venv .venv` instead. In **cmd.exe**, activation is `.\.venv\Scripts\activate.bat`. If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use **cmd.exe** with `activate.bat` above. Deactivate with `deactivate`.

### C. Conda or Mamba (any OS)

From the repo root, with your preferred Python version available:

```bash
conda create -n cs3751-viz python=3.12 -y
conda activate cs3751-viz
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Using **mamba**, replace `conda` with `mamba` in the lines above.

### Verify the install

With the environment **activated**:

```bash
python -c "import pandas, matplotlib, seaborn, scipy, statsmodels; print('OK')"
```

---

## Task 1 — five notebooks (one hypothesis each)

Use Jupyter from the repo root (`jupyter lab`), then open **one** notebook at a time and run **the single code cell** inside it.

| File | What you test |
|------|----------------|
| `notebooks/task1/01_H1_longer_time_higher_score.ipynb` | H1 — longer time → higher score |
| `notebooks/task1/02_H2_question_difficulty.ipynb` | H2 — some questions harder |
| `notebooks/task1/03_H3_high_vs_low_progression.ipynb` | H3 — high vs low progression over attempts |
| `notebooks/task1/04_H4_difficulty_time_performance.ipynb` | H4 — difficulty / time / performance (data limits noted in notebook) |
| `notebooks/task1/05_H5_optimal_time_window.ipynb` | H5 — optimal time window vs score |

Each notebook finds the project root on its own and writes PNGs to `outputs/`.

**EDA for all three quiz files:** `notebooks/00_EDA_three_quizzes.ipynb` (run all cells), or `python -m src eda` (`--no-show` to save only).

---

## Optional: run from the terminal (`src/`)

The same plots can be generated without Jupyter. Figures go to `outputs/`.

From the **repository root**, with your environment activated:

```bash
# Exploratory analysis (tables + figures → outputs/eda/)
python -m src eda

# Task 1 only (H1–H5)
python -m src task1

# Task 2 only (H6–H10)
python -m src task2

# Single hypothesis
python -m src h3

# Everything + printed summary table
python -m src all

# Summary table only (no data load)
python -m src summary
```

On servers or CI, save plots without opening windows:

```bash
python -m src task1 --no-show
```

Custom output directory: `python -m src h1 --output-dir ./figures`.

---

## Running Jupyter

With the environment activated, from the **repository root**:

```bash
jupyter lab
```

Use `notebooks/task1/` for Task 1 (see table above). For Task 2 you can copy the same pattern or use `python -m src task2`.

---

## Repository layout

| Path | Role |
|------|------|
| `data/` | Raw quiz CSVs — treat as read-only inputs |
| `docs/description.pdf` | Full assignment specification |
| `docs/claud.py` | Reference script / notebook-style blocks (superseded by `src/` for runs) |
| `notebooks/00_EDA_three_quizzes.ipynb` | EDA for all three CSVs; writes to `outputs/eda/` |
| `notebooks/task1/` | Five notebooks — Task 1, H1–H5, one runnable cell each |
| `src/` | Data loading + plot functions + `python -m src …` CLI (H1–H10) |
| `outputs/` | Generated PNGs (from notebooks or `python -m src`) |
| `requirements.txt` | Python dependencies |
| `.venv/` | Local venv (created by you; listed in `.gitignore`) |

Add Task 2 notebooks under `notebooks/task2/` when you define those hypotheses.

---

## Submission reminder (see PDF for details)

Group **PDF report** and a **zip** of source code that includes this README, per course instructions.
