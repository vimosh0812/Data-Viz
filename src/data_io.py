"""Load quiz CSVs from ``data/`` and build analysis-ready DataFrames."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

QUIZ_FILES = {
    1: DATA_DIR / "quiz1" / "quiz1_marks.csv",
    2: DATA_DIR / "quiz2" / "quiz2_marks.csv",
    3: DATA_DIR / "quiz3" / "quiz3_marks.csv",
}


def parse_time_to_seconds(text) -> float:
    """Parse Moodle-style duration strings into seconds (float, may be NaN)."""
    if pd.isna(text):
        return np.nan
    s = str(text).strip()
    if s in ("-", ""):
        return np.nan
    s_low = s.lower()
    total = 0.0
    m = re.search(r"(\d+)\s*days?", s_low)
    if m:
        total += int(m.group(1)) * 86400
    m = re.search(r"(\d+)\s*hours?", s_low)
    if m:
        total += int(m.group(1)) * 3600
    m = re.search(r"(\d+)\s*mins?", s_low)
    if m:
        total += int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*secs?", s_low)
    if m:
        total += int(m.group(1))
    return total if total > 0 else np.nan


def load_quiz(quiz_id: int) -> pd.DataFrame:
    """Load one quiz CSV: finished attempts only, numeric grades and per-Q marks."""
    path = QUIZ_FILES[quiz_id]
    df = pd.read_csv(path)

    df = df[df["State"] == "Finished"].copy()
    df["grade"] = pd.to_numeric(df["Grade/10.00"], errors="coerce")
    df["time_sec"] = df["Time taken"].apply(parse_time_to_seconds)
    df["time_min"] = df["time_sec"] / 60.0

    q_cols = [c for c in df.columns if c.startswith("Q.")]
    for c in q_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.rename(columns={"Student Code": "student_id"})
    df["quiz"] = quiz_id
    started = df["Started on"].astype(str).str.replace(r"\s+", " ", regex=True)
    df["Started on"] = pd.to_datetime(started, errors="coerce", dayfirst=True, format="mixed")
    df = df.rename(columns={"Started on": "started_on"})
    df = df.sort_values(["student_id", "started_on"])
    df["attempt_no"] = df.groupby("student_id").cumcount() + 1

    keep = (
        ["student_id", "quiz", "attempt_no", "grade", "time_sec", "time_min", "started_on"]
        + q_cols
    )
    return df[keep].dropna(subset=["grade", "time_sec"])


def load_all_quizzes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(Q1, Q2, Q3, ALL)``."""
    q1 = load_quiz(1)
    q2 = load_quiz(2)
    q3 = load_quiz(3)
    all_df = pd.concat([q1, q2, q3], ignore_index=True)
    return q1, q2, q3, all_df
