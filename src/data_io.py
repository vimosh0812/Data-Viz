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


def time_sec_to_minutes(time_sec: pd.Series) -> pd.Series:
    """Convert stored duration to minutes.

    ``time_sec`` from :func:`parse_time_to_seconds` is normally **seconds**
    (``minutes = time_sec / 60``).

    Some exports store **milliseconds** as unitless numbers. If interpreting
    as seconds yields an implausible duration (hundreds of minutes) while
    interpreting as milliseconds yields a plausible quiz length, use
    ``minutes = time_sec / 60_000``.
    """
    s = time_sec.astype(float)
    min_if_seconds = s / 60.0
    min_if_milliseconds = s / 60000.0
    # Row-wise: e.g. 420000 ms misread as 420000 s → 7000 min vs 7 min
    use_ms = (min_if_seconds > 720.0) & (min_if_milliseconds <= 720.0) & (min_if_seconds < 10000.0)
    out = np.where(use_ms, min_if_milliseconds, min_if_seconds)
    return pd.Series(out, index=time_sec.index, dtype=float)


def attempt_minutes_capped(time_sec: float, cap: float = 120.0) -> float:
    """Single-attempt duration in minutes for retake deltas: :func:`time_sec_to_minutes` then ``cap``.

    Extreme Moodle durations (e.g. multi-day open tabs) are clipped so paired
    deltas stay in a typical quiz range (default 120 minutes).
    """
    if not np.isfinite(time_sec) or time_sec <= 0:
        return 0.0
    m = float(time_sec_to_minutes(pd.Series([time_sec])).iloc[0])
    return float(np.clip(m, 0.0, cap))


def load_quiz(quiz_id: int) -> pd.DataFrame:
    """Load one quiz CSV: finished attempts only, numeric grades and per-Q marks."""
    path = QUIZ_FILES[quiz_id]
    df = pd.read_csv(path)

    df = df[df["State"] == "Finished"].copy()
    df["grade"] = pd.to_numeric(df["Grade/10.00"], errors="coerce")
    df["time_sec"] = df["Time taken"].apply(parse_time_to_seconds)
    df["time_min"] = time_sec_to_minutes(df["time_sec"])

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
