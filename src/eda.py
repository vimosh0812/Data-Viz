"""Exploratory analysis for three quiz CSVs — metrics and figures for hypothesis work.

All artefacts are written to a single directory (typically ``outputs/eda/``).
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_io import QUIZ_FILES, load_all_quizzes, parse_time_to_seconds
from src.viz_style import PALETTE, configure_matplotlib

_TRIPLES = lambda q1, q2, q3: [(q1, "Quiz 1"), (q2, "Quiz 2"), (q3, "Quiz 3")]
_COLORS = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]


def raw_vs_finished_summary() -> pd.DataFrame:
    rows = []
    for qid, path in QUIZ_FILES.items():
        raw = pd.read_csv(path)
        vc = raw["State"].value_counts(dropna=False)
        for state, n in vc.items():
            rows.append({"quiz": qid, "state": state, "rows": int(n)})
    return pd.DataFrame(rows).sort_values(["quiz", "rows"], ascending=[True, False])


def cleaned_row_counts() -> pd.DataFrame:
    q1, q2, q3, all_df = load_all_quizzes()
    return pd.DataFrame(
        {
            "dataset": ["Quiz 1", "Quiz 2", "Quiz 3", "All (stacked)"],
            "finished_with_grade_and_time": [len(q1), len(q2), len(q3), len(all_df)],
        }
    )


def data_quality_summary() -> pd.DataFrame:
    """Per raw CSV: how many rows are lost to state, bad grade, or unparseable time."""
    rows = []
    for qid, path in QUIZ_FILES.items():
        raw = pd.read_csv(path)
        n = len(raw)
        fin = raw["State"] == "Finished"
        n_fin = int(fin.sum())
        sub = raw.loc[fin].copy()
        g = pd.to_numeric(sub["Grade/10.00"], errors="coerce")
        t = sub["Time taken"].apply(parse_time_to_seconds)
        bad_grade = int(g.isna().sum())
        bad_time = int(t.isna().sum())
        usable = int((g.notna() & t.notna()).sum())
        rows.append(
            {
                "quiz": qid,
                "rows_total": n,
                "finished": n_fin,
                "finished_missing_grade": bad_grade,
                "finished_unparseable_time": bad_time,
                "rows_in_analysis_sample": usable,
            }
        )
    return pd.DataFrame(rows)


def cross_quiz_student_counts(q1: pd.DataFrame, q2: pd.DataFrame, q3: pd.DataFrame) -> pd.DataFrame:
    """How many distinct students appear in each quiz and in combinations (same ID across files)."""
    s1, s2, s3 = set(q1["student_id"]), set(q2["student_id"]), set(q3["student_id"])
    return pd.DataFrame(
        [
            {"scope": "Students in Quiz 1 only (cleaned)", "count": len(s1 - s2 - s3)},
            {"scope": "Students in Quiz 2 only (cleaned)", "count": len(s2 - s1 - s3)},
            {"scope": "Students in Quiz 3 only (cleaned)", "count": len(s3 - s1 - s2)},
            {"scope": "In all three quizzes", "count": len(s1 & s2 & s3)},
            {"scope": "In Quiz 1 and 2", "count": len(s1 & s2)},
            {"scope": "In Quiz 1 and 3", "count": len(s1 & s3)},
            {"scope": "In Quiz 2 and 3", "count": len(s2 & s3)},
        ]
    )


def student_attempt_behaviour(q1: pd.DataFrame, q2: pd.DataFrame, q3: pd.DataFrame) -> pd.DataFrame:
    """Per quiz: repeat-attempt rate and mean attempts — grounds H3/H7/H9."""
    rows = []
    for df, name in _TRIPLES(q1, q2, q3):
        mx = df.groupby("student_id")["attempt_no"].max()
        n_stu = mx.size
        multi = int((mx >= 2).sum())
        rows.append(
            {
                "quiz": name,
                "students": n_stu,
                "pct_with_2plus_attempts": round(100 * multi / max(n_stu, 1), 2),
                "mean_attempts_per_student": round(df.groupby("student_id").size().mean(), 3),
                "max_attempts_observed": int(mx.max()),
            }
        )
    return pd.DataFrame(rows)


def numeric_summary_table(q1: pd.DataFrame, q2: pd.DataFrame, q3: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for df, name in _TRIPLES(q1, q2, q3):
        tm = df["time_min"]
        g = df["grade"]
        rows.append(
            {
                "dataset": name,
                "n_attempts": len(df),
                "n_students": df["student_id"].nunique(),
                "grade_mean": round(g.mean(), 3),
                "grade_std": round(g.std(), 3),
                "grade_p25": round(g.quantile(0.25), 3),
                "grade_median": round(g.quantile(0.5), 3),
                "grade_p75": round(g.quantile(0.75), 3),
                "pct_grade_ge_6": round(100 * (g >= 6).mean(), 2),
                "pct_grade_ge_8": round(100 * (g >= 8).mean(), 2),
                "time_median_min": round(tm.median(), 3),
                "time_p75_min": round(tm.quantile(0.75), 3),
                "time_mean_min": round(tm.mean(), 3),
                "pct_time_over_60min": round(100 * (tm > 60).mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def first_to_last_grade_delta(q1: pd.DataFrame, q2: pd.DataFrame, q3: pd.DataFrame) -> pd.DataFrame:
    """Among students with ≥2 attempts: last minus first grade (informs H3 / reattempt stories)."""
    rows = []
    for df, name in _TRIPLES(q1, q2, q3):
        o = df.sort_values(["student_id", "attempt_no"])
        g = o.groupby("student_id", group_keys=False)
        first = g["grade"].first()
        last = g["grade"].last()
        n_att = g["attempt_no"].max()
        mask = n_att >= 2
        d = (last - first)[mask].dropna()
        rows.append(
            {
                "quiz": name,
                "n_students_multi": int(mask.sum()),
                "mean_delta_last_minus_first": round(float(d.mean()), 4) if len(d) else np.nan,
                "median_delta": round(float(d.median()), 4) if len(d) else np.nan,
                "pct_improved": round(100 * float((d > 0).mean()), 2) if len(d) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _qcols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Q.")]


def question_mean_matrix(q1: pd.DataFrame, q2: pd.DataFrame, q3: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rows = question index, columns = quiz; values = mean mark / max for that question."""
    cols = {}
    for df, qname in _TRIPLES(q1, q2, q3):
        series = {}
        for c in _qcols(df):
            try:
                mx = float(c.split("/")[1])
            except (IndexError, ValueError):
                mx = float(df[c].max())
            qix = c.split()[1].rstrip(".")
            series[qix] = df[c].mean() / mx
        cols[qname] = series
    # stable question order from first quiz column order
    order = [c.split()[1].rstrip(".") for c in _qcols(q1)]
    mat = pd.DataFrame({k: [cols[k][qi] for qi in order] for k in ["Quiz 1", "Quiz 2", "Quiz 3"]}, index=order)
    return mat, order


def plot_grade_histograms(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (df, title), c in zip(axes, _TRIPLES(q1, q2, q3), _COLORS):
        ax.hist(df["grade"].dropna(), bins=np.arange(0, 10.5, 0.5), color=c, edgecolor="white", linewidth=0.5)
        ax.axvline(df["grade"].mean(), color="black", linestyle="--", linewidth=1.2, label=f"Mean {df['grade'].mean():.2f}")
        ax.axvline(df["grade"].median(), color="grey", linestyle=":", linewidth=1.2, label=f"Median {df['grade'].median():.2f}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Grade / 10")
        ax.set_ylabel("Attempts (count)")
        ax.legend(fontsize=7)
    plt.suptitle(
        "Grade distribution per quiz — informs difficulty & pass-rate hypotheses",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "eda_grade_histograms.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_grade_ecdf(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    """ECDF overlay: compare overall score distributions across quizzes at a glance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (df, title), c in zip(_TRIPLES(q1, q2, q3), _COLORS):
        x = np.sort(df["grade"].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, drawstyle="steps-post", color=c, lw=2.2, label=title)
    ax.set_xlabel("Grade / 10")
    ax.set_ylabel("Cumulative proportion of attempts")
    ax.set_title(
        "ECDF of grades — steeper curve right = more high scores (compare quizzes)",
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    plt.tight_layout()
    fig.savefig(out_dir / "eda_grade_ecdf_by_quiz.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _time_long_capped(q1, q2, q3, cap_quantile: float = 0.995) -> pd.DataFrame:
    """Stack time_min with quiz label; cap each quiz at ``cap_quantile`` for fair comparison."""
    parts = []
    for df, name in _TRIPLES(q1, q2, q3):
        cap = df["time_min"].quantile(cap_quantile)
        d = df.loc[df["time_min"] <= cap, ["time_min"]].copy()
        d["quiz"] = name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def plot_time_histograms(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    """Single cross-quiz comparison (boxplots). Filename kept for backward compatibility."""
    out_dir.mkdir(parents=True, exist_ok=True)
    quiz_order = ["Quiz 1", "Quiz 2", "Quiz 3"]
    long = _time_long_capped(q1, q2, q3, 0.995)
    pal = dict(zip(quiz_order, _COLORS))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.boxplot(
        data=long,
        x="quiz",
        y="time_min",
        order=quiz_order,
        hue="quiz",
        palette=pal,
        width=0.52,
        ax=ax,
        legend=False,
        dodge=False,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.4),
        boxprops=dict(edgecolor="0.35", linewidth=1.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    ax.set_xlabel("")
    ax.set_ylabel("Time taken (minutes)")
    ax.set_title(
        "Time comparison between quizzes",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.text(
        0.5,
        -0.14,
        "Boxplots show median (black bar) and IQR; extremes above the 99.5th percentile\n"
        "within each quiz are omitted so one or two multi-day attempts do not compress the scale.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="0.35",
        linespacing=1.35,
    )

    y_hi = long["time_min"].max()
    y_lo = long["time_min"].min()
    pad = 0.05 * (y_hi - y_lo + 1e-6)
    for i, qz in enumerate(quiz_order):
        sub = long.loc[long["quiz"] == qz, "time_min"]
        med = float(sub.median())
        q1v, q3v = float(sub.quantile(0.25)), float(sub.quantile(0.75))
        top = float(sub.max())
        ax.text(
            i,
            top + pad,
            f"median {med:.1f} min\nIQR [{q1v:.1f}, {q3v:.1f}]",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.92),
        )

    ax.set_ylim(y_lo, y_hi + 4.5 * pad)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_dir / "eda_time_histograms.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_time_box_by_quiz(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    """Same capped data, horizontal boxplots — optional second layout for slides/appendix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    quiz_order = ["Quiz 1", "Quiz 2", "Quiz 3"]
    long = _time_long_capped(q1, q2, q3, 0.995)
    pal = dict(zip(quiz_order, _COLORS))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sns.boxplot(
        data=long,
        y="quiz",
        x="time_min",
        order=quiz_order,
        hue="quiz",
        palette=pal,
        width=0.55,
        ax=ax,
        legend=False,
        dodge=False,
        orient="h",
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.4),
        boxprops=dict(edgecolor="0.35", linewidth=1.0),
    )
    ax.set_xlabel("Time taken (minutes, ≤ 99.5th pctile per quiz)")
    ax.set_ylabel("")
    ax.set_title("Time comparison between quizzes (horizontal layout)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "eda_time_boxplot_by_quiz.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_attempts_per_student(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (df, title), c in zip(axes, _TRIPLES(q1, q2, q3), _COLORS):
        mx = df.groupby("student_id")["attempt_no"].max()
        vc = mx.value_counts().sort_index()
        ax.bar(vc.index.astype(int), vc.values, color=c, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Max attempts (per student)")
        ax.set_ylabel("Students")
    plt.suptitle(
        "Re-attempt pattern — who retakes informs H3, H7, H9",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "eda_attempts_per_student.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_first_to_last_grade_delta(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (df, title), c in zip(axes, _TRIPLES(q1, q2, q3), _COLORS):
        o = df.sort_values(["student_id", "attempt_no"])
        g = o.groupby("student_id", group_keys=False)
        first = g["grade"].first()
        last = g["grade"].last()
        n_att = g["attempt_no"].max()
        d = (last - first)[n_att >= 2].dropna()
        ax.hist(d, bins=np.arange(-10, 10.5, 0.5), color=c, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Last grade − first grade")
        ax.set_ylabel("Students")
    plt.suptitle(
        "Improvement from first to last attempt (students with 2+ tries) — H3 / H7 context",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "eda_first_to_last_grade_delta.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_question_mean_bars(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    long = []
    for df, qname in _TRIPLES(q1, q2, q3):
        for c in _qcols(df):
            qix = c.split()[1].rstrip(".")
            long.append({"question": f"Q{qix}", "quiz": qname, "mean_mark": df[c].mean()})
    long = pd.DataFrame(long)
    pivot = long.pivot(index="question", columns="quiz", values="mean_mark")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=ax, width=0.75, color=_COLORS)
    ax.set_title("Mean raw mark per question — H2 (item difficulty) context", fontweight="bold")
    ax.set_xlabel("Question")
    ax.set_ylabel("Mean mark (0–2)")
    ax.legend(fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / "eda_question_mean_marks.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_question_difficulty_heatmap(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    mat, _ = question_mean_matrix(q1, q2, q3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Normalised mean score (1 = full marks on average)\nH2: which items are hard in which quiz?", fontweight="bold")
    plt.ylabel("Question")
    plt.xlabel("Quiz")
    plt.tight_layout()
    fig.savefig(out_dir / "eda_question_difficulty_heatmap.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_grade_vs_time_hexbin(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    """Dense view of time vs grade (H1 / H5 exploratory)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (df, title), c in zip(axes, _TRIPLES(q1, q2, q3), _COLORS):
        cap = df["time_min"].quantile(0.995)
        d = df[df["time_min"] <= cap]
        hb = ax.hexbin(d["time_min"], d["grade"], gridsize=35, cmap="viridis", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Grade / 10")
    plt.suptitle("Grade vs time (density) — supports scatters in H1 / H5", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "eda_grade_vs_time_hexbin.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_attempts_timeline(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, df, title, c in zip(axes, [q1, q2, q3], ["Quiz 1", "Quiz 2", "Quiz 3"], _COLORS):
        d = df.dropna(subset=["started_on"]).copy()
        d["day"] = d["started_on"].dt.floor("D")
        daily = d.groupby("day").size()
        ax.fill_between(daily.index, daily.values, alpha=0.35, color=c)
        ax.plot(daily.index, daily.values, color=c, linewidth=1)
        ax.set_ylabel("Attempts / day")
        ax.set_title(title, fontweight="bold", loc="left")
    axes[-1].set_xlabel("Date")
    plt.suptitle("When attempts occur — H8 / cohort timing context", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "eda_attempts_timeline.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_best_score_vs_attempts(q1, q2, q3, out_dir: Path, show: bool = True) -> None:
    """Mean best grade by max attempt count (H9 exploratory)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (df, title), c in zip(axes, _TRIPLES(q1, q2, q3), _COLORS):
        s = df.groupby("student_id").agg(best=("grade", "max"), n=("attempt_no", "max")).reset_index()
        agg = s.groupby("n")["best"].agg(["mean", "count"]).reset_index()
        agg = agg[agg["count"] >= 5]
        ax.bar(agg["n"].astype(int).astype(str), agg["mean"], color=c, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Max attempts")
        ax.set_ylabel("Mean best grade / 10")
        ax.set_ylim(0, 10.5)
    plt.suptitle("Best score vs number of attempts (groups with n≥5) — H9 context", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "eda_best_grade_by_max_attempts.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def write_manifest(out_dir: Path) -> None:
    text = dedent(
        """
        EDA output folder — purpose of each file
        ==========================================

        CSV (tables for the report)
        -----------------------------
        eda_data_quality.csv          Raw vs finished; missing grade/time on finished rows
        eda_numeric_summary.csv      Grade & time distribution stats; pass-rate style columns
        eda_student_attempts.csv     Re-attempt rates (supports H3, H7, H9)
        eda_first_to_last_delta.csv  Mean/median change last−first grade for multi-attempters
        eda_cross_quiz_students.csv  Overlap of student IDs across quiz files

        PNG (figures)
        -------------
        eda_grade_histograms.png           Grade mass per quiz (+ mean/median lines)
        eda_grade_ecdf_by_quiz.png        Cumulative grades — compare quiz difficulty at once
eda_time_histograms.png          Cross-quiz time: boxplots (median, IQR); 99.5th pctile cap
eda_time_boxplot_by_quiz.png     Same data, horizontal boxplot layout
        eda_attempts_per_student.png      How many students take 1, 2, … attempts
        eda_first_to_last_grade_delta.png Improvement distribution for retakers
        eda_question_mean_marks.png       Mean raw marks per question (H2)
        eda_question_difficulty_heatmap.png Normalised means — hard vs easy items by quiz
        eda_grade_vs_time_hexbin.png      Density of grade vs time (H1 / H5)
        eda_attempts_timeline.png          Daily attempt volume (H8)
        eda_best_grade_by_max_attempts.png Exploratory link attempts → best score (H9)
        """
    ).strip()
    (out_dir / "README_EDA_FILES.txt").write_text(text, encoding="utf-8")


def run_all_eda_plots(out_dir: Path, show: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write all EDA tables and figures into ``out_dir`` (use ``.../outputs/eda``)."""
    configure_matplotlib()
    q1, q2, q3, _ = load_all_quizzes()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Data quality (raw → analysis) ===")
    dq = data_quality_summary()
    print(dq.to_string(index=False))
    dq.to_csv(out_dir / "eda_data_quality.csv", index=False)

    print("\n=== Raw CSV: rows by State ===")
    print(raw_vs_finished_summary().to_string(index=False))
    print("\n=== Analysis sample counts ===")
    print(cleaned_row_counts().to_string(index=False))

    print("\n=== Student overlap across quizzes (same student_id) ===")
    xc = cross_quiz_student_counts(q1, q2, q3)
    print(xc.to_string(index=False))
    xc.to_csv(out_dir / "eda_cross_quiz_students.csv", index=False)

    print("\n=== Student / attempt behaviour ===")
    sa = student_attempt_behaviour(q1, q2, q3)
    print(sa.to_string(index=False))
    sa.to_csv(out_dir / "eda_student_attempts.csv", index=False)

    print("\n=== First → last grade (multi-attempters only) ===")
    fl = first_to_last_grade_delta(q1, q2, q3)
    print(fl.to_string(index=False))
    fl.to_csv(out_dir / "eda_first_to_last_delta.csv", index=False)

    print("\n=== Numeric summary (cleaned attempts) ===")
    summary = numeric_summary_table(q1, q2, q3)
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "eda_numeric_summary.csv", index=False)

    plot_grade_histograms(q1, q2, q3, out_dir, show=show)
    plot_grade_ecdf(q1, q2, q3, out_dir, show=show)
    plot_time_histograms(q1, q2, q3, out_dir, show=show)
    plot_time_box_by_quiz(q1, q2, q3, out_dir, show=show)
    plot_attempts_per_student(q1, q2, q3, out_dir, show=show)
    plot_first_to_last_grade_delta(q1, q2, q3, out_dir, show=show)
    plot_question_mean_bars(q1, q2, q3, out_dir, show=show)
    plot_question_difficulty_heatmap(q1, q2, q3, out_dir, show=show)
    plot_grade_vs_time_hexbin(q1, q2, q3, out_dir, show=show)
    plot_attempts_timeline(q1, q2, q3, out_dir, show=show)
    plot_best_score_vs_attempts(q1, q2, q3, out_dir, show=show)

    write_manifest(out_dir)
    return q1, q2, q3, summary


# Backwards-compatible names used by older notebook imports
plot_question_means = plot_question_mean_bars
plot_grade_vs_time_scatter_small = plot_grade_vs_time_hexbin
