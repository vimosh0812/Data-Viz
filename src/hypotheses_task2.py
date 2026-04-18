"""Task 2 — student-generated hypotheses (H6–H10), adapted from ``docs/claud.py``."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr

from src.viz_style import PALETTE


def _triplets(q1: pd.DataFrame, q2: pd.DataFrame, q3: pd.DataFrame):
    return [(q1, "Quiz 1"), (q2, "Quiz 2"), (q3, "Quiz 3")]


def run_h6_q1_full_mark_effect(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H6: Students who score full marks on Q1 tend to score higher overall."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        q1_cols = [c for c in quiz_df.columns if c.startswith("Q. 1")]
        if not q1_cols:
            ax.set_title(f"{label} — Q1 column not found")
            continue
        q1_col = q1_cols[0]
        try:
            q1_max = float(q1_col.split("/")[1])
        except (IndexError, ValueError):
            q1_max = float(quiz_df[q1_col].max())
        d = quiz_df.copy()
        d["q1_full"] = d[q1_col] == q1_max

        for flag, color, lbl in [
            (True, PALETTE["success"], "Q1 full mark"),
            (False, PALETTE["secondary"], "Q1 partial / zero"),
        ]:
            sub = d.loc[d["q1_full"] == flag, "grade"].dropna()
            if len(sub) < 2:
                continue
            sns.kdeplot(sub, ax=ax, color=color, linewidth=2, fill=True, alpha=0.25, label=lbl)
            ax.axvline(sub.mean(), color=color, linestyle="--", linewidth=1.5)

        a = d.loc[d["q1_full"], "grade"].dropna()
        b = d.loc[~d["q1_full"], "grade"].dropna()
        if len(a) > 0 and len(b) > 0:
            _, pval = stats.mannwhitneyu(a, b, alternative="greater")
            ptxt = "< 0.001" if pval < 0.001 else f"= {pval:.3f}"
        else:
            ptxt = "n/a"
        ax.set_title(f"{label}\nMann–Whitney p {ptxt}", fontweight="bold")
        ax.set_xlabel("Grade / 10")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.suptitle("H6 — Full mark on Q1 → higher overall score?", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "h6_q1_fullmark_effect.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("\n── H6 — Mann–Whitney tests Q1-full vs not (alternative: greater). ──")


def run_h7_second_attempt_improvement(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H7: Students improve from attempt 1 to attempt 2."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        multi = quiz_df[quiz_df.groupby("student_id")["attempt_no"].transform("max") >= 2]
        pivot = (
            multi[multi["attempt_no"].isin([1, 2])]
            .pivot_table(index="student_id", columns="attempt_no", values="grade")
            .dropna()
        )
        if pivot.shape[1] < 2 or pivot.empty:
            ax.set_title(f"{label} — not enough paired attempts")
            continue
        pivot.columns = ["Attempt 1", "Attempt 2"]
        pivot["improved"] = pivot["Attempt 2"] > pivot["Attempt 1"]

        for _, row in pivot.iterrows():
            color = PALETTE["success"] if row["improved"] else PALETTE["secondary"]
            ax.plot([1, 2], [row["Attempt 1"], row["Attempt 2"]], color=color, alpha=0.3, linewidth=0.8)

        ax.plot(
            [1, 2],
            [pivot["Attempt 1"].mean(), pivot["Attempt 2"].mean()],
            color="black",
            linewidth=2.5,
            marker="o",
            markersize=8,
            label="Mean",
        )
        try:
            _, pval = stats.wilcoxon(pivot["Attempt 1"], pivot["Attempt 2"])
            ptxt = "< 0.001" if pval < 0.001 else f"= {pval:.3f}"
        except ValueError:
            pval = float("nan")
            ptxt = "n/a (no paired differences)"
        pct_improved = pivot["improved"].mean() * 100
        ax.set_title(f"{label}\nWilcoxon p {ptxt} | {pct_improved:.0f}% improved", fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Attempt 1", "Attempt 2"])
        ax.set_ylabel("Grade / 10")
        ax.set_ylim(0, 11)
        ax.plot([], [], color=PALETTE["success"], label="Improved")
        ax.plot([], [], color=PALETTE["secondary"], label="Declined / same")
        ax.legend(fontsize=9)

    plt.suptitle("H7 — Improvement from attempt 1 → 2?", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "h7_reattempt_improvement.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("\n── H7 — Wilcoxon signed-rank on paired grades (1 vs 2). ──")


def run_h8_early_starter_vs_grade(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H8: Students who start the quiz earlier in the window tend to score lower (or not)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        first = quiz_df[quiz_df["attempt_no"] == 1].dropna(subset=["started_on"]).sort_values("started_on")
        if len(first) < 10:
            ax.set_title(f"{label} — too few first attempts with timestamps")
            continue
        first = first.reset_index(drop=True)
        # 0% = earliest first-attempt starter in this quiz; 100% = latest
        first["time_order_pct"] = np.linspace(0, 100, num=len(first), endpoint=True)
        hb = ax.hexbin(
            first["time_order_pct"],
            first["grade"],
            gridsize=20,
            cmap="Blues",
            mincnt=1,
        )
        plt.colorbar(hb, ax=ax, label="Count")
        rho, pval = spearmanr(first["time_order_pct"], first["grade"])
        ptxt = "< 0.001" if pval < 0.001 else f"= {pval:.3f}"
        ax.set_title(f"{label}\nSpearman ρ = {rho:.3f} (p {ptxt})", fontweight="bold")
        ax.set_xlabel("Order of first attempt start (0 = earliest, 100 = latest)")
        ax.set_ylabel("Grade / 10")

    plt.suptitle(
        "H8 — Earlier starters vs grade? (first attempts, sorted by start time)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "h8_early_attempt_grade.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("\n── H8 — Negative ρ would support 'earlier → lower'; positive the opposite. ──")


def run_h9_attempts_vs_best_score(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H9: More attempts associates with a higher best score."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        summary = (
            quiz_df.groupby("student_id")
            .agg(total_attempts=("attempt_no", "max"), best_score=("grade", "max"))
            .reset_index()
        )
        agg = summary.groupby("total_attempts")["best_score"].agg(["mean", "sem", "count"]).reset_index()
        agg = agg[agg["count"] >= 5]
        if agg.empty:
            ax.set_title(f"{label} — sparse attempt counts")
            continue
        bars = ax.bar(
            agg["total_attempts"].astype(str),
            agg["mean"],
            yerr=agg["sem"] * 1.96,
            capsize=5,
            color=PALETTE["primary"],
            edgecolor="white",
            alpha=0.85,
        )
        for bar, cnt in zip(bars, agg["count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"n={int(cnt)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"{label} — mean best score by # attempts", fontweight="bold")
        ax.set_xlabel("Total attempts")
        ax.set_ylabel("Mean best score / 10")
        ax.set_ylim(0, 11)

    plt.suptitle("H9 — More attempts → better best score?", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "h9_attempts_vs_best_score.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("\n── H9 — Compare bar heights across attempt-count groups (n ≥ 5). ──")


def run_h10_question_score_variance_by_tier(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H10: Low performers have higher spread across normalised question scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        q_cols = [c for c in quiz_df.columns if c.startswith("Q.")]
        if not q_cols:
            continue
        d = quiz_df.copy()
        for c in q_cols:
            try:
                mx = float(c.split("/")[1])
            except (IndexError, ValueError):
                mx = float(d[c].max())
            d[c] = d[c] / mx
        d["q_std"] = d[q_cols].std(axis=1)
        med_grade = d["grade"].median()
        d["perf_tier"] = np.where(d["grade"] >= med_grade, "High", "Low")
        lo = d.loc[d["perf_tier"] == "Low", "q_std"].dropna()
        hi = d.loc[d["perf_tier"] == "High", "q_std"].dropna()
        if len(lo) < 2 or len(hi) < 2:
            ax.set_title(f"{label} — insufficient data")
            continue
        sns.violinplot(
            data=d,
            x="perf_tier",
            y="q_std",
            hue="perf_tier",
            order=["Low", "High"],
            palette={"Low": PALETTE["secondary"], "High": PALETTE["success"]},
            inner="box",
            ax=ax,
            cut=0,
            legend=False,
            dodge=False,
        )
        _, pval = stats.mannwhitneyu(lo, hi, alternative="greater")
        ptxt = "< 0.001" if pval < 0.001 else f"= {pval:.3f}"
        ax.set_title(f"{label}\nMann–Whitney p {ptxt}", fontweight="bold")
        ax.set_xlabel("Performance tier (by quiz median grade)")
        ax.set_ylabel("Std dev of normalised Q scores")

    plt.suptitle(
        "H10 — More uneven question marks among low performers?",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "h10_score_variance_by_tier.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("\n── H10 — Mann–Whitney: Low tier Q_std > High tier? ──")


def print_summary_table() -> None:
    summary = pd.DataFrame(
        {
            "ID": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"],
            "Hypothesis": [
                "Longer time → higher score",
                "Some questions consistently harder",
                "High performers improve; low performers erratic",
                "Hard Qs vs tiers; high performers faster (total time)",
                "Optimal time window for higher scores",
                "Full mark on Q1 → higher overall grade",
                "Students improve on 2nd attempt",
                "Earlier first starters vs grade",
                "More attempts → better best score",
                "Low performers have higher Q-score variance",
            ],
            "Visualization": [
                "Scatter + LOWESS (Spearman ρ)",
                "Bar (% full marks per Q)",
                "Line + band (mean grade by attempt & tier)",
                "Heatmap + violin (time by tier)",
                "Box plot (grade by time quantile)",
                "KDE comparison (Mann–Whitney)",
                "Slope chart (Wilcoxon)",
                "Hexbin (Spearman ρ)",
                "Bar with 95% CI",
                "Violin (Mann–Whitney)",
            ],
            "Task": ["Task 1"] * 5 + ["Task 2"] * 5,
        }
    )
    print(summary.to_string(index=False))


TASK2_RUNNERS = {
    "h6": run_h6_q1_full_mark_effect,
    "h7": run_h7_second_attempt_improvement,
    "h8": run_h8_early_starter_vs_grade,
    "h9": run_h9_attempts_vs_best_score,
    "h10": run_h10_question_score_variance_by_tier,
}
