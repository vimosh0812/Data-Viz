"""Task 1 hypotheses (H1–H5) from the assignment brief."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr

from src.viz_style import PALETTE


def _triplets(q1: pd.DataFrame, q2: pd.DataFrame, q3: pd.DataFrame):
    return [(q1, "Quiz 1"), (q2, "Quiz 2"), (q3, "Quiz 3")]


def run_h1_time_vs_score(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H1: Students who take longer tend to score higher."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]

    for ax, (quiz_df, label), color in zip(axes, _triplets(q1, q2, q3), colors):
        cap = quiz_df["time_min"].quantile(0.97)
        d = quiz_df[quiz_df["time_min"] <= cap].copy()
        rho, pval = spearmanr(d["time_min"], d["grade"])

        ax.scatter(d["time_min"], d["grade"], alpha=0.25, s=18, color=color, edgecolors="none")
        sm = lowess(d["grade"], d["time_min"], frac=0.3)
        ax.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2, label="LOWESS trend")

        ptxt = "< 0.001" if pval < 0.001 else f"= {pval:.3f}"
        ax.set_title(f"{label}\nSpearman ρ = {rho:.3f}  (p {ptxt})", fontweight="bold")
        ax.set_xlabel("Time taken (minutes)")
        ax.set_ylabel("Grade / 10")
        ax.legend(fontsize=9)

    plt.suptitle("H1 — Does longer time → higher score?", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "h1_time_vs_score.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print("\n── H1 verdict ──")
    for quiz_df, label in _triplets(q1, q2, q3):
        rho, pval = spearmanr(quiz_df["time_min"], quiz_df["grade"])
        verdict = "support (positive association)" if rho > 0.1 and pval < 0.05 else "weak / not supported"
        print(f"  {label}: ρ = {rho:.3f}, p = {pval:.4f} → {verdict}")


def _q_difficulty(quiz_df: pd.DataFrame) -> dict[str, float]:
    q_cols = [c for c in quiz_df.columns if c.startswith("Q.")]
    max_mark = {}
    for c in q_cols:
        try:
            max_mark[c] = float(c.split("/")[1].strip())
        except (IndexError, ValueError):
            max_mark[c] = float(quiz_df[c].max())
    correct_rate = {}
    for c in q_cols:
        full_marks = quiz_df[c] == max_mark[c]
        qid = c.split()[1].rstrip(".")
        correct_rate[qid] = full_marks.mean() * 100
    return correct_rate


def run_h2_question_difficulty(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H2: Some questions are consistently harder than others."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        cr = _q_difficulty(quiz_df)
        qs = list(cr.keys())
        vals = list(cr.values())
        colors = [
            PALETTE["success"] if v >= 70 else PALETTE["accent"] if v >= 40 else PALETTE["secondary"]
            for v in vals
        ]
        bars = ax.bar(qs, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
        ax.set_ylim(0, 105)
        ax.set_title(f"{label} — full-mark rate per question", fontweight="bold")
        ax.set_xlabel("Question")
        ax.set_ylabel("% students scoring full marks")
        ax.axhline(50, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    patches = [
        mpatches.Patch(color=PALETTE["success"], label="Easy (≥ 70%)"),
        mpatches.Patch(color=PALETTE["accent"], label="Medium (40–70%)"),
        mpatches.Patch(color=PALETTE["secondary"], label="Hard (< 40%)"),
    ]
    fig.legend(handles=patches, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=10)
    plt.suptitle("H2 — Are some questions consistently harder?", fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()
    fig.savefig(out_dir / "h2_question_difficulty.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("\n── H2 — Compare bar heights across Q1–Q5 within each quiz. ──")


def _tier_from_first_grade(first_grade: float) -> str:
    if first_grade >= 7:
        return "High"
    if first_grade >= 4:
        return "Mid"
    return "Low"


def run_h3_progression_by_tier(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H3: High performers improve consistently; low performers more erratic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors_tier = {"High": PALETTE["success"], "Mid": PALETTE["accent"], "Low": PALETTE["secondary"]}

    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        multi = quiz_df[quiz_df.groupby("student_id")["attempt_no"].transform("max") >= 2]
        if multi.empty:
            ax.set_title(f"{label} — no multi-attempt students")
            continue

        first_grade = (
            multi.sort_values(["student_id", "attempt_no"])
            .groupby("student_id", as_index=False)
            .first()[["student_id", "grade"]]
            .rename(columns={"grade": "first_grade"})
        )
        first_grade["tier"] = first_grade["first_grade"].map(_tier_from_first_grade)
        multi = multi.merge(first_grade[["student_id", "tier"]], on="student_id")

        agg = multi.groupby(["tier", "attempt_no"])["grade"].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0)

        for tier, grp in agg.groupby("tier"):
            ax.plot(
                grp["attempt_no"],
                grp["mean"],
                marker="o",
                linewidth=2.2,
                color=colors_tier[tier],
                label=tier,
            )
            ax.fill_between(
                grp["attempt_no"],
                grp["mean"] - grp["std"],
                grp["mean"] + grp["std"],
                alpha=0.15,
                color=colors_tier[tier],
            )

        ax.set_title(f"{label} — mean grade by attempt & tier", fontweight="bold")
        ax.set_xlabel("Attempt number")
        ax.set_ylabel("Mean grade / 10")
        ax.set_ylim(0, 11)
        ax.legend(title="Tier (from 1st attempt)", fontsize=9)

    plt.suptitle(
        "H3 — High vs low performers across attempts?",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "h3_improvement_by_tier.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("\n── H3 — Inspect whether High tier mean rises smoothly vs Low tier spread. ──")


def run_h4_difficulty_and_time_by_tier(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H4: Harder items vs performance; high performers faster (per-question time not in CSV).

    Per-question response times are not exported; we use mean *total* quiz time split by
    whether the attempt is below vs at/above the median mark on each question (proxy).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        q_cols = [c for c in quiz_df.columns if c.startswith("Q.")]
        med_grade = quiz_df["grade"].median()
        d = quiz_df.copy()
        d["perf_tier"] = np.where(d["grade"] >= med_grade, "High (≥ median)", "Low (< median)")
        agg = d.groupby("perf_tier")[q_cols].mean()
        for c in q_cols:
            try:
                mx = float(c.split("/")[1])
            except (IndexError, ValueError):
                mx = float(d[c].max())
            agg[c] = agg[c] / mx
        agg.columns = [f"Q{c.split()[1].rstrip('.')}" for c in q_cols]
        sns.heatmap(
            agg,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Normalised score"},
        )
        ax.set_title(f"{label} — mean normalised score by Q & tier", fontweight="bold")
        ax.set_ylabel("Performance tier")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("H4 (part A) — Question marks vs performance tier", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "h4_difficulty_perf_heatmap.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # Proxy for linking items to duration: total quiz time by median split on each question's mark
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        q_cols = [c for c in quiz_df.columns if c.startswith("Q.")]
        below_means = []
        above_means = []
        x_labels = []
        for c in q_cols:
            qid = f"Q{c.split()[1].rstrip('.')}"
            x_labels.append(qid)
            med_m = quiz_df[c].median()
            below = quiz_df[quiz_df[c] < med_m]["time_min"]
            above = quiz_df[quiz_df[c] >= med_m]["time_min"]
            below_means.append(below.mean() if len(below) else np.nan)
            above_means.append(above.mean() if len(above) else np.nan)
        x = np.arange(len(x_labels))
        w = 0.36
        ax.bar(x - w / 2, below_means, width=w, label="Below median mark on Q", color=PALETTE["secondary"], edgecolor="white")
        ax.bar(x + w / 2, above_means, width=w, label="At/above median mark on Q", color=PALETTE["success"], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{label}", fontweight="bold")
        ax.set_xlabel("Question")
        ax.set_ylabel("Mean total quiz time (min)")
        ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(
        "H4 (part B) — Mean total quiz time by score on each question (proxy; no per-Q timestamps)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "h4_mean_total_time_by_question_median_split.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors_scatter = {
        "High (≥ median)": PALETTE["success"],
        "Low (< median)": PALETTE["secondary"],
    }
    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        cap = quiz_df["time_min"].quantile(0.97)
        d = quiz_df[quiz_df["time_min"] <= cap].copy()
        med_grade = d["grade"].median()
        d["perf_tier"] = np.where(d["grade"] >= med_grade, "High (≥ median)", "Low (< median)")
        for tier in ("High (≥ median)", "Low (< median)"):
            sub = d[d["perf_tier"] == tier]
            ax.scatter(
                sub["time_min"],
                sub["grade"],
                alpha=0.28,
                s=16,
                color=colors_scatter[tier],
                edgecolors="none",
                label=tier,
            )
        ax.axhline(med_grade, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(f"{label} — grade vs total time", fontweight="bold")
        ax.set_xlabel("Time taken (minutes)")
        ax.set_ylabel("Grade / 10")
        ax.legend(fontsize=8, loc="best")

    plt.suptitle(
        "H4 (part C) — Grade vs time by overall performance tier (joint view)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "h4_grade_time_scatter_by_tier.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        d = quiz_df.copy()
        med_grade = d["grade"].median()
        d["perf_tier"] = np.where(d["grade"] >= med_grade, "High (≥ median)", "Low (< median)")
        high_t = d[d["perf_tier"] == "High (≥ median)"]["time_min"].dropna()
        low_t = d[d["perf_tier"] == "Low (< median)"]["time_min"].dropna()
        _, p_mw = mannwhitneyu(high_t, low_t, alternative="two-sided")

        cap = quiz_df["time_min"].quantile(0.97)
        d_plot = d[d["time_min"] <= cap].copy()
        ymax = d_plot["time_min"].quantile(0.95)
        sns.violinplot(
            data=d_plot,
            x="perf_tier",
            y="time_min",
            hue="perf_tier",
            ax=ax,
            palette={
                "High (≥ median)": PALETTE["success"],
                "Low (< median)": PALETTE["secondary"],
            },
            inner="box",
            cut=0,
            legend=False,
            dodge=False,
        )
        ax.set_ylim(0, ymax * 1.02)
        ptxt = "< 0.001" if p_mw < 0.001 else f"= {p_mw:.4f}"
        ax.set_title(f"{label}\nMann-Whitney U, p {ptxt}", fontweight="bold")
        ax.set_xlabel("Performance tier")
        ax.set_ylabel("Time (minutes)")

    plt.suptitle(
        "H4 (part D) — Total quiz time: high vs low performers (y-axis capped at 95th %ile for readability)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "h4_time_vs_perf_violin.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print("\n-- H4: Mann-Whitney U (total time: high vs low grade, median split) --")
    for quiz_df, label in _triplets(q1, q2, q3):
        d = quiz_df.copy()
        med_grade = d["grade"].median()
        high_t = d[d["grade"] >= med_grade]["time_min"].dropna()
        low_t = d[d["grade"] < med_grade]["time_min"].dropna()
        u_stat, p_mw = mannwhitneyu(high_t, low_t, alternative="two-sided")
        print(
            f"  {label}: U = {u_stat:.0f}, p = {p_mw:.4g} "
            f"(median time high={high_t.median():.2f} min, low={low_t.median():.2f} min)"
        )
    print(
        "-- H4: Per-question timestamps are not in the export; "
        "heatmap + median-split bars + joint scatter are indirect evidence. --"
    )


def run_h5_optimal_time_window(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H5: Optimal time band; very fast / very slow → lower scores.

    Complements H1: if there were an inverted-U “sweet spot”, LOWESS would bend
    downward at both ends; quintile box plots summarise the same axis.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (quiz_df, label) in zip(axes, _triplets(q1, q2, q3)):
        cap = quiz_df["time_min"].quantile(0.97)
        d = quiz_df[quiz_df["time_min"] <= cap].copy()
        try:
            d["time_bin"] = pd.qcut(d["time_min"], q=5, duplicates="drop")
        except ValueError:
            ax.set_title(f"{label} — insufficient distinct times for quintiles")
            continue
        bin_order = list(d["time_bin"].cat.categories)
        if len(bin_order) < 2:
            ax.set_title(f"{label} — not enough bins after qcut")
            continue
        pal_use = sns.color_palette("coolwarm_r", n_colors=len(bin_order))
        sns.boxplot(
            data=d,
            x="time_bin",
            y="grade",
            hue="time_bin",
            order=bin_order,
            palette=dict(zip(bin_order, pal_use)),
            ax=ax,
            width=0.55,
            medianprops=dict(color="black", linewidth=2),
            legend=False,
            dodge=False,
        )
        ax.set_title(f"{label} — grade by time quantile group", fontweight="bold")
        ax.set_xlabel("Time group (fast → slow)")
        ax.set_ylabel("Grade / 10")
        ax.tick_params(axis="x", rotation=25)

    plt.suptitle("H5 — Optimal time window for higher scores?", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "h5_optimal_time_window.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
    for ax, (quiz_df, label), color in zip(axes, _triplets(q1, q2, q3), colors):
        cap = quiz_df["time_min"].quantile(0.97)
        d = quiz_df[quiz_df["time_min"] <= cap].copy()
        ax.scatter(d["time_min"], d["grade"], alpha=0.25, s=18, color=color, edgecolors="none")
        sm = lowess(d["grade"], d["time_min"], frac=0.3)
        ax.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2, label="LOWESS")
        try:
            d["time_bin"] = pd.qcut(d["time_min"], q=5, duplicates="drop")
            codes = d["time_bin"].cat.codes
            valid = codes >= 0
            if valid.sum() >= 2 and codes[valid].nunique() >= 2:
                rho, p_sp = spearmanr(codes[valid], d.loc[valid, "grade"])
                ptxt = "< 0.001" if p_sp < 0.001 else f"= {p_sp:.3f}"
                ax.set_title(
                    f"{label}\nSpearman (time quintile vs grade): ρ = {rho:.3f} (p {ptxt})",
                    fontweight="bold",
                )
            else:
                ax.set_title(f"{label}", fontweight="bold")
        except (ValueError, TypeError):
            ax.set_title(f"{label}", fontweight="bold")
        ax.set_xlabel("Time taken (minutes)")
        ax.set_ylabel("Grade / 10")
        ax.legend(fontsize=9)

    plt.suptitle(
        "H5 — Grade vs time with LOWESS (inverted-U would show a clear mid-axis peak)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "h5_grade_vs_time_lowess.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print("\n-- H5: Median grade per time quintile (fast -> slow); compare shape to H1. --")
    for quiz_df, label in _triplets(q1, q2, q3):
        cap = quiz_df["time_min"].quantile(0.97)
        d = quiz_df[quiz_df["time_min"] <= cap].copy()
        try:
            d["time_bin"] = pd.qcut(d["time_min"], q=5, duplicates="drop")
        except ValueError:
            print(f"  {label}: insufficient distinct times for quintiles.")
            continue
        med_by_bin = d.groupby("time_bin", observed=True)["grade"].median()
        parts = [f"{idx.left:.1f}-{idx.right:.1f} min -> med {v:.1f}" for idx, v in med_by_bin.items()]
        print(f"  {label}: " + " | ".join(parts))
        codes = d["time_bin"].cat.codes
        valid = codes >= 0
        if valid.sum() >= 2 and codes[valid].nunique() >= 2:
            rho, p_sp = spearmanr(codes[valid], d.loc[valid, "grade"])
            print(f"    Spearman(ordinal time quintile, grade): rho = {rho:.3f}, p = {p_sp:.4g}")
    print(
        "-- H5: Consistent with H1: no smooth inverted-U is required to reject an "
        'optimal "middle time" band if LOWESS is monotone or flat. --'
    )


TASK1_RUNNERS = {
    "h1": run_h1_time_vs_score,
    "h2": run_h2_question_difficulty,
    "h3": run_h3_progression_by_tier,
    "h4": run_h4_difficulty_and_time_by_tier,
    "h5": run_h5_optimal_time_window,
}
