"""Task 2 — student-generated hypotheses (H6–H10), adapted from ``docs/claud.py``."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr

from src.data_io import attempt_minutes_capped
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


def run_h11_retake_improve_time_delta(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """Retakes: improved score vs change in time between consecutive attempts (by start time).

    Durations come from ``time_sec`` via :func:`attempt_minutes_capped` (seconds vs ms
    heuristic from :func:`time_sec_to_minutes`, then clip at 120 min per attempt) so
    ``delta_time`` is in minutes in a typical quiz range.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    out_dir.mkdir(parents=True, exist_ok=True)

    green = PALETTE["success"]
    red = PALETTE["secondary"]
    grey = "#9CA3AF"

    def _consecutive_pairs(quiz_df: pd.DataFrame) -> pd.DataFrame:
        multi = quiz_df.groupby("student_id").filter(lambda x: len(x) > 1)
        if multi.empty:
            return pd.DataFrame(columns=["delta_time", "delta_score"])
        multi = multi.dropna(subset=["started_on"])
        rows: list[dict] = []
        for _, grp in multi.groupby("student_id", sort=False):
            grp = grp.sort_values("started_on")
            for i in range(len(grp) - 1):
                t0 = attempt_minutes_capped(float(grp.iloc[i]["time_sec"]))
                t1 = attempt_minutes_capped(float(grp.iloc[i + 1]["time_sec"]))
                g0 = float(grp.iloc[i]["grade"])
                g1 = float(grp.iloc[i + 1]["grade"])
                rows.append({"delta_time": float(t1 - t0), "delta_score": float(g1 - g0)})
        return pd.DataFrame(rows)

    prepared: list[tuple[str, pd.DataFrame | None]] = []
    for quiz_df, label in _triplets(q1, q2, q3):
        pairs = _consecutive_pairs(quiz_df)
        if pairs.empty or len(pairs) < 3:
            prepared.append((label, None))
            continue
        cap = pairs["delta_time"].quantile(0.95)
        d = pairs.copy()
        d["delta_time_plot"] = d["delta_time"].clip(upper=cap)
        d["improved"] = np.where(d["delta_score"] > 0, "Improved", "Not improved")
        prepared.append((label, d))

    # --- Figure A: one row, three columns — scatter + LOWESS per quiz
    fig_s, axes_s = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax_s, (label, d) in zip(axes_s, prepared):
        if d is None:
            ax_s.set_title(f"{label} — insufficient consecutive pairs", fontweight="bold")
            continue

        col = np.where(
            d["delta_score"] > 0,
            green,
            np.where(d["delta_score"] < 0, red, grey),
        )
        ax_s.scatter(
            d["delta_time_plot"],
            d["delta_score"],
            c=list(col),
            alpha=0.45,
            s=22,
            edgecolors="none",
        )
        ax_s.axhline(0, color="grey", linestyle="--", linewidth=1.1, zorder=0)
        ax_s.axvline(0, color="grey", linestyle="--", linewidth=1.1, zorder=0)

        xl, xr = ax_s.get_xlim()
        yb, yt = ax_s.get_ylim()
        ax_s.set_xlim(min(xl, 0), max(xr, 0))
        ax_s.set_ylim(min(yb, 0), max(yt, 0))
        xl, xr = ax_s.get_xlim()
        yb, yt = ax_s.get_ylim()
        sm = lowess(d["delta_score"], d["delta_time_plot"], frac=0.35)
        ax_s.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2.2, label="LOWESS", zorder=5)
        ax_s.legend(loc="best", fontsize=9)

        rho, p_sp = spearmanr(d["delta_time_plot"], d["delta_score"])
        ptxt = "< 0.001" if p_sp < 0.001 else f"= {p_sp:.4g}"
        ax_s.set_title(f"{label}\nSpearman rho = {rho:.3f} (p {ptxt})", fontweight="bold", fontsize=11)

        ax_s.text(
            (0 + xr) / 2,
            (0 + yt) / 2,
            "More time, Better score",
            ha="center",
            va="center",
            fontsize=9,
            color="#374151",
            zorder=6,
        )
        ax_s.text(
            (xl + 0) / 2,
            (0 + yt) / 2,
            "Less time, Better score",
            ha="center",
            va="center",
            fontsize=9,
            color="#374151",
            zorder=6,
        )
        ax_s.text(
            (0 + xr) / 2,
            (yb + 0) / 2,
            "More time, Worse score",
            ha="center",
            va="center",
            fontsize=9,
            color="#374151",
            zorder=6,
        )
        ax_s.text(
            (xl + 0) / 2,
            (yb + 0) / 2,
            "Less time, Worse score",
            ha="center",
            va="center",
            fontsize=9,
            color="#374151",
            zorder=6,
        )
        ax_s.set_xlabel("Delta time (min); 120 min/attempt cap; 95th pct for points")
        ax_s.set_ylabel("Delta score (next minus previous)")

    fig_s.suptitle(
        "H11 (a) — Delta time vs delta score (scatter + LOWESS)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig_s.savefig(out_dir / "h11_retake_delta_scatter.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig_s)

    # --- Figure B: one row, three columns — box plots per quiz
    fig_b, axes_b = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax_b, (label, d) in zip(axes_b, prepared):
        if d is None:
            ax_b.set_title(f"{label} — insufficient data", fontweight="bold")
            continue

        imp = d.loc[d["improved"] == "Improved", "delta_time_plot"]
        not_imp = d.loc[d["improved"] == "Not improved", "delta_time_plot"]
        if len(imp) >= 1 and len(not_imp) >= 1:
            _, p_mw = mannwhitneyu(imp, not_imp, alternative="two-sided")
            mw_txt = "< 0.001" if p_mw < 0.001 else f"= {p_mw:.4g}"
        else:
            mw_txt = "n/a"

        sns.boxplot(
            data=d,
            x="improved",
            y="delta_time_plot",
            order=["Improved", "Not improved"],
            hue="improved",
            hue_order=["Improved", "Not improved"],
            palette={"Improved": green, "Not improved": red},
            ax=ax_b,
            width=0.45,
            dodge=False,
            legend=False,
        )
        ax_b.set_title(f"{label}\nMann-Whitney p {mw_txt}", fontweight="bold", fontsize=11)
        ax_b.set_xlabel("")
        ax_b.set_ylabel("Delta time (min); 95th pct cap for display")

    fig_b.suptitle(
        "H11 (b) — Delta time: improved vs not improved",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig_b.savefig(out_dir / "h11_retake_delta_box.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig_b)
    print(
        "\n-- H11: Consecutive pairs; attempt duration = time_sec_to_minutes + 120 min cap; "
        "plot delta_time capped at 95th pctile. --"
    )


def run_h12_knowledge_efficiency_across_attempts(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H12: Multi-attempt students — score and time trends; mean time vs mean score per student."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    out_dir.mkdir(parents=True, exist_ok=True)

    cap_min = 120.0
    min_n = 30

    fig = plt.figure(figsize=(20, 14), layout="constrained")
    fig.suptitle(
        "H12 — Knowledge and efficiency across multiple attempts",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    subfigs = fig.subfigures(3, 1, hspace=0.2)

    for subfig, (quiz_df, quiz_label) in zip(subfigs, _triplets(q1, q2, q3)):
        subfig.suptitle(quiz_label, fontsize=14, fontweight="bold")
        ax1, ax2, ax3 = subfig.subplots(1, 3, gridspec_kw={"wspace": 0.32})

        multi = quiz_df.groupby("student_id").filter(lambda x: len(x) > 1).copy()
        if multi.empty:
            for ax in (ax1, ax2, ax3):
                ax.set_title("No multi-attempt students")
            continue

        multi = multi.sort_values(["student_id", "started_on"])
        multi["att"] = multi.groupby("student_id").cumcount() + 1
        multi["time_capped"] = multi["time_sec"].map(
            lambda s: attempt_minutes_capped(float(s), cap_min)
        )

        # --- Column 1: mean score by attempt number
        sc = multi.groupby("att")["grade"].agg(["mean", "std", "count"])
        sc = sc[sc["count"] >= min_n]
        sc["std"] = sc["std"].fillna(0.0)
        if len(sc) > 0:
            xs = sc.index.to_numpy(dtype=float)
            ax1.plot(xs, sc["mean"], color="#2563EB", linewidth=2.4, marker="o", markersize=6)
            ax1.fill_between(
                xs,
                sc["mean"] - sc["std"],
                sc["mean"] + sc["std"],
                alpha=0.22,
                color="#2563EB",
            )
            ax1.set_xticks(xs)
            for xi, ni in zip(xs, sc["count"]):
                ax1.text(
                    xi,
                    -0.15,
                    f"n={int(ni)}",
                    transform=ax1.get_xaxis_transform(),
                    ha="center",
                    fontsize=9,
                )
        else:
            ax1.text(
                0.5,
                0.5,
                f"No attempt with n >= {min_n}",
                transform=ax1.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
        ax1.set_title(f"Mean score by attempt number — {quiz_label}", fontweight="bold", fontsize=11)
        ax1.set_xlabel("Attempt number")
        ax1.set_ylabel("Mean score (grade / 10)")
        ax1.grid(True, alpha=0.35)

        # --- Column 2: mean time (capped) by attempt number
        tm = multi.groupby("att")["time_capped"].agg(["mean", "std", "count"])
        tm = tm[tm["count"] >= min_n]
        tm["std"] = tm["std"].fillna(0.0)
        if len(tm) > 0:
            xt = tm.index.to_numpy(dtype=float)
            ax2.plot(xt, tm["mean"], color="#EA580C", linewidth=2.4, marker="o", markersize=6)
            ax2.fill_between(
                xt,
                tm["mean"] - tm["std"],
                tm["mean"] + tm["std"],
                alpha=0.22,
                color="#EA580C",
            )
            ax2.set_xticks(xt)
            for xi, ni in zip(xt, tm["count"]):
                ax2.text(
                    xi,
                    -0.15,
                    f"n={int(ni)}",
                    transform=ax2.get_xaxis_transform(),
                    ha="center",
                    fontsize=9,
                )
        else:
            ax2.text(
                0.5,
                0.5,
                f"No attempt with n >= {min_n}",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
        ax2.set_title(f"Mean time taken by attempt number — {quiz_label}", fontweight="bold", fontsize=11)
        ax2.set_xlabel("Attempt number")
        ax2.set_ylabel("Mean time (minutes, capped at 120 per attempt)")
        ax2.grid(True, alpha=0.35)

        # --- Column 3: per-student means, color by attempt count
        stu = (
            multi.groupby("student_id")
            .agg(mean_time=("time_capped", "mean"), mean_grade=("grade", "mean"), n_att=("att", "count"))
            .dropna()
        )
        if len(stu) >= 3:
            nmin, nmax = int(stu["n_att"].min()), int(stu["n_att"].max())
            scat = ax3.scatter(
                stu["mean_time"],
                stu["mean_grade"],
                c=stu["n_att"],
                cmap="Blues",
                alpha=0.75,
                s=36,
                edgecolors="white",
                linewidths=0.4,
                vmin=nmin,
                vmax=nmax,
            )
            subfig.colorbar(scat, ax=ax3, label="Number of attempts", shrink=0.75)
            sm = lowess(stu["mean_grade"], stu["mean_time"], frac=0.45)
            ax3.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2.0, label="LOWESS")
            ax3.legend(loc="best", fontsize=9)
            rho, p_sp = spearmanr(stu["mean_time"], stu["mean_grade"])
            ptxt = "< 0.001" if p_sp < 0.001 else f"= {p_sp:.4g}"
            ax3.set_title(
                f"Mean time vs mean score per student — {quiz_label}\nSpearman rho = {rho:.3f} (p {ptxt})",
                fontweight="bold",
                fontsize=11,
            )
        else:
            ax3.set_title(f"Mean time vs mean score per student — {quiz_label}\n(too few students)", fontsize=11)
        ax3.set_xlabel("Mean time across attempts (minutes)")
        ax3.set_ylabel("Mean score across attempts (grade / 10)")
        ax3.grid(True, alpha=0.35)

    fig.savefig(out_dir / "h12_knowledge_efficiency_across_attempts.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(
        f"\n-- H12: Multi-attempt students only; time capped at {cap_min:.0f} min per attempt; "
        f"line plots require n>={min_n} per attempt number. --"
    )


def run_h13_slow_fast_failers_recovery(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    out_dir: Path,
    show: bool = True,
) -> None:
    """H13: Slow vs fast first-attempt failers — recovery score (best minus first grade)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    blue = "#2563EB"
    red = "#DC2626"
    order = ["Slow Failers", "Fast Failers"]

    fig = plt.figure(figsize=(14, 12), layout="constrained")
    fig.suptitle(
        "H13 — Slow vs Fast Failers: recovery score comparison",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    subfigs = fig.subfigures(3, 1, hspace=0.22)

    for subfig, (quiz_df, quiz_label) in zip(subfigs, _triplets(q1, q2, q3)):
        subfig.suptitle(quiz_label, fontsize=13, fontweight="bold")
        ax_v, ax_b = subfig.subplots(1, 2, gridspec_kw={"wspace": 0.3})

        df = quiz_df.sort_values(["student_id", "started_on"])
        first = df.groupby("student_id", as_index=False).first()
        first["time_c"] = first["time_sec"].map(lambda s: attempt_minutes_capped(float(s), 120.0))

        med_t = float(first["time_c"].median())
        med_g = float(first["grade"].median())

        best_by_stu = df.groupby("student_id")["grade"].max()
        first_ix = first.set_index("student_id")

        n_per_stu = df.groupby("student_id").size()
        multi_ids = set(n_per_stu[n_per_stu >= 2].index)

        rows = []
        for sid in multi_ids:
            g0 = float(first_ix.loc[sid, "grade"])
            if g0 >= med_g:
                continue
            tc = float(first_ix.loc[sid, "time_c"])
            if tc > med_t:
                grp = "Slow Failers"
            elif tc < med_t:
                grp = "Fast Failers"
            else:
                continue
            rec = float(best_by_stu.loc[sid]) - g0
            rows.append({"group": grp, "recovery": rec})

        plot_df = pd.DataFrame(rows)
        n_slow = int((plot_df["group"] == "Slow Failers").sum()) if len(plot_df) else 0
        n_fast = int((plot_df["group"] == "Fast Failers").sum()) if len(plot_df) else 0

        if len(plot_df) == 0 or (n_slow == 0 and n_fast == 0):
            for ax in (ax_v, ax_b):
                ax.text(0.5, 0.5, "No students in groups", transform=ax.transAxes, ha="center", va="center")
            continue

        slow_s = plot_df.loc[plot_df["group"] == "Slow Failers", "recovery"]
        fast_s = plot_df.loc[plot_df["group"] == "Fast Failers", "recovery"]
        if len(slow_s) >= 1 and len(fast_s) >= 1:
            _, p_mw = mannwhitneyu(slow_s, fast_s, alternative="two-sided")
            ptxt = "< 0.001" if p_mw < 0.001 else f"= {p_mw:.4g}"
        else:
            ptxt = "n/a (one group empty)"

        palette = {"Slow Failers": blue, "Fast Failers": red}
        present = [g for g in order if (plot_df["group"] == g).any()]
        xlabels = [
            f"Slow Failers\nn={n_slow}" if g == "Slow Failers" else f"Fast Failers\nn={n_fast}" for g in present
        ]

        pal_map = {g: palette[g] for g in present}
        sns.violinplot(
            data=plot_df,
            x="group",
            y="recovery",
            order=present,
            hue="group",
            hue_order=present,
            palette=pal_map,
            ax=ax_v,
            cut=0,
            inner=None,
            width=0.7,
            dodge=False,
            legend=False,
        )
        sns.stripplot(
            data=plot_df,
            x="group",
            y="recovery",
            order=present,
            hue="group",
            hue_order=present,
            palette=pal_map,
            ax=ax_v,
            alpha=0.35,
            size=3.5,
            jitter=0.22,
            dodge=False,
            legend=False,
        )
        ax_v.axhline(0, color="grey", linestyle="--", linewidth=1.1, zorder=0)
        ax_v.set_xticks(range(len(present)))
        ax_v.set_xticklabels(xlabels)
        ax_v.set_title(
            f"Recovery score by first-attempt group — {quiz_label}\nMann-Whitney p {ptxt}",
            fontweight="bold",
            fontsize=11,
        )
        ax_v.set_xlabel("")
        ax_v.set_ylabel("Recovery score (best − first attempt)")

        sns.boxplot(
            data=plot_df,
            x="group",
            y="recovery",
            order=present,
            hue="group",
            hue_order=present,
            palette=pal_map,
            ax=ax_b,
            width=0.5,
            dodge=False,
            legend=False,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(markerfacecolor="grey", markersize=4, alpha=0.7),
        )
        ax_b.axhline(0, color="grey", linestyle="--", linewidth=1.1, zorder=0)
        ax_b.set_xticks(range(len(present)))
        ax_b.set_xticklabels(xlabels)
        ylo, yhi = ax_b.get_ylim()
        yspan = yhi - ylo if yhi > ylo else 1.0
        for i, grp in enumerate(present):
            sub = plot_df.loc[plot_df["group"] == grp, "recovery"]
            if len(sub) == 0:
                continue
            med = float(sub.median())
            ax_b.text(
                i,
                med + 0.03 * yspan,
                f"{med:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="black",
            )
        ax_b.set_title(
            f"Recovery score distribution — {quiz_label}\nMann-Whitney p {ptxt}",
            fontweight="bold",
            fontsize=11,
        )
        ax_b.set_xlabel("")
        ax_b.set_ylabel("Recovery score (best − first attempt)")

    fig.savefig(out_dir / "h13_slow_fast_failers_recovery.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("\n-- H13: First-attempt medians (all students); groups among multi-attempt failers; time capped at 120 min. --")


def print_summary_table() -> None:
    summary = pd.DataFrame(
        {
            "ID": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13"],
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
                "Improved retakes vs change in time (consecutive attempts)",
                "Knowledge and efficiency improve across attempts",
                "Slow failers recover more than fast failers",
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
                "Scatter + LOWESS; box (Mann–Whitney) — two PNGs",
                "Lines + band + scatter (Spearman)",
                "Violin + box (Mann–Whitney)",
            ],
            "Task": ["Task 1"] * 5 + ["Task 2"] * 8,
        }
    )
    print(summary.to_string(index=False))


TASK2_RUNNERS = {
    "h6": run_h6_q1_full_mark_effect,
    "h7": run_h7_second_attempt_improvement,
    "h8": run_h8_early_starter_vs_grade,
    "h9": run_h9_attempts_vs_best_score,
    "h10": run_h10_question_score_variance_by_tier,
    "h11": run_h11_retake_improve_time_delta,
    "h12": run_h12_knowledge_efficiency_across_attempts,
    "h13": run_h13_slow_fast_failers_recovery,
}
