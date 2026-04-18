# =============================================================================
# QUIZ MARKS - HYPOTHESIS TESTING & VISUALIZATION
# Each section is a standalone notebook cell block
# Dataset: quiz1_marks.csv, quiz2_marks.csv, quiz3_marks.csv
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — IMPORTS & SETUP
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings("ignore")

# Colour palette (accessible / print-friendly)
PALETTE = {
    "primary":   "#2E86AB",
    "secondary": "#E84855",
    "accent":    "#F4A261",
    "neutral":   "#6B7280",
    "light":     "#F3F4F6",
    "success":   "#22C55E",
}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

print("✅ Libraries loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_time_to_seconds(t):
    """Convert 'X mins Y secs' / 'Y secs' / 'X mins' strings to total seconds."""
    if pd.isna(t) or t.strip() in ["-", ""]:
        return np.nan
    t = t.strip()
    total = 0
    if "hour" in t:
        parts = t.split("hour")
        total += int(parts[0].strip()) * 3600
        t = parts[1]
    if "min" in t:
        parts = t.split("min")
        total += int(parts[0].strip()) * 60
        t = parts[1]
    if "sec" in t:
        secs = t.replace("s", "").strip()
        total += int(''.join(filter(str.isdigit, secs)))
    return total if total > 0 else np.nan


def load_quiz(filepath, quiz_id):
    """Load one quiz CSV, clean and standardise."""
    df = pd.read_csv(filepath)

    # Keep only finished attempts
    df = df[df["State"] == "Finished"].copy()

    # Parse numeric grade
    df["grade"] = pd.to_numeric(df["Grade/10.00"], errors="coerce")

    # Parse time taken → seconds
    df["time_sec"] = df["Time taken"].apply(parse_time_to_seconds)
    df["time_min"] = df["time_sec"] / 60

    # Identify question columns
    q_cols = [c for c in df.columns if c.startswith("Q.")]
    for c in q_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rename for consistency
    df = df.rename(columns={"Student Code": "student_id"})
    df["quiz"] = quiz_id

    # Attempt number per student (sorted by start time)
    df["Started on"] = pd.to_datetime(df["Started on"], errors="coerce")
    df = df.sort_values(["student_id", "Started on"])
    df["attempt_no"] = df.groupby("student_id").cumcount() + 1

    # Keep useful columns
    keep = ["student_id", "quiz", "attempt_no", "grade", "time_sec", "time_min"] + q_cols
    return df[keep].dropna(subset=["grade", "time_sec"])


# ── Load all three quizzes ──
Q1 = load_quiz("quiz1_marks.csv", 1)
Q2 = load_quiz("quiz2_marks.csv", 2)
Q3 = load_quiz("quiz3_marks.csv", 3)

ALL = pd.concat([Q1, Q2, Q3], ignore_index=True)

print(f"Quiz 1: {len(Q1)} rows | Quiz 2: {len(Q2)} rows | Quiz 3: {len(Q3)} rows")
print(f"Combined: {len(ALL)} rows")
print(ALL.describe().round(2))


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — EXPLORATORY OVERVIEW  (not a hypothesis — useful context)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (quiz_df, label) in zip(axes, [(Q1, "Quiz 1"), (Q2, "Quiz 2"), (Q3, "Quiz 3")]):
    ax.hist(quiz_df["grade"].dropna(), bins=15, color=PALETTE["primary"],
            edgecolor="white", linewidth=0.6)
    ax.set_title(f"{label} – Grade Distribution", fontweight="bold")
    ax.set_xlabel("Grade / 10")
    ax.set_ylabel("Count")
    ax.axvline(quiz_df["grade"].mean(), color=PALETTE["secondary"],
               linestyle="--", linewidth=1.5, label=f"Mean={quiz_df['grade'].mean():.1f}")
    ax.legend(fontsize=9)
plt.suptitle("Grade Distributions Across All Quizzes", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("overview_grade_dist.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4 — HYPOTHESIS 1 (Task 1)
# "Students who take longer to complete the quiz tend to score higher."
# ─────────────────────────────────────────────────────────────────────────────
# Approach: scatter + regression line + Spearman ρ per quiz

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

for ax, (quiz_df, label, color) in zip(
    axes,
    [(Q1, "Quiz 1", PALETTE["primary"]),
     (Q2, "Quiz 2", PALETTE["secondary"]),
     (Q3, "Quiz 3", PALETTE["accent"])]
):
    # Cap extreme outliers for cleaner plot
    cap = quiz_df["time_min"].quantile(0.97)
    d = quiz_df[quiz_df["time_min"] <= cap].copy()

    rho, pval = spearmanr(d["time_min"], d["grade"])

    ax.scatter(d["time_min"], d["grade"],
               alpha=0.25, s=18, color=color, edgecolors="none")

    # LOWESS smoothed trend
    from statsmodels.nonparametric.smoothers_lowess import lowess
    sm = lowess(d["grade"], d["time_min"], frac=0.3)
    ax.plot(sm[:, 0], sm[:, 1], color="black", linewidth=2, label="LOWESS trend")

    ax.set_title(f"{label}\nSpearman ρ = {rho:.3f}  (p {'< 0.001' if pval < 0.001 else f'= {pval:.3f}'})",
                 fontweight="bold")
    ax.set_xlabel("Time Taken (minutes)")
    ax.set_ylabel("Grade / 10")
    ax.legend(fontsize=9)

plt.suptitle("H1 – Does Longer Time → Higher Score?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("h1_time_vs_score.png", bbox_inches="tight")
plt.show()

# Decision helper
print("\n── H1 Verdict ──")
for quiz_df, label in [(Q1, "Quiz 1"), (Q2, "Quiz 2"), (Q3, "Quiz 3")]:
    rho, pval = spearmanr(quiz_df["time_min"], quiz_df["grade"])
    verdict = "ACCEPT" if rho > 0.1 and pval < 0.05 else "REJECT"
    print(f"  {label}: ρ={rho:.3f}, p={pval:.4f} → {verdict}")


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5 — HYPOTHESIS 2 (Task 1)
# "Some questions are consistently harder than others."
# ─────────────────────────────────────────────────────────────────────────────
# Approach: grouped bar chart of per-question correct-answer rate across quizzes

def q_difficulty(quiz_df, quiz_label):
    q_cols = [c for c in quiz_df.columns if c.startswith("Q.")]
    max_mark = {}
    for c in q_cols:
        # Extract max from column name e.g. "Q. 1 /2.00" → 2.0
        try:
            max_mark[c] = float(c.split("/")[1].strip())
        except Exception:
            max_mark[c] = quiz_df[c].max()
    
    correct_rate = {}
    for c in q_cols:
        full_marks = quiz_df[c] == max_mark[c]
        correct_rate[c.split()[1]] = full_marks.mean() * 100  # percent
    return correct_rate


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
quiz_data = [(Q1, "Quiz 1"), (Q2, "Quiz 2"), (Q3, "Quiz 3")]

for ax, (quiz_df, label) in zip(axes, quiz_data):
    cr = q_difficulty(quiz_df, label)
    qs = list(cr.keys())
    vals = list(cr.values())
    colors = [PALETTE["success"] if v >= 70 else PALETTE["accent"] if v >= 40
              else PALETTE["secondary"] for v in vals]
    bars = ax.bar(qs, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    ax.set_ylim(0, 105)
    ax.set_title(f"{label} – Full-mark Rate per Question", fontweight="bold")
    ax.set_xlabel("Question")
    ax.set_ylabel("% Students Scoring Full Marks")
    ax.axhline(50, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Legend
patches = [mpatches.Patch(color=PALETTE["success"],   label="Easy (≥70%)"),
           mpatches.Patch(color=PALETTE["accent"],    label="Medium (40–70%)"),
           mpatches.Patch(color=PALETTE["secondary"], label="Hard (<40%)")]
fig.legend(handles=patches, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=10)
plt.suptitle("H2 – Are Some Questions Consistently Harder?", fontsize=14, fontweight="bold", y=1.06)
plt.tight_layout()
plt.savefig("h2_question_difficulty.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6 — HYPOTHESIS 3 (Task 1)
# "High-performing students improve consistently; low performers show erratic progress."
# ─────────────────────────────────────────────────────────────────────────────
# Approach: line plots of grade across attempts, split by performer tier

def classify_performer(student_df):
    """Classify a student using their FIRST attempt grade."""
    first_grade = student_df.sort_values("attempt_no")["grade"].iloc[0]
    if first_grade >= 7:
        return "High"
    elif first_grade >= 4:
        return "Mid"
    else:
        return "Low"


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
colors_tier = {"High": PALETTE["success"], "Mid": PALETTE["accent"], "Low": PALETTE["secondary"]}

for ax, (quiz_df, label) in zip(axes, quiz_data):
    multi = quiz_df[quiz_df.groupby("student_id")["attempt_no"].transform("max") >= 2]
    if multi.empty:
        ax.set_title(f"{label} – No multi-attempt students")
        continue

    # Assign tier
    tier_map = multi.groupby("student_id").apply(classify_performer).rename("tier")
    multi = multi.join(tier_map, on="student_id")

    # Mean grade per (tier, attempt_no)
    agg = multi.groupby(["tier", "attempt_no"])["grade"].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0)

    for tier, grp in agg.groupby("tier"):
        ax.plot(grp["attempt_no"], grp["mean"], marker="o", linewidth=2.2,
                color=colors_tier[tier], label=tier)
        ax.fill_between(grp["attempt_no"],
                        grp["mean"] - grp["std"],
                        grp["mean"] + grp["std"],
                        alpha=0.15, color=colors_tier[tier])

    ax.set_title(f"{label} – Grade Progression by Tier", fontweight="bold")
    ax.set_xlabel("Attempt Number")
    ax.set_ylabel("Mean Grade / 10")
    ax.set_ylim(0, 11)
    ax.legend(title="Tier", fontsize=9)

plt.suptitle("H3 – High Performers Improve; Low Performers Erratic?", fontsize=14,
             fontweight="bold")
plt.tight_layout()
plt.savefig("h3_improvement_by_tier.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 7 — HYPOTHESIS 4 (Task 1)
# "Harder questions take longer; but high-performers answer them faster."
# ─────────────────────────────────────────────────────────────────────────────
# Approach: heatmap of mean per-question score split by low/high performer
# (We don't have per-question time, so proxy: overall time is compared
#  across question-difficulty groups & student tiers → grouped violin)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (quiz_df, label) in zip(axes, quiz_data):
    q_cols = [c for c in quiz_df.columns if c.startswith("Q.")]
    med_grade = quiz_df["grade"].median()
    quiz_df["perf_tier"] = quiz_df["grade"].apply(
        lambda g: "High (≥ median)" if g >= med_grade else "Low (< median)"
    )

    # Mean score on each question per performance tier
    agg = quiz_df.groupby("perf_tier")[q_cols].mean()

    # Normalise to 0–1 per question (max possible)
    for c in q_cols:
        try:
            mx = float(c.split("/")[1])
        except Exception:
            mx = quiz_df[c].max()
        agg[c] = agg[c] / mx

    agg.columns = [f"Q{c.split()[1]}" for c in q_cols]
    sns.heatmap(agg, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Normalised Score"})
    ax.set_title(f"{label}\nAvg Normalised Score by Q & Tier", fontweight="bold")
    ax.set_ylabel("Performance Tier")
    ax.tick_params(axis='x', rotation=45)

plt.suptitle("H4 – Question Difficulty vs Performance Tier", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("h4_difficulty_perf_heatmap.png", bbox_inches="tight")
plt.show()

# Companion: time distribution for high vs low performers
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (quiz_df, label) in zip(axes, quiz_data):
    cap = quiz_df["time_min"].quantile(0.97)
    d = quiz_df[quiz_df["time_min"] <= cap].copy()
    med_grade = d["grade"].median()
    d["perf_tier"] = d["grade"].apply(
        lambda g: "High (≥ median)" if g >= med_grade else "Low (< median)"
    )
    sns.violinplot(data=d, x="perf_tier", y="time_min", ax=ax,
                   palette=[PALETTE["success"], PALETTE["secondary"]],
                   inner="box", cut=0)
    ax.set_title(f"{label} – Time per Tier", fontweight="bold")
    ax.set_xlabel("Performance Tier")
    ax.set_ylabel("Time (minutes)")

plt.suptitle("H4 (companion) – Are High Performers Faster?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("h4_time_vs_perf_violin.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 8 — HYPOTHESIS 5 (Task 1)
# "There is an optimal time range; finishing too fast or too slow → lower score."
# ─────────────────────────────────────────────────────────────────────────────
# Approach: binned box plot of grade vs time quantile group (quintiles)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (quiz_df, label) in zip(axes, quiz_data):
    cap = quiz_df["time_min"].quantile(0.97)
    d = quiz_df[quiz_df["time_min"] <= cap].copy()
    d["time_bin"] = pd.qcut(d["time_min"], q=5,
                            labels=["Very Fast", "Fast", "Medium", "Slow", "Very Slow"])
    order = ["Very Fast", "Fast", "Medium", "Slow", "Very Slow"]
    pal = sns.color_palette("coolwarm_r", n_colors=5)
    sns.boxplot(data=d, x="time_bin", y="grade", order=order,
                palette=pal, ax=ax, width=0.55,
                medianprops=dict(color="black", linewidth=2))
    ax.set_title(f"{label} – Grade by Time Quintile", fontweight="bold")
    ax.set_xlabel("Time Group")
    ax.set_ylabel("Grade / 10")
    ax.tick_params(axis='x', rotation=20)

plt.suptitle("H5 – Optimal Time Window for Higher Scores?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("h5_optimal_time_window.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 9 — HYPOTHESIS 6 (Task 2 — Student Generated)
# "Students who score full marks on Q1 tend to score higher overall."
# ─────────────────────────────────────────────────────────────────────────────
# Rationale: Q1 may test foundational concepts; acing it signals mastery.
# Viz: Side-by-side KDE + rug for total grade, conditioned on Q1 full mark

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (quiz_df, label) in zip(axes, quiz_data):
    q1_col = [c for c in quiz_df.columns if c.startswith("Q. 1")]
    if not q1_col:
        ax.set_title(f"{label} – Q1 not found"); continue
    q1_col = q1_col[0]
    q1_max = float(q1_col.split("/")[1])
    quiz_df["q1_full"] = quiz_df[q1_col] == q1_max

    for flag, color, lbl in [(True, PALETTE["success"], "Q1 Full Mark"),
                              (False, PALETTE["secondary"], "Q1 Partial/Zero")]:
        sub = quiz_df[quiz_df["q1_full"] == flag]["grade"].dropna()
        sns.kdeplot(sub, ax=ax, color=color, linewidth=2, fill=True, alpha=0.25, label=lbl)
        ax.axvline(sub.mean(), color=color, linestyle="--", linewidth=1.5)

    stat, pval = stats.mannwhitneyu(
        quiz_df[quiz_df["q1_full"]]["grade"].dropna(),
        quiz_df[~quiz_df["q1_full"]]["grade"].dropna(),
        alternative="greater"
    )
    ax.set_title(f"{label}\nMann-Whitney p {'< 0.001' if pval < 0.001 else f'= {pval:.3f}'}",
                 fontweight="bold")
    ax.set_xlabel("Grade / 10")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

plt.suptitle("H6 – Full Mark on Q1 → Higher Overall Score?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("h6_q1_fullmark_effect.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 10 — HYPOTHESIS 7 (Task 2 — Student Generated)
# "Students who re-attempt a quiz significantly improve on their second attempt."
# ─────────────────────────────────────────────────────────────────────────────
# Viz: Paired dot/slope chart — each line = one student (attempt 1 → attempt 2)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (quiz_df, label) in zip(axes, quiz_data):
    multi = quiz_df[quiz_df.groupby("student_id")["attempt_no"].transform("max") >= 2]
    pivot = multi[multi["attempt_no"].isin([1, 2])].pivot_table(
        index="student_id", columns="attempt_no", values="grade"
    ).dropna()
    pivot.columns = ["Attempt 1", "Attempt 2"]
    pivot["improved"] = pivot["Attempt 2"] > pivot["Attempt 1"]

    for _, row in pivot.iterrows():
        color = PALETTE["success"] if row["improved"] else PALETTE["secondary"]
        ax.plot([1, 2], [row["Attempt 1"], row["Attempt 2"]],
                color=color, alpha=0.3, linewidth=0.8)

    # Mean lines
    ax.plot([1, 2], [pivot["Attempt 1"].mean(), pivot["Attempt 2"].mean()],
            color="black", linewidth=2.5, marker="o", markersize=8, label="Mean")

    stat, pval = stats.wilcoxon(pivot["Attempt 1"], pivot["Attempt 2"])
    pct_improved = pivot["improved"].mean() * 100
    ax.set_title(
        f"{label}\nWilcoxon p {'< 0.001' if pval < 0.001 else f'= {pval:.3f}'} "
        f"| {pct_improved:.0f}% improved",
        fontweight="bold"
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Attempt 1", "Attempt 2"])
    ax.set_ylabel("Grade / 10")
    ax.set_ylim(0, 11)

    # Custom legend
    ax.plot([], [], color=PALETTE["success"],  label="Improved")
    ax.plot([], [], color=PALETTE["secondary"], label="Declined")
    ax.legend(fontsize=9)

plt.suptitle("H7 – Do Students Improve on Their 2nd Attempt?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("h7_reattempt_improvement.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 11 — HYPOTHESIS 8 (Task 2 — Student Generated)
# "Students who attempt a quiz very early (relative to others) score lower."
# Rationale: Early birds may attempt before they are prepared.
# ─────────────────────────────────────────────────────────────────────────────
# Viz: scatter of attempt rank (by start time) vs grade, with hex-bin density

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (quiz_df, label) in zip(axes, quiz_data):
    d = quiz_df.copy()
    # Rank students by their FIRST attempt start time (use attempt_no == 1)
    first = d[d["attempt_no"] == 1].copy()
    first = first.reset_index(drop=True)
    first["relative_rank"] = first["grade"].rank(method="first")  # proxy using grade rank

    hb = ax.hexbin(first.index / len(first) * 100,
                   first["grade"],
                   gridsize=20, cmap="Blues", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")

    rho, pval = spearmanr(first.index, first["grade"])
    ax.set_title(
        f"{label}\nSpearman ρ = {rho:.3f}  (p {'< 0.001' if pval < 0.001 else f'= {pval:.3f}'})",
        fontweight="bold"
    )
    ax.set_xlabel("Student Percentile by Attempt Order (0=earliest)")
    ax.set_ylabel("Grade / 10")

plt.suptitle("H8 – Do Earlier Attempters Score Lower?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("h8_early_attempt_grade.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 12 — HYPOTHESIS 9 (Task 2 — Student Generated)
# "The number of attempts a student makes correlates with their final (best) score."
# ─────────────────────────────────────────────────────────────────────────────
# Viz: bar chart of mean best-score by total number of attempts

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (quiz_df, label) in zip(axes, quiz_data):
    summary = quiz_df.groupby("student_id").agg(
        total_attempts=("attempt_no", "max"),
        best_score=("grade", "max"),
        mean_score=("grade", "mean")
    ).reset_index()

    agg = summary.groupby("total_attempts")["best_score"].agg(["mean", "sem", "count"]).reset_index()
    agg = agg[agg["count"] >= 5]  # only reliable groups

    bars = ax.bar(agg["total_attempts"].astype(str), agg["mean"],
                  yerr=agg["sem"] * 1.96, capsize=5,
                  color=PALETTE["primary"], edgecolor="white", alpha=0.85)
    for bar, cnt in zip(bars, agg["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"n={cnt}", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"{label} – Best Score by # Attempts", fontweight="bold")
    ax.set_xlabel("Total Attempts")
    ax.set_ylabel("Mean Best Score / 10")
    ax.set_ylim(0, 11)

plt.suptitle("H9 – More Attempts → Better Final Score?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("h9_attempts_vs_best_score.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 13 — HYPOTHESIS 10 (Task 2 — Student Generated)
# "Score variance across questions is higher for low performers,
#  indicating inconsistent knowledge rather than uniform weakness."
# ─────────────────────────────────────────────────────────────────────────────
# Viz: violin/strip chart of per-student question-score standard deviation,
#      split by overall performance tier

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (quiz_df, label) in zip(axes, quiz_data):
    q_cols = [c for c in quiz_df.columns if c.startswith("Q.")]
    if not q_cols:
        continue
    # Normalise each Q score to 0–1
    d = quiz_df.copy()
    for c in q_cols:
        try:
            mx = float(c.split("/")[1])
        except Exception:
            mx = d[c].max()
        d[c] = d[c] / mx

    d["q_std"] = d[q_cols].std(axis=1)
    med_grade = d["grade"].median()
    d["perf_tier"] = d["grade"].apply(
        lambda g: "High" if g >= med_grade else "Low"
    )

    sns.violinplot(data=d, x="perf_tier", y="q_std", order=["Low", "High"],
                   palette=[PALETTE["secondary"], PALETTE["success"]],
                   inner="box", ax=ax, cut=0)
    stat, pval = stats.mannwhitneyu(
        d[d["perf_tier"] == "Low"]["q_std"].dropna(),
        d[d["perf_tier"] == "High"]["q_std"].dropna(),
        alternative="greater"
    )
    ax.set_title(
        f"{label}\nMann-Whitney p {'< 0.001' if pval < 0.001 else f'= {pval:.3f}'}",
        fontweight="bold"
    )
    ax.set_xlabel("Performance Tier")
    ax.set_ylabel("Std Dev of Normalised Q Scores")

plt.suptitle("H10 – Low Performers Have More Inconsistent Question Scores?",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("h10_score_variance_by_tier.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 14 — SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    "ID":  ["H1","H2","H3","H4","H5","H6","H7","H8","H9","H10"],
    "Hypothesis": [
        "Longer time → higher score",
        "Some questions consistently harder",
        "High performers improve; low performers erratic",
        "Hard Qs take longer; high performers faster",
        "Optimal time window for higher scores",
        "Full mark on Q1 → higher overall grade",
        "Students improve on 2nd attempt",
        "Early attempters score lower",
        "More attempts → better final score",
        "Low performers have higher Q-score variance",
    ],
    "Visualization": [
        "Scatter + LOWESS (Spearman ρ)",
        "Bar chart (full-mark rate per Q)",
        "Line plot (grade by attempt & tier)",
        "Heatmap + Violin (time by tier)",
        "Box plot (grade by time quintile)",
        "KDE comparison (Mann-Whitney U)",
        "Slope/paired chart (Wilcoxon)",
        "Hexbin density (Spearman ρ)",
        "Bar chart with 95% CI",
        "Violin (Mann-Whitney U)",
    ],
    "Task": ["Task 1"]*5 + ["Task 2"]*5,
})

print(summary.to_string(index=False))
