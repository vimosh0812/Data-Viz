"""Run one or all hypothesis analyses: ``python -m src h1`` or ``python -m src all``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python src/__main__.py`` from repo root
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import REPO_ROOT, load_all_quizzes
from src.hypotheses_task1 import TASK1_RUNNERS
from src.hypotheses_task2 import TASK2_RUNNERS, print_summary_table
from src.viz_style import configure_matplotlib

ALL_KEYS = [f"h{i}" for i in range(1, 11)]


def _parse_args():
    p = argparse.ArgumentParser(description="Run CS3751 quiz hypothesis plots (H1–H10).")
    p.add_argument(
        "hypothesis",
        nargs="?",
        default="all",
        help=f"Which to run: eda, {', '.join(ALL_KEYS)}, task1, task2, summary, or all",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Directory for PNG figures (default: ./outputs)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive figure windows (save only).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    key = args.hypothesis.strip().lower()
    out_dir = Path(args.output_dir).resolve()
    show = not args.no_show

    if key == "summary":
        print_summary_table()
        return

    if key == "eda":
        if not show:
            import matplotlib

            matplotlib.use("Agg")
        from src.eda import run_all_eda_plots

        eda_dir = out_dir / "eda"
        run_all_eda_plots(eda_dir, show=show)
        print(f"\nEDA tables and figures saved under {eda_dir}/ (see README_EDA_FILES.txt)")
        return

    # Non-interactive backends when not displaying (e.g. CI)
    if not show:
        import matplotlib

        matplotlib.use("Agg")

    configure_matplotlib()

    runners: dict[str, callable] = {**TASK1_RUNNERS, **TASK2_RUNNERS}
    allowed = {"all", "task1", "task2"} | set(runners.keys())
    if key not in allowed:
        print(
            f"Unknown choice {key!r}. Use: {', '.join(ALL_KEYS)}, task1, task2, summary, all",
            file=sys.stderr,
        )
        sys.exit(1)

    q1, q2, q3, _all = load_all_quizzes()
    print(f"Loaded rows — Quiz 1: {len(q1)} | Quiz 2: {len(q2)} | Quiz 3: {len(q3)}")

    def run_one(k: str):
        fn = runners[k]
        print(f"\n>>> Running {k.upper()} …")
        fn(q1, q2, q3, out_dir, show=show)
        print(f"    Saved under {out_dir}/")

    if key == "all":
        for k in ALL_KEYS:
            run_one(k)
        print_summary_table()
        return

    if key == "task1":
        for k in ("h1", "h2", "h3", "h4", "h5"):
            run_one(k)
        return

    if key == "task2":
        for k in ("h6", "h7", "h8", "h9", "h10"):
            run_one(k)
        return

    run_one(key)


if __name__ == "__main__":
    # When launched as script, fix path
    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
