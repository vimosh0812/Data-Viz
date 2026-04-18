"""Shared matplotlib / seaborn styling."""

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#E84855",
    "accent": "#F4A261",
    "neutral": "#6B7280",
    "light": "#F3F4F6",
    "success": "#22C55E",
}


def configure_matplotlib():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})
