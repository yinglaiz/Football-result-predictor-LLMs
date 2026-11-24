"""Consistent chart styling for notebooks and the dashboard."""

import matplotlib as mpl

# Fixed colour per outcome, used in every chart so H/D/A always look the same.
OUTCOME_COLORS = {"H": "#2a78d6", "D": "#eb6834", "A": "#1baf7a"}  # blue, orange, aqua
OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}
SINGLE = "#2a78d6"  # charts with one series
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def use_style() -> None:
    """Clean defaults: thin lines, light grid, no top/right borders."""
    mpl.rcParams.update(
        {
            "figure.figsize": (9, 4),
            "figure.dpi": 110,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_MUTED,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": TEXT,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
