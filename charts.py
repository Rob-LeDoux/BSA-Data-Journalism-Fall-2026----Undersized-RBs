from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import main

FIGURES_DIR = Path(__file__).parent / "figures"

# Colors and ink from the chart palette (series 1 = blue, series 2 = orange)
GROUP_COLORS = {"undersized": "#2a78d6", "other": "#eb6834"}
GROUP_LABELS = {"undersized": "Undersized (5'9\" or shorter, under 205 lbs)", "other": "Other RBs"}
TEXT = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def plot_success_rate(rb):
    # Share of plays that count as a "success" for the down and distance
    rates = rb.groupby(["touch_type", "size_group"])["success"].mean().unstack()
    touch_types = ["rush", "target"]
    groups = ["undersized", "other"]

    x = np.arange(len(touch_types))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, group in enumerate(groups):
        values = rates.loc[touch_types, group].to_numpy(dtype=float) * 100
        # Offset each group's bars, with a small gap between the pair
        bars = ax.bar(x + (i - 0.5) * (width + 0.02), values, width, color=GROUP_COLORS[group], label=GROUP_LABELS[group])
        ax.bar_label(bars, fmt="%.1f%%", padding=4, color=TEXT, fontsize=10)

    ax.set_xticks(x, ["Rushes", "Targets"])
    ax.set_ylabel("Success rate", color=MUTED)
    ax.set_ylim(0, 60)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title("RB success rate by size, 2000-2025 (garbage time removed)", color=TEXT, loc="left", fontsize=13)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT)
    fig.tight_layout()
    return fig


def main_chart():
    rbs = main.load_rb_roster(main.SEASONS)
    rb = main.remove_garbage_time(main.load_rb_touches(main.SEASONS, rbs))
    fig = plot_success_rate(rb)
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "success_rate_by_size.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.show()


if __name__ == "__main__":
    main_chart()
