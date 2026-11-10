from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import nflreadpy as nfl

import main

FIGURES_DIR = Path(__file__).parent / "figures"

# Pro Football Reference only charts contact from 2018 on
CONTACT_SEASONS = list(range(2018, 2026))
MIN_CARRIES = 50  # per season, so per-carry rates aren't driven by a handful of runs

GROUPS = ["undersized", "other"]
GROUP_LABELS = {"undersized": "Undersized", "other": "Other RBs"}
GROUP_COLORS = {"undersized": "#eb6834", "other": "#2a78d6"}
BG = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"


def load_contact_seasons():
    adv = nfl.load_pfr_advstats(seasons=CONTACT_SEASONS, stat_type="rush", summary_level="season")
    # Traded players have one row per team plus a combined "2TM"/"3TM" row; keep the combined one
    is_shared = pl.struct(["season", "pfr_id"]).is_duplicated()
    adv = adv.filter(~is_shared | pl.col("tm").str.ends_with("TM"))

    rbs = main.load_rb_roster(CONTACT_SEASONS).select(["season", "pfr_id", "size_group"]).unique(["season", "pfr_id"])
    return (
        adv.join(rbs, on=["season", "pfr_id"], how="inner")
        .filter((pl.col("att") >= MIN_CARRIES) & pl.col("ybc").is_not_null() & pl.col("yac").is_not_null())
        # Recompute per carry from totals; PFR's own columns are rounded to 0.1
        .with_columns(ybc_per_carry=pl.col("ybc") / pl.col("att"), yac_per_carry=pl.col("yac") / pl.col("att"))
        .to_pandas()
    )


def style_axis(ax):
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=TEXT_2)


def plot_box(ax, seasons, column, title):
    data = [seasons.loc[seasons["size_group"] == g, column].to_numpy() for g in GROUPS]
    box = ax.boxplot(
        data,
        tick_labels=[f"{GROUP_LABELS[g]}\n(n={len(d)})" for g, d in zip(GROUPS, data)],
        widths=0.5,
        patch_artist=True,
        medianprops={"color": TEXT, "linewidth": 2},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": MUTED, "markeredgecolor": "none", "alpha": 0.6},
    )
    for patch, group in zip(box["boxes"], GROUPS):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_edgecolor(GROUP_COLORS[group])

    # Median labels go on the outer side of each box so they don't run into the other box
    for i, values in enumerate(data, start=1):
        median = np.median(values)
        outer_left = i == 1
        ax.text(i - 0.3 if outer_left else i + 0.3, median, f"median {median:.2f}",
                ha="right" if outer_left else "left", va="center", color=TEXT, fontsize=9)

    style_axis(ax)
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylabel("Yards per carry", color=TEXT_2)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11.5, fontweight="bold")


def plot_contact_boxplots(seasons):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, (before, after) = plt.subplots(1, 2, figsize=(11, 6), facecolor=BG)
    plot_box(before, seasons, "ybc_per_carry", "Yards before contact")
    plot_box(after, seasons, "yac_per_carry", "Yards after contact")

    fig.text(0.06, 0.96, "Yards before and after contact, undersized vs. other RBs", fontsize=15, fontweight="bold", color=TEXT, va="top")
    fig.text(
        0.06, 0.905,
        f"Each box summarizes player-seasons from {CONTACT_SEASONS[0]}-{CONTACT_SEASONS[-1]} with {MIN_CARRIES}+ carries; dots are outlier seasons.",
        fontsize=10.5, color=TEXT_2, va="top",
    )
    fig.text(
        0.06, 0.02,
        "Source: Pro Football Reference advanced rushing stats via nflverse; roster heights/weights via nflverse.\n"
        "Undersized = 5'9\" or shorter and under 205 lbs (median roster listing). Includes all regular-season carries (garbage time not removed).",
        fontsize=8.5, color=MUTED, va="bottom",
    )
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.08, right=0.97, wspace=0.3)
    return fig


def main_chart():
    seasons = load_contact_seasons()
    summary = seasons.groupby("size_group")[["ybc_per_carry", "yac_per_carry"]].describe().T
    print(summary.round(2).to_string())
    fig = plot_contact_boxplots(seasons)
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "contact_yards_boxplots.png"
    fig.savefig(path, dpi=150, facecolor=BG)
    print(f"Saved {path}")
    plt.show()


if __name__ == "__main__":
    main_chart()
