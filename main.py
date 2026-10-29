from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import nflreadpy as nfl
from nflreadpy.config import update_config

# The default "memory" cache keeps every downloaded season in RAM (~0.2 GB each);
# cache to disk instead, which also makes reruns skip the download
update_config(cache_mode="filesystem")

SEASONS = list(range(2000, 2026))

# Garbage time: the offense's pre-snap win probability is outside 10%-90%, the
# standard nflfastR / rbsdm.com cutoff. Uses `wp` rather than `vegas_wp` so that
# pregame spreads don't mark early plays by big underdogs as "decided".
WP_MIN = 0.10
WP_MAX = 0.90

# Undersized: 5'9" or shorter AND under 205 lbs
UNDERSIZED_MAX_HEIGHT = 69  # inches, inclusive
UNDERSIZED_MAX_WEIGHT = 205  # lbs, exclusive

# Participation data (defenders_in_box) only exists from 2016 on
FIRST_PARTICIPATION_SEASON = 2016

FIGURES_DIR = Path(__file__).parent / "figures"

# Chart colors (series 1 = blue, series 2 = orange) and ink
GROUP_COLORS = {"undersized": "#2a78d6", "other": "#eb6834"}
TEXT = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

ROSTER_COLS = ["season", "gsis_id", "full_name", "height", "weight", "birth_date", "entry_year", "draft_number"]


def load_rb_roster(seasons):
    rosters = nfl.load_rosters(seasons)
    # Players who changed teams mid-season appear once per team
    rbs = (
        rosters.filter(pl.col("position") == "RB")
        .select(ROSTER_COLS)
        .unique(["season", "gsis_id"], keep="first", maintain_order=True)
        # Birth date is missing in some seasons; fill from the player's other seasons
        .with_columns(birth_date=pl.col("birth_date").max().over("gsis_id"))
    )
    # One height/weight per player (median across his seasons) so listings that
    # shift year to year can't move him between groups; also fills missing years
    sizes = rbs.group_by("gsis_id").agg(
        player_height=pl.col("height").median(),
        player_weight=pl.col("weight").median(),
    )
    is_undersized = (pl.col("player_height") <= UNDERSIZED_MAX_HEIGHT) & (pl.col("player_weight") < UNDERSIZED_MAX_WEIGHT)
    return rbs.join(sizes, on="gsis_id", how="left").with_columns(
        size_group=pl.when(is_undersized).then(pl.lit("undersized")).otherwise(pl.lit("other"))
    )


def load_season_rb_touches(season, rbs):
    # Rushes plus targeted passes (completions and incompletions) by RBs
    is_rush = pl.col("rush_attempt") == 1
    is_target = (pl.col("pass_attempt") == 1) & pl.col("receiver_player_id").is_not_null()
    touches = (
        nfl.load_pbp(season)
        .filter(is_rush | is_target)
        .with_columns(
            touch_type=pl.when(is_rush).then(pl.lit("rush")).otherwise(pl.lit("target")),
            player_id=pl.when(is_rush).then(pl.col("rusher_player_id")).otherwise(pl.col("receiver_player_id")),
            garbage_time=(pl.col("wp") < WP_MIN) | (pl.col("wp") > WP_MAX),
        )
        .join(rbs, left_on=["season", "player_id"], right_on=["season", "gsis_id"], how="inner")
        .with_columns(
            age=((pl.col("game_date").str.to_date() - pl.col("birth_date")).dt.total_days() / 365.25).round(2)
        )
    )
    if season >= FIRST_PARTICIPATION_SEASON:
        box = nfl.load_participation(season).select(
            game_id=pl.col("nflverse_game_id"),
            play_id=pl.col("play_id").cast(touches.schema["play_id"]),
            defenders_in_box=pl.col("defenders_in_box"),
        )
        touches = touches.join(box, on=["game_id", "play_id"], how="left")
    return touches


def load_rb_touches(seasons, rbs):
    # One season at a time so the full ~1.2M-play pbp is never in memory at once
    frames = []
    for season in seasons:
        print(f"Loading {season}...")
        frames.append(load_season_rb_touches(season, rbs))
    # Column types can differ slightly between seasons. Arrow-backed columns keep
    # pandas from turning every text value into a Python object (several GB more).
    return pl.concat(frames, how="diagonal_relaxed").to_pandas(use_pyarrow_extension_array=True)


def remove_garbage_time(df):
    return df[~df["garbage_time"]]


def plot_rush_yards_boxplot(rb):
    rushes = rb[rb["touch_type"] == "rush"]
    groups = ["undersized", "other"]
    # Two-point attempts have no rushing_yards, so drop missing values
    data = [rushes.loc[rushes["size_group"] == g, "rushing_yards"].dropna().to_numpy(dtype=float) for g in groups]

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # Whiskers reach 1.5x the box height; the thousands of long runs beyond them
    # are hidden because they would bury the boxes
    box = ax.boxplot(
        data,
        tick_labels=[f"Undersized\n(n={len(data[0]):,})", f"Other RBs\n(n={len(data[1]):,})"],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT, "linewidth": 2},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
    )
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_edgecolor(GROUP_COLORS[group])

    # Label each box's median, mean and quartiles
    for i, values in enumerate(data, start=1):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.text(i + 0.3, median, f"median {median:.0f}\nmean {values.mean():.2f}", va="center", color=TEXT, fontsize=9)
        ax.text(i + 0.3, q3, f"Q3 {q3:.0f}", va="center", color=MUTED, fontsize=9)
        ax.text(i + 0.3, q1, f"Q1 {q1:.0f}", va="center", color=MUTED, fontsize=9)

    ax.set_xlim(0.5, 2.9)
    ax.set_ylabel("Rushing yards per carry", color=MUTED)
    ax.set_title("Rushing yards per carry by RB size, 2000-2025\n(garbage time removed, outliers hidden)", color=TEXT, loc="left", fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout()
    return fig


def main():
    rbs = load_rb_roster(SEASONS)
    all_rb = load_rb_touches(SEASONS, rbs)
    rb = remove_garbage_time(all_rb)
    print(f"Loaded {len(all_rb):,} RB plays for seasons {SEASONS[0]}-{SEASONS[-1]}")
    print(f"Removed {len(all_rb) - len(rb):,} garbage-time plays, {len(rb):,} remain")
    print(rb["touch_type"].value_counts())
    print(f"Receptions: {int(rb['complete_pass'].sum()):,}")
    print(rb["size_group"].value_counts())
    print(rb[["season", "posteam", "full_name", "touch_type", "size_group", "age", "yards_gained"]].head())

    fig = plot_rush_yards_boxplot(rb)
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "rush_yards_boxplot.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.show()


if __name__ == "__main__":
    main()
