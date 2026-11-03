import pandas as pd
import polars as pl
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


if __name__ == "__main__":
    main()
