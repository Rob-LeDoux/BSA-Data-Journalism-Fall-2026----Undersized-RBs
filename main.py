import pandas as pd
import polars as pl
import nflreadpy as nfl
from nflreadpy.config import update_config

# The default "memory" cache keeps every downloaded season in RAM (~0.2 GB each);
# cache to disk instead, which also makes reruns skip the download
update_config(cache_mode="filesystem")

SEASONS = list(range(2000, 2026))

ROSTER_COLS = ["season", "gsis_id", "full_name", "height", "weight", "birth_date", "entry_year", "draft_number"]


def load_rb_roster(seasons):
    rosters = nfl.load_rosters(seasons)
    # Players who changed teams mid-season appear once per team
    return (
        rosters.filter(pl.col("position") == "RB")
        .select(ROSTER_COLS)
        .unique(["season", "gsis_id"], keep="first", maintain_order=True)
    )


def load_season_rb_touches(season, rbs):
    # Rushes plus targeted passes (completions and incompletions) by RBs
    is_rush = pl.col("rush_attempt") == 1
    is_target = (pl.col("pass_attempt") == 1) & pl.col("receiver_player_id").is_not_null()
    return (
        nfl.load_pbp(season)
        .filter(is_rush | is_target)
        .with_columns(
            touch_type=pl.when(is_rush).then(pl.lit("rush")).otherwise(pl.lit("target")),
            player_id=pl.when(is_rush).then(pl.col("rusher_player_id")).otherwise(pl.col("receiver_player_id")),
        )
        .join(rbs, left_on=["season", "player_id"], right_on=["season", "gsis_id"], how="inner")
    )


def load_rb_touches(seasons):
    # One season at a time so the full ~1.2M-play pbp is never in memory at once
    rbs = load_rb_roster(seasons)
    frames = []
    for season in seasons:
        print(f"Loading {season}...")
        frames.append(load_season_rb_touches(season, rbs))
    # Column types can differ slightly between seasons. Arrow-backed columns keep
    # pandas from turning every text value into a Python object (several GB more).
    return pl.concat(frames, how="diagonal_relaxed").to_pandas(use_pyarrow_extension_array=True)


def main():
    rb = load_rb_touches(SEASONS)
    print(f"Loaded {len(rb):,} RB plays for seasons {SEASONS[0]}-{SEASONS[-1]}")
    print(rb["touch_type"].value_counts())
    print(f"Receptions: {int(rb['complete_pass'].sum()):,}")
    print(rb[["season", "posteam", "full_name", "touch_type", "height", "weight", "yards_gained"]].head())


if __name__ == "__main__":
    main()
