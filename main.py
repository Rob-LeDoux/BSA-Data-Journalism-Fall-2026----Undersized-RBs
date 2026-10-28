import pandas as pd
import polars as pl
import nflreadpy as nfl

SEASONS = list(range(2000, 2026))

ROSTER_COLS = ["season", "gsis_id", "full_name", "height", "weight", "birth_date", "entry_year", "draft_number"]


def load_touches(seasons):
    # Rushes plus targeted passes (completions and incompletions). Filter in Polars
    # before converting so the full ~5 GB pbp never lands in pandas.
    is_rush = pl.col("rush_attempt") == 1
    is_target = (pl.col("pass_attempt") == 1) & pl.col("receiver_player_id").is_not_null()
    return (
        nfl.load_pbp(seasons)
        .filter(is_rush | is_target)
        .with_columns(
            touch_type=pl.when(is_rush).then(pl.lit("rush")).otherwise(pl.lit("target")),
            player_id=pl.when(is_rush).then(pl.col("rusher_player_id")).otherwise(pl.col("receiver_player_id")),
        )
        .to_pandas()
    )


def load_rb_roster(seasons):
    rosters = nfl.load_rosters(seasons).to_pandas()
    rbs = rosters[rosters["position"] == "RB"]
    # Players who changed teams mid-season appear once per team
    return rbs[ROSTER_COLS].drop_duplicates(["season", "gsis_id"])


def load_rb_touches(seasons):
    touches = load_touches(seasons)
    rbs = load_rb_roster(seasons)
    return touches.merge(
        rbs,
        left_on=["season", "player_id"],
        right_on=["season", "gsis_id"],
        how="inner",
    )


def main():
    rb = load_rb_touches(SEASONS)
    print(f"Loaded {len(rb):,} RB plays for seasons {SEASONS[0]}-{SEASONS[-1]}")
    print(rb["touch_type"].value_counts())
    print(f"Receptions: {int(rb['complete_pass'].sum()):,}")
    print(rb[["season", "posteam", "full_name", "touch_type", "height", "weight", "yards_gained"]].head())


if __name__ == "__main__":
    main()
