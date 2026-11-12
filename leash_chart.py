from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

import main

FIGURES_DIR = Path(__file__).parent / "figures"

# A "feature" season, counted in rushes + targets so receiving backs (often the
# undersized ones) aren't left out by a carries-only cutoff
MIN_TOUCHES = 100
MIN_TOUCHES_NEXT = 100  # kept a feature role the following season
THIRDS = ["Bottom third", "Middle third", "Top third"]

BG = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
COLORS = {False: "#2a78d6", True: "#eb6834"}
LABELS = {False: "Everyone else", True: "Undersized (5'9\" or shorter and under 205 lbs)"}


def load_player_seasons():
    # Cleaned project data: regular season, garbage time removed
    rbs = main.load_rb_roster(main.SEASONS)
    plays = main.remove_garbage_time(main.load_rb_touches(main.SEASONS, rbs))
    plays = plays[plays["season_type"] == "REG"]
    plays = plays[["season", "player_id", "size_group", "touch_type", "epa"]].astype({"epa": "float64"})
    plays["is_rush"] = plays["touch_type"] == "rush"

    seasons = plays.groupby(["player_id", "season"]).agg(
        size_group=("size_group", "first"),
        touches=("is_rush", "size"),
        epa_per_play=("epa", "mean"),
    ).reset_index()

    # Touches the following season, 0 if he didn't play
    next_touches = seasons[["player_id", "season", "touches"]].assign(season=lambda d: d["season"] - 1)
    seasons = seasons.merge(next_touches.rename(columns={"touches": "touches_next"}), on=["player_id", "season"], how="left")
    seasons["touches_next"] = seasons["touches_next"].fillna(0)

    # Feature seasons with a following season in the data
    feature = seasons[(seasons["touches"] >= MIN_TOUCHES) & (seasons["season"] < main.SEASONS[-1])].copy()
    feature["undersized"] = feature["size_group"] == "undersized"
    feature["kept"] = (feature["touches_next"] >= MIN_TOUCHES_NEXT).astype(int)
    # EPA per play relative to the other feature backs that year, split into thirds within each season
    feature["epa_above_avg"] = feature["epa_per_play"] - feature.groupby("season")["epa_per_play"].transform("mean")
    feature["third"] = feature.groupby("season")["epa_above_avg"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)
    ).map(dict(enumerate(THIRDS)))
    feature["third"] = pd.Categorical(feature["third"], categories=THIRDS, ordered=True)
    return feature


def gap_test(feature, third):
    group = feature[feature["third"] == third]
    counts = group.groupby("undersized")["kept"].sum()
    totals = group.groupby("undersized")["kept"].size()
    return proportions_ztest([counts[True], counts[False]], [totals[True], totals[False]])[1]


def style_axis(ax):
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0, colors=TEXT_2)


def plot_leash(feature):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=BG)
    style_axis(ax)

    x = np.arange(len(THIRDS))
    width = 0.36
    shares = {}
    for offset, undersized in ((-width / 2, False), (width / 2, True)):
        group = feature[feature["undersized"] == undersized].groupby("third", observed=False)["kept"]
        counts, totals = group.sum(), group.size()
        share = counts / totals * 100
        low, high = proportion_confint(counts, totals, method="wilson")
        ax.bar(x + offset, share, width, color=COLORS[undersized], edgecolor=BG, linewidth=2, label=LABELS[undersized])
        ax.vlines(x + offset, low * 100, high * 100, color=TEXT, linewidth=1.2)
        for xi, s, hi, n in zip(x + offset, share, high * 100, totals):
            ax.text(xi, hi + 1.5, f"{s:.0f}%", ha="center", va="bottom", color=TEXT, fontsize=10)
            ax.text(xi, -3, f"n={n}", ha="center", va="center", color=MUTED, fontsize=8.5)
        shares[undersized] = share
    print("Kept a feature role next season (%):\n", pd.DataFrame({"Everyone else": shares[False], "Undersized": shares[True]}).round(0).to_string())

    p_values = {third: gap_test(feature, third) for third in THIRDS}
    print("Two-proportion test p:", {t: round(p, 3) for t, p in p_values.items()})

    ax.set_xticks(x, THIRDS)
    ax.set_ylim(-6, 105)
    ax.set_ylabel(f"Kept a {MIN_TOUCHES_NEXT}+ touch role next season (%)", color=TEXT_2)
    ax.set_xlabel(f"EPA per play that season, relative to all {MIN_TOUCHES}+ touch backs that year", color=TEXT_2)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT_2)

    bottom = THIRDS[0]
    n_small_bottom = int(((feature["third"] == bottom) & feature["undersized"]).sum())
    fig.text(0.04, 0.96, "Undersized backs kept feature roles less often, most of all after a bad year", fontsize=15, fontweight="bold", color=TEXT, va="top")
    fig.text(
        0.04, 0.9,
        f"After a bottom-third season for EPA per play, undersized backs kept a feature role {shares[True][bottom]:.0f}% of the time vs. "
        f"{shares[False][bottom]:.0f}% for everyone else,\nbut only {n_small_bottom} undersized seasons fall in that group "
        f"(two-proportion test p = {p_values[bottom]:.2f}).\nAfter average or good seasons the gap is smaller and not statistically clear. Error bars: 95% intervals.",
        fontsize=10, color=TEXT_2, va="top",
    )
    fig.text(
        0.04, 0.02,
        f"Source: nflverse (nflreadpy), NFL regular seasons {main.SEASONS[0]}-{main.SEASONS[-1]}, garbage time removed. "
        "Undersized = 5'9\" or shorter and under 205 lbs\n(median roster listing). "
        f"Seasons {main.SEASONS[0]}-{main.SEASONS[-2]} with {MIN_TOUCHES}+ rushes and targets. "
        f"Kept = {MIN_TOUCHES_NEXT}+ rushes and targets the following season.\nEPA per play covers rushes and targets.",
        fontsize=8.5, color=MUTED, va="bottom",
    )
    fig.subplots_adjust(top=0.74, bottom=0.22, left=0.08, right=0.98)
    return fig


def main_chart():
    feature = load_player_seasons()
    print(f"{len(feature)} feature seasons ({int(feature['undersized'].sum())} undersized)")
    fig = plot_leash(feature)
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "shorter_leash.png"
    fig.savefig(path, dpi=200, facecolor=BG)
    print(f"Saved {path}")
    plt.show()


if __name__ == "__main__":
    main_chart()
