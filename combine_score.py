from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
import nflreadpy as nfl
from nflreadpy.config import update_config

import main

# Read only from the local nflreadpy cache; never re-download
update_config(cache_mode="filesystem", cache_duration=10**9)

FIGURES_DIR = Path(__file__).parent / "figures"

# Combine-based size definition (inclusive on both)
UNDERSIZED_MAX_HEIGHT = 69  # inches
UNDERSIZED_MAX_WEIGHT = 210  # lbs

FIT_FIRST_CLASS = 2000
FIT_LAST_CLASS = 2022  # later classes don't have 4 seasons yet
DRILLS = ["forty", "broad_jump", "cone"]
DRILL_NAMES = {"forty": "40", "broad_jump": "broad jump", "cone": "3-cone"}
QUINTILES = ["Bottom 20%", "20-40%", "40-60%", "60-80%", "Top 20%"]
ROUND_GROUPS = ["Rounds 1-2", "Rounds 3-4", "Rounds 5-7"]

BG = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
COLORS = {False: "#2a78d6", True: "#eb6834"}
LABELS = {False: "Everyone else", True: "Undersized (5'9\" or shorter and 210 lbs or lighter)"}


def load_rb_combine():
    combine = nfl.load_combine().to_pandas()
    combine = combine[combine["pos"] == "RB"].copy()
    feet_inches = combine["ht"].str.split("-", expand=True).astype(float)
    combine["height"] = feet_inches[0] * 12 + feet_inches[1]
    combine["undersized"] = (combine["height"] <= UNDERSIZED_MAX_HEIGHT) & (combine["wt"] <= UNDERSIZED_MAX_WEIGHT)
    combine["drafted"] = combine["draft_ovr"].notna().astype(int)
    return combine


def load_season_epa():
    # Regular-season RB EPA per player-season, all plays (like nflverse season stats)
    rbs = main.load_rb_roster(main.SEASONS)
    plays = main.load_rb_touches(main.SEASONS, rbs)
    plays = plays[(plays["season_type"] == "REG") & (plays["two_point_attempt"] != 1)]
    plays = plays[["season", "pfr_id", "touch_type", "epa"]].astype({"epa": "float64"})
    plays["is_rush"] = plays["touch_type"] == "rush"
    plays["rush_epa"] = plays["epa"].where(plays["is_rush"], 0.0)
    plays["rec_epa"] = plays["epa"].where(~plays["is_rush"], 0.0)

    seasons = plays.groupby(["pfr_id", "season"]).agg(
        carries=("is_rush", "sum"),
        targets=("is_rush", lambda s: (~s).sum()),
        rush_epa=("rush_epa", "sum"),
        rec_epa=("rec_epa", "sum"),
    ).reset_index()

    league = plays.groupby("season").agg(rush_epa=("rush_epa", "sum"), rec_epa=("rec_epa", "sum"), carries=("is_rush", "sum"))
    league["targets"] = plays.groupby("season")["is_rush"].apply(lambda s: (~s).sum())
    avg_rush = (league["rush_epa"] / league["carries"]).rename("avg_rush_epa")
    avg_rec = (league["rec_epa"] / league["targets"]).rename("avg_rec_epa")

    seasons = seasons.join(avg_rush, on="season").join(avg_rec, on="season")
    seasons["epa_above_avg"] = (
        seasons["rush_epa"] - seasons["carries"] * seasons["avg_rush_epa"]
        + seasons["rec_epa"] - seasons["targets"] * seasons["avg_rec_epa"]
    )
    return seasons[["pfr_id", "season", "epa_above_avg"]]


def add_careers(combine, season_epa):
    # First 4 NFL seasons = draft class year through class year + 3
    joined = combine[["pfr_id", "season"]].dropna(subset=["pfr_id"]).merge(
        season_epa, on="pfr_id", suffixes=("_class", "")
    )
    in_window = joined[(joined["season"] >= joined["season_class"]) & (joined["season"] <= joined["season_class"] + 3)]
    careers = in_window.groupby("pfr_id")["epa_above_avg"].sum().rename("epa_above_avg_yr1to4").reset_index()

    combine = combine.merge(careers, on="pfr_id", how="left")
    in_fit_classes = combine["season"].between(FIT_FIRST_CLASS, FIT_LAST_CLASS)
    # Never played (or never linked) = no value added
    combine.loc[in_fit_classes, "epa_above_avg_yr1to4"] = combine.loc[in_fit_classes, "epa_above_avg_yr1to4"].fillna(0)
    return combine


def fit_drill_weights(combine):
    for drill in DRILLS:
        z = (combine[drill] - combine[drill].mean()) / combine[drill].std()
        # Faster 40 and 3-cone times are better, so flip them
        combine[f"z_{drill}"] = -z if drill in ("forty", "cone") else z

    fit_rows = combine[
        combine["season"].between(FIT_FIRST_CLASS, FIT_LAST_CLASS) & combine[DRILLS].notna().all(axis=1)
    ]
    model = smf.ols("epa_above_avg_yr1to4 ~ z_forty + z_broad_jump + z_cone + wt", data=fit_rows).fit()
    print(model.summary())

    coefs = model.params[[f"z_{d}" for d in DRILLS]]
    if (coefs < 0).any():
        raise SystemExit(f"Negative drill coefficient(s), stopping:\n{coefs[coefs < 0]}")
    weights = (coefs / coefs.sum()).rename(lambda name: name.removeprefix("z_"))
    print(f"\nFit on {len(fit_rows)} backs. Weights: " + ", ".join(f"{DRILL_NAMES[d]} = {w:.2f}" for d, w in weights.items()))
    return weights


def score_backs(combine, weights):
    can_score = combine["forty"].notna() & (combine["broad_jump"].notna() | combine["cone"].notna())
    z = combine[[f"z_{d}" for d in DRILLS]].to_numpy()
    w = weights[DRILLS].to_numpy()
    has = ~np.isnan(z)
    raw = np.nansum(z * w, axis=1) / (has * w).sum(axis=1)

    scored = combine[can_score].copy()
    scored["raw_score"] = raw[can_score.to_numpy()]
    scored["score"] = scored["raw_score"].rank(pct=True) * 100
    scored["quintile"] = pd.qcut(scored["score"], 5, labels=QUINTILES)
    scored["round_group"] = pd.cut(scored["draft_round"], [0, 2, 4, 7], labels=ROUND_GROUPS)
    return scored


def style_axis(ax):
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0, colors=TEXT_2)


def plot_share_drafted(ax, scored):
    x = np.arange(len(QUINTILES))
    width = 0.38
    rows = []
    for offset, undersized in ((-width / 2, False), (width / 2, True)):
        group = scored[scored["undersized"] == undersized].groupby("quintile", observed=False)["drafted"]
        counts, totals = group.sum(), group.size()
        share = counts / totals * 100
        low, high = proportion_confint(counts, totals, method="wilson")
        ax.bar(x + offset, share, width, color=COLORS[undersized], edgecolor=BG, linewidth=2, label=LABELS[undersized])
        ax.vlines(x + offset, low * 100, high * 100, color=TEXT, linewidth=1)
        for xi, s, hi, n in zip(x + offset, share, high * 100, totals):
            ax.text(xi, hi + 1.5, f"{s:.0f}%", ha="center", va="bottom", color=TEXT, fontsize=9)
            ax.text(xi, -3, f"n={n}", ha="center", va="center", color=MUTED, fontsize=8)
        rows.append(share.round(0).rename(LABELS[undersized].split(" (")[0]))
    print("\nDrafted % by quintile:\n", pd.concat(rows, axis=1).to_string())

    ax.set_xticks(x, QUINTILES)
    ax.set_ylim(-6, 105)
    ax.set_ylabel("Drafted (%)", color=TEXT_2)
    ax.set_xlabel("RB combine score (percentile among RB combine invitees)", color=TEXT_2)
    ax.set_title("Share drafted, by combine score", loc="left", color=TEXT, fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=9)


def plot_drafted_scores(ax, scored):
    drafted = scored[scored["drafted"] == 1]
    x = np.arange(len(ROUND_GROUPS))
    rows = []
    for offset, undersized in ((-0.15, False), (0.15, True)):
        group = drafted[drafted["undersized"] == undersized].groupby("round_group", observed=False)["score"]
        means, totals = group.mean(), group.size()
        ax.plot(x + offset, means, "o", markersize=10, color=COLORS[undersized], markeredgecolor=BG, markeredgewidth=2, linestyle="none")
        for xi, m, n in zip(x + offset, means, totals):
            ax.text(xi, m - 3.5, f"{m:.0f}", ha="center", va="top", color=TEXT, fontsize=9)
            ax.text(xi, 32.5, f"n={n}", ha="center", va="center", color=MUTED, fontsize=8)
        rows.append(means.round(0).rename(LABELS[undersized].split(" (")[0]))
    print("\nAverage score of drafted backs:\n", pd.concat(rows, axis=1).to_string())

    ax.set_xticks(x, ROUND_GROUPS)
    ax.set_xlim(-0.5, len(ROUND_GROUPS) - 0.5)
    ax.set_ylim(30, 95)
    ax.set_ylabel("Average combine score of drafted backs\n(percentile; axis starts at 30)", color=TEXT_2)
    # Two lines: the full title is wider than this narrower panel
    ax.set_title("Drafted undersized backs tested a bit higher\nin each round group", loc="left", color=TEXT, fontsize=11.5, fontweight="bold")


def plot_figure(scored, weights):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.5, 1]}, facecolor=BG)
    for ax in (left, right):
        style_axis(ax)
    plot_share_drafted(left, scored)
    plot_drafted_scores(right, scored)

    pct = {d: f"{weights[d] * 100:.0f}%" for d in DRILLS}
    fig.text(0.07, 0.955, "Top-testing undersized backs may get less draft credit", fontsize=15, fontweight="bold", color=TEXT, va="top")
    fig.text(
        0.07, 0.895,
        "Among the top 40% of combine scores, undersized backs were drafted less often than equally athletic backs, but the 95% intervals overlap;\n"
        "lower down there is no consistent gap. Drafted undersized backs tested a bit higher in each round group. "
        f"Weights: 40 {pct['forty']}, broad jump {pct['broad_jump']}, 3-cone {pct['cone']}.",
        fontsize=10.5, color=TEXT_2, va="top",
    )
    fig.text(
        0.07, 0.03,
        "Source: nflverse combine and play-by-play data. Undersized = 5'9\" or shorter (inclusive) and 210 lbs or lighter at the combine.\n"
        "Weights = each drill's link to first-4-year EPA above average, controlling for weight. Score needs the 40 plus the broad jump or 3-cone. Error bars: 95% intervals.",
        fontsize=8.5, color=MUTED, va="bottom",
    )
    fig.subplots_adjust(top=0.76, bottom=0.2, left=0.07, right=0.98, wspace=0.28)
    return fig


def main_chart():
    combine = add_careers(load_rb_combine(), load_season_epa())
    weights = fit_drill_weights(combine)
    scored = score_backs(combine, weights)
    print(f"\nScored {len(scored)} backs ({int(scored['undersized'].sum())} undersized)")
    fig = plot_figure(scored, weights)
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "fig6_combine_score_vs_draft.png"
    fig.savefig(path, dpi=200, facecolor=BG)
    print(f"Saved {path}")


if __name__ == "__main__":
    main_chart()
