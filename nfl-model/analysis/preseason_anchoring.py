"""Test whether the market underreacts to in-season surprise relative to preseason
priors, and for how long.

Motivating claim (Patterson, Shank & Fodor 2025, Economics Letters, "Anchoring Bias in
the NFL Gambling Market"): sportsbooks keep incorporating preseason Super Bowl odds into
their week-by-week line-setting through roughly week 8, not just bettors -- a claim about
the market's own updating SPEED, independent of the schedule-structure effects this repo
already tested and ruled out (divisional/rest/travel regressed against market residual,
R^2 = 0.0008, GATES.md).

NOT wired into any gate or the live model. A standalone check on data already cached from
the model build (schedules, walk-forward ratings) -- no new ingestion, no simulator, no
API calls, no change to `model_spread` / `model_total`.

Method
------
For each graded REG-season game (week >= FIRST_GRADED_WEEK):

  - `preseason_power[team]` = (off_rating - def_rating) at that team's FIRST walk-forward
    snapshot of the season -- i.e. the ratings used to predict that season's Week 1 games,
    before any of that season's own games have been played. This is mostly the season-
    prior injection (`ratings.py::season_priors`, last season's final rating regressed
    toward zero) blended with the still-decaying tail of the previous season's real games
    (half-life 6 games) -- not identical to what a sportsbook uses, but it is the model's
    own principled, already-computed stand-in for "preseason expectation," on the same
    scale its own ratings are built from.
  - `current_power[team]` = (off_rating - def_rating) at the walk-forward snapshot for
    the week actually being predicted -- the model's live, in-season view of that team.
  - `surprise[team] = current_power - preseason_power` -- how far this season's evidence
    has moved the team from where it started. `power = off - def` is a deliberately crude
    single-number team-strength proxy (equal-weighted, not the model's real
    `net_epa(off, opponent_def)` matchup combination) -- adequate for this exploratory
    check, not a production feature.
  - `market_error = result - spread_line` (home perspective; nflverse `spread_line` is
    positive when home is favored and aligns with `result`, per nfl-model/CLAUDE.md).

Hypothesis: if the market is still anchored on preseason expectation, `market_error`
should correlate POSITIVELY with `surprise_diff = surprise[home] - surprise[away]` in the
weeks the paper names (2-8) -- the model already "knows" what the market has not priced
in -- and that relationship should fade toward zero once the market has had time to
update. A significant early window and a null late window is the claimed signature; a
null everywhere, or a significant window that does NOT fade, are both evidence against it
as stated.

This is a market-error regression, not a model backtest -- if real and durable it would
be a genuine, previously-untested lead, but it does not become an edge claim until it
clears the same bar (power-checked significance, held-out replication) every other claim
in this repo has to clear.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from robust_inference import fit_ols  # noqa: E402

import os

CACHE_DIR = Path(os.environ.get("NFL_MODEL_CACHE_DIR") or (Path.home() / ".cache" / "nfl-model"))
RATINGS_FILE = "walkforward_ratings_default_o750_d550_p10_2016_2026.parquet"
FIRST_GRADED_WEEK = 2  # week 1 has no "surprise" yet: no current-season evidence exists


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    wf = pd.read_parquet(CACHE_DIR / RATINGS_FILE)
    sched = pd.read_parquet(CACHE_DIR / "schedules_2016_2026.parquet")
    return wf, sched


def team_power(wf: pd.DataFrame) -> pd.DataFrame:
    p = wf.copy()
    p["power"] = p["off_rating"] - p["def_rating"]
    return p[["season", "week", "team", "power"]]


def build_frame(wf: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    power = team_power(wf)

    preseason = (
        power.sort_values(["season", "team", "week"])
        .groupby(["season", "team"], as_index=False)
        .first()[["season", "team", "power"]]
        .rename(columns={"power": "preseason_power"})
    )

    games = sched[
        sched["game_type"].eq("REG")
        & sched["result"].notna()
        & sched["spread_line"].notna()
        & (sched["week"] >= FIRST_GRADED_WEEK)
    ][["game_id", "season", "week", "home_team", "away_team", "result", "spread_line"]].copy()

    cur_h = power.rename(columns={"team": "home_team", "power": "current_power_home"})
    cur_a = power.rename(columns={"team": "away_team", "power": "current_power_away"})
    games = games.merge(cur_h, on=["season", "week", "home_team"], how="inner")
    games = games.merge(cur_a, on=["season", "week", "away_team"], how="inner")

    pre_h = preseason.rename(columns={"team": "home_team", "preseason_power": "preseason_power_home"})
    pre_a = preseason.rename(columns={"team": "away_team", "preseason_power": "preseason_power_away"})
    games = games.merge(pre_h, on=["season", "home_team"], how="inner")
    games = games.merge(pre_a, on=["season", "away_team"], how="inner")

    games["surprise_home"] = games["current_power_home"] - games["preseason_power_home"]
    games["surprise_away"] = games["current_power_away"] - games["preseason_power_away"]
    games["surprise_diff"] = games["surprise_home"] - games["surprise_away"]
    games["market_error"] = games["result"] - games["spread_line"]
    return games


def report(games: pd.DataFrame, label: str) -> "float | None":
    if len(games) < 50:
        print(f"{label:<28} only {len(games)} games -- too few to fit")
        return None
    fit = fit_ols(
        games["market_error"], games["surprise_diff"],
        covariance="cluster", groups=games["season"].to_numpy(),
    )
    b, se_b, t_b = float(fit.beta[1]), float(fit.se[1]), float(fit.t_values[1])
    sig = "  <-- |t|>=1.96" if abs(t_b) >= 1.96 else ""
    print(
        f"{label:<28} n={fit.n:>5}  b={b:+.4f}  se={se_b:.4f}  t={t_b:+.2f}  "
        f"r2={fit.r_squared:.4f}{sig}"
    )
    return t_b


def main() -> int:
    wf, sched = load()
    games = build_frame(wf, sched)
    print(f"total gradeable games (week >= {FIRST_GRADED_WEEK}): {len(games):,}")
    print(f"seasons: {sorted(games['season'].unique())}")
    print()

    t_early = report(games[games["week"].between(2, 8)], "weeks 2-8 (claimed window)")
    t_late = report(games[games["week"] >= 9], "weeks 9+ (should fade if real)")
    report(games, "all graded weeks (2+)")
    print()
    for wk in range(2, 9):
        report(games[games["week"] == wk], f"  week {wk} only")

    print()
    if t_early is not None and t_late is not None:
        diff = t_early - t_late  # informal; not a proper difference-of-b test
        print(f"t(weeks 2-8) - t(weeks 9+) = {diff:+.2f}  (informal comparison, not a power-checked difference test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
