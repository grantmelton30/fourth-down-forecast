"""Baselines a prop model has to beat before it deserves any further work.

WHY BASELINES COME FIRST. `GATE_BEATS_ELO` has been failing in this repo for a year, and
that gate is the most useful thing in it: it says the drive simulator, the EPA ratings and
the context adjustments together are worth less than a 538-style Elo anyone could write in
an afternoon. A prop model needs the same check before it earns a line of further work, and
this one is free — it needs no market data at all.

The estimators here are deliberately dumb:

    last_n        what he averaged over his last N games
    ewma          the same, weighted toward recent games
    season_mean   his average so far this season
    usage_volume  usage share x predicted team volume x conversion rate

Only the last is "the model". If it cannot beat "what did he average recently", nothing
downstream matters and the project stops here for the cost of an afternoon.

EVERY ESTIMATOR IS WALK-FORWARD ON KICKOFF TIMESTAMPS. A baseline that peeks is not a
baseline, it is a trap: it would look unbeatable and kill a model that was actually fine.
`CLAUDE.md` §1 governs here exactly as it does in nfl-model.

WHAT THIS CANNOT ANSWER. Whether the model beats the PRICE. This repo has already proved
those are different questions -- Elo is more accurate than the NFL model and still returns
48.9 per 100. Accuracy is necessary, not sufficient, and a good result here is permission to
buy market history, not evidence of an edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prior(series: pd.Series, fn) -> pd.Series:
    """Apply `fn` to strictly earlier values only. shift(1) before anything else."""
    return fn(series.shift(1))


def add_baselines(
    frame: pd.DataFrame, *, stat: str, player_col: str = "player_id",
    last_n: int = 4, halflife: float = 3.0,
) -> pd.DataFrame:
    """Attach per-player walk-forward baselines for `stat`.

    `frame` must already be sorted by kickoff — ordering by week label is not the same
    thing and leaks a postponed game backwards in time.
    """
    out = frame.copy()
    g = out.groupby(player_col)[stat]

    out["base_last_n"] = g.transform(
        lambda s: _prior(s, lambda p: p.rolling(last_n, min_periods=1).mean()))
    out["base_ewma"] = g.transform(
        lambda s: _prior(s, lambda p: p.ewm(halflife=halflife, min_periods=1).mean()))
    out["base_season_mean"] = g.transform(
        lambda s: _prior(s, lambda p: p.expanding().mean()))
    out["prior_games"] = g.transform(
        lambda s: _prior(s, lambda p: p.expanding().count())).fillna(0.0)
    return out


def add_team_volume(
    frame: pd.DataFrame, *, denom: str, team_col: str = "team", last_n: int = 4,
) -> pd.DataFrame:
    """Predicted team opportunity in `denom`, from the team's own strictly earlier games.

    The model needs a volume forecast and must not be handed the real one: using the actual
    team total would be lookahead of the most flattering kind, since usage share x TRUE
    volume reconstructs the answer almost exactly and would look like a triumph.
    """
    out = frame.copy()
    per_game = (out.groupby([team_col, "season", "week"], as_index=False)
                   .agg(team_total=(denom, "sum"), kickoff=("kickoff", "first"))
                   .sort_values("kickoff"))
    per_game["team_volume_pred"] = (
        per_game.groupby(team_col)["team_total"]
        .transform(lambda s: _prior(s, lambda p: p.rolling(last_n, min_periods=1).mean()))
    )
    return out.merge(
        per_game[[team_col, "season", "week", "team_volume_pred"]],
        on=[team_col, "season", "week"], how="left")


def add_usage_model(
    frame: pd.DataFrame, *, share_col: str, stat: str, denom: str,
    player_col: str = "player_id",
) -> pd.DataFrame:
    """The actual model: usage share x predicted team volume x conversion rate.

    Conversion is the rate at which an opportunity becomes the stat -- catches per target,
    for instance. For a stat that IS the opportunity (carries, pass attempts) it is 1 by
    construction and the term drops out.
    """
    out = frame.copy()
    if stat == denom:
        conv = pd.Series(1.0, index=out.index)
    else:
        rate = (out[stat] / out[denom].replace(0.0, np.nan)).astype(float)
        conv = out.groupby(player_col)[stat].transform(lambda s: np.nan)  # placeholder shape
        conv = rate.groupby(out[player_col]).transform(
            lambda s: _prior(s, lambda p: p.expanding().mean()))
        # A player with no prior opportunities has no personal rate; fall back to the
        # league's prior mean rather than to zero, which would predict he never converts.
        league = rate.groupby(out["season"]).transform(
            lambda s: _prior(s, lambda p: p.expanding().mean()))
        conv = conv.fillna(league).fillna(rate.mean())

    out["model_pred"] = out[share_col] * out["team_volume_pred"] * conv
    out["model_conv_rate"] = conv
    return out


ESTIMATORS = ["base_last_n", "base_ewma", "base_season_mean", "model_pred"]


def score(frame: pd.DataFrame, *, stat: str, estimators=None, min_prior: int = 4) -> pd.DataFrame:
    """MAE and RMSE per estimator on the same rows, so the comparison is like-for-like.

    Restricted to rows where every estimator is defined. Scoring each on whatever subset it
    happens to cover would flatter whichever one is most often missing.
    """
    estimators = estimators or ESTIMATORS
    sub = frame[frame["prior_games"] >= min_prior]
    sub = sub.dropna(subset=[stat, *estimators])
    rows = []
    for est in estimators:
        err = sub[est] - sub[stat]
        rows.append({
            "estimator": est,
            "n": int(len(sub)),
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()),
        })
    out = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best_base = out[out["estimator"].str.startswith("base_")]["mae"].min()
    out["vs_best_baseline"] = (out["mae"] - best_base).round(4)
    return out


__all__ = ["add_baselines", "add_team_volume", "add_usage_model", "score", "ESTIMATORS"]
