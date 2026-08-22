"""Prop projection: a player's own form, adjusted for the defence, as a distribution.

THIS IS A REBUILD, AND THE REASON MATTERS. The first model reconstructed a player's output
from parts -- usage share x predicted team volume x conversion rate -- and lost to an
exponentially-weighted average of his own recent games on all three markets tested
(receptions -3.3%, carries -12.2%, pass attempts -9.5%). No bug was behind it: the
components measured within 4% of their realised values. The decomposition itself was the
error. A player's recent output already encodes his usage, his team's volume and his
conversion rate in one number that reality computed exactly; splitting it into three
separately-estimated pieces and multiplying them adds variance without adding information.

Published implementations all share the opposite shape -- the rolling average is the BASE
and the model adjusts it:

    mu = player_avg + defence_effect            (Rome, Bayesian hierarchical FF projections)
    "a measured tweak from the receiver's baseline projection"   (FTN's WR/CB tool)

So that is what this does. It never competes with the baseline; it corrects it.

AND IT RETURNS A DISTRIBUTION, NOT A NUMBER. "Props never ask you for point estimates"
(Atkinson, Poisson touchdown props). The bet is P(stat > 5.5), so a mean of 6.1 is useless
without the spread around it. Counts get a negative binomial: Poisson is the natural first
choice but receptions are overdispersed relative to it, and understating the tail is exactly
how a prop model prices the over wrong.

THE DEFENCE EFFECT IS PARTIALLY POOLED, and it has to be. A team faces a given position
about seventeen times a season. Estimating 32 teams x 4 positions independently on that is
noise; shrinking each toward "no effect" in proportion to sample size is what makes it
usable. The null here is deliberately zero rather than the league mean -- a defence is
assumed ordinary until its own record says otherwise.

EVERYTHING IS WALK-FORWARD ON KICKOFF TIMESTAMPS, including the defence effects, which are
built only from games that kicked off strictly earlier. A defensive rating that includes the
game being predicted would make this model look excellent and be worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def add_opponent(frame: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Attach the opposing team for each player-week."""
    g = schedules[["season", "week", "home_team", "away_team"]]
    home = g.rename(columns={"home_team": "team", "away_team": "opponent"})
    away = g.rename(columns={"away_team": "team", "home_team": "opponent"})
    long = pd.concat([home.assign(is_home=1), away.assign(is_home=0)], ignore_index=True)
    return frame.merge(long, on=["season", "week", "team"], how="left")


def defense_effects(
    frame: pd.DataFrame, *, stat: str, baseline_col: str, prior_games: float = 10.0,
) -> pd.DataFrame:
    """Walk-forward defence-vs-position effect, shrunk toward zero.

    For each row: how much better or worse have players of this position done against this
    defence than their own baselines predicted, over games that kicked off strictly earlier.

    `prior_games` is the shrinkage strength -- an effect estimated from `prior_games`
    observations is weighted half. Set to 10 because a defence faces a position roughly
    seventeen times a season, so ten is about "most of one season of evidence before you
    trust it fully". Chosen from that arithmetic, NOT tuned against the result: tuning it
    until the model wins is the multiplicity trap this repo has already been caught by once.
    """
    out = frame.sort_values("kickoff").copy()
    resid = out[stat] - out[baseline_col]
    out["_resid"] = resid

    key = out.groupby(["opponent", "position"])["_resid"]
    # Strictly earlier: shift(1) before any accumulation, so a game never informs itself.
    prior_sum = key.transform(lambda s: s.shift(1).expanding().sum())
    prior_n = key.transform(lambda s: s.shift(1).expanding().count())

    raw = prior_sum / prior_n.replace(0, np.nan)
    weight = prior_n / (prior_n + prior_games)
    out["def_effect"] = (weight * raw).fillna(0.0)      # unknown defence == ordinary
    out["def_effect_n"] = prior_n.fillna(0.0)
    return out.drop(columns=["_resid"])


def home_effect(frame: pd.DataFrame, *, stat: str, baseline_col: str) -> pd.DataFrame:
    """League-wide home/away residual, walk-forward. Small, but free to include."""
    out = frame.sort_values("kickoff").copy()
    resid = out[stat] - out[baseline_col]
    out["_r"] = resid
    eff = out.groupby("is_home")["_r"].transform(
        lambda s: s.shift(1).expanding().mean())
    out["home_effect"] = eff.fillna(0.0)
    return out.drop(columns=["_r"])


def predict(frame: pd.DataFrame, *, baseline_col: str, floor: float = 0.0) -> pd.DataFrame:
    """baseline + defence + home, floored at zero.

    A negative projection is not a smaller projection, it is an impossible one: a receiver
    cannot catch -0.3 passes. Clipping matters because the defence effect is additive and
    can otherwise push a low-usage player below zero.
    """
    out = frame.copy()
    out["pred_mean"] = (out[baseline_col].astype(float)
                        + out["def_effect"].astype(float)
                        + out.get("home_effect", 0.0)).clip(lower=floor)
    return out


def fit_dispersion(frame: pd.DataFrame, *, stat: str, mean_col: str = "pred_mean") -> float:
    """Negative-binomial dispersion from realised over/under-dispersion.

    For a Poisson, variance == mean. Receptions are overdispersed: a receiver's week-to-week
    spread exceeds his average, because targets themselves vary. Fitting `r` from the data
    rather than assuming Poisson is what keeps the tail -- and therefore the price of an
    over -- honest.
    """
    s = frame.dropna(subset=[stat, mean_col])
    mu = s[mean_col].mean()
    var = ((s[stat] - s[mean_col]) ** 2).mean()
    if var <= mu or mu <= 0:
        return float("inf")          # not overdispersed; Poisson is the limiting case
    return float(mu ** 2 / (var - mu))


def prob_over(mean, line, r: float) -> np.ndarray:
    """P(stat > line) under a negative binomial with mean `mean` and dispersion `r`.

    Lines are half-points in practice, so P(X > 5.5) == P(X >= 6) == 1 - cdf(5).
    """
    mean = np.asarray(mean, dtype=float)
    line = np.asarray(line, dtype=float)
    k = np.floor(line)
    if not np.isfinite(r):
        return 1.0 - stats.poisson.cdf(k, np.maximum(mean, 1e-9))
    p = r / (r + np.maximum(mean, 1e-9))
    return 1.0 - stats.nbinom.cdf(k, r, p)


def calibration(frame: pd.DataFrame, *, stat: str, mean_col: str, r: float,
                bins: int = 10) -> pd.DataFrame:
    """Do things predicted to happen 30% of the time happen 30% of the time?

    The only score that matters for a prop. A model can have excellent MAE and be useless
    for betting if its probabilities are wrong, and this repo already has GATE_CALIBRATED
    failing on team totals for exactly that reason.

    Evaluated against each row's own median as a pseudo-line, so it needs no market data.
    """
    s = frame.dropna(subset=[stat, mean_col]).copy()
    lines = np.floor(s[mean_col]) + 0.5          # a realistic half-point line
    s["p_over"] = prob_over(s[mean_col], lines, r)
    s["hit"] = (s[stat] > lines).astype(float)
    s["bucket"] = pd.cut(s["p_over"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = s.groupby("bucket", observed=True).agg(
        n=("hit", "size"), predicted=("p_over", "mean"), actual=("hit", "mean"))
    g["gap"] = (g["actual"] - g["predicted"]).round(4)
    return g.reset_index()


COUNT_MARKETS = ("receptions", "carries", "attempts", "completions",
                 "passing_tds", "rushing_tds", "receiving_tds", "targets")


def is_count_market(stat: str) -> bool:
    """Whether a negative binomial is a legitimate model for this stat.

    IT IS NOT FOR YARDAGE, and the failure is loud rather than subtle: fitting a negative
    binomial to receiving yards produces -inf log scores, because outcomes the data contains
    are assigned zero probability. Yardage has a point mass at zero (targeted, no catch),
    rushing yards go negative, and the shape is heavy-tailed and role-dependent -- a hurdle,
    mixture or empirical-residual model is required, which this module does not yet have.

    Reported honestly instead of silently: a distribution that cannot represent the outcome
    must not be used to price a bet on it.
    """
    return stat in COUNT_MARKETS


def log_score(frame: pd.DataFrame, *, stat: str, mean_col: str, r: float) -> float:
    """Mean log-likelihood of the realised outcome. Rewards a sharp distribution that is
    also right, and punishes confidence in the wrong place -- which MAE cannot see.

    Returns NaN for a market this distribution cannot represent, rather than -inf.
    """
    if not is_count_market(stat):
        return float("nan")
    s = frame.dropna(subset=[stat, mean_col])
    mu = np.maximum(s[mean_col].to_numpy(dtype=float), 1e-9)
    y = s[stat].to_numpy(dtype=float)
    if not np.isfinite(r):
        return float(np.mean(stats.poisson.logpmf(y, mu)))
    p = r / (r + mu)
    return float(np.mean(stats.nbinom.logpmf(y, r, p)))


__all__ = ["add_opponent", "defense_effects", "home_effect", "predict", "fit_dispersion",
           "prob_over", "calibration", "log_score", "is_count_market", "COUNT_MARKETS"]
