"""Injury burden: how much of a team's actual playing time is unavailable.

WHY NOT COUNT PLAYERS. "Four players out" says nothing on its own -- four special-teamers
and four starters are the same number and a very different team. What predicts the result
is the share of snaps the missing players would have taken, so each player on the injury
report is weighted by the fraction of snaps he had actually been playing before he got
hurt. One unit of burden is roughly one full-time starter.

Measured on 1,535 walk-forward games, against the residual of the current model:

    snap-weighted burden differential   -0.518 pts of margin per unit   t = -2.15
    raw count of players out            -0.303 pts per player           t = -2.04
    both in the same regression         neither survives (collinear)

So the weighting is worth having but only modestly -- the two measure the same thing and
a plain count is a decent proxy. A 1-sd injury differential is about 0.76 points.

INJURIES MOVE THE SPREAD, NOT THE TOTAL. Against the total residual the burden sum is
-0.011 (t = -0.05), i.e. nothing. A banged-up team scores less AND concedes more, and the
two cancel in the total. Only `spread_points` is adjusted.

NO LOOKAHEAD. A player's snap share is an expanding mean over strictly earlier weeks, and
the injury report is published before kickoff, so this is available prospectively -- which
is the point of it. The as-of join is load-bearing: a player who is Out has no snap-count
row for that week, so joining on the exact week matches nobody. Taking his most recent
earlier week instead moves the match rate from 0.1% to 78%.
"""

from __future__ import annotations

import re

import pandas as pd

# Statuses meaning the player is not expected to take a meaningful share of snaps.
# "Questionable" is deliberately excluded: it is the league's most strategically vague
# designation and most Questionable players do in fact play.
UNAVAILABLE = ("Out", "Doubtful")

BURDEN_COLUMNS = ["season", "week", "team", "burden", "n_out"]


def _norm(name: object) -> str:
    """Injuries key on gsis_id and snap counts on pfr_player_id -- there is no shared id,
    so the join is on a normalised name within a team-season, where collisions are rare."""
    return re.sub(r"[^a-z]", "", str(name).lower())


def build_injury_burden(
    injuries: pd.DataFrame, snap_counts: pd.DataFrame
) -> pd.DataFrame:
    """One row per (season, week, team): snap-weighted burden and raw count out."""
    if (
        injuries is None or snap_counts is None
        or injuries.empty or snap_counts.empty
    ):
        return pd.DataFrame(columns=BURDEN_COLUMNS)

    inj, snaps = injuries.copy(), snap_counts.copy()
    for d in (inj, snaps):
        d["season"] = d["season"].astype(int)
        d["week"] = d["week"].astype(int)
    inj["k"] = inj["full_name"].map(_norm)
    snaps["k"] = snaps["player"].map(_norm)

    # A player's importance is the larger of his offensive and defensive share -- he is
    # one or the other, and taking the max avoids halving everyone's weight.
    snaps["play_pct"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    snaps = snaps.sort_values(["season", "team", "k", "week"])
    snaps["prior_pct"] = (
        snaps.groupby(["season", "team", "k"])["play_pct"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    prior = (
        snaps[["season", "team", "k", "week", "prior_pct"]]
        .dropna(subset=["prior_pct"])
        .sort_values("week")
    )

    out = inj[inj["report_status"].isin(UNAVAILABLE)].copy().sort_values("week")
    if out.empty:
        return pd.DataFrame(columns=BURDEN_COLUMNS)

    # AS-OF, not exact -- see the module docstring. An unmatched player contributes zero
    # rather than an assumed share; he is usually a week-one or practice-squad case.
    matched = pd.merge_asof(
        out, prior, on="week", by=["season", "team", "k"], direction="backward"
    )
    matched["w"] = matched["prior_pct"].fillna(0.0)

    burden = matched.groupby(["season", "week", "team"])["w"].sum().rename("burden")
    count = out.groupby(["season", "week", "team"]).size().rename("n_out")
    return (
        pd.concat([burden, count], axis=1)
        .fillna({"burden": 0.0, "n_out": 0})
        .reset_index()[BURDEN_COLUMNS]
    )


def burden_lookup(burden: pd.DataFrame) -> dict:
    """(season, week, team) -> burden, for row-at-a-time use inside the walk."""
    if burden is None or burden.empty:
        return {}
    return {
        (int(r.season), int(r.week), r.team): float(r.burden)
        for r in burden.itertuples(index=False)
    }


def injury_spread_points(home_burden: float, away_burden: float, cfg) -> float:
    """Points of margin, home perspective. Negative when the home team is the sicker one.

    Capped, because the tail is not something 1,535 games can speak to: the worst observed
    team-week is 9.8 starter-equivalents, which uncapped would be a 5-point swing off a
    coefficient fitted at t = -2.15.
    """
    diff = float(home_burden) - float(away_burden)
    pts = -cfg.context.injury_points_per_burden * diff
    cap = cfg.context.injury_max_points
    return max(-cap, min(cap, pts))
