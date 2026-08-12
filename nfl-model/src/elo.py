"""Baseline Elo (§8b, GATE_BEATS_ELO). Baseline only -- never an input to the model.

Deliberately plain: a standard 538-style NFL Elo with margin-of-victory scaling and
between-season regression. If the drive simulator cannot beat this, the extra machinery is
not earning its keep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .ingest import CANONICAL_TEAMS


@dataclass(frozen=True)
class EloConfig:
    k: float = 20.0
    hfa_points: float = 1.7
    points_per_elo: float = 25.0     # 25 Elo points ~ 1 point of spread
    start: float = 1500.0
    season_regression: float = 0.25  # toward the league mean each offseason


def run_elo(schedules: pd.DataFrame, cfg: Config, elo_cfg: "EloConfig | None" = None):
    """Walk forward through every game, predicting before updating.

    Returns (per-game predictions, final ratings). The prediction for a game is always
    made from ratings containing only earlier games, so this is lookahead-free by
    construction.
    """
    ec = elo_cfg or EloConfig()
    ratings = {t: ec.start for t in CANONICAL_TEAMS}
    games = schedules[
        schedules["result"].notna() & schedules["kickoff"].notna()
    ].sort_values("kickoff")

    rows = []
    prev_season = None
    for g in games.itertuples(index=False):
        if prev_season is not None and g.season != prev_season:
            for t in ratings:
                ratings[t] += (ec.start - ratings[t]) * ec.season_regression
        prev_season = g.season

        home, away = g.home_team, g.away_team
        if home not in ratings or away not in ratings:
            continue
        hfa = 0.0 if g.location == "Neutral" else ec.hfa_points
        elo_diff = ratings[home] - ratings[away] + hfa * ec.points_per_elo

        rows.append({
            "game_id": g.game_id,
            "season": g.season,
            "week": g.week,
            "elo_spread": elo_diff / ec.points_per_elo,   # positive = home favored
            "elo_home_pre": ratings[home],
            "elo_away_pre": ratings[away],
        })

        # Update, with 538's margin-of-victory multiplier.
        expected_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        actual_home = 1.0 if g.result > 0 else (0.5 if g.result == 0 else 0.0)
        mov = np.log(abs(g.result) + 1.0) * (2.2 / (0.001 * abs(elo_diff) + 2.2))
        shift = ec.k * mov * (actual_home - expected_home)
        ratings[home] += shift
        ratings[away] -= shift

    return pd.DataFrame(rows), pd.Series(ratings, name="elo")
