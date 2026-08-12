from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (PRESEASON_FEATURES, add_preseason_matchup_features,
                          maturity_phase)
from src.live_mean import fit_live_mean


def _frame(useful_preseason: bool) -> pd.DataFrame:
    rng = np.random.default_rng(14)
    rows = []
    for season in range(2020, 2026):
        for week in range(1, 7):
            for game in range(55):
                net = rng.normal(0, .05)
                prior = rng.normal()
                margin = 3 + 180 * net + (5 * prior if useful_preseason else 0) + rng.normal(0, 10)
                rows.append({"season": season, "week": week, "game_id": f"{season}-{week}-{game}",
                    "net_diff": net, "is_home": 1., "pace_sum": rng.normal(0, .03),
                    "eff_sum": rng.normal(0, .05), "returning_production_diff": prior,
                    "actual_margin": margin, "actual_total": 54 + rng.normal(0, 11)})
    return pd.DataFrame(rows)


def test_preseason_challenger_requires_held_out_early_season_improvement():
    model = fit_live_mean(_frame(True), week=1)
    assert model.spread.challenger_promoted is True
    assert "returning_production_diff" in model.spread.features
    assert model.validation_season == 2025


def test_noise_preseason_feature_is_not_promoted():
    model = fit_live_mean(_frame(False), week=1)
    assert model.spread.challenger_promoted is False
    assert model.spread.features == ("net_diff", "is_home")


def test_preseason_features_are_interacted_with_historical_maturity_phase():
    games = pd.DataFrame([
        {"game_id": 1, "season": 2025, "week": 1, "home_team": "A", "away_team": "B"},
        {"game_id": 2, "season": 2025, "week": 6, "home_team": "A", "away_team": "B"},
    ])
    preseason = pd.DataFrame([
        {"season": 2025, "team": "A", "returning_production": .8},
        {"season": 2025, "team": "B", "returning_production": .4},
    ])
    out = add_preseason_matchup_features(games, preseason)
    assert out.loc[0, "returning_production_diff_preseason"] == .4
    assert pd.isna(out.loc[0, "returning_production_diff_established"])
    assert out.loc[1, "returning_production_diff_established"] == .4
    assert maturity_phase(1) == "preseason"
    assert maturity_phase(3) == "early"
    assert maturity_phase(5) == "developing"
    assert maturity_phase(8) == "established"
    assert set(PRESEASON_FEATURES)
