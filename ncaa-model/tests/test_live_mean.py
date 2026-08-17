from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (PRESEASON_FEATURES, add_preseason_matchup_features,
                          maturity_phase, previous_season_power)
from src.live_mean import LiveMeanModel, fit_live_mean
from validated_model import ValidatedRidge


def _constant_ridge(value: float, promoted: bool = False) -> ValidatedRidge:
    """A ValidatedRidge that always predicts `value`, regardless of input -- for testing
    LiveMeanModel.predict()'s own logic in isolation from the real fitting machinery."""
    return ValidatedRidge(
        features=(), center=np.array([]), scale=np.array([]), medians=np.array([]),
        beta=np.array([value]), lower=np.array([]), upper=np.array([]),
        challenger_promoted=promoted, validation_base_rmse=0.0,
        validation_challenger_rmse=0.0,
    )


def test_predict_clips_an_infeasible_spread_total_pair_to_the_physical_floor():
    """Regression guard for the 2026-08-17 bug: fit_live_mean fits spread and total as
    two separate ridge models with nothing constraining their combination to be
    physically possible. Real FCS ratings made this a real, not just theoretical, failure
    -- Georgia vs Tennessee State predicted spread +47.6, total +47.0, which implies a
    losing score of (47.0-47.6)/2 = -0.3. total can never be less than |spread|; a shutout
    is the most lopsided a real game can be. Reproduces that exact pair directly against
    LiveMeanModel.predict() rather than the full fit, so this stays fast and deterministic
    even if the live rating data that first exposed it changes."""
    model = LiveMeanModel(
        spread=_constant_ridge(47.55294280197436, promoted=True),
        total=_constant_ridge(47.01042655031927, promoted=False),
        phase="preseason", validation_season=2025,
    )

    spread, total = model.predict(pd.DataFrame({"x": [0.0]}))

    assert spread == pytest.approx(47.55294280197436)
    assert total >= abs(spread), "total must never be less than |spread| -- negative score"
    assert total == pytest.approx(abs(spread))


def test_predict_leaves_a_feasible_pair_untouched():
    model = LiveMeanModel(
        spread=_constant_ridge(7.0), total=_constant_ridge(48.5),
        phase="preseason", validation_season=2025,
    )

    spread, total = model.predict(pd.DataFrame({"x": [0.0]}))

    assert spread == pytest.approx(7.0)
    assert total == pytest.approx(48.5)


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


def test_previous_season_power_is_available_before_week_one():
    ratings = pd.DataFrame([
        {"season": 2024, "week": 12, "team": "A", "off_rating": .2, "def_rating": -.1},
        {"season": 2024, "week": 12, "team": "B", "off_rating": -.1, "def_rating": .1},
        {"season": 2025, "week": 1, "team": "A", "off_rating": 9, "def_rating": 9},
    ])
    prior = previous_season_power(ratings, seasons=[2025])
    assert prior.loc[prior.team.eq("A"), "prior_power_rating"].iloc[0] == pytest.approx(.3)
    assert prior.loc[prior.team.eq("B"), "prior_power_rating"].iloc[0] == pytest.approx(-.2)
