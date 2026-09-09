from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction_contract import Forecast, MarketEvidence
from publication import assess_quality
from slate_builder import (calibration_from_weights, calibration_permissions,
                           forecast_from_projection)


class Sim:
    margins = np.array([-7, 0, 3, 7, 14], dtype=float)
    totals = np.array([35, 42, 45, 49, 56], dtype=float)
    home_scores = (totals + margins) / 2
    away_scores = (totals - margins) / 2
    weights = np.full(5, .2)

    @property
    def mean_margin(self):
        return float(self.weights @ self.margins)

    @property
    def mean_total(self):
        return float(self.weights @ self.totals)

    def recentered(self, margin):
        self._margin_target = float(margin)
        return self

    def retotaled(self, total):
        self._total_target = float(total)
        return self


def test_validated_mean_replaces_simulation_mean_without_destroying_distribution():
    sim = Sim()
    raw_mean = sim.mean_margin
    forecast = forecast_from_projection(sim, spread=9.0, total=49.0)
    assert raw_mean != pytest.approx(9.0)
    assert forecast.spread == pytest.approx(9.0)
    assert sim._margin_target == 9.0
    assert sim._total_target == 49.0
    assert forecast.source == "football model + drive simulation"
    assert forecast.interval_80_low is not None


def test_roster_uncertainty_widens_interval_but_preserves_projection_mean():
    baseline = forecast_from_projection(Sim(), spread=9.0, total=49.0)
    uncertain = forecast_from_projection(
        Sim(), spread=9.0, total=49.0, interval_multiplier=1.35,
    )
    assert uncertain.spread == baseline.spread
    assert uncertain.total == baseline.total
    assert uncertain.interval_80_low < baseline.interval_80_low
    assert uncertain.interval_80_high > baseline.interval_80_high


def test_ncaa_calibration_permissions_are_market_specific_and_positive_only():
    """Significance (`t > 2`) was dropped from `calibration_permissions` 2026-08-18 --
    a directionally positive weight now blends toward the market regardless of whether it
    also clears statistical significance. The only real exclusion criterion left is sign:
    an anti-predictive (negative) weight must still never be blended. `t_model_spread`/
    `t_model_total` are omitted from this fixture entirely since they're no longer read."""
    weights = {
        "b_model_spread": -.10,
        "b_model_total": .20,
        "a_spread": -.4, "a_total": .3,
    }
    permissions = calibration_permissions(weights, "ncaa")
    assert permissions == {"spread": False, "total": True}
    model = Forecast.from_spread_total(14, 55, None, "model")
    market = Forecast.from_spread_total(35, 58, None, "market")

    # Permissions stay market-specific -- the total's weight is directionally positive and
    # the spread's is not. What happens when they disagree: publish null rather than
    # half-copy the market under a column that claims a blend, not "the market re-emitted
    # under a column that claims validation" (the substitution the contract forbids).
    assert calibration_from_weights(model, market, weights, "ncaa") is None


def test_negative_significant_slope_never_enables_calibration():
    weights = {"b_model_spread": 0, "b_model_spread_raw": -.3,
               "t_model_spread": -3.2, "b_model_total": 0,
               "t_model_total": 0}
    assert calibration_permissions(weights, "ncaa") == {
        "spread": False, "total": False}


def test_game_specific_quality_and_disagreement_fail_closed():
    evidence = MarketEvidence("three-book median", True, 3, 3, ("a", "b", "c"))
    quality = assess_quality(
        league="ncaa", week=1, home_games_observed=0, away_games_observed=0,
        unavailable_features=("confirmed QB/coordinator continuity",),
        market_evidence=evidence, spread_difference=-21.8,
        calibration_status={"spread": False, "total": True}, bets_allowed=True,
    )
    assert quality.label == "incomplete"
    assert quality.pick_eligible is False
    assert quality.out_of_distribution is True
    assert any("Extreme" in warning for warning in quality.warnings)
    assert any("spread experimental blend" in reason for reason in quality.reasons)


def test_only_extreme_disagreement_gets_warning_icon_policy():
    evidence = MarketEvidence("two-book reference", False, 2, 1, ("a", "b"))
    review = assess_quality(
        league="ncaa", week=1, home_games_observed=0, away_games_observed=0,
        unavailable_features=(), market_evidence=evidence, spread_difference=14.1,
        calibration_status={"spread": False, "total": False}, bets_allowed=False,
    )
    extreme = assess_quality(
        league="ncaa", week=1, home_games_observed=0, away_games_observed=0,
        unavailable_features=(), market_evidence=evidence, spread_difference=20.1,
        calibration_status={"spread": False, "total": False}, bets_allowed=False,
    )
    assert review.out_of_distribution is False
    assert review.warnings == ()
    assert any("Extreme" in warning for warning in extreme.warnings)


@pytest.mark.parametrize("week,observed,label", [
    (1, 0, "preseason"), (2, 1, "early-season"),
    (4, 3, "developing"), (7, 6, "established"),
])
def test_quality_maturity_progresses_but_does_not_claim_betting_edge(week, observed, label):
    evidence = MarketEvidence("three-book median", True, 3, 3, ("a", "b", "c"))
    quality = assess_quality(
        league="ncaa", week=week, home_games_observed=observed,
        away_games_observed=observed, unavailable_features=(),
        market_evidence=evidence, spread_difference=2.0,
        calibration_status={"spread": True, "total": True}, bets_allowed=False,
    )
    assert quality.label == label
    assert quality.pick_eligible is False
