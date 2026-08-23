from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slate_builder import calibration_from_weights, forecast_from_sim, market_for_game


class Sim:
    margins = np.array([-3, 3, 7])
    totals = np.array([40, 44, 48])
    weights = np.array([.1, .2, .7])


def test_sim_forecast_uses_calibrated_weights_not_raw_average():
    forecast = forecast_from_sim(Sim())
    assert forecast.spread == pytest.approx(5.2)
    assert forecast.total == pytest.approx(46.4)
    assert forecast.home_win_probability == pytest.approx(.9)


def test_three_sources_still_quote_ONE_book_not_their_middle():
    """Three books used to yield the median. It is still one book's real line -- coverage
    is reported separately via is_consensus, and never folded into the number."""
    game = pd.Series({"game_id": "g", "spread_line": 1, "total_line": 40})
    lines = pd.DataFrame([
        {"league": "nfl", "game_id": "g", "provider": p, "spread": s, "total": t,
         "observed_at": f"2026-09-01T1{i}:00:00Z"}
        for i, (p, s, t) in enumerate([("a", 2, 44), ("b", 3, 45), ("c", 4, 46)])
    ])
    market, evidence = market_for_game(
        game, lines, league="nfl", as_of="2026-09-02T00:00:00Z")
    assert (market.spread, market.total) == (2, 44), "book 'a' wins on alphabetical fallback"
    assert market.spread != 3, "the median of 2, 3, 4 must not appear"
    assert evidence.is_consensus, "three books is still well-covered, it just is not blended"


def test_calibration_remains_separate():
    model = forecast_from_sim(Sim())
    game = pd.Series({"game_id": "g", "spread_line": 2, "total_line": 44})
    market, _ = market_for_game(game, pd.DataFrame(), league="nfl")
    calibrated = calibration_from_weights(
        model, market, {"a_spread": 0, "b_model": .25, "a_total": 0,
                        "b_model_total": .5, "t_model": 2.5,
                        "t_model_total": 2.5}, "nfl")
    assert calibrated.spread != model.spread
    assert model.spread == pytest.approx(5.2)


def _weights(**overrides):
    base = {"a_spread": 0, "b_model": .25, "a_total": 0, "b_model_total": .5,
            "t_model": 2.5, "t_model_total": 2.5}
    base.update(overrides)
    return base


def _calibrate(**overrides):
    model = forecast_from_sim(Sim())
    game = pd.Series({"game_id": "g", "spread_line": 2, "total_line": 44})
    market, _ = market_for_game(game, pd.DataFrame(), league="nfl")
    return calibration_from_weights(model, market, _weights(**overrides), "nfl"), market


def test_low_significance_but_positive_weight_still_produces_a_blend():
    """Significance (`t > 2`) is no longer required, dropped 2026-08-18: a directionally
    positive weight is blended toward the market even when it isn't yet statistically
    proven, since a market-anchored blend is the right move regardless of whether this
    repo's own edge is independently demonstrated. `t_model`/`t_model_total` are no longer
    read by `calibration_permissions` at all -- low values here are asserting they are
    now ignored, not exercising a real gate."""
    calibrated, market = _calibrate(t_model=0.4, t_model_total=0.4)
    assert calibrated is not None
    assert calibrated.spread != market.spread
    assert calibrated.total != market.total


def test_missing_spread_weight_yields_no_calibrated_forecast_at_all():
    """A market copy wearing a "calibrated" label is the thing to avoid.

    Weighting an unvalidated market at zero silently republishes the market under a
    column that claims validation. `Forecast` also derives both scores from spread AND
    total, so a half-calibrated object cannot even be internally coherent. Absent
    evidence for either market, there is no calibrated projection -- publish null.
    """
    calibrated, _ = _calibrate(b_model=None)
    assert calibrated is None, "a missing spread weight must not be filled from the market"


def test_missing_total_weight_yields_no_calibrated_forecast_at_all():
    calibrated, _ = _calibrate(b_model_total=None)
    assert calibrated is None


def test_anti_predictive_spread_never_produces_a_calibrated_number():
    calibrated, _ = _calibrate(b_model=-0.3, b_model_raw=-0.3)
    assert calibrated is None


def test_both_markets_validated_still_produces_a_blend():
    calibrated, market = _calibrate()
    assert calibrated is not None
    assert calibrated.spread != market.spread


def test_market_snapshot_drops_observations_after_cutoff():
    game = pd.Series({"game_id": "g", "spread_line": 1, "total_line": 40,
                      "kickoff": "2026-09-02T00:00:00Z"})
    lines = pd.DataFrame([
        {"league": "nfl", "game_id": "g", "provider": "early", "spread": 2,
         "total": 44, "observed_at": "2026-09-01T10:00:00Z"},
        {"league": "nfl", "game_id": "g", "provider": "future", "spread": 9,
         "total": 60, "observed_at": "2026-09-01T20:00:00Z"},
    ])
    market, evidence = market_for_game(
        game, lines, league="nfl", as_of="2026-09-01T12:00:00Z")
    assert market.spread == 2
    assert evidence.providers == ("early",)


def test_two_sources_pick_one_book_rather_than_splitting_them():
    """The defect both audits found: a two-book median is the midpoint, and the midpoint of
    2 and 4 is 3 -- a number neither book offered."""
    game = pd.Series({"game_id": "g", "spread_line": 1, "total_line": 40})
    lines = pd.DataFrame([
        {"league": "ncaa", "game_id": "g", "provider": "a", "spread": 2,
         "total": 44, "observed_at": "2026-08-01T10:00:00Z"},
        {"league": "ncaa", "game_id": "g", "provider": "b", "spread": 4,
         "total": 46, "observed_at": "2026-08-01T10:01:00Z"},
    ])
    market, evidence = market_for_game(
        game, lines, league="ncaa", as_of="2026-08-02T00:00:00Z")
    assert market.spread == 2, "book 'a', not the 3 that neither book posted"
    assert evidence.book_count_spread == 2
    assert evidence.is_consensus is False
    assert evidence.label == "a (of 2 books)", "the label names the book that was taken"
