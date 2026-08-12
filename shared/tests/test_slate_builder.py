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


def test_three_manual_sources_create_consensus():
    game = pd.Series({"game_id": "g", "spread_line": 1, "total_line": 40})
    lines = pd.DataFrame([
        {"league": "nfl", "game_id": "g", "provider": p, "spread": s, "total": t,
         "observed_at": f"2026-09-01T1{i}:00:00Z"}
        for i, (p, s, t) in enumerate([("a", 2, 44), ("b", 3, 45), ("c", 4, 46)])
    ])
    market, evidence = market_for_game(
        game, lines, league="nfl", as_of="2026-09-02T00:00:00Z")
    assert market.spread == 3
    assert market.total == 45
    assert evidence.is_consensus


def test_calibration_remains_separate():
    model = forecast_from_sim(Sim())
    game = pd.Series({"game_id": "g", "spread_line": 2, "total_line": 44})
    market, _ = market_for_game(game, pd.DataFrame(), league="nfl")
    calibrated = calibration_from_weights(
        model, market, {"a_spread": 0, "b_model": .25, "a_total": 0,
                        "b_model_total": .5}, "nfl")
    assert calibrated.spread != model.spread
    assert model.spread == pytest.approx(5.2)


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
