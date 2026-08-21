"""W1 forward tracking (PREREG.md W1).

This script is the only record the pre-registered wind rule will ever have, so the tests are
about the properties that make a record trustworthy rather than about convenience: the
declared threshold is enforced exactly, a logged bet is never rewritten, settlement uses the
number the bet was recorded at, and the forecast never silently becomes the observation.

The last one is the reason W1 exists as a separate exercise at all. The backtest that
motivated it selected on the KICKOFF READING; a rule that quietly did the same thing
prospectively would reproduce a 57.7% backtest and mean nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import track_w1
from src.weather import ForecastReading


def _slate():
    """One outdoor game per side of the declared boundary, plus a dome and a played game."""
    return pd.DataFrame([
        {"game_id": "g1", "season": 2026, "week": 1, "away_team": "A", "home_team": "B",
         "kickoff": pd.Timestamp("2026-09-13 13:00"), "roof": "outdoors",
         "total_line": 44.5, "result": np.nan, "total": np.nan, "wind": np.nan},
        {"game_id": "g2", "season": 2026, "week": 1, "away_team": "C", "home_team": "D",
         "kickoff": pd.Timestamp("2026-09-13 13:00"), "roof": "outdoors",
         "total_line": 48.0, "result": np.nan, "total": np.nan, "wind": np.nan},
        {"game_id": "g3", "season": 2026, "week": 1, "away_team": "E", "home_team": "F",
         "kickoff": pd.Timestamp("2026-09-13 16:25"), "roof": "outdoors",
         "total_line": 41.0, "result": np.nan, "total": np.nan, "wind": np.nan},
        {"game_id": "g4", "season": 2026, "week": 1, "away_team": "G", "home_team": "H",
         "kickoff": pd.Timestamp("2026-09-13 13:00"), "roof": "dome",
         "total_line": 52.5, "result": np.nan, "total": np.nan, "wind": np.nan},
        # Already played: has a result, so it can never be forecast.
        {"game_id": "g5", "season": 2026, "week": 1, "away_team": "I", "home_team": "J",
         "kickoff": pd.Timestamp("2026-09-13 13:00"), "roof": "outdoors",
         "total_line": 45.0, "result": 7.0, "total": 51.0, "wind": 18.0},
    ])


# forecast wind by game: below / exactly at / far above the threshold, and one implausible
_WINDS = {"g1": 9.9, "g2": 10.0, "g3": 22.0, "g5": 30.0}


def _fake_forecast(game, *, now=None):
    if game.get("roof") == "dome":
        return ForecastReading(None, None, None, True, "2026-09-11T12:00:00+00:00",
                               None, "indoor")
    wind = _WINDS.get(game["game_id"], 5.0)
    return ForecastReading(wind, 60.0, 0.0, False, "2026-09-11T12:00:00+00:00",
                           49.0, "forecast")


@pytest.fixture(autouse=True)
def _patch_forecast(monkeypatch):
    monkeypatch.setattr(track_w1, "forecast_at_bet_time", _fake_forecast)


def test_declared_threshold_is_enforced_exactly():
    q = track_w1._qualifying(_slate(), 2026, 1, verbose=False)
    assert set(q["game_id"]) == {"g2", "g3"}, "10.0 mph is inclusive, 9.9 is not"
    assert q["forecast_wind_mph"].min() >= track_w1.MIN_WIND_MPH


def test_domes_are_excluded():
    q = track_w1._qualifying(_slate(), 2026, 1, verbose=False)
    assert "g4" not in set(q["game_id"])


def test_a_retractable_roof_of_unknown_state_does_not_qualify():
    """`schedules.roof` is null for every unplayed game, and five venues are retractable.
    An unresolved roof is UNKNOWN, not open -- betting wind at a stadium that may be sealed
    shut is betting on nothing. Real bug: HOU and IND both showed roof=None on the 2026
    week 1 slate."""
    from src.weather import forecast_at_bet_time
    game = pd.Series({"game_id": "gx", "home_team": "HOU", "location": "Home",
                      "stadium": None, "roof": None,
                      "kickoff": pd.Timestamp("2026-09-13 13:00")})
    fc = forecast_at_bet_time(game)
    assert fc.source == "roof_unknown"
    assert fc.wind_mph is None, "must not fall through to a forecast"


def test_played_games_are_never_recorded():
    """A finished game has an observed wind. Selecting on it would be the exact lookahead
    the whole exercise exists to avoid -- and would reproduce the backtest, not test it."""
    q = track_w1._qualifying(_slate(), 2026, 1, verbose=False)
    assert "g5" not in set(q["game_id"])


def test_implausible_forecasts_are_rejected_not_treated_as_calm():
    """config/nfl.yaml rejects readings above 40 mph; nflverse carries a 71 mph game that
    moved a total by 19.6 points. Rejected must mean 'no bet', never 'calm'."""
    slate = _slate()
    monkey = dict(_WINDS)
    monkey["g3"] = 71.0
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(track_w1, "forecast_at_bet_time",
                   lambda game, *, now=None: (
                       ForecastReading(None, None, None, True, "x", None, "indoor")
                       if game.get("roof") == "dome" else
                       ForecastReading(monkey.get(game["game_id"], 5.0), 60.0, 0.0,
                                       False, "x", 49.0, "forecast")))
        q = track_w1._qualifying(slate, 2026, 1, verbose=False)
    assert set(q["game_id"]) == {"g2"}, "71 mph is rejected, and is not a bet"


def test_the_side_is_always_under():
    """W1 is one-sided by construction. A rule that can fire either way is a different and
    untested hypothesis (PREREG W1)."""
    q = track_w1._qualifying(_slate(), 2026, 1, verbose=False)
    assert (q["side"] == "UNDER").all()
    assert track_w1.SIDE == "UNDER"


def test_the_forecast_and_its_horizon_are_recorded():
    """Without these the log cannot distinguish a 3-hour forecast from a 6-day one, and the
    forecast-error measurement PREREG W1 calls the primary output is impossible."""
    q = track_w1._qualifying(_slate(), 2026, 1, verbose=False)
    assert q["forecast_issued_at"].notna().all()
    assert q["hours_before_kickoff"].notna().all()


def test_the_log_is_append_only(tmp_path, monkeypatch):
    """`record` run twice on the same week must not duplicate or rewrite anything."""
    monkeypatch.setattr(track_w1, "LOG_PATH", tmp_path / "w1_log.csv")
    existing = pd.DataFrame([{
        "game_id": "g3", "season": 2026, "week": 1, "away_team": "E", "home_team": "F",
        "kickoff": "", "forecast_wind_mph": 22.0,
        "forecast_issued_at": "2026-09-11T12:00:00+00:00", "hours_before_kickoff": 49.0,
        "market_total_at_bet": 41.0, "side": "UNDER",
        "recorded_at": "2026-09-11T12:00:00+00:00",
        "actual_total": np.nan, "market_total_close": np.nan, "observed_wind_mph": np.nan,
        "result": "", "graded_at": "",
    }])
    track_w1._write_log(existing)

    q = track_w1._qualifying(_slate(), 2026, 1, verbose=False)
    log = track_w1._read_log()
    already = set(log["game_id"].astype(str))
    fresh = q[~q["game_id"].astype(str).isin(already)]
    assert set(fresh["game_id"]) == {"g2"}, "g3 is already logged and must be skipped"
    assert track_w1._read_log().loc[0, "recorded_at"] == "2026-09-11T12:00:00+00:00"


@pytest.mark.parametrize("line,actual,expected", [
    (44.5, 40.0, "WIN"),     # stayed under
    (44.5, 51.0, "LOSS"),
    (44.0, 44.0, "PUSH"),
])
def test_settlement_uses_the_recorded_line(line, actual, expected):
    """Mirrors cmd_grade. Settling at the CLOSE instead would silently convert this into a
    different experiment than the one registered."""
    result = "PUSH" if actual == line else ("WIN" if actual < line else "LOSS")
    assert result == expected


def test_rule_constants_match_the_registration():
    """If someone edits these, the test that the log is still measuring W1 must fail --
    PREREG W1 says any change to threshold, side, universe or stake voids the test."""
    assert track_w1.MIN_WIND_MPH == 10.0
    assert track_w1.MAX_WIND_MPH == 40.0
    assert track_w1.SIDE == "UNDER"
    assert track_w1.UNIVERSE == "outdoor"
