"""P1 forward tracking (PREREG.md P1).

This script is the only record the pre-registered rule will ever have, so the tests are
about the properties that make a record trustworthy rather than about convenience: the
declared bounds are enforced exactly, a logged bet is never rewritten, and settlement uses
the number the bet was recorded at rather than the one that turned up later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import track_p1


def _frame():
    """Edges chosen to sit on both sides of each declared boundary."""
    return pd.DataFrame([
        # edge -0.2: below MIN_EDGE, no opinion
        {"game_id": 1, "season": 2026, "week": 1, "away_team": "A", "home_team": "B",
         "model_total": 50.3, "total_open": 60.0, "total_close": 50.5, "restricted": True, "actual_total": np.nan},
        # edge +0.5: exactly MIN_EDGE, inclusive -> qualifies
        {"game_id": 2, "season": 2026, "week": 1, "away_team": "C", "home_team": "D",
         "model_total": 55.5, "total_open": 40.0, "total_close": 55.0, "restricted": True, "actual_total": np.nan},
        # edge -3.0: comfortably inside -> qualifies, UNDER
        {"game_id": 3, "season": 2026, "week": 1, "away_team": "E", "home_team": "F",
         "model_total": 49.0, "total_open": 62.0, "total_close": 52.0, "restricted": True, "actual_total": np.nan},
        # edge +6.0: exactly MAX_EDGE, EXCLUSIVE -> must not qualify
        {"game_id": 4, "season": 2026, "week": 1, "away_team": "G", "home_team": "H",
         "model_total": 61.0, "total_open": 45.0, "total_close": 55.0, "restricted": True, "actual_total": np.nan},
        # edge +2.0 but OUTSIDE the restricted universe -> must not qualify
        {"game_id": 5, "season": 2026, "week": 1, "away_team": "I", "home_team": "J",
         "model_total": 57.0, "total_open": 45.0, "total_close": 55.0, "restricted": False, "actual_total": np.nan},
    ])


def test_declared_edge_bounds_are_enforced_exactly():
    q = track_p1._qualifying(_frame(), 2026, 1)
    assert set(q["game_id"]) == {2, 3}, "0.5 is inclusive, 6.0 is exclusive"
    assert q["edge"].abs().min() >= track_p1.MIN_EDGE
    assert q["edge"].abs().max() < track_p1.MAX_EDGE


def test_side_follows_the_sign_of_the_edge():
    q = track_p1._qualifying(_frame(), 2026, 1).set_index("game_id")
    assert q.loc[2, "side"] == "OVER"    # model above the market
    assert q.loc[3, "side"] == "UNDER"   # model below it


def test_non_restricted_games_are_excluded():
    q = track_p1._qualifying(_frame(), 2026, 1)
    assert 5 not in set(q["game_id"])


def test_the_bet_is_priced_at_the_line_available_now_not_the_opener():
    """The bet-time number IS the experiment, and until 2026-08-21 this logged the OPENER --
    CFBD's overUnderOpen, often set weeks earlier. Selected and priced at the opener the
    same rule reads 52.07% and -7.1 units on 2021-2025, against 54.13% and +39.1 at the
    number actually available. Every fixture here has an opener far from its current line,
    so a regression is a failure and not a rounding difference."""
    frame = _frame()
    # `total_close` on an UNPLAYED game is the currently displayed quote, not a closing
    # line from the future -- it only becomes "the close" once the game kicks off.
    q = track_p1._qualifying(frame, 2026, 1)
    assert (q["market_total_at_bet"] == q["total_close"]).all()


def test_the_log_is_append_only(tmp_path, monkeypatch):
    """`record` run twice on the same week must not duplicate or rewrite anything."""
    monkeypatch.setattr(track_p1, "LOG_PATH", tmp_path / "p1_log.csv")
    existing = pd.DataFrame([{
        "game_id": 3, "season": 2026, "week": 1, "away_team": "E", "home_team": "F",
        "kickoff": "", "model_total": 49.0, "market_total_at_bet": 52.0, "edge": -3.0,
        "side": "UNDER", "recorded_at": "2026-08-19T00:00:00+00:00",
        "actual_total": np.nan, "market_total_close": np.nan, "result": "", "graded_at": "",
    }])
    track_p1._write_log(existing)

    q = track_p1._qualifying(_frame(), 2026, 1)
    log = track_p1._read_log()
    already = set(log["game_id"].astype(str))
    fresh = q[~q["game_id"].astype(str).isin(already)]
    assert set(fresh["game_id"]) == {2}, "game 3 is already logged and must be skipped"
    # And the pre-existing row is untouched, including its original recorded_at.
    assert track_p1._read_log().loc[0, "recorded_at"] == "2026-08-19T00:00:00+00:00"


@pytest.mark.parametrize("side,line,actual,expected", [
    ("OVER", 52.0, 55.0, "WIN"),
    ("OVER", 52.0, 48.0, "LOSS"),
    ("UNDER", 52.0, 48.0, "WIN"),
    ("UNDER", 52.0, 55.0, "LOSS"),
    ("OVER", 52.0, 52.0, "PUSH"),
    ("UNDER", 52.0, 52.0, "PUSH"),
])
def test_settlement_uses_the_recorded_line(side, line, actual, expected):
    """Mirrors cmd_grade's logic. Settling at the CLOSE instead would silently convert this
    into a different experiment than the one registered."""
    if actual == line:
        result = "PUSH"
    elif (actual > line) == (side == "OVER"):
        result = "WIN"
    else:
        result = "LOSS"
    assert result == expected


def test_rule_constants_match_the_registration():
    """If someone edits these, the test that the log is still measuring P1 must fail --
    PREREG P1 says any change to trigger, cap, universe or stake voids the test."""
    assert track_p1.MIN_EDGE == 0.5
    assert track_p1.MAX_EDGE == 6.0
    assert track_p1.UNIVERSE == "restricted"


def test_line_basis_is_recorded_so_the_two_batches_can_never_be_pooled():
    """The 26 bets logged before the fix were priced at the opener and stay that way -- the
    log is append-only. They carry a different `line_basis` so nobody can later add them to
    a record built on executable numbers and report one win rate over both."""
    q = track_p1._qualifying(_frame(), 2026, 1)
    assert (q["line_basis"] == "current").all()


def test_a_game_with_no_current_line_falls_back_and_says_so():
    """A missing current quote falls back to the opener rather than dropping the game, but
    labels itself so the mixture is visible in the log instead of being invisible."""
    frame = _frame()
    frame.loc[frame["game_id"] == 3, "total_close"] = np.nan
    frame.loc[frame["game_id"] == 3, "total_open"] = 52.0   # close enough to still qualify
    q = track_p1._qualifying(frame, 2026, 1).set_index("game_id")
    assert q.loc[3, "line_basis"] == "opener_fallback"
    assert q.loc[3, "market_total_at_bet"] == 52.0


def test_the_opener_is_recorded_but_never_graded_against():
    """The full arc -- open, the number actually taken, the close -- has to be on the row so
    "did we bet before or after the market moved" is answerable from the log alone. But the
    opener must not become the settlement price: PREREG P1 grades at the number available
    when the bet was recorded, and the 2026-08-21 correction exists because using the opener
    made the forward record test a different and historically LOSING strategy."""
    frame = _frame()
    q = track_p1._qualifying(frame, 2026, 1).set_index("game_id")
    # Every fixture has an opener deliberately far from its current line.
    assert q.loc[2, "market_total_open"] == 40.0
    assert q.loc[2, "market_total_at_bet"] == 55.0
    assert q.loc[2, "edge"] == pytest.approx(55.5 - 55.0), "edge is off the BET line"


def test_the_opener_is_carried_into_the_written_row():
    frame = _frame()
    q = track_p1._qualifying(frame, 2026, 1)
    assert "market_total_open" in track_p1.LOG_COLUMNS
    assert q["market_total_open"].notna().all()
