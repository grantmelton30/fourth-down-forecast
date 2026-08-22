"""Walk-forward usage shares (prop-model/src/player_stats.py).

The whole prop model rests on "what share of his offence does this player get", and the one
way to make that number look good and be worthless is to let it see the game it is
predicting. These tests are mostly about that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.player_stats import attach_kickoffs, usage_shares


def _stats():
    """Two receivers on one team over three weeks. A's share climbs, B's falls, so any
    leakage of the current game into its own share is visible as a number that is too
    close to the truth."""
    rows = []
    for week, (a_tgt, b_tgt) in enumerate([(2, 8), (5, 5), (8, 2)], start=1):
        for pid, name, tgt in (("A", "Player A", a_tgt), ("B", "Player B", b_tgt)):
            rows.append({
                "player_id": pid, "player_display_name": name, "position": "WR",
                "team": "KC", "season": 2026, "week": week,
                "targets": tgt, "receptions": tgt - 1, "carries": 0, "attempts": 0,
            })
    return pd.DataFrame(rows)


def _schedule():
    return pd.DataFrame([
        {"season": 2026, "week": w, "home_team": "KC", "away_team": "DEN",
         "kickoff": pd.Timestamp(f"2026-09-{6 + 7 * w:02d} 13:00")}
        for w in (1, 2, 3)
    ])


def test_the_first_game_has_no_prior_and_says_so():
    """Week 1 must carry zero prior games. A model that quietly treats an unknown player as
    league-average is making a claim it has no evidence for."""
    u = usage_shares(attach_kickoffs(_stats(), _schedule()))
    wk1 = u[u["week"] == 1]
    assert (wk1["receptions_prior_games"] == 0).all()


def test_a_share_never_includes_its_own_game():
    """The load-bearing test. Player A's week-3 share must be built from weeks 1-2 only.
    His weeks 1-2 target shares are 0.2 and 0.5, so any prior mean at or above 0.5 means
    week 3 (0.8) leaked into its own prediction."""
    u = usage_shares(attach_kickoffs(_stats(), _schedule()), prior_games=0.0)
    a3 = u[(u["player_id"] == "A") & (u["week"] == 3)].iloc[0]
    assert a3["receptions_prior_games"] == 2
    assert a3["receptions_share"] == pytest.approx(0.35, abs=1e-9), (
        "expected mean(0.2, 0.5); anything higher means the current game leaked in")


def test_shrinkage_pulls_a_thin_sample_toward_the_team_mean():
    """A player with one game is not a 20%-share player because he was targeted twice once.
    With shrinkage on, his estimate must sit strictly between his own rate and the mean."""
    strong = usage_shares(attach_kickoffs(_stats(), _schedule()), prior_games=0.0)
    shrunk = usage_shares(attach_kickoffs(_stats(), _schedule()), prior_games=8.0)
    a2_strong = strong[(strong["player_id"] == "A") & (strong["week"] == 2)].iloc[0]
    a2_shrunk = shrunk[(shrunk["player_id"] == "A") & (shrunk["week"] == 2)].iloc[0]
    assert a2_shrunk["receptions_share"] != a2_strong["receptions_share"]
    assert abs(a2_shrunk["receptions_share"] - 0.5) < abs(a2_strong["receptions_share"] - 0.5)


def test_ordering_is_by_kickoff_not_by_week_number():
    """A postponed game kicks off after a later week's games. Ordering on the week LABEL
    would feed a game's result into a prediction made before it was played -- exactly what
    CLAUDE.md section 1 forbids, and invisible if you only ever look at week numbers."""
    sched = _schedule().copy()
    # Week 2 is postponed to after week 3.
    sched.loc[sched["week"] == 2, "kickoff"] = pd.Timestamp("2026-10-01 13:00")
    u = usage_shares(attach_kickoffs(_stats(), sched), prior_games=0.0)

    a_wk2 = u[(u["player_id"] == "A") & (u["week"] == 2)].iloc[0]
    # Chronologically week 2 is now LAST, so it must see weeks 1 and 3 (shares 0.2 and 0.8).
    assert a_wk2["receptions_prior_games"] == 2
    assert a_wk2["receptions_share"] == pytest.approx(0.5, abs=1e-9)


def test_usage_requires_kickoffs_and_refuses_to_guess():
    with pytest.raises(ValueError, match="attach_kickoffs"):
        usage_shares(_stats())


def test_a_player_week_with_no_scheduled_game_is_dropped_not_defaulted():
    """A stat line that matches no game cannot be placed in time. Keeping it with a guessed
    timestamp would silently reorder the walk-forward window."""
    stats = _stats()
    stats.loc[len(stats)] = {"player_id": "C", "player_display_name": "Player C",
                             "position": "WR", "team": "LAR", "season": 2026, "week": 1,
                             "targets": 5, "receptions": 4, "carries": 0, "attempts": 0}
    joined = attach_kickoffs(stats, _schedule())
    assert "C" not in set(joined["player_id"])
