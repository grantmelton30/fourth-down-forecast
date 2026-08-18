"""Pass-only/rush-only EPA as a challenger feature (GATES.md 2026-08-17).

Tests a competitor's claimed methodology: that pooling pass and rush plays into one
`ppa_per_play` dilutes a strong passing signal with a weak (near-noise) rushing one.
`ingest.py::build_game_offense`'s `play_type` filter and `ratings.py::split_net_epa_matchup`
are the two new pieces; whether either signal actually promotes is a question for a real
rebuild (`run_backtest.py`), not a unit test -- this file only pins the arithmetic and
filtering, matching the split between `test_key_numbers_gate.py` (arithmetic) and
`pooled_margin_pmf`/`run_backtest.py` (the real read) elsewhere in this repo.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import ingest
from src.config import load_config
from src.ratings import net_epa_vec, split_net_epa_matchup


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """`build_game_offense` writes to its cache unconditionally -- redirect it to a
    throwaway directory so a synthetic fixture never overwrites the real, expensive
    game_offense*.parquet artifacts this repo's other tests and scripts depend on."""
    monkeypatch.setattr(ingest, "CACHE_DIR", tmp_path)


def _games():
    return pd.DataFrame([{
        "game_id": 1, "season": 2025, "week": 1,
        "kickoff": pd.Timestamp("2025-09-01", tz="UTC"),
        "neutralSite": False, "completed": True,
        "homeTeam": "A", "awayTeam": "B",
        "homeConference": "X", "awayConference": "Y",
        "homeClassification": "fbs", "awayClassification": "fbs",
        "homePoints": 30, "awayPoints": 20,
    }])


def _drives():
    return pd.DataFrame([
        {"game_id": 1, "offense": "A", "driveNumber": 1},
        {"game_id": 1, "offense": "A", "driveNumber": 2},
        {"game_id": 1, "offense": "B", "driveNumber": 1},
    ])


def _plays():
    """Team A: one pass (ppa=1.0), one rush (ppa=-1.0), one interception (ppa=-2.0, a
    turnover-outcome play excluded from both split views). Team B: one pass, one rush."""
    return pd.DataFrame([
        {"game_id": 1, "offense": "A", "defense": "B", "competitive": True,
         "ppa": 1.0, "playType": "Pass Reception"},
        {"game_id": 1, "offense": "A", "defense": "B", "competitive": True,
         "ppa": -1.0, "playType": "Rush"},
        {"game_id": 1, "offense": "A", "defense": "B", "competitive": True,
         "ppa": -2.0, "playType": "Interception"},
        {"game_id": 1, "offense": "B", "defense": "A", "competitive": True,
         "ppa": 0.5, "playType": "Passing Touchdown"},
        {"game_id": 1, "offense": "B", "defense": "A", "competitive": True,
         "ppa": 0.2, "playType": "Rushing Touchdown"},
    ])


def test_pooled_default_is_unchanged_by_the_new_parameter():
    cfg = load_config()
    pooled = ingest.build_game_offense(_plays(), _drives(), _games(), cfg)
    row = pooled[pooled["offense"] == "A"].iloc[0]
    assert row.ppa_per_play == pytest.approx((1.0 - 1.0 - 2.0) / 3)
    assert row.n_plays == 3


def test_pass_view_excludes_rush_and_turnover_plays():
    cfg = load_config()
    passed = ingest.build_game_offense(_plays(), _drives(), _games(), cfg, play_type="pass")
    row = passed[passed["offense"] == "A"].iloc[0]
    assert row.ppa_per_play == pytest.approx(1.0)
    assert row.n_plays == 1


def test_rush_view_excludes_pass_and_turnover_plays():
    cfg = load_config()
    rushed = ingest.build_game_offense(_plays(), _drives(), _games(), cfg, play_type="rush")
    row = rushed[rushed["offense"] == "A"].iloc[0]
    assert row.ppa_per_play == pytest.approx(-1.0)
    assert row.n_plays == 1


def test_touchdown_variants_are_classified_by_their_play_type():
    """'Passing Touchdown'/'Rushing Touchdown' must match the same substring classifier
    `features.py::build_team_game_features` uses for explosive-play detection, not fall
    into the turnover-only 'neither' bucket."""
    cfg = load_config()
    passed = ingest.build_game_offense(_plays(), _drives(), _games(), cfg, play_type="pass")
    rushed = ingest.build_game_offense(_plays(), _drives(), _games(), cfg, play_type="rush")
    assert passed[passed["offense"] == "B"].iloc[0].ppa_per_play == pytest.approx(0.5)
    assert rushed[rushed["offense"] == "B"].iloc[0].ppa_per_play == pytest.approx(0.2)


def test_unknown_play_type_raises():
    cfg = load_config()
    with pytest.raises(ValueError):
        ingest.build_game_offense(_plays(), _drives(), _games(), cfg, play_type="special")


def _walkforward(team_off_def: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2025, "week": 1, "team": team, "off_rating": o, "def_rating": d}
        for team, (o, d) in team_off_def.items()
    ])


def test_split_net_epa_matchup_matches_hand_computed_diff_and_sum():
    cfg = load_config()
    games = _games()  # A home vs B away
    wf_pass = _walkforward({"A": (0.30, -0.10), "B": (0.10, 0.05)})
    wf_rush = _walkforward({"A": (-0.05, 0.02), "B": (0.02, -0.02)})

    out = split_net_epa_matchup(wf_pass, wf_rush, games, cfg)
    row = out.iloc[0]

    pass_net_home = net_epa_vec(pd.Series([0.30]), pd.Series([0.05]), cfg).item()
    pass_net_away = net_epa_vec(pd.Series([0.10]), pd.Series([-0.10]), cfg).item()
    assert row.pass_net_epa_diff == pytest.approx(pass_net_home - pass_net_away)
    assert row.pass_net_epa_sum == pytest.approx(pass_net_home + pass_net_away)

    rush_net_home = net_epa_vec(pd.Series([-0.05]), pd.Series([-0.02]), cfg).item()
    rush_net_away = net_epa_vec(pd.Series([0.02]), pd.Series([0.02]), cfg).item()
    assert row.rush_net_epa_diff == pytest.approx(rush_net_home - rush_net_away)
    assert row.rush_net_epa_sum == pytest.approx(rush_net_home + rush_net_away)


def test_split_net_epa_matchup_output_is_challenger_ready():
    """Column names must end in `_diff`/`_sum` for `_validated_challenger`'s suffix-driven
    candidate detection (`src/backtest.py`) to pick them up with no extra wiring."""
    cfg = load_config()
    wf = _walkforward({"A": (0.0, 0.0), "B": (0.0, 0.0)})
    out = split_net_epa_matchup(wf, wf, _games(), cfg)
    assert list(out.columns) == [
        "game_id", "pass_net_epa_diff", "pass_net_epa_sum",
        "rush_net_epa_diff", "rush_net_epa_sum",
    ]
