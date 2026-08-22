"""Player identity resolution (prop-model/src/player_ids.py).

Identity is the foundation everything else sits on: if a player's usage history attaches to
the wrong key, every projection built on it is wrong in a way no downstream test can catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.player_ids import attach_player_key, coverage_report, normalize_name


def test_normalize_matches_the_team_model_exactly():
    """Deliberately identical to nfl-model/src/injuries.py::_norm. If these diverge, the two
    models silently disagree about who a player is, and nothing would flag it.

    Loaded BY PATH, not by import: both repos have a package called `src`, so a plain
    `from src.injuries import ...` resolves to prop-model's own and the test would compare a
    function to itself and always pass. `shared/sport.py::_load_model_package` exists for the
    same reason.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "nfl-model" / "src" / "injuries.py"
    spec = importlib.util.spec_from_file_location("_nfl_injuries", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for name in ["A.J. Brown", "Amon-Ra St. Brown", "Ja'Marr Chase", "  Josh  Allen  ",
                 "D.K. Metcalf", "Marvin Harrison Jr."]:
        assert normalize_name(name) == mod._norm(name), name


def test_player_key_is_scoped_to_team_and_season():
    """Two different players sharing a name must not collide, and one player must not carry
    usage history across a trade -- his share of a new offence is a different quantity."""
    frame = pd.DataFrame([
        {"season": 2026, "team": "PHI", "full_name": "A.J. Brown"},
        {"season": 2026, "team": "TEN", "full_name": "A.J. Brown"},
        {"season": 2025, "team": "PHI", "full_name": "A.J. Brown"},
    ])
    keys = attach_player_key(frame, name_col="full_name")["player_key"]
    assert keys.nunique() == 3, "team and season must both scope the key"


def test_punctuation_and_case_do_not_create_separate_players():
    frame = pd.DataFrame([
        {"season": 2026, "team": "DET", "full_name": "Amon-Ra St. Brown"},
        {"season": 2026, "team": "DET", "full_name": "amon ra st brown"},
        {"season": 2026, "team": "DET", "full_name": "AMONRA STBROWN"},
    ])
    keys = attach_player_key(frame, name_col="full_name")["player_key"]
    assert keys.nunique() == 1


def test_an_unusable_name_is_unresolved_not_silently_bucketed():
    """An empty or punctuation-only name must become NA. Normalising it to "" would put
    every such row in one shared bucket and pool unrelated players' usage."""
    frame = pd.DataFrame([
        {"season": 2026, "team": "KC", "full_name": ""},
        {"season": 2026, "team": "KC", "full_name": "..."},
        {"season": 2026, "team": "KC", "full_name": "Travis Kelce"},
    ])
    out = attach_player_key(frame, name_col="full_name")
    assert out["player_key"].isna().sum() == 2
    assert (out["player_key_source"] == "unresolved").sum() == 2


def test_coverage_report_states_the_mix_rather_than_assuming_it():
    frame = pd.DataFrame([
        {"season": 2026, "team": "KC", "full_name": "Travis Kelce"},
        {"season": 2026, "team": "KC", "full_name": ""},
    ])
    rep = coverage_report(attach_player_key(frame, name_col="full_name"))
    assert rep["n"] == 2
    assert rep["resolved_pct"] == 50.0
    assert "unresolved" in rep["by_source"]


def test_coverage_report_refuses_a_frame_it_has_not_seen():
    with pytest.raises(ValueError):
        coverage_report(pd.DataFrame({"x": [1]}))


@pytest.mark.integration
def test_the_crosswalk_is_not_used_as_a_replacement_for_the_name_join():
    """Measured 2026-08-21 on 2,448 Out/Doubtful player-weeks: name join 93.0%, crosswalk
    alone 80.5%, union 96.3%. The crosswalk carries gsis_id on only 64% of its rows, so
    promoting it to primary would lose 387 rows to gain 81. This test pins the ORDERING --
    the name must remain primary -- because the intuitive change is the wrong one."""
    frame = pd.DataFrame([{"season": 2026, "team": "KC", "full_name": "Travis Kelce",
                           "gsis_id": "00-0030506"}])
    out = attach_player_key(frame, name_col="full_name", gsis_col="gsis_id")
    assert out.loc[0, "player_key_source"] == "name", (
        "a usable name must win; the crosswalk is a fallback, not the primary key")
