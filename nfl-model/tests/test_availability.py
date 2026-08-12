from __future__ import annotations

import pandas as pd
import pytest

from src.availability import (build_availability_features, matchup_availability,
                              read_manual_availability)


def test_position_weighting_uses_strictly_prior_snap_share():
    snaps = pd.DataFrame([
        {"season": 2026, "week": 1, "team": "A", "player": "Left Tackle",
         "offense_pct": .90, "defense_pct": 0},
        {"season": 2026, "week": 2, "team": "A", "player": "Left Tackle",
         "offense_pct": .50, "defense_pct": 0},
        {"season": 2026, "week": 1, "team": "A", "player": "Receiver",
         "offense_pct": .60, "defense_pct": 0},
        {"season": 2026, "week": 2, "team": "A", "player": "Receiver",
         "offense_pct": .20, "defense_pct": 0},
    ])
    reports = pd.DataFrame([
        {"season": 2026, "week": 2, "team": "A", "full_name": "Left Tackle",
         "position": "OT", "report_status": "Out", "source": "manual",
         "observed_at": pd.Timestamp("2026-09-15T18:00:00Z")},
        {"season": 2026, "week": 2, "team": "A", "full_name": "Receiver",
         "position": "WR", "report_status": "Questionable", "source": "manual",
         "observed_at": pd.Timestamp("2026-09-15T18:00:00Z")},
    ])
    row = build_availability_features(reports, snaps).iloc[0]
    assert row.offensive_line_burden == pytest.approx(.90)
    assert row.skill_burden == pytest.approx(.15)
    assert row.availability_confirmed


def test_missing_manual_file_is_unknown_not_healthy(tmp_path):
    assert read_manual_availability(tmp_path / "missing.csv").empty


def test_matchup_burden_is_home_minus_away():
    features = pd.DataFrame([
        {"season": 2026, "week": 2, "team": "A", "offensive_line_burden": .9,
         "skill_burden": .1, "quarterback_burden": 0, "defense_burden": .2},
        {"season": 2026, "week": 2, "team": "B", "offensive_line_burden": .3,
         "skill_burden": .2, "quarterback_burden": .5, "defense_burden": .1},
    ])
    schedule = pd.DataFrame([{"game_id": "g", "season": 2026, "week": 2,
                              "home_team": "A", "away_team": "B"}])
    row = matchup_availability(features, schedule).iloc[0]
    assert row.offensive_line_burden_diff == pytest.approx(.6)
    assert row.quarterback_burden_diff == pytest.approx(-.5)
