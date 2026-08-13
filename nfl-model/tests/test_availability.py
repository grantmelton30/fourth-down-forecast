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


def test_week_one_absence_uses_last_seasons_snap_share():
    """Within-season priors made Week 1 structurally blind.

    Snap share was averaged over strictly earlier weeks of the *same* season, so in
    Week 1 no player had an established share and every burden was exactly 0.0 --
    measured across all 283 historical Week 1 team-weeks, not merely sparse. A starter
    who took 95% of snaps last season must carry weight the moment he is ruled out,
    which is precisely when the model has no current-season evidence to fall back on.
    """
    snaps = pd.DataFrame([
        {"season": 2025, "week": w, "team": "A", "player": "Left Tackle",
         "offense_pct": .95, "defense_pct": 0} for w in (1, 2, 3)
    ])
    reports = pd.DataFrame([
        {"season": 2026, "week": 1, "team": "A", "full_name": "Left Tackle",
         "position": "OT", "report_status": "Out", "source": "manual",
         "observed_at": pd.Timestamp("2026-09-10T18:00:00Z")},
    ])
    row = build_availability_features(reports, snaps).iloc[0]
    assert row.offensive_line_burden == pytest.approx(.95)


def test_current_season_snaps_supersede_the_prior_season_carry_forward():
    """Once this season has evidence, it governs; the carry-forward is only a prior."""
    snaps = pd.DataFrame([
        {"season": 2025, "week": 1, "team": "A", "player": "Receiver",
         "offense_pct": .90, "defense_pct": 0},
        {"season": 2026, "week": 1, "team": "A", "player": "Receiver",
         "offense_pct": .20, "defense_pct": 0},
    ])
    reports = pd.DataFrame([
        {"season": 2026, "week": 2, "team": "A", "full_name": "Receiver",
         "position": "WR", "report_status": "Out", "source": "manual",
         "observed_at": pd.Timestamp("2026-09-17T18:00:00Z")},
    ])
    row = build_availability_features(reports, snaps).iloc[0]
    assert row.skill_burden == pytest.approx(.20)


def test_carry_forward_never_reaches_across_a_later_season():
    """A 2026 report must not borrow snap share recorded in 2027."""
    snaps = pd.DataFrame([
        {"season": 2027, "week": 1, "team": "A", "player": "Rookie",
         "offense_pct": .80, "defense_pct": 0},
    ])
    reports = pd.DataFrame([
        {"season": 2026, "week": 1, "team": "A", "full_name": "Rookie",
         "position": "WR", "report_status": "Out", "source": "manual",
         "observed_at": pd.Timestamp("2026-09-10T18:00:00Z")},
    ])
    row = build_availability_features(reports, snaps).iloc[0]
    assert row.skill_burden == pytest.approx(0.0)


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
