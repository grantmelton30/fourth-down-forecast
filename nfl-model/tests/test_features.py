from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_team_game_features, matchup_feature_table, rolling_team_features


def _pbp():
    return pd.DataFrame([
        dict(game_id="g1", season=2025, week=1, posteam="A", defteam="B", down=1,
             play_type="pass", epa=.5, success=1, yards_gained=25, qb_dropback=1,
             sack=0, interception=0, fumble_lost=0, yardline_100=80, touchdown=0,
             special=0, competitive=True),
        dict(game_id="g1", season=2025, week=1, posteam="A", defteam="B", down=2,
             play_type="run", epa=-.2, success=0, yards_gained=4, qb_dropback=0,
             sack=0, interception=0, fumble_lost=0, yardline_100=15, touchdown=1,
             special=0, competitive=True),
        dict(game_id="g1", season=2025, week=1, posteam="B", defteam="A", down=1,
             play_type="pass", epa=-1.0, success=0, yards_gained=-7, qb_dropback=1,
             sack=1, interception=0, fumble_lost=0, yardline_100=75, touchdown=0,
             special=0, competitive=True),
        dict(game_id="g2", season=2025, week=2, posteam="A", defteam="C", down=1,
             play_type="pass", epa=1.0, success=1, yards_gained=30, qb_dropback=1,
             sack=0, interception=0, fumble_lost=0, yardline_100=70, touchdown=0,
             special=0, competitive=True, fixed_drive=1),
    ])


def test_team_game_features_include_requested_signal_families():
    out = build_team_game_features(_pbp())
    a = out[(out.game_id == "g1") & (out.team == "A")].iloc[0]
    assert a.early_down_pass_epa == pytest.approx(.5)
    assert a.early_down_rush_epa == pytest.approx(-.2)
    assert a.success_rate == pytest.approx(.5)
    assert a.explosive_rate == pytest.approx(.5)
    assert a.sack_avoidance == pytest.approx(1.0)
    assert a.red_zone_td_rate == pytest.approx(1.0)
    assert a.turnover_opportunity_rate == pytest.approx(0.0)


def test_rolling_features_are_strictly_prior_to_current_game():
    game = build_team_game_features(_pbp())
    rolling = rolling_team_features(game, half_life_games=6)
    current = rolling[(rolling.game_id == "g2") & (rolling.team == "A")].iloc[0]
    assert current.early_down_pass_epa == pytest.approx(.5)
    assert current.games_observed == 1


def test_first_game_has_no_fabricated_history():
    rolling = rolling_team_features(build_team_game_features(_pbp()))
    first = rolling[(rolling.game_id == "g1") & (rolling.team == "A")].iloc[0]
    assert np.isnan(first.early_down_pass_epa)
    assert first.games_observed == 0


def test_matchup_table_uses_strictly_prior_team_form():
    schedule = pd.DataFrame([
        {"game_id": "g1", "season": 2025, "week": 1, "home_team": "A", "away_team": "B"},
        {"game_id": "g2", "season": 2025, "week": 2, "home_team": "A", "away_team": "C"},
    ])
    out = matchup_feature_table(_pbp(), schedule)
    g2 = out[out.game_id == "g2"].iloc[0]
    assert g2.home_early_down_pass_epa == pytest.approx(.5)


def test_starting_field_position_uses_drive_starts_not_all_plays():
    plays = _pbp().copy()
    plays["fixed_drive"] = [1, 1, 2, 1]
    out = build_team_game_features(plays)
    a = out[(out.game_id == "g1") & (out.team == "A")].iloc[0]
    assert a.starting_field_position == pytest.approx(80.0)


def test_matchup_features_include_prior_opponent_allowed_form():
    plays = _pbp().copy()
    plays["fixed_drive"] = [1, 1, 1, 1]
    schedule = pd.DataFrame([
        {"game_id": "g1", "season": 2025, "week": 1, "home_team": "A", "away_team": "B"},
        {"game_id": "g2", "season": 2025, "week": 2, "home_team": "A", "away_team": "C"},
    ])
    out = matchup_feature_table(plays, schedule)
    g2 = out[out.game_id == "g2"].iloc[0]
    # C's only prior defensive game is g2 itself, so strict shifting leaves it unknown.
    assert np.isnan(g2.away_allowed_early_down_pass_epa)
