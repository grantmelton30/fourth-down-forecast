from __future__ import annotations

import pandas as pd
import pytest

from src.cfbd_client import APIBudgetExceeded
from src.features import (build_team_game_features, game_uncertainty_multiplier,
                          load_free_preseason,
                          matchup_feature_table, normalize_preseason_sources,
                          rolling_team_features)


def _plays():
    return pd.DataFrame([
        {"game_id": 1, "season": 2025, "week": 1, "offense": "A", "defense": "B",
         "playType": "Pass Reception", "yardsGained": 22, "yardsToGoal": 80,
         "ppa": .4, "driveId": 1, "down": 1, "garbage": False},
        {"game_id": 1, "season": 2025, "week": 1, "offense": "A", "defense": "B",
         "playType": "Rushing Touchdown", "yardsGained": 8, "yardsToGoal": 8,
         "ppa": .7, "driveId": 1, "down": 2, "garbage": False},
        {"game_id": 2, "season": 2025, "week": 2, "offense": "A", "defense": "C",
         "playType": "Sack", "yardsGained": -5, "yardsToGoal": 70,
         "ppa": -.8, "driveId": 2, "down": 1, "garbage": False},
    ])


def test_play_features_and_prior_shift():
    game = build_team_game_features(_plays())
    first = game[game.game_id.eq(1)].iloc[0]
    assert first.success_rate == 1.0
    assert first.explosive_rate == pytest.approx(.5)
    assert first.starting_field_position == 80
    current = rolling_team_features(game)[lambda d: d.game_id.eq(2)].iloc[0]
    assert current.success_rate == 1.0


def test_preseason_sources_preserve_missing_and_aggregate_portal():
    returning = pd.DataFrame([{"year": 2026, "team": "A", "percentPPA": .62,
                               "percentPassingPPA": .70}])
    portal = pd.DataFrame([
        {"season": 2026, "origin": "B", "destination": "A", "rating": .91},
        {"season": 2026, "origin": "A", "destination": "C", "rating": .80},
    ])
    talent = pd.DataFrame([{"year": 2026, "school": "A", "talent": 850.0}])
    out = normalize_preseason_sources(returning=returning, portal=portal, talent=talent)
    a = out[out.team.eq("A")].iloc[0]
    assert a.returning_production == pytest.approx(.62)
    assert a.qb_continuity == pytest.approx(.70)
    assert a.portal_net_rating == pytest.approx(.11)
    assert pd.isna(a.recruiting_rating)


def test_uncertainty_widens_early_and_cross_tier():
    baseline = game_uncertainty_multiplier(8, preseason_available=True, cross_tier=False)
    uncertain = game_uncertainty_multiplier(1, preseason_available=False, cross_tier=True)
    assert baseline == 1.0
    assert uncertain > 1.5


def test_significant_turnover_increases_uncertainty_without_inventing_a_point_penalty():
    stable = game_uncertainty_multiplier(
        1, preseason_available=True, cross_tier=False,
        returning_production=.82, qb_continuity=.90,
    )
    high_turnover = game_uncertainty_multiplier(
        1, preseason_available=True, cross_tier=False,
        returning_production=.28, qb_continuity=.10,
    )
    assert high_turnover > stable
    assert high_turnover - stable >= .15
    mature_stable = game_uncertainty_multiplier(
        8, preseason_available=True, cross_tier=False,
        returning_production=.82, qb_continuity=.90,
    )
    mature_turnover = game_uncertainty_multiplier(
        8, preseason_available=True, cross_tier=False,
        returning_production=.28, qb_continuity=.10,
    )
    assert mature_turnover == mature_stable
    assert game_uncertainty_multiplier(
        8, preseason_available=False, cross_tier=False,
    ) == 1.0


def test_matchup_feature_is_strictly_prior():
    games = pd.DataFrame([
        {"game_id": 1, "season": 2025, "week": 1, "homeTeam": "A", "awayTeam": "B"},
        {"game_id": 2, "season": 2025, "week": 2, "homeTeam": "A", "awayTeam": "C"},
    ])
    row = matchup_feature_table(_plays(), games).query("game_id == 2").iloc[0]
    assert row.home_success_rate == 1.0


def test_optional_preseason_endpoint_rate_limit_stays_missing_instead_of_blocking_refresh():
    class Client:
        def frame(self, endpoint, cache_key, **params):
            if endpoint == "coaches":
                raise APIBudgetExceeded("free tier rate limited")
            return pd.DataFrame([{"year": params["year"], "team": "A"}])

    sources = load_free_preseason(Client(), [2026])
    assert len(sources["returning"]) == 1
    assert sources["coaching"].empty
