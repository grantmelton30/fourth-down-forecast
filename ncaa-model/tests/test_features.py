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


def test_preseason_poll_points_covers_both_fbs_polls_and_the_fcs_poll():
    """FCS Coaches Poll added 2026-08-17, once FCS teams got real ratings and a reason
    for this to matter. Safe to average alongside the two FBS polls with no
    cross-contamination guard: a team is never listed in both an FBS and the FCS poll by
    construction of the source data."""
    rankings = pd.DataFrame([{
        "season": 2026,
        "polls": [
            {"poll": "AP Top 25", "ranks": [{"school": "Georgia", "points": 1550}]},
            {"poll": "Coaches Poll", "ranks": [{"school": "Georgia", "points": 1600}]},
            {"poll": "FCS Coaches Poll",
             "ranks": [{"school": "North Dakota State", "points": 700}]},
        ],
    }])
    universe = pd.DataFrame([
        {"season": 2026, "team": "Georgia"},
        {"season": 2026, "team": "North Dakota State"},
        {"season": 2026, "team": "Some Unranked Team"},
    ])
    out = normalize_preseason_sources(rankings=rankings, recruiting=universe.assign(
        points=100.0))

    by_team = out.set_index("team")["preseason_poll_points"]
    assert by_team["Georgia"] == pytest.approx(1575.0)
    assert by_team["North Dakota State"] == pytest.approx(700.0)
    assert by_team["Some Unranked Team"] == 0.0


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


def _trench_plays():
    """Game 1: a sack, a stuffed run, a positive run, and a clean completion -- enough to
    hand-compute both trench rates and confirm each is masked to its own play-type context
    rather than diluted by the other. Game 2 gives team A a second appearance so the
    strictly-prior shift in rolling_team_features has something to carry forward, mirroring
    _plays()'s two-game shape above."""
    return pd.DataFrame([
        {"game_id": 1, "season": 2025, "week": 1, "offense": "A", "defense": "B",
         "playType": "Sack", "yardsGained": -7, "yardsToGoal": 60,
         "ppa": -.9, "driveId": 1, "down": 2, "garbage": False},
        {"game_id": 1, "season": 2025, "week": 1, "offense": "A", "defense": "B",
         "playType": "Pass Reception", "yardsGained": 9, "yardsToGoal": 53,
         "ppa": .3, "driveId": 1, "down": 3, "garbage": False},
        {"game_id": 1, "season": 2025, "week": 1, "offense": "A", "defense": "B",
         "playType": "Rush", "yardsGained": -2, "yardsToGoal": 53,
         "ppa": -.4, "driveId": 1, "down": 1, "garbage": False},
        {"game_id": 1, "season": 2025, "week": 1, "offense": "A", "defense": "B",
         "playType": "Rush", "yardsGained": 5, "yardsToGoal": 55,
         "ppa": .2, "driveId": 1, "down": 2, "garbage": False},
        {"game_id": 2, "season": 2025, "week": 2, "offense": "A", "defense": "C",
         "playType": "Pass Reception", "yardsGained": 4, "yardsToGoal": 40,
         "ppa": .1, "driveId": 2, "down": 1, "garbage": False},
    ])


def test_pass_block_and_run_block_success_are_hand_computed_and_denominator_masked():
    game = build_team_game_features(_trench_plays())
    row = game[game.game_id.eq(1)].iloc[0]
    # pass_block_success: 1 sack out of 2 dropbacks (Sack + Pass Reception) -- not diluted
    # by the two Rush plays.
    assert row.pass_block_success == pytest.approx(0.5)
    # run_block_success: 1 non-stuffed run out of 2 rush attempts -- not diluted by the
    # dropback plays.
    assert row.run_block_success == pytest.approx(0.5)


def test_trench_matchup_columns_are_produced_and_strictly_prior():
    games = pd.DataFrame([
        {"game_id": 1, "season": 2025, "week": 1, "homeTeam": "A", "awayTeam": "B"},
        {"game_id": 2, "season": 2025, "week": 2, "homeTeam": "C", "awayTeam": "A"},
    ])
    out = matchup_feature_table(_trench_plays(), games)
    assert {"pass_block_success_matchup_diff", "pass_block_success_matchup_sum",
            "run_block_success_matchup_diff", "run_block_success_matchup_sum"} <= set(out)
    # Week 2: A's week-1 rates (0.5/0.5) should carry forward as its own prior form, the
    # same shift-by-one contract test_matchup_feature_is_strictly_prior checks for
    # success_rate above -- game 2's own plays must not leak into this value.
    row = out.query("game_id == 2").iloc[0]
    assert row.away_pass_block_success == pytest.approx(0.5)
    assert row.away_run_block_success == pytest.approx(0.5)


def test_optional_preseason_endpoint_rate_limit_stays_missing_instead_of_blocking_refresh():
    class Client:
        def frame(self, endpoint, cache_key, **params):
            if endpoint == "coaches":
                raise APIBudgetExceeded("free tier rate limited")
            return pd.DataFrame([{"year": params["year"], "team": "A"}])

    sources = load_free_preseason(Client(), [2026])
    assert len(sources["returning"]) == 1
    assert sources["coaching"].empty


def _coaches_payload():
    """CFBD's real shape: team and year live inside a nested `seasons` array.

    Nothing identifies a team at the top level, so a normalizer looking for flat
    `team`/`school` columns finds nothing and yields all-NaN. That is why head-coach
    continuity has never once been computed despite the data downloading correctly --
    138 coaches for 2026 sat on disk and were discarded every run.
    """
    return pd.DataFrame([
        {"firstName": "Ada", "lastName": "Lovelace", "hireDate": "2023-12-01",
         "seasons": [{"school": "A", "year": 2025}, {"school": "A", "year": 2026}]},
        {"firstName": "Alan", "lastName": "Turing", "hireDate": "2025-12-10",
         "seasons": [{"school": "B", "year": 2026}]},
        {"firstName": "Grace", "lastName": "Hopper", "hireDate": "2019-01-01",
         "seasons": [{"school": "B", "year": 2025}]},
    ])


def test_head_coach_continuity_is_read_from_the_nested_coaches_schema():
    out = normalize_preseason_sources(coaching=_coaches_payload())
    by_team = out[out["season"] == 2026].set_index("team")["head_coach_continuity"]

    # A kept her job from 2025 into 2026; B replaced Hopper with Turing.
    assert by_team.loc["A"] == 1.0
    assert by_team.loc["B"] == 0.0


def test_a_first_observed_season_has_unknown_continuity_not_zero():
    """With no prior year on record we do not know whether the coach changed."""
    out = normalize_preseason_sources(coaching=_coaches_payload())
    first = out[(out["season"] == 2025) & (out["team"] == "A")]
    assert first["head_coach_continuity"].isna().all()


def test_coaching_features_without_a_free_source_stay_absent():
    """CFBD's coaches endpoint returns head coaches only.

    Coordinator continuity has no free source, so it must remain missing rather than
    being silently defaulted -- a fabricated 'coordinators unchanged' would be worse
    than an honest gap.
    """
    out = normalize_preseason_sources(coaching=_coaches_payload())
    for column in ("offensive_coordinator_continuity",
                   "defensive_coordinator_continuity"):
        assert column in out
        assert out[column].isna().all()


def test_preseason_normalization_drops_rows_without_team_keys():
    returning = pd.DataFrame([{"year": 2026, "team": "A", "percentPPA": .60}])
    changed_portal_schema = pd.DataFrame([{"season": 2026, "unexpected": "value"}])
    changed_talent_schema = pd.DataFrame([{"year": 2026, "talent": 800.0}])
    out = normalize_preseason_sources(
        returning=returning, portal=changed_portal_schema,
        talent=changed_talent_schema,
    )
    assert out["team"].tolist() == ["A"]
    assert out.iloc[0]["returning_production"] == pytest.approx(.60)
    assert pd.isna(out.iloc[0]["portal_net_rating"])
