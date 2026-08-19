"""Quarterback adjustment (DECISIONS.md D17).

The properties pinned here are the ones that make the feature safe to ship rather than the
ones that make it look good: an incumbent who plays produces exactly zero, an absence is
read only from a full zero-attempt line, the replacement is never sourced from the game
being predicted, and a missing reading degrades to today's behaviour instead of quietly
moving a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.qb import (apply_manual_status, build_passer_games, game_qb_delta, qb_points,
                    qb_points_for_game, qb_ratings_table)


def _passers():
    """Team A: `starter` throws weeks 1-2 well, `backup` mops up. Week 3 the starter is
    absent entirely -- the announced case this feature exists for."""
    return pd.DataFrame([
        {"game_id": 1, "season": 2025, "week": 1, "team": "A", "qb_id": "starter",
         "qb_name": "Starter", "attempts": 30, "ppa": 0.40},
        {"game_id": 1, "season": 2025, "week": 1, "team": "A", "qb_id": "backup",
         "qb_name": "Backup", "attempts": 3, "ppa": -0.20},
        {"game_id": 2, "season": 2025, "week": 2, "team": "A", "qb_id": "starter",
         "qb_name": "Starter", "attempts": 30, "ppa": 0.40},
        {"game_id": 2, "season": 2025, "week": 2, "team": "A", "qb_id": "backup",
         "qb_name": "Backup", "attempts": 3, "ppa": -0.20},
        {"game_id": 3, "season": 2025, "week": 3, "team": "A", "qb_id": "starter",
         "qb_name": "Starter", "attempts": 0, "ppa": None},
        {"game_id": 3, "season": 2025, "week": 3, "team": "A", "qb_id": "backup",
         "qb_name": "Backup", "attempts": 28, "ppa": -0.10},
    ])


def test_incumbent_is_the_passer_with_the_most_prior_work_not_this_games():
    cfg = load_config()
    t = qb_ratings_table(_passers(), cfg).set_index("game_id")
    # Week 3: the backup threw every pass, but the INCUMBENT is still the starter --
    # that is the whole point, since the team rating describes the starter.
    assert t.loc[3, "incumbent_id"] == "starter"
    assert t.loc[3, "replacement_id"] == "backup"


def test_no_adjustment_when_the_incumbent_actually_plays():
    cfg = load_config()
    t = qb_ratings_table(_passers(), cfg).set_index("game_id")
    assert t.loc[2, "incumbent_absent"] == False  # noqa: E712
    assert t.loc[2, "qb_delta"] == pytest.approx(0.0)


def test_absence_produces_a_negative_delta_when_the_backup_is_worse():
    cfg = load_config()
    t = qb_ratings_table(_passers(), cfg).set_index("game_id")
    assert bool(t.loc[3, "incumbent_absent"]) is True
    # Backup rated below the starter, so replacing him must lower the team's expectation.
    assert t.loc[3, "qb_delta"] < 0.0


def test_absence_is_detected_when_the_starter_has_NO_ROW_at_all():
    """REGRESSION, 2026-08-19. This is the shape real CFBD data actually has.

    `games/players` only returns players who appeared, so an absent quarterback is not a
    zero-attempt row -- he is missing from the game entirely. The first version of
    `qb_ratings_table` ranked passers within the rows a game happened to contain, so an
    absent starter could never be seen, and the feature was a silent no-op across all
    11,441 team-games. Every other test in this file passed throughout, because the
    fixtures wrote an explicit `attempts: 0` row that real data never contains.
    """
    cfg = load_config()
    p = _passers()
    p = p[~((p.game_id == 3) & (p.qb_id == "starter"))]  # he is simply not in the box score
    t = qb_ratings_table(p, cfg).set_index("game_id")
    assert t.loc[3, "incumbent_id"] == "starter", "incumbent must come from PRIOR games"
    assert bool(t.loc[3, "incumbent_absent"]) is True
    assert t.loc[3, "qb_delta"] < 0.0


def test_a_partial_workload_is_not_treated_as_an_absence():
    """The -6.46 point mid-game group. Real, far larger, and unknowable before kickoff, so
    it must never fire this adjustment."""
    cfg = load_config()
    p = _passers()
    p.loc[(p.game_id == 3) & (p.qb_id == "starter"), "attempts"] = 5  # played, then pulled
    t = qb_ratings_table(p, cfg).set_index("game_id")
    assert bool(t.loc[3, "incumbent_absent"]) is False
    assert t.loc[3, "qb_delta"] == pytest.approx(0.0)


def test_ratings_are_strictly_prior():
    """Week 1 carries no prior workload for anyone, so nobody has an established rating and
    no adjustment can be produced from a team's first game."""
    cfg = load_config()
    t = qb_ratings_table(_passers(), cfg).set_index("game_id")
    assert t.loc[1, "incumbent_rating"] == pytest.approx(0.0)
    assert t.loc[1, "qb_delta"] == pytest.approx(0.0)


def _league(flash_attempts: int) -> pd.DataFrame:
    """A league of ordinary passers plus one hot hand, so the centring has something to
    centre against. With a single passer in the frame he IS the league mean and every
    rating is correctly zero -- which is right, but tests nothing."""
    rows = []
    for wk, gid in ((1, 1), (2, 2)):
        for team in ("C", "D", "E"):
            rows.append({"game_id": gid, "season": 2025, "week": wk, "team": team,
                         "qb_id": f"avg_{team}", "qb_name": "Avg", "attempts": 30,
                         "ppa": 0.0})
        rows.append({"game_id": gid, "season": 2025, "week": wk, "team": "B",
                     "qb_id": "flash", "qb_name": "Flash",
                     "attempts": flash_attempts, "ppa": 1.5})
    return pd.DataFrame(rows)


def test_shrinkage_pulls_a_small_sample_toward_average():
    """A passer with a handful of attempts must not be rated like a season-long starter."""
    cfg = load_config()
    small = qb_ratings_table(_league(4), cfg)
    big = qb_ratings_table(_league(300), cfg)
    r_small = small[(small.game_id == 2) & (small.team == "B")]["incumbent_rating"].iloc[0]
    r_big = big[(big.game_id == 2) & (big.team == "B")]["incumbent_rating"].iloc[0]
    assert r_small > 0.0          # genuinely above league average
    assert abs(r_small) < abs(r_big)   # but credited far less of it on 4 attempts


def test_points_are_capped_and_missing_deltas_are_zero():
    cfg = load_config()
    pts = qb_points([0.0, -1.0, 1.0, np.nan], cfg)
    assert pts[0] == pytest.approx(0.0)
    # delta is the binary starter-out indicator: -1 for the side missing its starter.
    assert pts[1] == pytest.approx(-cfg.qb.points_per_starter_out)
    assert pts[2] == pytest.approx(cfg.qb.points_per_starter_out)
    # And the cap still binds if a delta ever arrives out of range.
    assert qb_points([-99.0], cfg)[0] == pytest.approx(-cfg.qb.max_adjustment_points)
    # A missing reading must be neutral, never a penalty -- this is what makes a failed
    # ESPN pull degrade to today's behaviour instead of moving a line.
    assert pts[3] == pytest.approx(0.0)


def test_disabled_config_is_a_hard_off_switch():
    cfg = load_config()
    off = type(cfg)(**{**cfg.__dict__, "qb": type(cfg.qb)(
        **{**cfg.qb.__dict__, "enabled": False})})
    assert qb_points([-1.0, 1.0], off).tolist() == [0.0, 0.0]


def test_game_delta_is_home_minus_away():
    table = pd.DataFrame([
        {"game_id": 9, "team": "A", "qb_delta": -1.0},
        {"game_id": 9, "team": "B", "qb_delta": 0.0},
    ])
    games = pd.DataFrame([{"game_id": 9, "homeTeam": "A", "awayTeam": "B"}])
    out = game_qb_delta(table, games)
    assert out["qb_delta_gap"].iloc[0] == pytest.approx(-1.0)


def test_manual_status_overrides_the_box_score():
    """The live path has no box score to read an absence from, so a human (or an ESPN pull)
    supplies it and must win."""
    cfg = load_config()
    t = qb_ratings_table(_passers(), cfg)
    manual = pd.DataFrame([{"season": 2025, "week": 2, "team": "A", "starter_out": True,
                            "observed_at": "2025-09-10T12:00:00Z", "source": "manual"}])
    out = apply_manual_status(t, manual).set_index("game_id")
    assert bool(out.loc[2, "incumbent_absent"]) is True
    assert out.loc[2, "qb_delta"] < 0.0


def test_build_passer_games_survives_missing_efficiency():
    """Attempts exist for every passer; PPA does not always. Missing efficiency must not
    drop the row, because the zero-attempt absence reading lives on the attempts side."""
    attempts = pd.DataFrame([{
        "game_id": 1, "season": 2025, "week": 1, "team": "A", "qb_id": "x",
        "qb_name": "X", "attempts": 20}])
    out = build_passer_games(attempts, pd.DataFrame())
    assert len(out) == 1
    assert pd.isna(out["ppa"].iloc[0])


# --------------------------------------------------------------------------------------
# THE LIVE PATH. Everything above tests the table; these test the only route by which a
# missing quarterback reaches a number a user actually reads.
# --------------------------------------------------------------------------------------

def _schedule():
    return pd.DataFrame([
        {"game_id": 1, "season": 2025, "week": 1, "homeTeam": "A", "awayTeam": "Z"},
        {"game_id": 2, "season": 2025, "week": 2, "homeTeam": "A", "awayTeam": "Z"},
        {"game_id": 3, "season": 2025, "week": 3, "homeTeam": "A", "awayTeam": "Z"},
        # scheduled, not yet played: no box score exists for it at all
        {"game_id": 4, "season": 2025, "week": 4, "homeTeam": "A", "awayTeam": "Z"},
    ])


def test_scheduled_but_unplayed_games_get_a_row_and_are_not_called_absent():
    """The gap that made this feature unable to affect a single future projection.

    A game that has not happened has no box score, so it appeared nowhere in the passer
    data and got no row -- meaning no adjustment could fire and a manual override had
    nothing to attach to. It must now get a row, carry an incumbent identified from prior
    weeks, and default to "the starter is playing".
    """
    cfg = load_config()
    t = qb_ratings_table(_passers(), cfg, games=_schedule())
    future = t[t["game_id"] == 4]
    assert len(future) >= 1, "a scheduled game must produce a row"
    row = future[future["team"] == "A"].iloc[0]
    assert bool(row["incumbent_absent"]) is False, "unplayed is unknown, not absent"
    assert row["incumbent_id"] == "starter", "incumbent comes from prior weeks"


def test_manual_override_can_reach_a_future_game():
    """The point of the row above: a human can now say a starter is out for a game that
    has not been played, which is the entire live use case."""
    cfg = load_config()
    t = qb_ratings_table(_passers(), cfg, games=_schedule())
    manual = pd.DataFrame([{"season": 2025, "week": 4, "team": "A", "starter_out": True,
                            "observed_at": "2025-09-24T12:00:00Z", "source": "manual"}])
    out = apply_manual_status(t, manual)
    row = out[(out["game_id"] == 4) & (out["team"] == "A")].iloc[0]
    assert bool(row["incumbent_absent"]) is True
    assert row["qb_delta"] == pytest.approx(-1.0)


def test_qb_points_for_game_is_home_perspective_and_fails_soft():
    cfg = load_config()
    table = pd.DataFrame([
        {"game_id": 7, "team": "H", "qb_delta": -1.0},
        {"game_id": 7, "team": "V", "qb_delta": 0.0},
    ])
    home_out = qb_points_for_game(7, "H", "V", table, cfg)
    assert home_out.points == pytest.approx(-cfg.qb.points_per_starter_out)
    away_out = qb_points_for_game(
        7, "V", "H", table, cfg)          # same table, sides swapped
    assert away_out.points == pytest.approx(+cfg.qb.points_per_starter_out)
    # Every failure path is a no-op, never a penalty.
    assert qb_points_for_game(7, "H", "V", None, cfg).points == 0.0
    assert qb_points_for_game(999, "H", "V", table, cfg).points == 0.0
    assert qb_points_for_game(7, "H", "V", pd.DataFrame(), cfg).points == 0.0


def test_with_qb_moves_the_spread_only_and_is_identity_at_zero():
    """`ContextAdjustment` is the sole injection point into the simulator mean that
    project_game.py and the viewer both report."""
    from src.context import NULL_CONTEXT

    moved = NULL_CONTEXT.with_qb(-1.75, "H starter out")
    assert moved.spread_points == pytest.approx(-1.75)
    assert moved.total_points == pytest.approx(NULL_CONTEXT.total_points)
    assert moved.components["qb_points"] == pytest.approx(-1.75)
    # A zero adjustment must not even allocate a new object, so an unaffected game is
    # provably untouched.
    assert NULL_CONTEXT.with_qb(0.0) is NULL_CONTEXT
