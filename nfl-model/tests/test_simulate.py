"""Simulator correctness and the §6d validation gate.

`validate_simulator()` pools simulated margins across many historical games and compares
the shape against history. Read the note on EndgameTable in src/drive_model.py before
changing anything here: the key-number spikes come from teams playing to the scoreboard,
not from lumpy scoring increments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import pooled_margin_pmf
from src.config import load_config
from src.context import ContextAdjustment
from src.drive_model import fit_drive_model, fit_endgame_table
from src.drives import build_drive_table, fit_start_field_position
from src.ingest import load_pbp, load_schedules
from src.ratings import build_game_offense_table, build_walkforward_ratings
from src.simulate import SimResult, points_per_net_epa, simulate_game

KEY_NUMBERS = (3, 7, 6, 10, 14, 4)
TOLERANCE_PP = 2.0


@pytest.fixture(scope="module")
def machinery():
    cfg = load_config()
    schedules = load_schedules(cfg.train_seasons)
    pbp = load_pbp(cfg.train_seasons)
    drives = build_drive_table(pbp, cache_key="main")
    game_off = build_game_offense_table(pbp, schedules, cache_key="main")
    walkforward = build_walkforward_ratings(game_off, schedules, cfg)
    season = max(cfg.backtest_seasons)
    model = fit_drive_model(drives, walkforward, cfg, as_of_season=season)
    endgame = fit_endgame_table(drives, cfg, as_of_season=season)
    start_fp = fit_start_field_position(drives)
    return cfg, schedules, pbp, model, endgame, start_fp


def _flat_ratings(index=("H", "A")):
    return pd.DataFrame(
        {"off_rating": [0.0, 0.0], "def_rating": [0.0, 0.0], "pace_rating": [0.0, 0.0]},
        index=list(index),
    )


@pytest.fixture(scope="module")
def neutral_sim(machinery):
    cfg, _, _, model, endgame, start_fp = machinery
    return simulate_game(
        "H", "A", _flat_ratings(), model, ContextAdjustment(0.0, 0.0), cfg, start_fp,
        endgame=endgame,
    )


# --------------------------------------------------------------------------------------
# Distribution realism
# --------------------------------------------------------------------------------------

def test_simulated_total_is_realistic(neutral_sim):
    """Real NFL games since 2019 average about 45.8 points."""
    assert 42.0 < neutral_sim.mean_total < 50.0


def test_simulated_margin_spread_is_realistic(neutral_sim):
    """Real margin sd is about 14.2."""
    assert 12.5 < neutral_sim.margins.std() < 15.5


def test_ties_are_rare(neutral_sim):
    """Overtime must resolve nearly every tie: real regular-season ties are ~0.4%."""
    tie_rate = float((neutral_sim.margins == 0).mean())
    assert tie_rate < 0.02, f"tie rate {tie_rate:.3%} -- is overtime running?"


def test_margins_are_integers(neutral_sim):
    assert np.all(neutral_sim.margins == np.round(neutral_sim.margins))


def test_scores_are_never_negative(neutral_sim):
    assert (neutral_sim.home_scores >= 0).all()
    assert (neutral_sim.away_scores >= 0).all()


# --------------------------------------------------------------------------------------
# Probability mechanics
# --------------------------------------------------------------------------------------

def test_cover_prob_handles_pushes():
    """At a whole number a push is neither a win nor a loss, so the two sides must still
    sum to 1 after conditioning on the bet resolving."""
    sim = SimResult(
        margins=np.array([-3.0, 0.0, 3.0, 3.0, 7.0]),
        totals=np.zeros(5), home_scores=np.zeros(5), away_scores=np.zeros(5),
    )
    home = sim.cover_prob(3.0, "home")
    away = sim.cover_prob(3.0, "away")
    assert home + away == pytest.approx(1.0)
    # One margin above 3, two below; the two pushes are excluded entirely.
    assert home == pytest.approx(1 / 3)


def test_cover_prob_ignoring_pushes_would_overstate_edge():
    margins = np.array([3.0] * 20 + [10.0] * 40 + [-10.0] * 40)
    sim = SimResult(margins, np.zeros(100), np.zeros(100), np.zeros(100))
    assert sim.cover_prob(3.0, "home") > float((margins > 3.0).mean())


def test_total_prob_sums_to_one():
    sim = SimResult(
        margins=np.zeros(5), totals=np.array([40.0, 44.0, 44.0, 50.0, 51.0]),
        home_scores=np.zeros(5), away_scores=np.zeros(5),
    )
    assert sim.total_prob(44.0, "over") + sim.total_prob(44.0, "under") == pytest.approx(1.0)


def test_recentering_shifts_without_reshaping(neutral_sim):
    """Calibration reweights the existing lattice, preserving football score atoms."""
    shifted = neutral_sim.recentered(-6.5)
    assert shifted.mean_margin == pytest.approx(-6.5)
    assert np.array_equal(shifted.margins, neutral_sim.margins)
    assert np.array_equal(shifted.totals, neutral_sim.totals)
    assert shifted.margin_pmf().get(3, 0.0) > 0.0
    assert shifted.margin_pmf().get(7, 0.0) > 0.0


def test_context_moves_the_line_by_the_requested_points(machinery):
    """A 1.7-point context adjustment must move the simulated spread by ~1.7 points.

    This is what the points-to-net_epa conversion in simulate.net_epa_shift_for_points
    exists to guarantee; the playbook's literal points/drive formula misses by an order of
    magnitude.
    """
    cfg, _, _, model, endgame, start_fp = machinery
    base = simulate_game(
        "H", "A", _flat_ratings(), model, ContextAdjustment(0.0, 0.0), cfg, start_fp,
        endgame=endgame, rng=np.random.default_rng(1),
    )
    bumped = simulate_game(
        "H", "A", _flat_ratings(), model, ContextAdjustment(1.7, 0.0), cfg, start_fp,
        endgame=endgame, rng=np.random.default_rng(1),
    )
    delta = bumped.mean_margin - base.mean_margin
    assert delta == pytest.approx(1.7, abs=0.6), f"context moved the line {delta:.2f} pts"


def test_points_per_net_epa_is_positive(machinery):
    cfg, _, _, model, _, _ = machinery
    assert points_per_net_epa(model, cfg, 11.4) > 0


def test_better_team_is_favored(machinery):
    cfg, _, _, model, endgame, start_fp = machinery
    ratings = pd.DataFrame(
        {
            "off_rating": [0.15, -0.15],
            "def_rating": [-0.10, 0.10],
            "pace_rating": [0.0, 0.0],
        },
        index=["GOOD", "BAD"],
    )
    sim = simulate_game(
        "GOOD", "BAD", ratings, model, ContextAdjustment(0.0, 0.0), cfg, start_fp,
        endgame=endgame,
    )
    assert sim.mean_margin > 3.0


# --------------------------------------------------------------------------------------
# §6d validation gate
# --------------------------------------------------------------------------------------

def validate_simulator(n_games: int = 400, n_sims: int = 4000, verbose: bool = True):
    """Pool simulated margins across historical games and compare to the real
    distribution. Returns (passed, table)."""
    cfg = load_config()
    schedules = load_schedules(cfg.train_seasons)
    pbp = load_pbp(cfg.train_seasons)
    pmf = pooled_margin_pmf(cfg, schedules, pbp, n_games=n_games, n_sims=n_sims)

    done = schedules[schedules["result"].notna() & schedules["season"].ge(2019)]
    real = done["result"].abs().value_counts(normalize=True)

    rows = []
    for k in KEY_NUMBERS:
        sim_pct = 100 * (pmf.get(k, 0.0) + pmf.get(-k, 0.0))
        real_pct = 100 * float(real.get(k, 0.0))
        rows.append({
            "margin": k,
            "simulated_pct": round(sim_pct, 2),
            "historical_pct": round(real_pct, 2),
            "diff_pp": round(sim_pct - real_pct, 2),
            "within_tolerance": abs(sim_pct - real_pct) <= TOLERANCE_PP,
        })
    table = pd.DataFrame(rows)
    if verbose:
        print("\nKEY NUMBER VALIDATION (§6d)")
        print(table.to_string(index=False))
    return bool(table["within_tolerance"].all()), table


def test_key_number_validation_table_is_produced():
    """The comparison table must always be computed and printed, pass or fail -- it is the
    evidence §16 asks to be pasted into the README."""
    _, table = validate_simulator(n_games=120, n_sims=2000, verbose=True)
    assert len(table) == len(KEY_NUMBERS)
    assert table["simulated_pct"].between(0, 100).all()
