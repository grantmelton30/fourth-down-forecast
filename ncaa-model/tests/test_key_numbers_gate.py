"""GATE_KEY_NUMBERS arithmetic -- the piece of `pooled_margin_pmf` + `gate_key_numbers`
that does not need a live CFBD-backed cache or the drive simulator to exercise.

`pooled_margin_pmf` itself (the simulator-driving half) is not unit-tested here -- it
needs real ratings, a real drive table, and the shared Monte Carlo, exactly like
nfl-model's `pooled_margin_pmf` has no dedicated unit test either. Its first real reading
comes from a full `run_backtest.py` rebuild. This file pins the comparison arithmetic
`gate_key_numbers` performs on whatever PMF it is handed, so that piece is verified
without needing that rebuild.
"""

from __future__ import annotations

import pandas as pd

from src.backtest import KEY_NUMBERS, TAIL_THRESHOLD, gate_key_numbers


def _market(margins: list[float], fbs_only: bool = True) -> pd.DataFrame:
    return pd.DataFrame({
        "actual_margin": margins,
        "fbs_only": [fbs_only] * len(margins),
    })


def test_perfect_match_passes_with_zero_gap():
    """A sim PMF that reproduces the empirical distribution exactly must clear every key
    number and report a zero worst gap."""
    margins = [3.0, -3.0, 3.0, 7.0, -7.0, 10.0, 1.0, 1.0]  # |3|:3/8, |7|:2/8, |10|:1/8
    market = _market(margins)
    sim_pmf = {3: 3 / 8, -3: 0.0, 7: 1 / 8, -7: 1 / 8, 10: 1 / 8, 1: 1 / 8, -1: 1 / 8}

    result = gate_key_numbers(sim_pmf, market)

    assert result.passed
    assert result.name == "GATE_KEY_NUMBERS"
    assert "worst gap 0.00pp" in result.observed


def test_large_gap_at_a_key_number_fails():
    """A simulator that puts zero mass at |margin|=3, when a quarter of real games land
    there, must fail -- and must name 3 as the offender."""
    margins = [3.0] * 25 + [1.0] * 75  # |3| is 25% of real games
    market = _market(margins)
    sim_pmf = {1: 1.0}  # simulator never produces a 3

    result = gate_key_numbers(sim_pmf, market, tol=2.0)

    assert not result.passed
    assert "|margin|=3" in result.observed
    assert "worst gap 25.00pp" in result.observed


def test_tail_is_checked_independently_of_the_named_key_numbers():
    """A simulator that matches every named key number but misjudges the >28 blowout
    tail must still fail, on the tail specifically."""
    margins = [3.0] * 50 + [35.0] * 50  # half key-number-3, half deep blowout
    market = _market(margins)
    # Sim matches the |3| spike exactly but puts no mass past the tail threshold at all.
    sim_pmf = {3: 0.5, 1: 0.5}

    result = gate_key_numbers(sim_pmf, market, tol=2.0)

    assert not result.passed
    assert f">{TAIL_THRESHOLD}" in result.observed


def test_non_fbs_games_are_excluded_from_the_real_distribution():
    """A non-FBS-only row must not contaminate the historical comparison -- the same
    restriction every other NCAA gate applies."""
    fbs = _market([3.0] * 10, fbs_only=True)
    non_fbs = _market([21.0] * 90, fbs_only=False)
    market = pd.concat([fbs, non_fbs], ignore_index=True)
    sim_pmf = {3: 1.0}  # sim says every game is a 3-point game

    result = gate_key_numbers(sim_pmf, market, tol=2.0)

    # If the FCS/non-FBS rows leaked in, |3| would read as ~10% real instead of 100%,
    # and this would fail. It must pass, proving the filter held.
    assert result.passed


def test_key_number_set_matches_the_documented_college_set():
    """Pins the college key-number set against silent drift -- GATES.md documents
    3/7/10/14/17/21 plus the >28 tail, and this is a different set from the NFL's
    (3/7/6/10/14/4), which is the whole point of not sharing the NFL's gate."""
    assert KEY_NUMBERS == (3, 7, 10, 14, 17, 21)
    assert TAIL_THRESHOLD == 28
