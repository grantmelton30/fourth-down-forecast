"""Bet selection (prop-model/src/selection.py), following PLOS ONE 2023.

The paper's central result is that a correctly-priced line cannot be beaten: when the book
has the median right, the win rate is bounded between 47.6% and 52.4% and the vig takes the
difference. So the job is not predicting well, it is finding lines placed away from the true
median. These tests pin that logic, including the direction error that made every market
look unbeatable on the first run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.selection import breakeven_prob, ev_per_unit, required_edge, select_bets


def test_breakeven_matches_the_published_bounds():
    """PLOS 2023 bounds the win rate at 47.6%-52.4% against a correct line, which is exactly
    the -110 break-even. If this constant drifts, every selection decision moves with it."""
    assert 100 * breakeven_prob(-110) == pytest.approx(52.38, abs=0.01)
    assert 100 * breakeven_prob(-115) == pytest.approx(53.49, abs=0.01)
    assert 100 * breakeven_prob(-125) == pytest.approx(55.56, abs=0.01)
    assert 100 * breakeven_prob(+120) == pytest.approx(45.45, abs=0.01)


def test_a_coin_flip_at_minus_110_loses_money():
    """The paper's headline: betting a correctly-priced line is negative EV even when you
    take the side more likely to win. This is the whole reason selection exists."""
    assert ev_per_unit(0.50, -110) < 0
    assert ev_per_unit(0.5237, -110) < 0, "just under break-even is still a loser"
    assert ev_per_unit(0.5239, -110) > 0


def test_a_line_sitting_on_the_median_is_never_worth_betting():
    rng = np.random.default_rng(0)
    sims = rng.normal(80, 30, 40000)
    picks = select_bets([sims], [float(np.median(sims))], -110)
    assert picks[0] is None, "no edge exists when the book has the median right"


def test_the_book_being_wrong_in_either_direction_produces_the_right_side():
    """A line below the median is an OVER, above it is an UNDER. Reversing this is the bug
    that made required_edge return infinity for every market."""
    rng = np.random.default_rng(1)
    sims = rng.normal(80, 20, 40000)
    low = select_bets([sims], [50.0], -110)[0]
    high = select_bets([sims], [110.0], -110)[0]
    assert low["side"] == "OVER" and low["p"] > 0.9
    assert high["side"] == "UNDER" and high["p"] > 0.9


def test_required_edge_is_finite_and_small_for_a_realistic_distribution():
    """If this returns infinity the market is unbeatable at any model quality. It did on the
    first run, because the line was shifted away from the median and the FAR side taken,
    so every probability fell instead of rising."""
    rng = np.random.default_rng(2)
    sims = rng.normal(80, 30, 40000)
    need = required_edge(sims, -110)
    assert np.isfinite(need) and 0 < need < 20


def test_a_worse_price_demands_a_bigger_error_from_the_book():
    """Vig is the hurdle. At -125 you need break-even 55.6% instead of 52.4%, so the book
    must be further wrong before a bet clears -- which is why the price must be recorded."""
    rng = np.random.default_rng(3)
    sims = rng.normal(80, 30, 40000)
    assert required_edge(sims, -125) > required_edge(sims, -110)


def test_a_tighter_distribution_needs_a_smaller_error():
    """Required edge is a property of the SHAPE, not just the mean. A player whose output is
    predictable swings past break-even sooner, so the same book error is worth more on him."""
    rng = np.random.default_rng(4)
    tight = rng.normal(80, 10, 40000)
    wide = rng.normal(80, 40, 40000)
    assert required_edge(tight, -110) < required_edge(wide, -110)


def test_selection_declines_most_of_a_fairly_priced_board():
    """The paper argues the majority of any board is not worth betting. A selector that
    finds a bet everywhere is not selecting."""
    rng = np.random.default_rng(5)
    boards = [rng.normal(80, 30, 6000) for _ in range(60)]
    lines = [float(np.median(b)) for b in boards]
    picks = select_bets(boards, lines, -110)
    assert all(p is None for p in picks)


def test_ev_is_reported_per_unit_risked():
    """+100 doubles a winner, so a 60% shot returns 0.2 per unit risked."""
    assert ev_per_unit(0.60, +100) == pytest.approx(0.20, abs=1e-9)
    assert ev_per_unit(1.0, -200) == pytest.approx(0.5, abs=1e-9)


def test_all_push_is_not_a_winning_under():
    assert select_bets([np.full(100,5)], [5])[0] is None
    assert ev_per_unit(0,-110,p_push=1) == 0


def test_integer_line_ev_matches_exhaustive_settlement():
    sims = np.array([4]*20+[5]*30+[6]*50)
    r = select_bets([sims],[5],over_prices=[-120],under_prices=[-105])[0]
    assert r["side"] == "OVER"
    assert r["p_push"] == pytest.approx(.3)
    assert r["ev"] == pytest.approx((50*100/120-20)/100)


def test_side_specific_odds_can_reverse_the_choice():
    sims = np.array([0]*45+[1]*55)
    r = select_bets([sims],[.5],over_prices=[-200],under_prices=[150])[0]
    assert r["side"] == "UNDER" and r["price"] == 150


@pytest.mark.parametrize("sims", [[],[float("nan")],[float("inf")]])
def test_bad_simulations_are_rejected(sims):
    with pytest.raises(ValueError):
        select_bets([sims],[5])


def test_missing_side_price_cannot_be_assumed():
    assert select_bets([[0,1]],[.5],over_prices=[float("nan")],under_prices=[100]) == [None]
    with pytest.raises(ValueError):
        select_bets([[0,1]],[.5],over_prices=[-110])
    with pytest.raises(ValueError):
        select_bets([[0,1]],[.5,1.5])
