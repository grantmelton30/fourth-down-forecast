"""The forward-record arithmetic (scripts/build_tracker.py).

The numbers on the Record tab are the only scoreboard this project has, and the combined
view pools two rules, so the maths has to be right in one place rather than two.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "build_tracker", ROOT / "scripts" / "build_tracker.py")
bt = importlib.util.module_from_spec(spec)
sys.modules["build_tracker"] = bt
spec.loader.exec_module(bt)


def test_no_settled_bets_is_not_zero_percent():
    """An empty record must read 'no data', never 0% -- which would look like total failure
    on the day a rule is registered."""
    s = bt._stats(0, 0)
    assert s["win_pct"] is None and s["units"] == 0.0


def test_units_price_a_loss_at_minus_110():
    """A loss costs 1.1 units, a win pays 1.0. Treating them symmetrically would show a
    50% record as break-even when it is actually losing money."""
    assert bt._stats(1, 1)["units"] == pytest.approx(-0.1)
    assert bt._stats(10, 0)["units"] == pytest.approx(10.0)
    assert bt._stats(0, 10)["units"] == pytest.approx(-11.0)


def test_break_even_rate_is_roughly_units_neutral():
    """52.38% is the break-even claim the whole site is measured against; at that rate the
    units should sit near zero, which is what makes the number meaningful."""
    wins, losses = 5238, 4762
    assert bt._stats(wins, losses)["units"] == pytest.approx(0.0, abs=5.0)


def test_interval_needs_more_than_one_bet():
    assert bt._stats(1, 0)["ci_low"] is None
    assert bt._stats(30, 20)["ci_low"] is not None


def test_win_rate_and_interval_are_sane():
    s = bt._stats(60, 40)
    assert s["win_pct"] == 60.0
    assert s["ci_low"] < 60.0 < s["ci_high"]


def test_breakeven_constant_matches_minus_110():
    """-110 both sides: 110/210. If this drifts, every 'vs break-even' figure on the site
    silently moves with it."""
    assert bt.BREAKEVEN == pytest.approx(100 * 110 / 210, abs=0.01)
