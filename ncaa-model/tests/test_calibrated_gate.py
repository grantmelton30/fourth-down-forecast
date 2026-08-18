"""GATE_CALIBRATED arithmetic -- the piece of `attach_calibration` + `calibration_table`
+ `gate_calibrated` that does not need a live CFBD-backed cache or the drive simulator to
exercise, matching `test_key_numbers_gate.py`'s split. `attach_calibration` itself (the
simulator-driving half) needs real ratings, a real drive table, and the shared Monte
Carlo, exactly like `pooled_margin_pmf` -- its first real reading comes from a full
`run_backtest.py` rebuild.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import calibration_table, gate_calibrated


def _frame(cover_probs: list[float], covered: list[bool]) -> pd.DataFrame:
    """`covered[i]` True means the home side beat `spread_close` (actual_margin > line)."""
    n = len(cover_probs)
    return pd.DataFrame({
        "cover_prob_home": cover_probs,
        "spread_close": [0.0] * n,
        "actual_margin": [1.0 if c else -1.0 for c in covered],
    })


def test_well_calibrated_bin_passes():
    """100 games all predicted at ~55% cover, 55 of which actually cover: a perfect bin."""
    frame = _frame([0.55] * 100, [True] * 55 + [False] * 45)

    calib = calibration_table(frame)
    result = gate_calibrated(calib)

    assert result.passed
    assert "GATE_CALIBRATED" == result.name


def test_overconfident_bin_fails():
    """100 games predicted at 90% cover that only cover 60% of the time -- a 30pp gap,
    over the 6pp default tolerance -- must fail and name the offending bin."""
    frame = _frame([0.90] * 100, [True] * 60 + [False] * 40)

    calib = calibration_table(frame)
    result = gate_calibrated(calib)

    assert not result.passed
    assert "worst bin off by 30.00pp" in result.observed


def test_bins_under_the_size_floor_are_not_tested():
    """A wildly miscalibrated bin with only 50 observations must not fail the gate --
    the 100-observation floor exists so a handful of games cannot swing the verdict."""
    thin_bad = _frame([0.90] * 50, [True] * 5 + [False] * 45)  # 10% realized vs 90% predicted
    thick_good = _frame([0.55] * 100, [True] * 55 + [False] * 45)
    frame = pd.concat([thin_bad, thick_good], ignore_index=True)

    calib = calibration_table(frame)
    result = gate_calibrated(calib)

    assert result.passed


def test_no_bin_meets_the_size_floor_fails_closed():
    frame = _frame([0.55] * 10, [True] * 6 + [False] * 4)

    calib = calibration_table(frame)
    result = gate_calibrated(calib)

    assert not result.passed
    assert "no bin with 100+ observations" in result.observed


def test_pushes_are_excluded_from_the_denominator():
    """A game where actual_margin equals the close line is neither a cover nor a loss and
    must not be counted toward calibration in either direction."""
    resolved = _frame([0.55] * 100, [True] * 55 + [False] * 45)
    pushes = pd.DataFrame({
        "cover_prob_home": [0.55] * 20,
        "spread_close": [0.0] * 20,
        "actual_margin": [0.0] * 20,
    })
    frame = pd.concat([resolved, pushes], ignore_index=True)

    calib = calibration_table(frame)

    assert int(calib["n"].sum()) == 100


def test_anchors_on_the_close_by_default_not_the_opener():
    """A game that covers the close but not the open must be graded against the close --
    NCAA's calibration evidence is close-anchored (CALIBRATION_ANCHOR, run_backtest.py)
    because the opener-anchored totals signal was a measured false positive."""
    frame = pd.DataFrame({
        "cover_prob_home": [0.6] * 100,
        "spread_open": [5.0] * 100,
        "spread_close": [0.0] * 100,
        "actual_margin": [2.0] * 100,  # covers the close (>0) but not the open (<5)
    })

    calib = calibration_table(frame)

    assert calib.loc[calib["n"] > 0, "realized"].iloc[0] == pytest.approx(1.0)
