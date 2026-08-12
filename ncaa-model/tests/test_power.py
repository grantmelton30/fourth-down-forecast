"""Assertions on the power module — the guard against unpowered claims.

Each test pins a failure mode that produces a plausible-looking sentence in a write-up
while being unsupported by the sample it was read off:

1. `required_b` and `mde` are different numbers. Conflating them understates what it
   takes to detect an effect by a factor of ~1.43, which is exactly the gap that turns
   "we would have seen it" into a false statement.
2. An underpowered sample must never be reported as a null. This is the reason the module
   exists; if `verdict` ever returns a null for a sample that could not have seen the
   reference effect, every downstream write-up inherits the error.
3. A difference is noisier than either side of it. Reading two `b`s side by side is the
   specific mistake D6d made.
4. The recorded Appendix A coefficients must not drift silently. They are cited in four
   documents and in the standing conclusion.

Requires the parquet cache:
    NCAA_MODEL_CACHE_DIR=~/.cache/ncaa-model pytest tests/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.power import Difference, Power, analyse, analyse_frame, compare
from src.config import CACHE_DIR

# Appendix A, NCAA_PLAYBOOK.md — the identical model on two anchors.
APPENDIX_A_CLOSE_B, APPENDIX_A_CLOSE_T = 0.076, 0.89
APPENDIX_A_OPEN_B, APPENDIX_A_OPEN_T = 0.195, 2.23
DOCUMENTED_RESTRICTED_UNIVERSE = 1543


def _need(path):
    if not path.exists():
        pytest.skip(f"cache artifact missing: {path.name}; set NCAA_MODEL_CACHE_DIR")
    return path


def _frame():
    frame = pd.read_parquet(_need(CACHE_DIR / "backtest_frame_default.parquet"))
    return frame[frame["has_opener"] & frame["season"].between(2021, 2025)]


def _synthetic(n: int, b_true: float, noise_sd: float = 16.0, seed: int = 7):
    """A sample with a known true `b`, in the residual form the blend uses."""
    rng = np.random.default_rng(seed)
    market = pd.Series(rng.normal(55.0, 8.0, n))
    disagreement = pd.Series(rng.normal(0.0, 5.0, n))
    model = market + disagreement
    actual = market + b_true * disagreement + rng.normal(0.0, noise_sd, n)
    return model, market, actual


# --------------------------------------------------------------------------------------
# 1. The two thresholds are distinct, and ordered
# --------------------------------------------------------------------------------------

def test_required_b_and_mde_are_distinct_and_ordered():
    """`mde` must exceed `required_b`, because detection is not the same as reporting.

    An effect exactly at `required_b` is detected only half the time — the point estimate
    lands below the threshold on half of samples. `mde` is the size at which detection
    reaches the target power. Collapsing them is the arithmetic that makes an underpowered
    study look adequate.
    """
    p = Power(label="synthetic", n=1000, b=0.0, se_b=0.05, t_b=0.0,
              disagreement_sd=5.0, resid_sd=16.0)

    assert p.required_b == pytest.approx(1.959964 * 0.05, rel=1e-6)
    assert p.mde == pytest.approx((1.959964 + 0.841621) * 0.05, rel=1e-6)
    assert p.mde > p.required_b
    assert p.mde / p.required_b == pytest.approx(1.4294, rel=1e-3)

    # An effect exactly at `required_b` is a coin flip to detect.
    assert p.power_at(p.required_b) == pytest.approx(0.50, abs=0.01)
    # An effect at `mde` is detected at the target rate.
    assert p.power_at(p.mde) == pytest.approx(p.target_power, abs=0.01)


def test_required_n_scales_as_inverse_sqrt():
    """Quadrupling n halves the MDE, so required_n must scale as 1/b^2."""
    p = Power(label="s", n=1000, b=0.0, se_b=0.05, t_b=0.0,
              disagreement_sd=5.0, resid_sd=16.0)
    assert p.required_n(p.mde) == pytest.approx(p.n, rel=0.01)
    # Halving the detectable effect costs 4x the sample.
    assert p.required_n(p.mde / 2) == pytest.approx(4 * p.n, rel=0.01)


# --------------------------------------------------------------------------------------
# 2. An underpowered sample is never reported as a null
# --------------------------------------------------------------------------------------

def test_underpowered_zero_is_not_called_a_null():
    """The core guarantee. A tiny sample measuring b = 0 supports no claim.

    Without this distinction, "we found nothing" and "we could not have found anything"
    are written down identically — and the second one has twice been mistaken for the
    first in this build.
    """
    model, market, actual = _synthetic(n=60, b_true=0.0)
    p = analyse(model, market, actual, label="tiny")

    assert abs(p.t_b) < 1.96
    assert not p.adequately_powered
    assert p.verdict.startswith("UNDERPOWERED")
    assert "NULL" not in p.verdict


def test_adequately_powered_zero_is_a_real_null():
    """The complement: a large sample measuring zero IS evidence of absence."""
    model, market, actual = _synthetic(n=20000, b_true=0.0)
    p = analyse(model, market, actual, label="large")

    assert abs(p.t_b) < 1.96
    assert p.adequately_powered
    assert p.verdict == "NULL (adequately powered)"


def test_real_effect_is_recovered_and_detected():
    """A true effect at the reference size must be found in an adequately powered sample."""
    model, market, actual = _synthetic(n=20000, b_true=0.20)
    p = analyse(model, market, actual, label="real")

    assert p.b == pytest.approx(0.20, abs=0.03)
    assert p.verdict == "DETECTED"
    assert p.power_at() > 0.95


# --------------------------------------------------------------------------------------
# 3. Differences are noisier than their components
# --------------------------------------------------------------------------------------

def test_difference_se_exceeds_both_components():
    """se(a - b) = hypot(se_a, se_b) is strictly larger than either.

    D6d read b_early = +0.242 against b_late = -0.146 and called the effect absent late.
    The difference carried se 0.247 and t = 1.53. This assertion is the arithmetic that
    makes that impossible to miss.
    """
    a = Power("a", n=1289, b=0.1205, se_b=0.0900, t_b=1.34,
              disagreement_sd=5.0, resid_sd=16.0)
    b = Power("b", n=254, b=-0.2661, se_b=0.2300, t_b=-1.16,
              disagreement_sd=5.0, resid_sd=16.0)
    d = Difference("early", "late", a, b)

    assert d.se_diff > a.se_b and d.se_diff > b.se_b
    assert d.se_diff == pytest.approx(np.hypot(0.0900, 0.2300), rel=1e-9)
    assert abs(d.t_diff) < 1.96
    assert d.verdict.startswith("UNDERPOWERED")
    lo, hi = d.ci
    assert lo < 0 < hi, "a non-significant difference must have a CI containing zero"


def test_week_partition_claim_stays_retracted():
    """The D6d partition must never again read as a difference on real data."""
    graded = _frame()
    graded = graded[graded["restricted"]].dropna(
        subset=["model_total", "total_close", "actual_total"]
    )
    d = compare(graded, graded["week"] < 13, graded["week"] >= 13,
                "weeks 4-12", "weeks 13+")

    assert abs(d.t_diff) < 1.96, (
        f"the week-13 partition now reads t = {d.t_diff:+.2f}. D6d retracted this claim "
        "as underpowered; if it has become significant that is a finding requiring its "
        "own pre-registration, not a silently passing test."
    )
    assert d.mde_diff > d.a.reference_b, (
        "the week partition has become adequately powered — revisit D6d deliberately"
    )


# --------------------------------------------------------------------------------------
# 4. The Appendix A coefficients must not drift
# --------------------------------------------------------------------------------------

def test_appendix_a_anchor_coefficients_hold():
    """Pins the two numbers the standing conclusion rests on.

    The whole null turns on these being different from each other: an identical model
    reads t = +2.23 anchored on the opener and t = +0.89 anchored on the close. If either
    drifts, Appendix A describes a model that no longer exists.
    """
    graded = _frame()
    graded = graded[graded["restricted"]].dropna(
        subset=["model_total", "total_open", "total_close", "actual_total"]
    )
    assert len(graded) == DOCUMENTED_RESTRICTED_UNIVERSE

    close = analyse_frame(graded, market_col="total_close", label="close")
    opener = analyse_frame(graded, market_col="total_open", label="open")

    assert close.b == pytest.approx(APPENDIX_A_CLOSE_B, abs=0.005)
    assert close.t_b == pytest.approx(APPENDIX_A_CLOSE_T, abs=0.02)
    assert opener.b == pytest.approx(APPENDIX_A_OPEN_B, abs=0.005)
    assert opener.t_b == pytest.approx(APPENDIX_A_OPEN_T, abs=0.02)

    # And the restricted close-anchored read is NOT adequately powered — which is why the
    # full-FBS window below is the one the standing conclusion must cite.
    assert not close.adequately_powered


def test_full_fbs_close_anchored_null_is_adequately_powered():
    """The properly powered close-anchored test — the one that settles the null.

    The restricted universe (n=1,543) has 65% power against the reference effect, so its
    null was never conclusive. The full FBS window declared in D2a carries n=2,985 and
    93% power, and reads b indistinguishable from zero. This is the assertion that makes
    the NCAA totals null evidence of absence rather than absence of evidence.
    """
    base = _frame()
    full = base[base["fbs_only"]].dropna(
        subset=["model_total", "total_close", "actual_total"]
    )
    p = analyse_frame(full, market_col="total_close", label="full FBS")

    assert p.n > 2900, f"full FBS window is {p.n} rows, expected ~2,985"
    assert p.adequately_powered, (
        f"the settling test has lost power (now {p.power_at():.1%}) — the standing "
        "conclusion can no longer cite it"
    )
    assert abs(p.b) < 0.05, f"b = {p.b:+.4f}; the recorded null is b ~ 0"
    assert p.verdict == "NULL (adequately powered)"
