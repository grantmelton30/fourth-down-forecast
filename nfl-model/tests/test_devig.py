"""Devig correctness (§7a) and blend mechanics (§7c)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.market import (
    BlendWeights,
    american_to_decimal,
    american_to_prob,
    apply_blend,
    devig_two_way,
    fit_blend_weights,
    key_number_value,
)


def test_american_to_prob():
    assert american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert american_to_prob(100) == pytest.approx(0.5)
    assert american_to_prob(150) == pytest.approx(0.4)


def test_american_to_decimal():
    assert american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-4)
    assert american_to_decimal(150) == pytest.approx(2.5)


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devig_sums_to_one(method):
    a, b = devig_two_way(-110, -110, method)
    assert a + b == pytest.approx(1.0, abs=1e-6)
    # A balanced market must devig to a coin flip under every method.
    assert a == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devig_preserves_ordering(method):
    a, b = devig_two_way(-200, 170, method)
    assert a > b
    assert a + b == pytest.approx(1.0, abs=1e-6)


def test_methods_agree_on_near_balanced_markets():
    """The reason multiplicative is the default: at -110/-105 the book sum is ~1.048 and
    all three methods land within a fraction of a percent of each other."""
    probs = [
        devig_two_way(-110, -105, m)[0]
        for m in ("multiplicative", "power", "shin")
    ]
    assert max(probs) - min(probs) < 0.01


def test_methods_diverge_on_lopsided_markets():
    """...and the reason the other two exist: on a heavy favorite they disagree, which is
    where favorite-longshot bias lives."""
    mult = devig_two_way(-2000, 1100, "multiplicative")[0]
    shin = devig_two_way(-2000, 1100, "shin")[0]
    assert abs(mult - shin) > 0.001


def test_blend_recovers_known_coefficients():
    """Synthetic data where the true model weight is 0.4 and the market is unbiased."""
    cfg = load_config()
    rng = np.random.default_rng(7)
    n = 4000
    market = rng.normal(0, 6, n)
    disagreement = rng.normal(0, 3, n)
    frame = pd.DataFrame({
        "model_spread": market + disagreement,
        "market_spread": market,
        "actual_margin": market + 0.4 * disagreement + rng.normal(0, 12, n),
    })
    w = fit_blend_weights(frame, cfg)
    assert w.b_model == pytest.approx(0.4, abs=0.08)
    assert w.t_model > 2.0

def test_blend_separates_level_bias_from_incremental_signal():
    cfg=load_config(); n=200; market=np.linspace(-7,7,n); disagreement=np.tile([1.,2.,3.,4.],n//4)
    totals=np.linspace(39,51,n); td=np.tile([2.,3.,4.,5.],n//4)
    frame=pd.DataFrame({"model_spread":market+disagreement,"market_spread":market,"actual_margin":market+2.25,"model_total":totals+td,"market_total":totals,"actual_total":totals-1.5})
    w=fit_blend_weights(frame,cfg)
    assert w.a_spread==pytest.approx(2.25); assert w.b_model_raw==pytest.approx(0,abs=1e-12)
    assert apply_blend(-3,1,w)==pytest.approx(-.75)
    assert w.a_total==pytest.approx(-1.5); assert w.b_model_total_raw==pytest.approx(0,abs=1e-12)

def test_loader_rejects_legacy_or_incomplete_v2(tmp_path):
    legacy=tmp_path/"legacy.json"; legacy.write_text('{"b_model": 0.2}')
    assert BlendWeights.load(legacy) is None
    incomplete=tmp_path/"incomplete.json"; incomplete.write_text('{"schema_version": 2, "b_model": 0.2}')
    assert BlendWeights.load(incomplete) is None


def test_blend_clamps_negative_weight():
    """A negative fitted weight means anti-predictive on this sample -- clamp to zero and
    say so, never bet it."""
    cfg = load_config()
    rng = np.random.default_rng(11)
    n = 3000
    market = rng.normal(0, 6, n)
    disagreement = rng.normal(0, 3, n)
    frame = pd.DataFrame({
        "model_spread": market + disagreement,
        "market_spread": market,
        "actual_margin": market - 0.5 * disagreement + rng.normal(0, 10, n),
    })
    w = fit_blend_weights(frame, cfg)
    assert w.b_model == 0.0
    assert w.clamped


def test_apply_blend_is_market_plus_shrunken_disagreement():
    w = BlendWeights(
        b_model=0.3, se_model=0.05, t_model=6.0,
        r_squared=0.01, resid_sd=13.0, n=1000,
    )
    # The model disagrees by 3 points; 0.9 of that is treated as real.
    assert apply_blend(-3.0, 0.0, w) == pytest.approx(-2.1)


def test_key_number_value_counts_mass_between_lines():
    pmf = {2: 0.05, 3: 0.15, 4: 0.05, 7: 0.09}
    # Crossing 3 picks up the spike; a move that crosses nothing picks up nothing.
    assert key_number_value(2.0, 4.0, pmf) == pytest.approx(0.15)
    assert key_number_value(4.5, 6.5, pmf) == pytest.approx(0.0)
