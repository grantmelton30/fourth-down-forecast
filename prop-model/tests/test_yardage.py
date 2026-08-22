"""Compound yardage model (prop-model/src/yardage.py).

The distribution is the part that was wrong before, so these tests are mostly about the
properties that made the previous attempt collapse: negative gains, a real mass at zero,
and a right tail heavy enough that a normal or a count distribution cannot represent it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.yardage import (ald_cdf, ald_logpdf, ald_ppf, fit_ald, fit_touch_dispersion,
                         prob_over_from_sims, sample_rounded_ald, simulate_game_yards)

ALD = (2.0, 1.45, 0.313)          # the fit measured on 44,957 real carries


def test_the_distribution_allows_negative_gains():
    """8.8% of real carries lose yards. Any distribution on the non-negative integers gives
    them probability zero, which is precisely why the negative binomial produced -inf."""
    rng = np.random.default_rng(0)
    draws = sample_rounded_ald(*ALD, 20000, rng)
    assert (draws < 0).mean() > 0.03, "a model that cannot lose yards is not modelling carries"


def test_it_produces_a_real_mass_at_zero():
    """A gamma has no mass at zero, but 8.4% of carries gain exactly nothing."""
    rng = np.random.default_rng(1)
    draws = sample_rounded_ald(*ALD, 20000, rng)
    assert (draws == 0).mean() > 0.02


def test_the_right_tail_is_heavy_enough_to_matter():
    """~10% of carries go for more than ten yards, and that tail is where an over cashes.
    A normal fitted to the same mean and sd puts far too little mass there."""
    rng = np.random.default_rng(2)
    draws = sample_rounded_ald(*ALD, 40000, rng)
    assert draws.mean() > 0
    assert (draws > 10).mean() > 0.05
    assert stats.skew(draws) > 1.0, "real per-carry skew is +3.7; a normal is 0"


def test_cdf_and_inverse_cdf_agree():
    u = np.linspace(0.01, 0.99, 60)
    round_trip = ald_cdf(ald_ppf(u, *ALD), *ALD)
    assert np.allclose(round_trip, u, atol=1e-8)


def test_the_density_peaks_at_mu():
    """Real per-carry yardage is sharply peaked, which is why the paper prefers this over a
    normal -- the mode carries much more mass than a bell curve would put there."""
    mu = ALD[0]
    at_mode = ald_logpdf([mu], *ALD)[0]
    assert at_mode > ald_logpdf([mu - 3], *ALD)[0]
    assert at_mode > ald_logpdf([mu + 3], *ALD)[0]


def test_fitting_recovers_parameters_it_generated():
    rng = np.random.default_rng(3)
    y = ald_ppf(rng.random(30000), *ALD)
    mu, sigma, tau = fit_ald(y)
    assert mu == pytest.approx(ALD[0], abs=0.25)
    assert sigma == pytest.approx(ALD[1], abs=0.25)
    assert tau == pytest.approx(ALD[2], abs=0.05)


def test_fitting_refuses_a_sample_too_small_to_support_three_parameters():
    with pytest.raises(ValueError):
        fit_ald(np.arange(10.0))


def test_zero_touches_gives_zero_yards_with_no_special_case():
    """The point mass at zero for a game prop is 'he did not get the ball'. It has to fall
    out of the compound structure rather than be bolted on, or a player who is inactive
    would need handling everywhere downstream."""
    rng = np.random.default_rng(4)
    sims = simulate_game_yards(mean_touches=1e-9, touch_r=float("inf"), ald=ALD,
                               n_sims=2000, rng=rng)
    assert (sims == 0).mean() > 0.99


def test_more_touches_means_more_yards():
    rng = np.random.default_rng(5)
    few = simulate_game_yards(mean_touches=5, touch_r=8.0, ald=ALD, n_sims=8000, rng=rng)
    many = simulate_game_yards(mean_touches=18, touch_r=8.0, ald=ALD, n_sims=8000, rng=rng)
    assert many.mean() > few.mean() * 2.5


def test_game_yardage_inherits_the_skew_of_its_touches():
    """A prop's whole value is the shape. If the compound sum came out symmetric, the
    per-touch distribution would have been wasted effort."""
    rng = np.random.default_rng(6)
    sims = simulate_game_yards(mean_touches=14, touch_r=6.0, ald=ALD, n_sims=20000, rng=rng)
    assert stats.skew(sims) > 0.3
    assert sims.min() < sims.mean(), "a bad game must be possible"


def test_prob_over_moves_the_right_way():
    rng = np.random.default_rng(7)
    sims = simulate_game_yards(mean_touches=14, touch_r=6.0, ald=ALD, n_sims=20000, rng=rng)
    low = prob_over_from_sims(sims, 30.0)
    high = prob_over_from_sims(sims, 90.0)
    assert 0.0 < high < low < 1.0


def test_touch_dispersion_detects_overdispersion_and_falls_back_to_poisson():
    rng = np.random.default_rng(8)
    over = rng.negative_binomial(5, 5 / (5 + 12), size=6000)
    assert np.isfinite(fit_touch_dispersion(over))
    assert not np.isfinite(fit_touch_dispersion(np.full(500, 7.0)))


def test_simulation_is_reproducible_from_its_seed():
    """A prop price that changes when nothing changed is untraceable, and this repo grades
    bets from recorded numbers."""
    a = simulate_game_yards(mean_touches=12, touch_r=6.0, ald=ALD, n_sims=3000,
                            rng=np.random.default_rng(11))
    b = simulate_game_yards(mean_touches=12, touch_r=6.0, ald=ALD, n_sims=3000,
                            rng=np.random.default_rng(11))
    assert np.array_equal(a, b)
