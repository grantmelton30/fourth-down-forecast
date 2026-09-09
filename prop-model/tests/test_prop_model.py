"""The rebuilt prop model (prop-model/src/prop_model.py).

The rebuild's whole premise is that the defence effect must be estimated from games that
already happened. A defensive rating that includes the game it is predicting would make this
model look excellent and be worth nothing, and no downstream score would reveal it. Most of
these tests exist for that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prop_model import (add_opponent, calibration, defense_effects, fit_dispersion,
                            home_effect, is_count_market, log_score, predict, prob_over)


def _frame():
    """Four players facing the same defence in successive weeks, every one beating his
    baseline by exactly 2. A defence effect built correctly can only see earlier games."""
    return pd.DataFrame([
        {"player_id": f"p{w}", "position": "WR", "team": "KC", "opponent": "DEN",
         "is_home": 1, "season": 2026, "week": w,
         "kickoff": pd.Timestamp("2026-09-06") + pd.Timedelta(days=7 * w),
         "receptions": 6.0, "base_ewma": 4.0}
        for w in range(1, 5)
    ])


def test_the_first_game_against_a_defence_gets_no_effect():
    """With no prior evidence a defence must be treated as ordinary, not as whatever the
    game being predicted happens to show."""
    out = defense_effects(_frame(), stat="receptions", baseline_col="base_ewma")
    assert out.sort_values("week").iloc[0]["def_effect"] == 0.0


def test_a_defence_effect_never_includes_the_game_it_predicts():
    """Every player beats his baseline by exactly 2.0. Week 2's effect may only reflect
    week 1, so with shrinkage off it must be exactly 2.0 x (1/(1+0)) -- and crucially the
    count of prior observations must be 1, not 2."""
    out = defense_effects(_frame(), stat="receptions", baseline_col="base_ewma",
                          prior_games=0.0).sort_values("week")
    w2 = out.iloc[1]
    assert w2["def_effect_n"] == 1
    assert w2["def_effect"] == pytest.approx(2.0)


def test_shrinkage_pulls_a_thin_defence_sample_toward_no_effect():
    """A defence seen once is not a proven matchup. The null is zero -- ordinary -- not the
    league mean, because 'this defence is average' is the honest default."""
    strong = defense_effects(_frame(), stat="receptions", baseline_col="base_ewma",
                             prior_games=0.0).sort_values("week")
    shrunk = defense_effects(_frame(), stat="receptions", baseline_col="base_ewma",
                             prior_games=10.0).sort_values("week")
    assert abs(shrunk.iloc[1]["def_effect"]) < abs(strong.iloc[1]["def_effect"])
    assert shrunk.iloc[1]["def_effect"] > 0, "shrunk toward zero, not through it"


def test_defences_are_tracked_separately_per_position():
    """A defence that smothers tight ends may be ordinary against receivers. Pooling them
    would average away the only thing the adjustment is for."""
    f = _frame()
    f2 = f.assign(position="TE", receptions=2.0, player_id=f["player_id"] + "_te")
    out = defense_effects(pd.concat([f, f2], ignore_index=True),
                          stat="receptions", baseline_col="base_ewma", prior_games=0.0)
    wr = out[(out["position"] == "WR") & (out["week"] == 3)].iloc[0]["def_effect"]
    te = out[(out["position"] == "TE") & (out["week"] == 3)].iloc[0]["def_effect"]
    assert wr > 0 > te, "WR beat their baselines, TE missed theirs"


def test_a_projection_is_never_negative():
    """A receiver cannot catch -0.3 passes. The defence term is additive, so a strong
    defence could otherwise push a low-usage player below zero."""
    f = _frame()
    f["base_ewma"] = 0.2
    f["def_effect"] = -5.0
    out = predict(f, baseline_col="base_ewma")
    assert (out["pred_mean"] >= 0).all()


def test_prob_over_treats_a_half_point_line_as_a_strict_count():
    """P(X > 5.5) is P(X >= 6). Off-by-one here misprices every bet by one whole reception."""
    p = prob_over([5.0], [5.5], r=10.0)[0]
    q = prob_over([5.0], [5.0], r=10.0)[0]
    assert p == pytest.approx(q), "floor(5.5) == floor(5.0) == 5"
    assert 0.0 < p < 1.0


def test_prob_over_rises_with_the_projection():
    lo = prob_over([3.0], [5.5], r=8.0)[0]
    hi = prob_over([8.0], [5.5], r=8.0)[0]
    assert hi > lo


def test_overdispersion_widens_the_distribution():
    """A smaller r means fatter tails. Understating dispersion is exactly how a prop model
    prices an over wrong -- it makes extreme outcomes look rarer than they are."""
    tight = prob_over([5.0], [9.5], r=100.0)[0]
    fat = prob_over([5.0], [9.5], r=2.0)[0]
    assert fat > tight


def test_yardage_is_refused_a_count_distribution():
    """Fitting a negative binomial to receiving yards produced -inf log scores in practice:
    outcomes present in the data are assigned zero probability. Yardage has a point mass at
    zero and rushing yards go negative. Returning NaN says so; -inf would look like a very
    bad model rather than an invalid one."""
    assert is_count_market("receptions") and not is_count_market("receiving_yards")
    f = pd.DataFrame({"receiving_yards": [40.0, 0.0], "pred_mean": [35.0, 30.0]})
    assert np.isnan(log_score(f, stat="receiving_yards", mean_col="pred_mean", r=5.0))


def test_dispersion_falls_back_to_poisson_when_not_overdispersed():
    f = pd.DataFrame({"receptions": [5.0, 5.0, 5.0], "pred_mean": [5.0, 5.0, 5.0]})
    assert not np.isfinite(fit_dispersion(f, stat="receptions"))


def test_calibration_reports_the_gap_rather_than_a_single_score():
    """One aggregate number hides where a model is wrong. A prop model that is well
    calibrated in the middle and badly wrong in the tails is useless precisely where the
    lines are."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(2, 8, 4000)
    obs = rng.poisson(mu)
    f = pd.DataFrame({"receptions": obs.astype(float), "pred_mean": mu})
    cal = calibration(f, stat="receptions", mean_col="pred_mean", r=float("inf"))
    assert {"n", "predicted", "actual", "gap"} <= set(cal.columns)
    big = cal[cal["n"] >= 100]
    assert (big["gap"].abs() < 0.10).all(), "a correctly specified model must calibrate"


def test_home_effect_is_walk_forward_too():
    f = _frame()
    out = home_effect(f, stat="receptions", baseline_col="base_ewma").sort_values("week")
    assert out.iloc[0]["home_effect"] == 0.0, "no prior games, no effect"


def test_opponent_is_attached_from_the_schedule_both_ways():
    sched = pd.DataFrame([{"season": 2026, "week": 1, "home_team": "KC", "away_team": "DEN"}])
    f = pd.DataFrame([{"season": 2026, "week": 1, "team": "KC"},
                      {"season": 2026, "week": 1, "team": "DEN"}])
    out = add_opponent(f, sched).set_index("team")
    assert out.loc["KC", "opponent"] == "DEN" and out.loc["KC", "is_home"] == 1
    assert out.loc["DEN", "opponent"] == "KC" and out.loc["DEN", "is_home"] == 0


def test_same_kickoff_outcomes_cannot_change_other_predictions():
    f = _frame()
    f.loc[1, "kickoff"] = f.loc[0, "kickoff"]
    first = defense_effects(f, stat="receptions", baseline_col="base_ewma")
    h_first = home_effect(f, stat="receptions", baseline_col="base_ewma")
    changed = f.copy()
    changed.loc[0, "receptions"] = 1000
    second = defense_effects(changed, stat="receptions", baseline_col="base_ewma")
    h_second = home_effect(changed, stat="receptions", baseline_col="base_ewma")
    assert first.loc[1,"def_effect"] == second.loc[1,"def_effect"] == 0
    assert h_first.loc[1,"home_effect"] == h_second.loc[1,"home_effect"] == 0
    assert second.loc[2,"def_effect"] != first.loc[2,"def_effect"]


def test_same_kickoff_order_does_not_change_effects():
    f = _frame()
    f.loc[1,"kickoff"] = f.loc[0,"kickoff"]
    a = defense_effects(f, stat="receptions", baseline_col="base_ewma").set_index("player_id")
    b = defense_effects(f.iloc[::-1], stat="receptions", baseline_col="base_ewma").set_index("player_id")
    np.testing.assert_allclose(a.sort_index().def_effect,b.sort_index().def_effect)
