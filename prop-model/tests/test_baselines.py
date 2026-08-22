"""Baselines and scoring (prop-model/src/baselines.py).

A baseline that peeks is worse than no baseline: it looks unbeatable and would kill a model
that was actually fine. Most of these tests are about that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import add_baselines, add_team_volume, add_usage_model, score


def _frame():
    """One player, receptions climbing 1..5, so any leak of the current game shows up as a
    baseline that is too close to the truth."""
    return pd.DataFrame([
        {"player_id": "A", "team": "KC", "season": 2026, "week": w,
         "kickoff": pd.Timestamp("2026-09-06") + pd.Timedelta(days=7 * w),
         "receptions": w, "targets": w + 1}
        for w in range(1, 6)
    ])


def test_no_baseline_can_see_its_own_game():
    b = add_baselines(_frame(), stat="receptions")
    # Week 3's baselines must come from weeks 1-2 only: mean(1,2) = 1.5.
    w3 = b[b["week"] == 3].iloc[0]
    assert w3["base_last_n"] == pytest.approx(1.5)
    assert w3["base_season_mean"] == pytest.approx(1.5)
    assert w3["prior_games"] == 2


def test_the_first_game_has_no_baseline_at_all():
    b = add_baselines(_frame(), stat="receptions")
    w1 = b[b["week"] == 1].iloc[0]
    assert pd.isna(w1["base_last_n"]) and w1["prior_games"] == 0


def test_last_n_forgets_beyond_its_window():
    b = add_baselines(_frame(), stat="receptions", last_n=2)
    w5 = b[b["week"] == 5].iloc[0]
    assert w5["base_last_n"] == pytest.approx(3.5), "mean(3,4), not mean(1..4)"


def test_ewma_leans_on_recent_games_more_than_a_flat_mean():
    b = add_baselines(_frame(), stat="receptions", halflife=1.0)
    w5 = b[b["week"] == 5].iloc[0]
    assert w5["base_ewma"] > w5["base_season_mean"], "series is rising, so recency wins"


def test_team_volume_is_predicted_never_observed():
    """Handing the model the real team volume would reconstruct the answer: usage share x
    TRUE volume is almost exactly the stat. It would look like a triumph and mean nothing."""
    f = _frame()
    f = pd.concat([f, f.assign(player_id="B", receptions=1, targets=2)], ignore_index=True)
    out = add_team_volume(f.sort_values("kickoff"), denom="targets")
    wk1 = out[out["week"] == 1]
    assert wk1["team_volume_pred"].isna().all(), "week 1 has no prior team games"
    wk2 = out[out["week"] == 2].iloc[0]
    assert wk2["team_volume_pred"] == pytest.approx(4.0), "week 1 targets were 2 + 2"


def test_conversion_is_one_when_the_stat_is_its_own_opportunity():
    """Carries are both the opportunity and the stat, so the conversion term must drop out
    rather than introduce a spurious rate."""
    f = add_team_volume(add_baselines(_frame(), stat="targets"), denom="targets")
    f["share"] = 0.5
    out = add_usage_model(f, share_col="share", stat="targets", denom="targets")
    assert (out["model_conv_rate"] == 1.0).all()


def test_scoring_uses_the_same_rows_for_every_estimator():
    """Scoring each estimator on whatever subset it happens to cover would flatter whichever
    one is most often missing."""
    f = add_team_volume(add_baselines(_frame(), stat="receptions"), denom="targets")
    f["share"] = 0.5
    f = add_usage_model(f, share_col="share", stat="receptions", denom="targets")
    table = score(f, stat="receptions", min_prior=1)
    assert table["n"].nunique() == 1, "every estimator must be scored on identical rows"


def test_score_reports_the_gap_to_the_best_baseline():
    """The headline number is not the model's MAE, it is whether the model beat the best
    thing you could have done without it."""
    f = add_team_volume(add_baselines(_frame(), stat="receptions"), denom="targets")
    f["share"] = 0.5
    f = add_usage_model(f, share_col="share", stat="receptions", denom="targets")
    table = score(f, stat="receptions", min_prior=1)
    assert "vs_best_baseline" in table.columns
    best = table[table["estimator"].str.startswith("base_")]["vs_best_baseline"].min()
    assert best == pytest.approx(0.0), "the best baseline is the zero point"
