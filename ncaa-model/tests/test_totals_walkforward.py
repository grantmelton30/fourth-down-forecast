from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[1] / "analysis" / "totals_walkforward.py"
SPEC = importlib.util.spec_from_file_location("totals_walkforward", PATH)
totals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(totals)


def frame():
    rows = []
    for season in range(2019, 2026):
        for game in range(340):
            pace = 20 + (game % 11) / 5
            efficiency = ((game * 7) % 17 - 8) / 10
            market = 48 + (game % 9)
            residual = .35 * efficiency + .08 * (pace - 21) + np.sin(game) * .2
            rows.append({
                "game_id": f"{season}-{game}", "season": season, "week": 1 + game % 13,
                "fbs_only": True, "restricted": game % 3 > 0, "home_p5": game % 4 == 0,
                "away_p5": game % 5 == 0, "actual_total": market + residual,
                "total_close": market, "spread_close": game % 40 - 20,
                "model_total_pure": market + efficiency,
                "model_total_market_adjusted": market + efficiency / 2,
                "pace_sum": pace, "eff_sum": efficiency, "net_diff": game % 15 - 7,
                "success_rate_matchup_sum": efficiency / 3,
                "explosive_rate_matchup_sum": efficiency / 4,
                "havoc_avoidance_matchup_sum": efficiency / 5,
                "red_zone_td_rate_matchup_sum": efficiency / 6,
            })
    return pd.DataFrame(rows)


def test_current_season_outcomes_cannot_change_its_predictions():
    original = frame()
    changed = original.copy()
    changed.loc[changed.season == 2025, "actual_total"] += 1000
    columns = ["game_id", "market", "bias", *totals.FAMILIES, "adaptive", "adaptive_choice"]
    first = totals.walkforward_predictions(original)
    second = totals.walkforward_predictions(changed)
    first = first[first.season == 2025][columns].reset_index(drop=True)
    second = second[second.season == 2025][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_adaptive_choice_uses_two_prior_seasons_only():
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for game in range(200):
            actual = 50.0
            rows.append({"season": season, "week": 1, "actual_total": actual,
                         "market": 51.0, "bias": 52.0,
                         "pure_residual": 50.0 if season < 2024 else 80.0,
                         "tempo_efficiency": 53.0, "play_quality": 54.0})
    result = totals.apply_adaptive(pd.DataFrame(rows))
    assert result.groupby("season").adaptive_choice.first().to_dict() == {
        2022: "market", 2023: "market", 2024: "pure_residual", 2025: "market",
    }


def test_prepare_rejects_duplicate_game_identity():
    duplicated = pd.concat([frame().iloc[:1], frame().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per game_id"):
        totals.prepare(duplicated)


def test_report_frame_label_never_exposes_an_absolute_path(tmp_path):
    assert totals._frame_label(totals.ROOT / "data" / "cache" / "x.parquet") == "data/cache/x.parquet"
    assert totals._frame_label(tmp_path / "external.parquet") == "external.parquet"


def test_metrics_reward_a_better_paired_forecast():
    predictions = totals.walkforward_predictions(frame())
    result = {row["model"]: row for row in totals.metrics(predictions, n_boot=100)}
    assert result["tempo_efficiency"]["rmse_gain_vs_market"] > 0
    assert result["market"]["rmse_gain_vs_market"] == pytest.approx(0)
    assert all(row["coverage"] <= 1 for row in result.values())


def test_p1_reproduction_uses_pure_model_and_full_registered_window():
    data = frame()
    data.loc[data.season == 2021, "model_total_market_adjusted"] = (
        data.loc[data.season == 2021, "total_close"] - 5
    )
    result = totals.p1_summary(totals.prepare(data))
    assert result["n"] > 0
    assert result["wins"] > result["losses"]


def test_unavailable_fixed_diagnostic_is_reported():
    predictions = totals.walkforward_predictions(frame())
    predictions = predictions[predictions.week >= 4]
    early = [row for row in totals.diagnostics(predictions) if row["group"] == "weeks_1_3"]
    assert early and all(row["n"] == 0 and row["available"] is False for row in early)


def test_market_total_diagnostics_use_only_pregame_line_membership():
    predictions = totals.walkforward_predictions(frame())
    diagnostics = totals.diagnostics(predictions)
    low = [row for row in diagnostics
           if row["group"] == "market_total_at_most_50" and row["model"] == "market"]
    high = [row for row in diagnostics
            if row["group"] == "market_total_over_50" and row["model"] == "market"]
    assert low[0]["n"] == int(predictions.total_close.le(50).sum())
    assert high[0]["n"] == int(predictions.total_close.gt(50).sum())
    assert not any(row["group"] in {"market_over", "market_under"} for row in diagnostics)
