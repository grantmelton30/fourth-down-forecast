import numpy as np
import pandas as pd
import pytest

from src.totals_shadow import FEATURES, fit_totals_shadow


def history():
    rows = []
    for season in range(2019, 2027):
        for game in range(160):
            pace = 20 + game % 7
            efficiency = (game % 9 - 4) / 10
            market = 50 + game % 5
            rows.append({"game_id": f"{season}-{game}", "season": season,
                         "fbs_only": True, "actual_total": market + efficiency,
                         "total_close": market, "pace_sum": pace, "eff_sum": efficiency,
                         "net_diff": game % 11 - 5})
    return pd.DataFrame(rows)


def test_fit_excludes_forecast_season_outcomes():
    original = history()
    changed = original.copy()
    changed.loc[changed.season == 2026, "actual_total"] += 1000
    row = original[original.season == 2026].iloc[[0]]
    first = fit_totals_shadow(original, 2026).predict(row, 52.5)
    second = fit_totals_shadow(changed, 2026).predict(row, 52.5)
    assert first == second
    assert first["training_through"] == 2025


def test_shadow_is_market_anchored_and_keeps_bias_comparator():
    model = fit_totals_shadow(history(), 2026)
    row = history().iloc[[0]]
    result = model.predict(row, 55.0)
    assert result["selected_model"] == "tempo_efficiency"
    assert result["shadow_total"] == result["tempo_total"]
    assert result["bias_total"] == pytest.approx(55.0 + model.bias)
    assert result["features"] == list(FEATURES)


def test_live_shadow_fails_closed_on_missing_feature():
    model = fit_totals_shadow(history(), 2026)
    row = history().iloc[[0]].copy()
    row["pace_sum"] = np.nan
    with pytest.raises(ValueError, match="must be complete"):
        model.predict(row, 55.0)
