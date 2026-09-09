import pandas as pd

from src.totals_shadow_evaluation import evaluate_totals_shadow


def rows(n=320, *, tempo_error=1.0, bias_error=3.0, market_error=4.0):
    return pd.DataFrame([
        {
            "league": "ncaa", "game_id": f"g{i}", "season": 2026,
            "week": i % 8 + 1, "decision_hours": 1, "actual_total": 50 + i % 5,
            "shadow_spec_version": "T1-shadow-v1", "shadow_model_version": "frozen",
            "shadow_market_total": 50 + i % 5 + market_error,
            "shadow_bias_total": 50 + i % 5 + bias_error,
            "shadow_tempo_total": 50 + i % 5 + tempo_error,
        }
        for i in range(n)
    ])


def test_clear_paired_improvement_earns_review_only():
    result = evaluate_totals_shadow(rows(), replicates=200)
    assert result["status"] == "review_candidate"
    assert result["eligible_for_review"] is True
    assert result["tempo_gain_vs_market_ci90"][0] > 0
    assert result["tempo_gain_vs_bias_ci90"][0] > 0


def test_small_sample_remains_collecting_even_with_perfect_result():
    result = evaluate_totals_shadow(rows(40), replicates=50)
    assert result["status"] == "collecting"
    assert result["eligible_for_review"] is False


def test_repeated_game_or_changed_model_invalidates_evaluation():
    duplicate = pd.concat([rows(40), rows(1)], ignore_index=True)
    assert evaluate_totals_shadow(duplicate, replicates=20)["status"] == "invalid"
    changed = rows(40)
    changed.loc[0, "shadow_model_version"] = "changed"
    assert evaluate_totals_shadow(changed, replicates=20)["status"] == "invalid"


def test_wrong_horizon_and_spec_are_excluded():
    frame = rows(40)
    frame.loc[:19, "decision_hours"] = 24
    frame.loc[20:, "shadow_spec_version"] = "other"
    assert evaluate_totals_shadow(frame, replicates=20)["n_games"] == 0
