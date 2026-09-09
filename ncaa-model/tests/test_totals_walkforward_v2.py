from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "analysis" / "totals_walkforward_v2.py"
SPEC = importlib.util.spec_from_file_location("totals_walkforward_v2", PATH)
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)

V1_TEST_PATH = Path(__file__).with_name("test_totals_walkforward.py")
V1_TEST_SPEC = importlib.util.spec_from_file_location("totals_walkforward_v1_test", V1_TEST_PATH)
v1_test = importlib.util.module_from_spec(V1_TEST_SPEC)
V1_TEST_SPEC.loader.exec_module(v1_test)


def frame():
    data = v1_test.frame()
    data["home_team"] = "Home-" + (data.index % 40).astype(str)
    data["away_team"] = "Away-" + (data.index % 45).astype(str)
    data["kickoff"] = pd.to_datetime(
        data.season.astype(str) + "-09-01", utc=True
    ) + pd.to_timedelta(data.week * 7 + data.index % 5, unit="D")
    data["total_open"] = data.total_close - (data.index % 3 - 1)
    data["indoor"] = (data.index % 5 == 0).astype(float)
    data["wind_mph"] = (data.index % 17).astype(float)
    data["qb_delta_gap"] = (data.index % 9 - 4).astype(float)
    for column, divisor in (
        ("early_down_efficiency_matchup_sum", 2),
        ("starting_field_position_matchup_sum", 7),
        ("pass_block_success_matchup_sum", 8),
        ("run_block_success_matchup_sum", 9),
    ):
        data[column] = data.eff_sum / divisor
    return data


def test_same_kickoff_games_cannot_inform_each_other():
    data = frame().iloc[:2].copy()
    data["home_team"] = ["Shared", "Shared"]
    data["away_team"] = ["A", "B"]
    data["kickoff"] = pd.Timestamp("2021-09-01T18:00:00Z")
    data["actual_total"] = [100, 0]
    prepared = v2.prepare(data)
    assert prepared.team_memory_feature.tolist() == [0.0, 0.0]


def test_prior_result_changes_only_later_team_memory():
    data = frame().iloc[:3].copy()
    data["home_team"] = "Shared"
    data["away_team"] = ["A", "B", "C"]
    data["kickoff"] = pd.to_datetime([
        "2021-09-01T18:00:00Z", "2021-09-08T18:00:00Z", "2021-09-15T18:00:00Z"
    ])
    changed = data.copy()
    changed.loc[changed.index[1], "actual_total"] += 100
    first = v2.prepare(data).team_memory_feature
    second = v2.prepare(changed).team_memory_feature
    assert first.iloc[:2].equals(second.iloc[:2])
    assert first.iloc[2] != second.iloc[2]


def test_current_game_outcome_cannot_change_its_prediction():
    data = frame()
    target = data[(data.season == 2025)].index[100]
    game_id = data.loc[target, "game_id"]
    first = v2.walkforward_predictions(data).set_index("game_id")
    data.loc[target, "actual_total"] += 500
    second = v2.walkforward_predictions(data).set_index("game_id")
    for model in ("bias", "tempo_efficiency", *v2.FAMILIES):
        assert first.loc[game_id, model] == second.loc[game_id, model]


def test_walkforward_models_have_full_synthetic_coverage():
    predictions = v2.walkforward_predictions(frame())
    assert set(predictions.season.unique()) == {2022, 2023, 2024, 2025}
    assert predictions[["market", "bias", "tempo_efficiency", *v2.FAMILIES]].notna().all().all()
