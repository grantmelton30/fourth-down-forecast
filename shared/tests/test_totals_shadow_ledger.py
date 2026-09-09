import json

import pytest

from totals_shadow_ledger import append_shadow_record, validate_shadow_record


def record():
    return {
        "prediction_id": "p1", "game_id": "g1", "season": 2026, "week": 3,
        "home_team": "Home", "away_team": "Away",
        "kickoff": "2026-09-12T17:00:00Z", "generated_at": "2026-09-12T12:00:00Z",
        "data_cutoff": "2026-09-12T11:59:00Z",
        "market_observed_at": "2026-09-12T11:55:00Z",
        "market_label": "Book", "providers": ["Book"], "market_total": 52.5,
        "bias_total": 52.1, "tempo_total": 51.9, "shadow_total": 51.9,
        "spec_version": "T1-shadow-v1", "model_version": "model1",
        "selected_model": "tempo_efficiency", "training_through": 2025,
        "status": "prospective_shadow_not_a_pick",
    }


def test_append_is_idempotent_and_round_trips(tmp_path):
    path = tmp_path / "shadow.jsonl"
    assert append_shadow_record(path, record()) is True
    assert append_shadow_record(path, record()) is False
    assert validate_shadow_record(json.loads(path.read_text()))["prediction_id"] == "p1"


def test_conflicting_prediction_is_rejected(tmp_path):
    path = tmp_path / "shadow.jsonl"
    append_shadow_record(path, record())
    changed = record()
    changed["tempo_total"] = changed["shadow_total"] = 54.0
    with pytest.raises(ValueError, match="conflicting"):
        append_shadow_record(path, changed)


@pytest.mark.parametrize("field,value", [
    ("market_observed_at", "2026-09-12T18:00:00Z"),
    ("market_observed_at", "2026-09-12 11:00:00"),
    ("market_observed_at", "2026-09-12T05:59:59Z"),
    ("training_through", 2026),
    ("market_total", float("nan")),
])
def test_invalid_or_future_evidence_fails_closed(field, value):
    row = record()
    row[field] = value
    with pytest.raises(ValueError):
        validate_shadow_record(row)
