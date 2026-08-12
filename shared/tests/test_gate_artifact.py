from __future__ import annotations

import json

import pytest

from gate_artifact import GateArtifactError, load_gate_artifact, write_gate_artifact


def test_missing_required_gate_is_written_as_blocker(tmp_path):
    path = write_gate_artifact(
        tmp_path,
        league="nfl",
        model_version="test",
        data_cutoff="2025-12-31",
        gates=[{"name": "GATE_BLEND_INFORMATIVE", "passed": True}],
        promotion_names={"GATE_BLEND_INFORMATIVE", "GATE_RMSE_SPREAD"},
    )
    payload = load_gate_artifact(path, expected_league="nfl")
    assert payload["bets_allowed"] is False
    assert "GATE_RMSE_SPREAD" in payload["blockers"]
    assert "GATE_NO_LOOKAHEAD" in payload["blockers"]


def test_malformed_or_internally_inconsistent_artifact_fails_closed(tmp_path):
    path = write_gate_artifact(
        tmp_path,
        league="ncaa",
        model_version="test",
        data_cutoff=None,
        gates=[{"name": "GATE_RMSE_TOTAL", "passed": False}],
        promotion_names={"GATE_RMSE_TOTAL"},
    )
    payload = json.loads(path.read_text())
    payload["bets_allowed"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(GateArtifactError, match="disagrees"):
        load_gate_artifact(path, expected_league="ncaa")


def test_unknown_schema_fails_closed(tmp_path):
    path = tmp_path / "gate_results.json"
    path.write_text(json.dumps({"schema_version": 999, "gates": [{}]}))
    with pytest.raises(GateArtifactError, match="unknown gate schema"):
        load_gate_artifact(path)


def test_artifact_from_different_model_version_fails_closed(tmp_path):
    path = write_gate_artifact(
        tmp_path,
        league="nfl",
        model_version="nfl-old",
        data_cutoff="2025-12-31",
        gates=[{"name": "GATE_BLEND_INFORMATIVE", "passed": True}],
        promotion_names={"GATE_BLEND_INFORMATIVE"},
    )
    with pytest.raises(GateArtifactError, match="different model"):
        load_gate_artifact(
            path,
            expected_league="nfl",
            expected_model_version="nfl-new",
        )
