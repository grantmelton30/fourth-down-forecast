"""Calibration weights must carry the same provenance the gate artifact already does.

`calibration_status` is a public claim: it decides whether the site labels the calibrated
output validated.  Before this contract existed the weights behind that claim were a bare,
unversioned JSON dict that any stale or foreign build could leave in the cache, while the
betting decision sitting next to it was rigorously bound to a model version.  These tests
close that asymmetry, and require a rejected artifact to be distinguishable from an absent
one so the difference can be disclosed rather than silently swallowed.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from calibration_evidence import (FILENAME, CalibrationEvidenceError,
                                  load_calibration_evidence,
                                  write_calibration_evidence)
from model_identity import build_model_version
from sport import SportAdapter, SportProfile


class _Adapter(SportAdapter):
    def __init__(self, repo, cache, frame):
        super().__init__(SportProfile(key="ncaa", label="NCAA", repo=repo))
        self.cache = cache
        self.frame = frame

    def _cache_dir(self):
        return self.cache

    def backtest_frame(self):
        return self.frame


def _repo(tmp_path):
    repo = tmp_path / "ncaa-model"
    config = repo / "config" / "ncaa.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("revision: 1\n")
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("")
    (repo / "run_backtest.py").write_text("")
    return repo, config


def _frame():
    return pd.DataFrame({
        "game_id": ["g1"],
        "season": [2025],
        "week": [14],
        "kickoff": [pd.Timestamp("2025-12-06", tz="UTC")],
        "model_spread": [3.0],
        "market_spread": [2.5],
        "actual_margin": [7.0],
    })


def _weights():
    return {"b_model_total": 0.4, "b_model_total_raw": 0.4, "t_model_total": 3.1,
            "b_model_spread": 0.0, "b_model_spread_raw": -0.2, "t_model_spread": 0.9}


def _write(cache, repo, config, frame, weights=None):
    return write_calibration_evidence(
        cache, league="ncaa",
        model_version=build_model_version("ncaa", repo, config, frame),
        data_cutoff="2025-12-06", weights=weights or _weights(),
    )


# -- the envelope ----------------------------------------------------------------------

def test_round_trip_preserves_weights_under_matching_identity(tmp_path):
    repo, config = _repo(tmp_path)
    frame = _frame()
    cache = tmp_path / "cache"
    path = _write(cache, repo, config, frame)
    assert path.name == FILENAME

    payload = load_calibration_evidence(
        path, expected_league="ncaa",
        expected_model_version=build_model_version("ncaa", repo, config, frame),
    )
    assert payload["weights"] == _weights()
    assert payload["league"] == "ncaa"


def test_load_rejects_weights_from_a_different_model_version(tmp_path):
    repo, config = _repo(tmp_path)
    frame = _frame()
    path = _write(tmp_path / "cache", repo, config, frame)

    with pytest.raises(CalibrationEvidenceError, match="different model"):
        load_calibration_evidence(
            path, expected_league="ncaa", expected_model_version="ncaa-somethingelse")


def test_load_rejects_a_legacy_unversioned_weight_dict(tmp_path):
    """The pre-contract format: a bare dict of coefficients with no provenance at all."""
    path = tmp_path / FILENAME
    path.write_text(json.dumps(_weights()))

    with pytest.raises(CalibrationEvidenceError, match="schema"):
        load_calibration_evidence(path, expected_league="ncaa")


def test_load_rejects_a_corrupt_or_absent_artifact(tmp_path):
    corrupt = tmp_path / FILENAME
    corrupt.write_text("{not json")
    with pytest.raises(CalibrationEvidenceError, match="unreadable"):
        load_calibration_evidence(corrupt, expected_league="ncaa")

    with pytest.raises(CalibrationEvidenceError, match="missing"):
        load_calibration_evidence(tmp_path / "absent" / FILENAME, expected_league="ncaa")


def test_load_rejects_a_league_mismatch(tmp_path):
    repo, config = _repo(tmp_path)
    path = _write(tmp_path / "cache", repo, config, _frame())
    with pytest.raises(CalibrationEvidenceError, match="league"):
        load_calibration_evidence(path, expected_league="nfl")


def test_weights_must_be_a_mapping_of_finite_numbers(tmp_path):
    repo, config = _repo(tmp_path)
    with pytest.raises(CalibrationEvidenceError, match="numeric"):
        _write(tmp_path / "cache", repo, config, _frame(),
               weights={"b_model_total": float("nan")})


def test_fit_metadata_survives_the_round_trip(tmp_path):
    """A real record carries the estimator, window and clamp list alongside the numbers."""
    repo, config = _repo(tmp_path)
    frame = _frame()
    record = {**_weights(), "covariance_type_total": "cluster", "n_clusters_total": 41,
              "window": "pre-2025", "clamped": ["b_model_spread"], "n_total": 812}
    path = _write(tmp_path / "cache", repo, config, frame, weights=record)

    payload = load_calibration_evidence(
        path, expected_league="ncaa",
        expected_model_version=build_model_version("ncaa", repo, config, frame),
    )
    assert payload["weights"] == record


# -- the adapter reads it fail-closed --------------------------------------------------

def test_adapter_returns_weights_only_for_the_matching_build(tmp_path):
    repo, config = _repo(tmp_path)
    frame = _frame()
    cache = tmp_path / "cache"
    _write(cache, repo, config, frame)
    adapter = _Adapter(repo, cache, frame)

    assert adapter.blend_weights() == _weights()
    assert adapter.calibration_block_reason() is None

    # Editing the config changes the model version; weights fitted against the previous
    # build must stop authorizing a "validated" calibration label.
    config.write_text("revision: 2\n")
    assert adapter.blend_weights() == {}


def test_adapter_distinguishes_absent_evidence_from_rejected_evidence(tmp_path):
    repo, config = _repo(tmp_path)
    frame = _frame()
    cache = tmp_path / "cache"
    adapter = _Adapter(repo, cache, frame)

    assert adapter.blend_weights() == {}
    absent = adapter.calibration_block_reason()
    assert absent is not None and "missing" in absent.lower()

    _write(cache, repo, config, frame)
    config.write_text("revision: 2\n")
    rejected = adapter.calibration_block_reason()
    assert rejected is not None and rejected != absent
    assert "model" in rejected.lower()


def test_published_reason_never_leaks_a_filesystem_path(tmp_path):
    """The operator message may name the file; the reason shown to readers may not."""
    repo, config = _repo(tmp_path)
    cache = tmp_path / "cache"
    adapter = _Adapter(repo, cache, _frame())

    reasons = [adapter.calibration_block_reason()]
    cache.mkdir()
    (cache / FILENAME).write_text("{not json")
    reasons.append(adapter.calibration_block_reason())
    _write(cache, repo, config, _frame())
    config.write_text("revision: 2\n")
    reasons.append(adapter.calibration_block_reason())

    for reason in reasons:
        assert reason
        assert str(tmp_path) not in reason
        assert "/" not in reason and FILENAME not in reason


def test_unversioned_cache_artifact_closes_calibration(tmp_path):
    """A stale cache left by an older build must not open the calibration gate."""
    repo, _ = _repo(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / FILENAME).write_text(json.dumps(_weights()))
    adapter = _Adapter(repo, cache, _frame())

    assert adapter.blend_weights() == {}
    assert "schema" in adapter.calibration_block_reason().lower()
