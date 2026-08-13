"""The evidence that authorized a publication must be committed alongside it.

Both provenance artifacts -- the gate decision and the calibration weights -- lived only
in an ephemeral CI cache. Nothing in version control recorded which gate results or which
blend weights produced a given published slate, so a claim on the site could not be traced
back to the measurement behind it, and an evicted cache changed the answer silently.

The snapshot is deliberately not allowed to invent anything: an artifact that is absent is
recorded as absent, because "we did not measure this" is itself part of the record.
"""
from __future__ import annotations

import hashlib
import json

from evidence_snapshot import snapshot_evidence


def _cache(tmp_path, league="ncaa", *, gate=True, weights=True):
    cache = tmp_path / league / "cache"
    cache.mkdir(parents=True)
    if gate:
        (cache / "gate_results.json").write_text(json.dumps(
            {"schema_version": 1, "league": league, "model_version": f"{league}-abc",
             "bets_allowed": False, "blockers": ["GATE_RMSE_SPREAD"], "gates": []}))
    if weights:
        (cache / "blend_weights.json").write_text(json.dumps(
            {"schema_version": 1, "league": league, "model_version": f"{league}-abc",
             "weights": {"b_model_total": 0.4}}))
    return cache


def test_snapshot_copies_both_artifacts_and_records_their_digests(tmp_path):
    cache = _cache(tmp_path)
    out = tmp_path / "evidence"

    manifest = snapshot_evidence(out, [("ncaa", cache)])

    copied = out / "ncaa" / "gate_results.json"
    assert copied.exists()
    assert (out / "ncaa" / "blend_weights.json").exists()

    entry = manifest["leagues"]["ncaa"]["artifacts"]["gate_results.json"]
    assert entry["present"] is True
    assert entry["sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert entry["model_version"] == "ncaa-abc"


def test_absent_evidence_is_recorded_as_absent_not_omitted(tmp_path):
    cache = _cache(tmp_path, weights=False)
    out = tmp_path / "evidence"

    manifest = snapshot_evidence(out, [("ncaa", cache)])

    entry = manifest["leagues"]["ncaa"]["artifacts"]["blend_weights.json"]
    assert entry["present"] is False
    assert entry["sha256"] is None
    assert not (out / "ncaa" / "blend_weights.json").exists()


def test_snapshot_surfaces_a_disagreement_between_the_two_artifacts(tmp_path):
    """Gate results and weights from different builds must not be published together."""
    cache = _cache(tmp_path)
    (cache / "blend_weights.json").write_text(json.dumps(
        {"schema_version": 1, "league": "ncaa", "model_version": "ncaa-STALE",
         "weights": {"b_model_total": 0.4}}))

    manifest = snapshot_evidence(tmp_path / "evidence", [("ncaa", cache)])

    assert manifest["leagues"]["ncaa"]["consistent"] is False
    assert "ncaa" in manifest["inconsistent_leagues"]


def test_consistent_when_both_artifacts_share_a_model_version(tmp_path):
    manifest = snapshot_evidence(tmp_path / "evidence", [("ncaa", _cache(tmp_path))])
    assert manifest["leagues"]["ncaa"]["consistent"] is True
    assert manifest["inconsistent_leagues"] == []


def test_stale_snapshot_files_are_replaced_not_merged(tmp_path):
    out = tmp_path / "evidence"
    (out / "ncaa").mkdir(parents=True)
    (out / "ncaa" / "blend_weights.json").write_text('{"stale": true}')

    snapshot_evidence(out, [("ncaa", _cache(tmp_path, weights=False))])

    # The previous run's weights must not linger and appear to authorize this build.
    assert not (out / "ncaa" / "blend_weights.json").exists()


def test_manifest_is_written_to_disk_and_is_stable_json(tmp_path):
    out = tmp_path / "evidence"
    manifest = snapshot_evidence(out, [("ncaa", _cache(tmp_path))])

    written = json.loads((out / "manifest.json").read_text())
    assert written["leagues"] == manifest["leagues"]
