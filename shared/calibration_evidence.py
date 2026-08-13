"""Provenance-bound calibration weights.

The blend weights decide `calibration_status`, which is the difference between the site
publishing a calibrated number as *validated* and publishing it as the market baseline.
That is a public claim, so it needs the same binding the betting decision already has in
`gate_artifact.py`: weights are only usable by the exact model version, configuration and
backtest evidence that produced them.

Before this envelope existed the weights were a bare dict of coefficients.  Nothing
recorded which build fitted them, so a cache left behind by an older configuration --
or restored from another run entirely -- silently authorized a validated label.  Every
rejection path here fails closed, and every rejection carries a distinct reason so the
publication layer can disclose *why* calibration is unavailable instead of treating a
missing artifact and a repudiated one as the same silence.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

SCHEMA_VERSION = 1
FILENAME = "blend_weights.json"


class CalibrationEvidenceError(ValueError):
    """Raised for any artifact that cannot be proven to belong to this build.

    Carries two texts on purpose. The message is for operators and may name paths; the
    `reason` is published to readers of the site, so it states the category of failure
    and never leaks a filesystem layout.
    """

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason


MISSING = "calibration evidence is missing for this build"
UNREADABLE = "calibration evidence is unreadable"
UNVERSIONED = "calibration evidence predates the provenance schema"
WRONG_LEAGUE = "calibration evidence belongs to another league"
WRONG_BUILD = "calibration evidence was fitted by a different model build"
BAD_WEIGHTS = "calibration weights are not usable numbers"


def _clean_value(key: str, value: object) -> object:
    """Coerce one weight-record field to a JSON-native value, rejecting junk numbers.

    A real fit record is not purely numeric: it carries the covariance estimator used,
    the fitting window, and the list of coefficients that were clamped.  That context is
    part of the evidence and is kept.  What must never survive is a non-finite
    coefficient, because NaN compares false against every threshold and would sail
    through the promotion checks as a silent "no".
    """
    if hasattr(value, "item"):  # NumPy/pandas scalars are not JSON-native.
        value = value.item()
    if isinstance(value, (list, tuple)):
        return [_clean_value(key, item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise CalibrationEvidenceError(
                f"calibration weights must be finite numeric values; {key!r} is not",
                reason=BAD_WEIGHTS)
        return value
    raise CalibrationEvidenceError(
        f"calibration weights must be JSON-native values; {key!r} is {type(value).__name__}",
        reason=BAD_WEIGHTS)


def _clean_weights(weights: object) -> dict:
    if not isinstance(weights, Mapping) or not weights:
        raise CalibrationEvidenceError(
            "calibration weights must be a non-empty mapping", reason=BAD_WEIGHTS)
    return {str(key): _clean_value(str(key), value) for key, value in weights.items()}


def write_calibration_evidence(
    cache_dir: Path, *, league: str, model_version: str, data_cutoff: str | None,
    weights: Mapping,
) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "league": str(league).lower(),
        "model_version": str(model_version),
        "data_cutoff": data_cutoff,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "producer": os.environ.get("CODEX_MODEL_BUILD", "run_backtest.py"),
        "weights": _clean_weights(weights),
    }
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def load_calibration_evidence(
    path: Path, *, expected_league: str | None = None,
    expected_model_version: str | None = None,
) -> dict:
    path = Path(path)
    if not path.exists():
        raise CalibrationEvidenceError(
            f"calibration evidence missing: {path}", reason=MISSING)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationEvidenceError(
            f"calibration evidence unreadable: {path}", reason=UNREADABLE) from exc
    if not isinstance(payload, dict):
        raise CalibrationEvidenceError(
            "unknown calibration evidence schema", reason=UNVERSIONED)
    # A pre-contract artifact is a bare coefficient dict with no schema marker.  It is
    # not upgradeable: nothing in it records which build produced it.
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationEvidenceError(
            "unknown calibration evidence schema", reason=UNVERSIONED)
    if expected_league and payload.get("league") != expected_league.lower():
        raise CalibrationEvidenceError(
            "calibration evidence league mismatch", reason=WRONG_LEAGUE)
    if (expected_model_version is not None
            and payload.get("model_version") != expected_model_version):
        raise CalibrationEvidenceError(
            "calibration evidence was produced by a different model/configuration version",
            reason=WRONG_BUILD)
    payload["weights"] = _clean_weights(payload.get("weights"))
    return payload


__all__ = ["BAD_WEIGHTS", "CalibrationEvidenceError", "FILENAME", "MISSING",
           "SCHEMA_VERSION", "UNREADABLE", "UNVERSIONED", "WRONG_BUILD", "WRONG_LEAGUE",
           "load_calibration_evidence", "write_calibration_evidence"]
