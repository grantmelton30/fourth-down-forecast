"""Copy the artifacts that authorized a publication into version control.

`gate_results.json` decides whether picks are allowed; `blend_weights.json` decides
whether the calibrated output may be called validated.  Both were written into a
gitignored cache that only GitHub Actions preserved, which left two problems: a published
claim could not be traced to the measurement behind it, and an evicted cache changed the
answer with nothing in the history to show it had changed.

Snapshotting them turns those artifacts into a committed audit trail.  Two rules keep the
trail honest.  Absent evidence is recorded as absent rather than skipped, because a gap in
measurement is a fact worth publishing.  And a stale copy from an earlier run is deleted
rather than left in place, so last week's weights can never sit in the directory looking
like this week's authorization.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from calibration_evidence import FILENAME as CALIBRATION_FILENAME
from gate_artifact import FILENAME as GATE_FILENAME

ARTIFACTS = (GATE_FILENAME, CALIBRATION_FILENAME)
MANIFEST = "manifest.json"


def _read_model_version(path: Path) -> "str | None":
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("model_version") if isinstance(payload, dict) else None


def snapshot_evidence(destination: Path, leagues: Iterable[tuple[str, Path]]) -> dict:
    """Copy each league's provenance artifacts under `destination` and describe them."""
    destination = Path(destination)
    manifest: dict = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "leagues": {},
        "inconsistent_leagues": [],
    }
    for league, cache_dir in leagues:
        cache_dir = Path(cache_dir)
        league_dir = destination / league
        league_dir.mkdir(parents=True, exist_ok=True)
        entries: dict = {}
        versions = set()
        for name in ARTIFACTS:
            source, target = cache_dir / name, league_dir / name
            if not source.exists():
                # Remove any copy left by an earlier run; an absent artifact is safer
                # than a stale one that reads as current authorization.
                target.unlink(missing_ok=True)
                entries[name] = {"present": False, "sha256": None, "model_version": None}
                continue
            payload = source.read_bytes()
            target.write_bytes(payload)
            version = _read_model_version(target)
            if version:
                versions.add(version)
            entries[name] = {
                "present": True,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "model_version": version,
            }
        # Two artifacts naming different builds cannot both describe this publication.
        consistent = len(versions) <= 1
        manifest["leagues"][league] = {
            "consistent": consistent,
            "model_version": next(iter(versions)) if len(versions) == 1 else None,
            "artifacts": entries,
        }
        if not consistent:
            manifest["inconsistent_leagues"].append(league)

    destination.mkdir(parents=True, exist_ok=True)
    (destination / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


__all__ = ["ARTIFACTS", "MANIFEST", "snapshot_evidence"]
