#!/usr/bin/env python3
"""Fail-closed validation for candidate public prediction and explorer artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from prediction_contract import PredictionLedger, SCHEMA_VERSION  # noqa: E402


def validate(ledger_path: Path, predictions_path: Path, explorer_path: Path) -> None:
    records = PredictionLedger(ledger_path).records()
    if not records:
        raise ValueError("candidate ledger has no predictions")
    feed = json.loads(predictions_path.read_text())
    explorer = json.loads(explorer_path.read_text())
    if feed.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("prediction feed schema does not match the immutable contract")
    if int(feed.get("record_count", -1)) != len(feed.get("records", [])):
        raise ValueError("prediction feed record_count is inconsistent")
    if set(feed.get("leagues", ())) != {"nfl", "ncaa"}:
        raise ValueError("prediction feed must retain both leagues")
    if set(explorer.get("leagues", {})) != {"nfl", "ncaa"}:
        raise ValueError("explorer must retain both leagues")
    for league, payload in explorer["leagues"].items():
        if not payload.get("ratings") or not payload.get("schedule"):
            raise ValueError(f"{league} explorer is missing ratings or schedule")
    for record in feed["records"]:
        independent = record.get("independent") or {}
        for field in ("spread", "total", "home_score", "away_score"):
            value = independent.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"non-finite independent {field}")
        low, high = independent.get("interval_80_low"), independent.get("interval_80_high")
        if low is not None and high is not None and not low < high:
            raise ValueError("forecast interval is invalid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=ROOT / "data/predictions.jsonl")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "web/api/v1/predictions.json")
    parser.add_argument("--explorer", type=Path,
                        default=ROOT / "web/api/v1/explorer.json")
    args = parser.parse_args()
    validate(args.ledger, args.predictions, args.explorer)
    print("publication validation passed for NFL and NCAA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
