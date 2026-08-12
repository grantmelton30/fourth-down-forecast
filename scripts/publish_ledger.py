#!/usr/bin/env python3
"""Publish the latest immutable prediction per game as a versioned static API."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from prediction_contract import PredictionLedger, SCHEMA_VERSION  # noqa: E402


def build_feed(ledger: PredictionLedger) -> dict:
    latest = {}
    for record in ledger.records():
        key = (record.league, record.game_id)
        if key not in latest or record.generated_at > latest[key].generated_at:
            latest[key] = record
    records = sorted(
        latest.values(), key=lambda r: (r.kickoff, r.league, r.game_id)
    )
    generated = max((r.generated_at for r in records), default=None)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "record_count": len(records),
        "leagues": sorted({r.league for r in records}),
        "records": [r.to_dict() for r in records],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=ROOT / "data" / "predictions.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "web" / "api" / "v1" / "predictions.json")
    args = parser.parse_args()
    feed = build_feed(PredictionLedger(args.ledger))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(feed, indent=2, sort_keys=True) + "\n")
    print(f"published {feed['record_count']} records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
