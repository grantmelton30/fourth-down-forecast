#!/usr/bin/env python3
"""Refresh sources/models and transactionally build the public static artifacts."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from evidence_snapshot import snapshot_evidence  # noqa: E402
from gate_artifact import FILENAME as GATE_FILENAME  # noqa: E402
from sport import load_adapter  # noqa: E402


def checked(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True, env=os.environ.copy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-only", action="store_true",
        help="reuse validated model caches and refresh the public slate/market quotes",
    )
    args = parser.parse_args()
    python = sys.executable
    checked(python, "scripts/refresh_market_data.py")
    if not args.market_only:
        checked(python, "nfl-model/run_backtest.py", "--refresh",
                "--diagnostic-exit-zero")
        checked(python, "ncaa-model/run_backtest.py", "--diagnostic-exit-zero")

    with tempfile.TemporaryDirectory(prefix="football-publish-") as directory:
        stage = Path(directory)
        ledger = stage / "predictions.jsonl"
        predictions = stage / "predictions.json"
        explorer = stage / "explorer.json"
        shadow = stage / "ncaa_totals_shadow.jsonl"
        source_ledger = ROOT / "data/predictions.jsonl"
        source_shadow = ROOT / "data/internal/ncaa_totals_shadow.jsonl"
        if source_ledger.exists():
            shutil.copy2(source_ledger, ledger)
        if source_shadow.exists():
            shutil.copy2(source_shadow, shadow)
        checked(python, "scripts/build_slate.py", "--ledger", str(ledger),
                "--totals-shadow-ledger", str(shadow))
        checked(python, "scripts/publish_ledger.py", "--ledger", str(ledger),
                "--output", str(predictions))
        checked(python, "scripts/build_explorer.py", "--output", str(explorer))
        checked(python, "scripts/validate_publication.py", "--ledger", str(ledger),
                "--predictions", str(predictions), "--explorer", str(explorer))

        targets = {
            ledger: source_ledger,
            predictions: ROOT / "web/api/v1/predictions.json",
            explorer: ROOT / "web/api/v1/explorer.json",
        }
        if shadow.exists():
            targets[shadow] = source_shadow
        for source, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)

    # Commit the evidence that authorized what was just published, so the claim and the
    # measurement behind it travel together in history rather than in a cache that
    # expires.  Runs on market-only passes too: those republish under evidence they did
    # not regenerate, which is exactly the case worth having on record.
    manifest = snapshot_evidence(ROOT / "data/evidence", [
        ("nfl", ROOT / "nfl-model/data/cache"),
        ("ncaa", ROOT / "ncaa-model/data/cache"),
    ])
    for league, entry in sorted(manifest["leagues"].items()):
        missing = [name for name, a in entry["artifacts"].items() if not a["present"]]
        if missing:
            print(f"{league}: no validation evidence for {', '.join(sorted(missing))}")
    # An artifact that exists but is refused is the dangerous case: it looks measured and
    # behaves unmeasured. That state shipped undetected -- the model version was hashed
    # from an in-memory frame the reader could never reproduce, so every gate artifact was
    # repudiated and "no bets" looked identical whether verification worked or not.
    for league, entry in sorted(manifest["leagues"].items()):
        if not entry["artifacts"][GATE_FILENAME]["present"]:
            continue
        adapter = load_adapter(league, ROOT)
        if adapter.gate_artifact() is None:
            print(f"WARNING: {league} wrote a gate artifact that the adapter refuses; "
                  "picks are blocked by broken verification, not by a measured gate")
    if manifest["inconsistent_leagues"]:
        print("WARNING: gate results and calibration weights disagree on the model "
              f"version for: {', '.join(manifest['inconsistent_leagues'])}")
    print("source refresh and fail-closed publication completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
