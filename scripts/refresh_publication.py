#!/usr/bin/env python3
"""Refresh sources/models and transactionally build the public static artifacts."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checked(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True, env=os.environ.copy())


def main() -> int:
    python = sys.executable
    checked(python, "nfl-model/run_backtest.py", "--refresh",
            "--diagnostic-exit-zero")
    checked(python, "ncaa-model/run_backtest.py", "--diagnostic-exit-zero")

    with tempfile.TemporaryDirectory(prefix="football-publish-") as directory:
        stage = Path(directory)
        ledger = stage / "predictions.jsonl"
        predictions = stage / "predictions.json"
        explorer = stage / "explorer.json"
        source_ledger = ROOT / "data/predictions.jsonl"
        if source_ledger.exists():
            shutil.copy2(source_ledger, ledger)
        checked(python, "scripts/build_slate.py", "--ledger", str(ledger))
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
        for source, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
    print("source refresh and fail-closed publication completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
