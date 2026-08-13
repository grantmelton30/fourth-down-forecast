#!/usr/bin/env python3
"""Refresh only lightweight live schedules/lines for daily public publication."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from sport import load_adapter  # noqa: E402


def main() -> int:
    # NFL refreshes schedules (including nflverse's published line) without rebuilding
    # historical PBP. NCAA's budgeted client refreshes only the live games and lines
    # payloads here; historical artifacts remain permanent.
    nfl = load_adapter("nfl", ROOT)
    nfl._module("ingest").load_schedules(nfl._cfg().train_seasons, refresh=True)

    ncaa = load_adapter("ncaa", ROOT)
    cfg = ncaa._cfg()
    client = ncaa._module("cfbd_client").BudgetedCFBD(cfg)
    ingest = ncaa._module("ingest")
    ingest.load_games(client, [cfg.seasons.current])
    ingest.load_lines(client, cfg, [cfg.seasons.current])
    print("refreshed live NFL schedule and NCAA games/market lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
