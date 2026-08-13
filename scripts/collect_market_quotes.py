#!/usr/bin/env python3
"""Capture timestamped market quotes into the append-only ledger (Objective 1).

Deliberately independent of the model. This runs on its own schedule, spends at most a
couple of API calls, touches no cached football data and rebuilds nothing -- because the
one thing that cannot be recovered later is an observation nobody wrote down, and that
capture must not be coupled to whether a backtest happens to be healthy.

A run that cannot reach a source records the failure against the affected games rather
than returning quietly, so a gap in the archive can always be attributed to an outage
rather than to a market that never opened.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from market_ledger import MarketQuoteLedger  # noqa: E402
from market_sources import (CFBD_SOURCE, cfbd_quotes, nflverse_quotes,  # noqa: E402
                            source_failures)
from sport import load_adapter  # noqa: E402

COLLECTOR_VERSION = "market-collector/1"


def collect_ncaa(ledger: MarketQuoteLedger, observed_at: str) -> int:
    adapter = load_adapter("ncaa", ROOT)
    cfg = adapter._cfg()
    season = int(cfg.seasons.current)
    client = adapter._module("cfbd_client").BudgetedCFBD(cfg)
    try:
        # Forced fresh: a cached payload would be stamped with an observation time it
        # did not have. One call per run is a negligible share of the free tier.
        payload = client.call("lines", f"lines_{season}", ttl_hours=0.0, year=season)
    except Exception as exc:  # noqa: BLE001 - an outage is data, not a crash
        print(f"ncaa: market source failed -- {type(exc).__name__}: {exc}")
        period = adapter.default_period()
        games = adapter.games(season, period[1]) if period else pd.DataFrame()
        rows = [{"season": season, "week": int(g.get("week", 0)),
                 "game_id": str(g["game_id"]), "home_team": g.get("home_team"),
                 "away_team": g.get("away_team"), "kickoff": g["kickoff"]}
                for _, g in games.iterrows()] if len(games) else []
        return ledger.extend_if_changed(source_failures(
            rows, league="ncaa", observed_at=observed_at,
            collector_version=COLLECTOR_VERSION, source_id=CFBD_SOURCE,
            detail=f"{type(exc).__name__}: {exc}"))
    quotes = cfbd_quotes(payload, observed_at=observed_at,
                         collector_version=COLLECTOR_VERSION)
    written = ledger.extend_if_changed(quotes)
    print(f"ncaa: {len(payload)} games, {len(quotes)} observations, {written} new")
    return written


def collect_nfl(ledger: MarketQuoteLedger, observed_at: str) -> int:
    adapter = load_adapter("nfl", ROOT)
    try:
        schedules = adapter._module("ingest").load_schedules(
            adapter._cfg().train_seasons, refresh=True)
    except Exception as exc:  # noqa: BLE001
        print(f"nfl: market source failed -- {type(exc).__name__}: {exc}")
        return 0
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    upcoming = schedules[pd.to_datetime(schedules["gameday"], errors="coerce") >= today]
    quotes = nflverse_quotes(upcoming, observed_at=observed_at,
                             collector_version=COLLECTOR_VERSION)
    written = ledger.extend_if_changed(quotes)
    print(f"nfl: {len(upcoming)} upcoming games, {len(quotes)} observations, "
          f"{written} new")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("nfl", "ncaa", "all"), default="all")
    parser.add_argument("--ledger", type=Path,
                        default=ROOT / "data/market_quotes.jsonl")
    args = parser.parse_args()

    ledger = MarketQuoteLedger(args.ledger)
    observed_at = datetime.now(timezone.utc).isoformat()
    written = 0
    if args.league in ("ncaa", "all"):
        written += collect_ncaa(ledger, observed_at)
    if args.league in ("nfl", "all"):
        written += collect_nfl(ledger, observed_at)
    print(f"appended {written} new market observations to {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
