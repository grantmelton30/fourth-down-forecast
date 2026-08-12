#!/usr/bin/env python3
"""Build the next priced NFL/NCAA slate and append immutable public records."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from prediction_contract import PredictionLedger  # noqa: E402
from publication import build_public_record  # noqa: E402
from slate_builder import (calibration_from_weights, forecast_from_sim, market_for_game,
                           source_digest)  # noqa: E402
from sport import load_adapter  # noqa: E402


def manual_lines(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["league", "game_id", "provider", "spread", "total",
                                     "observed_at"])
    frame = pd.read_csv(path)
    required = {"league", "game_id", "provider", "spread", "total", "observed_at"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"market_lines.csv missing columns: {sorted(missing)}")
    return frame


def unavailable_for(league: str, evidence) -> tuple[str, ...]:
    missing = []
    if evidence is None or not evidence.is_consensus:
        missing.append("three-source market consensus")
    if league == "nfl" and not (ROOT / "nfl-model/data/manual/availability.csv").exists():
        missing.append("confirmed live player availability")
    if league == "ncaa" and not (ROOT / "ncaa-model/data/manual/preseason_features.csv").exists():
        missing.append("confirmed QB/coordinator continuity")
    return tuple(missing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=("nfl", "ncaa", "all"), default="all")
    parser.add_argument("--ledger", type=Path, default=ROOT / "data/predictions.jsonl")
    parser.add_argument("--market-lines", type=Path,
                        default=ROOT / "data/manual/market_lines.csv")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    lines, ledger = manual_lines(args.market_lines), PredictionLedger(args.ledger)
    built, skipped = 0, 0
    leagues = ("nfl", "ncaa") if args.league == "all" else (args.league,)
    for league in leagues:
        adapter = load_adapter(league, ROOT)
        period = adapter.default_period()
        if period is None:
            print(f"{league}: no priced period available")
            continue
        season, week = period
        games = adapter.games(season, week)
        for _, game in games.iterrows():
            kickoff = pd.to_datetime(game.get("kickoff"), utc=True, errors="coerce")
            if pd.isna(kickoff) or kickoff.to_pydatetime() <= now:
                skipped += 1
                continue
            sim = adapter.simulate(str(game["game_id"]))
            if sim is None:
                skipped += 1
                continue
            independent = forecast_from_sim(sim)
            market, evidence = market_for_game(game, lines, league=league, as_of=now)
            calibrated = calibration_from_weights(
                independent, market, adapter.blend_weights(), league)
            unavailable = unavailable_for(league, evidence)
            record = build_public_record(
                league=league, game_id=str(game["game_id"]), season=season, week=week,
                kickoff=kickoff.isoformat(), home_team=str(game["home_team"]),
                away_team=str(game["away_team"]), independent=independent, market=market,
                market_evidence=evidence, calibrated=calibrated,
                model_version=f"{league}-{source_digest(adapter.profile.repo)}",
                data_cutoff=now.isoformat(), unavailable_features=unavailable,
                bets_allowed=adapter.bets_allowed(), generated_at=now.isoformat(),
            )
            built += int(ledger.append(record))
    print(f"appended {built} predictions; skipped {skipped} unavailable/past games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
