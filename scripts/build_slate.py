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
from publication import assess_quality, build_public_record  # noqa: E402
from slate_builder import (calibration_from_weights, calibration_permissions,
                           forecast_from_projection, forecast_from_sim, market_for_game,
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


def unavailable_for(league: str, adapter=None, game_id=None) -> tuple[str, ...]:
    """Reasons this game's evidence is short of what a fully-mature build could have.

    "Three-source market consensus" used to be a permanent entry here: this system deliberately
    sources 1-2 books per game (see market_consensus.py's minimum_books=3, which nothing here
    ever meets), so it was flagging every single published game as "incomplete" forever rather
    than describing a gap that closes with more data. Removed 2026-08-17; the exact book count
    and provider list are still fully disclosed via market_evidence, which says more than a
    boolean ever did.
    """
    missing = []
    if league == "nfl" and not (ROOT / "nfl-model/data/manual/availability.csv").exists():
        missing.append("confirmed live player availability")
    if league == "ncaa":
        if adapter is not None and hasattr(adapter, "preseason_missing"):
            missing.extend(adapter.preseason_missing(str(game_id)))
        elif not (ROOT / "ncaa-model/data/manual/preseason_features.csv").exists():
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
        league_lines = lines
        if league == "ncaa" and hasattr(adapter, "market_snapshots"):
            league_lines = pd.concat(
                [lines, adapter.market_snapshots()], ignore_index=True, sort=False
            )
        weights = adapter.blend_weights()
        permissions = calibration_permissions(weights, league)
        calibration_block = adapter.calibration_block_reason()
        if calibration_block:
            print(f"{league}: calibration closed -- {calibration_block}")
        for _, game in games.iterrows():
            kickoff = pd.to_datetime(game.get("kickoff"), utc=True, errors="coerce")
            if pd.isna(kickoff) or kickoff.to_pydatetime() <= now:
                skipped += 1
                continue
            sim = adapter.simulate(str(game["game_id"]))
            if sim is None:
                skipped += 1
                continue
            mean = (
                adapter.projection_mean(str(game["game_id"]))
                if league == "ncaa" and hasattr(adapter, "projection_mean") else None
            )
            uncertainty = (
                adapter.projection_uncertainty(str(game["game_id"]))
                if league == "ncaa" and hasattr(adapter, "projection_uncertainty")
                else {"multiplier": 1.0, "reasons": ()}
            )
            independent = (
                forecast_from_projection(
                    sim, spread=mean["spread"], total=mean["total"],
                    interval_multiplier=uncertainty["multiplier"],
                )
                if mean is not None else forecast_from_sim(sim)
            )
            market, evidence = market_for_game(
                game, league_lines, league=league, as_of=now)
            calibrated = calibration_from_weights(
                independent, market, weights, league)
            unavailable = unavailable_for(
                league, adapter=adapter, game_id=game["game_id"])
            observed = (
                adapter.games_observed(str(game["game_id"]))
                if hasattr(adapter, "games_observed") else {"home": 0, "away": 0}
            )
            spread_difference = (
                independent.spread - market.spread if market is not None else None
            )
            quality = assess_quality(
                league=league, week=week,
                home_games_observed=observed["home"],
                away_games_observed=observed["away"],
                unavailable_features=unavailable, market_evidence=evidence,
                spread_difference=spread_difference,
                calibration_status=permissions, bets_allowed=adapter.bets_allowed(),
                calibration_block_reason=calibration_block,
            )
            record = build_public_record(
                league=league, game_id=str(game["game_id"]), season=season, week=week,
                kickoff=kickoff.isoformat(), home_team=str(game["home_team"]),
                away_team=str(game["away_team"]), independent=independent, market=market,
                market_evidence=evidence, calibrated=calibrated,
                model_version=f"{league}-{source_digest(adapter.profile.repo)}",
                data_cutoff=now.isoformat(), unavailable_features=unavailable,
                bets_allowed=adapter.bets_allowed(), generated_at=now.isoformat(),
                confidence=quality.label,
                quality_reasons=tuple((*quality.reasons, *uncertainty["reasons"])),
                warnings=quality.warnings,
                pick_eligible=quality.pick_eligible,
                out_of_distribution=quality.out_of_distribution,
                calibration_status=permissions, games_observed=observed,
            )
            built += int(ledger.append(record))
    print(f"appended {built} predictions; skipped {skipped} unavailable/past games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
