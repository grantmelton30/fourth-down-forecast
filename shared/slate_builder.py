"""Pure helpers for converting a sport adapter's slate into prediction records."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from market_consensus import aggregate_market_consensus
from prediction_contract import Forecast, MarketEvidence
from publication import (build_public_record, calibrated_forecast,
                         weighted_quantile)


def source_digest(repo: Path) -> str:
    digest = hashlib.sha256()
    paths = (sorted((repo / "src").glob("*.py"))
             + sorted((repo / "config").glob("*.yaml"))
             + sorted((repo.parent / "shared").glob("*.py")))
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def market_for_game(
    game: pd.Series, manual_lines: pd.DataFrame, *, league: str,
    as_of: "str | pd.Timestamp | None" = None,
) -> tuple[Forecast | None, MarketEvidence | None]:
    lines = pd.DataFrame()
    if manual_lines is not None and len(manual_lines):
        lines = manual_lines[
            manual_lines["league"].astype(str).str.lower().eq(league)
            & manual_lines["game_id"].astype(str).eq(str(game["game_id"]))
        ]
        lines = lines.copy()
        lines["observed_at"] = pd.to_datetime(lines["observed_at"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        kickoff = pd.to_datetime(game.get("kickoff"), utc=True, errors="coerce")
        if pd.notna(kickoff):
            cutoff = min(cutoff, kickoff - pd.Timedelta(microseconds=1))
        lines = lines[lines["observed_at"] <= cutoff]
    if len(lines):
        consensus = aggregate_market_consensus(lines).iloc[0]
        if pd.notna(consensus["spread"]) and pd.notna(consensus["total"]):
            evidence = MarketEvidence(
                label=("multi-book median" if consensus["is_consensus"]
                       else "submitted line baseline"),
                is_consensus=bool(consensus["is_consensus"]),
                book_count_spread=int(consensus["book_count_spread"]),
                book_count_total=int(consensus["book_count_total"]),
                providers=tuple(consensus["providers"]),
                observed_at=consensus["observed_at"].isoformat(),
                spread_dispersion=float(consensus["spread_dispersion"]),
                total_dispersion=float(consensus["total_dispersion"]),
            )
            return Forecast.from_spread_total(
                consensus["spread"], consensus["total"], None, evidence.label
            ), evidence
    spread, total = game.get("spread_line"), game.get("total_line")
    if pd.isna(spread) or pd.isna(total):
        return None, None
    provider = str(game.get("provider") or (
        "nflverse published line" if league == "nfl" else "CFBD selected provider"))
    evidence = MarketEvidence("published line baseline", False, 1, 1, (provider,))
    return Forecast.from_spread_total(spread, total, None, evidence.label), evidence


def forecast_from_sim(sim) -> Forecast:
    weights = np.asarray(getattr(sim, "weights", np.ones(len(sim.margins))), dtype=float)
    weights = weights / weights.sum()
    spread = float(weights @ np.asarray(sim.margins, dtype=float))
    total = float(weights @ np.asarray(sim.totals, dtype=float))
    win = float(weights[np.asarray(sim.margins) > 0].sum())
    return Forecast.from_spread_total(
        spread, total, win, "independent football simulation",
        interval_80_low=weighted_quantile(sim.margins, weights, .10),
        interval_80_high=weighted_quantile(sim.margins, weights, .90),
    )


def calibration_from_weights(independent: Forecast, market: Forecast | None,
                             weights: dict, league: str) -> Forecast | None:
    if market is None or not weights:
        return None
    spread_weight = weights.get("b_model" if league == "nfl" else "b_model_spread")
    total_weight = weights.get("b_model_total")
    if spread_weight is None or total_weight is None:
        return None
    return calibrated_forecast(
        independent, market, spread_intercept=weights.get("a_spread", 0),
        spread_weight=spread_weight, total_intercept=weights.get("a_total", 0),
        total_weight=total_weight,
    )


__all__ = ["calibration_from_weights", "forecast_from_sim", "market_for_game",
           "source_digest"]
