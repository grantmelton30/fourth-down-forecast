"""Transform immutable model/market inputs into the public four-output contract."""
from __future__ import annotations

import numpy as np

from prediction_contract import (Forecast, MarketEvidence, PredictionRecord,
                                 build_prediction_record)


def calibrated_forecast(
    independent: Forecast, market: Forecast, *, spread_intercept: float,
    spread_weight: float, total_intercept: float, total_weight: float,
    source: str = "expanding-season residual blend",
) -> Forecast:
    """Market plus only the historically validated share of model disagreement."""
    spread = market.spread + float(spread_intercept) + float(spread_weight) * (
        independent.spread - market.spread
    )
    total = market.total + float(total_intercept) + float(total_weight) * (
        independent.total - market.total
    )
    return Forecast.from_spread_total(spread, total, None, source)


def confidence_label(
    *, unavailable_features: tuple[str, ...], market_evidence: MarketEvidence | None,
    calibrated: Forecast | None, bets_allowed: bool,
) -> str:
    """A disclosure label, never a disguised betting grade."""
    if unavailable_features or market_evidence is None:
        return "limited data"
    if not market_evidence.is_consensus:
        return "single-line baseline"
    if calibrated is None:
        return "uncalibrated"
    return "validated" if bets_allowed else "projection only"


def build_public_record(
    *, league: str, game_id: str, season: int, week: int, kickoff: str,
    home_team: str, away_team: str, independent: Forecast,
    market: Forecast | None, market_evidence: MarketEvidence | None,
    calibrated: Forecast | None, model_version: str, data_cutoff: str,
    unavailable_features: tuple[str, ...] = (), bets_allowed: bool = False,
    generated_at: str | None = None,
) -> PredictionRecord:
    return build_prediction_record(
        league=league, game_id=game_id, season=season, week=week, kickoff=kickoff,
        home_team=home_team, away_team=away_team, independent=independent,
        market=market, calibrated=calibrated, model_version=model_version,
        data_cutoff=data_cutoff, generated_at=generated_at,
        confidence=confidence_label(
            unavailable_features=unavailable_features, market_evidence=market_evidence,
            calibrated=calibrated, bets_allowed=bets_allowed,
        ),
        unavailable_features=unavailable_features, market_evidence=market_evidence,
    )


def weighted_quantile(values, weights, probability: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cumulative, probability, side="left")])


__all__ = ["build_public_record", "calibrated_forecast", "confidence_label",
           "weighted_quantile"]
