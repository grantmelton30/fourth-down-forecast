"""Transform immutable model/market inputs into the public four-output contract."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prediction_contract import (Forecast, MarketEvidence, PredictionRecord,
                                 build_prediction_record)


@dataclass(frozen=True)
class QualityAssessment:
    label: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    pick_eligible: bool
    out_of_distribution: bool


def assess_quality(
    *, league: str, week: int, home_games_observed: int, away_games_observed: int,
    unavailable_features: tuple[str, ...], market_evidence: MarketEvidence | None,
    spread_difference: float | None, calibration_status: dict[str, bool],
    bets_allowed: bool, calibration_block_reason: str | None = None,
) -> QualityAssessment:
    reasons = list(unavailable_features)
    warnings: list[str] = []
    observed = min(int(home_games_observed), int(away_games_observed))
    if unavailable_features:
        label = "incomplete"
    elif league == "ncaa" and (int(week) <= 1 or observed == 0):
        label = "preseason"
    elif league == "ncaa" and (int(week) <= 3 or observed <= 2):
        label = "early-season"
    elif league == "ncaa" and (int(week) <= 5 or observed <= 4):
        label = "developing"
    else:
        label = "established"
    if market_evidence is None:
        reasons.append("no market reference")
    elif not market_evidence.is_consensus:
        reasons.append("market is not a three-source consensus")
    unvalidated = [m for m in ("spread", "total") if not calibration_status.get(m, False)]
    for market in unvalidated:
        reasons.append(f"{market} calibration is not validated")
    # Say why the evidence was refused, not merely that the label is absent. A build
    # that never fitted weights and a build whose weights were repudiated as belonging
    # to a different model both show no calibration; only one of them is routine.
    if unvalidated and calibration_block_reason:
        reasons.append(str(calibration_block_reason))
    difference = abs(float(spread_difference)) if spread_difference is not None else 0.0
    # A warning icon is an interruption, not a generic uncertainty badge.  Reserve it
    # for truly exceptional disagreement; ordinary Week 1 model/market differences are
    # already disclosed numerically and by the preseason quality label.
    out_of_distribution = difference >= 20.0
    if out_of_distribution:
        warnings.append("Extreme model/market disagreement; pick suppressed")
    pick_eligible = bool(
        bets_allowed and label == "established" and not reasons
        and not out_of_distribution and all(calibration_status.values())
    )
    return QualityAssessment(label, tuple(dict.fromkeys(reasons)), tuple(warnings),
                             pick_eligible, out_of_distribution)


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
    confidence: str | None = None, quality_reasons: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (), pick_eligible: bool = False,
    out_of_distribution: bool = False,
    calibration_status: dict[str, bool] | None = None,
    games_observed: dict[str, int] | None = None,
) -> PredictionRecord:
    return build_prediction_record(
        league=league, game_id=game_id, season=season, week=week, kickoff=kickoff,
        home_team=home_team, away_team=away_team, independent=independent,
        market=market, calibrated=calibrated, model_version=model_version,
        data_cutoff=data_cutoff, generated_at=generated_at,
        confidence=confidence or confidence_label(
            unavailable_features=unavailable_features, market_evidence=market_evidence,
            calibrated=calibrated, bets_allowed=bets_allowed,
        ),
        unavailable_features=unavailable_features, market_evidence=market_evidence,
        quality_reasons=quality_reasons, warnings=warnings,
        pick_eligible=pick_eligible, out_of_distribution=out_of_distribution,
        calibration_status=calibration_status, games_observed=games_observed,
    )


def weighted_quantile(values, weights, probability: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cumulative, probability, side="left")])


__all__ = ["QualityAssessment", "assess_quality", "build_public_record",
           "calibrated_forecast", "confidence_label", "weighted_quantile"]
