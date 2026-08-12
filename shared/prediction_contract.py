"""Versioned, immutable records shared by model pipelines and public presentation.

The independent forecast is structurally separate from market data.  A website may render
these fields but must never derive or overwrite them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


def _iso(value: str | datetime) -> str:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Forecast:
    spread: float
    total: float
    home_score: float
    away_score: float
    home_win_probability: float | None
    source: str
    interval_80_low: float | None = None
    interval_80_high: float | None = None

    @classmethod
    def from_spread_total(
        cls, spread: float, total: float, home_win_probability: float | None, source: str,
        interval_80_low: float | None = None, interval_80_high: float | None = None,
    ) -> "Forecast":
        spread, total = float(spread), float(total)
        if total < 0:
            raise ValueError("forecast total cannot be negative")
        if home_win_probability is not None and not 0 <= home_win_probability <= 1:
            raise ValueError("home_win_probability must be between zero and one")
        return cls(
            spread=spread, total=total,
            home_score=(total + spread) / 2.0,
            away_score=(total - spread) / 2.0,
            home_win_probability=(None if home_win_probability is None
                                  else float(home_win_probability)),
            source=str(source), interval_80_low=interval_80_low,
            interval_80_high=interval_80_high,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Forecast":
        return cls(**payload)


@dataclass(frozen=True)
class ForecastDifference:
    spread: float
    total: float


@dataclass(frozen=True)
class MarketEvidence:
    label: str
    is_consensus: bool
    book_count_spread: int
    book_count_total: int
    providers: tuple[str, ...]
    observed_at: str | None = None
    spread_dispersion: float | None = None
    total_dispersion: float | None = None


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    schema_version: int
    league: str
    game_id: str
    season: int
    week: int
    kickoff: str
    home_team: str
    away_team: str
    independent: Forecast
    market: Forecast | None
    calibrated: Forecast | None
    model_market_difference: ForecastDifference | None
    model_version: str
    data_cutoff: str
    generated_at: str
    confidence: str
    unavailable_features: tuple[str, ...]
    market_evidence: MarketEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictionRecord":
        data = dict(payload)
        data["independent"] = Forecast.from_dict(data["independent"])
        for name in ("market", "calibrated"):
            if data.get(name) is not None:
                data[name] = Forecast.from_dict(data[name])
        if data.get("model_market_difference") is not None:
            data["model_market_difference"] = ForecastDifference(
                **data["model_market_difference"]
            )
        if data.get("market_evidence") is not None:
            evidence = dict(data["market_evidence"])
            evidence["providers"] = tuple(evidence.get("providers", ()))
            data["market_evidence"] = MarketEvidence(**evidence)
        data["unavailable_features"] = tuple(data.get("unavailable_features", ()))
        return cls(**data)


def build_prediction_record(
    *, league: str, game_id: str, season: int, week: int, kickoff: str,
    home_team: str, away_team: str, independent: Forecast,
    market: Forecast | None, calibrated: Forecast | None, model_version: str,
    data_cutoff: str, generated_at: str | None = None, confidence: str = "unrated",
    unavailable_features: tuple[str, ...] = (),
    market_evidence: MarketEvidence | None = None,
) -> PredictionRecord:
    kickoff, data_cutoff = _iso(kickoff), _iso(data_cutoff)
    generated_at = _iso(generated_at or datetime.now(timezone.utc))
    if data_cutoff >= kickoff:
        raise ValueError("data_cutoff must be strictly before kickoff")
    difference = (
        ForecastDifference(
            spread=independent.spread - market.spread,
            total=independent.total - market.total,
        )
        if market is not None else None
    )
    identity = {
        "schema_version": SCHEMA_VERSION, "league": league.lower(),
        "game_id": str(game_id), "model_version": str(model_version),
        "data_cutoff": data_cutoff, "generated_at": generated_at,
    }
    prediction_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return PredictionRecord(
        prediction_id=prediction_id, schema_version=SCHEMA_VERSION,
        league=league.lower(), game_id=str(game_id), season=int(season), week=int(week),
        kickoff=kickoff, home_team=str(home_team), away_team=str(away_team),
        independent=independent, market=market, calibrated=calibrated,
        model_market_difference=difference, model_version=str(model_version),
        data_cutoff=data_cutoff, generated_at=generated_at, confidence=str(confidence),
        unavailable_features=tuple(sorted(set(unavailable_features))),
        market_evidence=market_evidence,
    )


class PredictionLedger:
    """Append-only JSONL ledger with idempotent retries and conflict rejection."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> list[PredictionRecord]:
        if not self.path.exists():
            return []
        return [PredictionRecord.from_dict(json.loads(line))
                for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, record: PredictionRecord) -> bool:
        encoded = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
        for existing in self.records():
            if existing.prediction_id == record.prediction_id:
                if existing.to_dict() != record.to_dict():
                    raise ValueError("prediction ledger is immutable; conflicting overwrite")
                return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        return True


__all__ = ["Forecast", "ForecastDifference", "MarketEvidence", "PredictionLedger", "PredictionRecord",
           "SCHEMA_VERSION", "build_prediction_record"]
