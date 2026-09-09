"""Append-only validation for prospective NCAA totals shadow forecasts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


MAX_MARKET_AGE_HOURS = 6


def validate_shadow_record(record: dict) -> dict:
    out = dict(record)
    required_text = (
        "prediction_id", "game_id", "home_team", "away_team", "spec_version",
        "model_version", "market_label", "selected_model", "status",
    )
    for key in required_text:
        if not str(out.get(key) or "").strip():
            raise ValueError(f"totals shadow requires {key}")
        out[key] = str(out[key])
    for key in ("kickoff", "generated_at", "data_cutoff", "market_observed_at"):
        try:
            value = pd.Timestamp(out.get(key))
        except (TypeError, ValueError):
            value = pd.NaT
        if pd.isna(value) or value.tzinfo is None:
            raise ValueError("totals shadow timestamps must be known and timezone-aware")
        out[key] = value.tz_convert("UTC").isoformat()
    kickoff = pd.Timestamp(out["kickoff"])
    generated = pd.Timestamp(out["generated_at"])
    cutoff = pd.Timestamp(out["data_cutoff"])
    observed = pd.Timestamp(out["market_observed_at"])
    if not cutoff <= generated < kickoff or not observed <= generated < kickoff:
        raise ValueError("totals shadow inputs must be observed before kickoff")
    if observed < generated - pd.Timedelta(hours=MAX_MARKET_AGE_HOURS):
        raise ValueError("totals shadow market quote is too old")
    out["season"], out["week"], out["training_through"] = (
        int(out[key]) for key in ("season", "week", "training_through")
    )
    if out["training_through"] >= out["season"]:
        raise ValueError("totals shadow training must end before the forecast season")
    for key in ("market_total", "bias_total", "tempo_total", "shadow_total"):
        out[key] = float(out[key])
        if not math.isfinite(out[key]) or out[key] <= 0:
            raise ValueError("totals shadow totals must be positive and finite")
    out["providers"] = sorted(str(value) for value in out.get("providers", []))
    payload = {key: value for key, value in out.items() if key != "record_id"}
    out["record_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return out


def append_shadow_record(path: str | Path, record: dict) -> bool:
    row = validate_shadow_record(record)
    path = Path(path)
    existing = []
    if path.exists():
        existing = [validate_shadow_record(json.loads(line))
                    for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for prior in existing:
        if prior["prediction_id"] == row["prediction_id"]:
            if prior != row:
                raise ValueError("totals shadow ledger refuses a conflicting prediction")
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return True


__all__ = ["MAX_MARKET_AGE_HOURS", "append_shadow_record", "validate_shadow_record"]
