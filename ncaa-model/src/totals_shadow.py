"""Frozen NCAA totals shadow model for prospective evidence collection.

The shadow is anchored to a timestamped market total and never replaces the independent
forecast. Its only question is whether fixed tempo/efficiency features improve on an
earlier-season mean-bias correction during untouched 2026 games.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


SPEC_VERSION = "T1-shadow-v1"
FEATURES = ("pace_sum", "eff_sum", "pace_efficiency_interaction", "abs_net_diff")
RIDGE_ALPHA = 45.0
MIN_TRAIN_ROWS = 300


def _derived(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"pace_sum", "eff_sum", "net_diff"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"totals shadow missing features: {sorted(missing)}")
    out = frame.copy()
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["pace_efficiency_interaction"] = out["pace_sum"] * out["eff_sum"]
    out["abs_net_diff"] = out["net_diff"].abs()
    return out


@dataclass(frozen=True)
class TotalsShadowModel:
    forecast_season: int
    training_through: int
    bias: float
    medians: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    model_version: str

    def predict(self, row: pd.DataFrame, market_total: float) -> dict:
        if len(row) != 1:
            raise ValueError("totals shadow prediction requires exactly one game")
        market_total = float(market_total)
        if not np.isfinite(market_total) or market_total <= 0:
            raise ValueError("totals shadow requires a positive finite market total")
        values = _derived(row).loc[:, FEATURES].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("totals shadow live features must be complete")
        design = np.column_stack([np.ones(1), (values - self.center) / self.scale])
        correction = float((design @ self.beta)[0])
        return {
            "spec_version": SPEC_VERSION,
            "status": "prospective_shadow_not_a_pick",
            "market_total": market_total,
            "bias_total": market_total + self.bias,
            "tempo_total": market_total + correction,
            "shadow_total": market_total + correction,
            "selected_model": "tempo_efficiency",
            "training_through": self.training_through,
            "model_version": self.model_version,
            "features": list(FEATURES),
        }


def fit_totals_shadow(frame: pd.DataFrame, forecast_season: int) -> TotalsShadowModel:
    required = {"game_id", "season", "fbs_only", "actual_total", "total_close", *FEATURES[:2],
                "net_diff"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"totals shadow training frame missing columns: {sorted(missing)}")
    data = _derived(frame)
    data["season"] = pd.to_numeric(data["season"], errors="coerce")
    data["actual_total"] = pd.to_numeric(data["actual_total"], errors="coerce")
    data["total_close"] = pd.to_numeric(data["total_close"], errors="coerce")
    train = data[
        data["fbs_only"].fillna(False)
        & (data["season"] < int(forecast_season))
        & data["actual_total"].notna()
        & data["total_close"].notna()
        & data.loc[:, FEATURES].notna().any(axis=1)
    ].copy()
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError("totals shadow has insufficient earlier-season training rows")
    target = (train["actual_total"] - train["total_close"]).to_numpy(float)
    values = train.loc[:, FEATURES].to_numpy(float)
    medians = np.nanmedian(values, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("totals shadow contains an entirely missing training feature")
    values = np.where(np.isnan(values), medians, values)
    center = values.mean(axis=0)
    scale = np.where(values.std(axis=0) > 1e-9, values.std(axis=0), 1.0)
    design = np.column_stack([np.ones(len(values)), (values - center) / scale])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    fingerprint_columns = ["game_id", "season", "actual_total", "total_close", *FEATURES]
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(
            train.loc[:, fingerprint_columns].sort_values(["season", "game_id"]), index=False
        ).values.tobytes()
    ).hexdigest()
    identity = {
        "spec_version": SPEC_VERSION, "features": FEATURES, "alpha": RIDGE_ALPHA,
        "forecast_season": int(forecast_season), "training_through": int(train.season.max()),
        "training_fingerprint": fingerprint,
    }
    model_version = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return TotalsShadowModel(
        forecast_season=int(forecast_season), training_through=int(train.season.max()),
        bias=float(target.mean()), medians=medians, center=center, scale=scale, beta=beta,
        model_version=model_version,
    )


__all__ = ["FEATURES", "SPEC_VERSION", "TotalsShadowModel", "fit_totals_shadow"]
