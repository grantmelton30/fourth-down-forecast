"""Availability-dated feature records and leakage-safe joins."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def _utc(value):
    return pd.Timestamp(value).tz_localize("UTC") if pd.Timestamp(value).tzinfo is None \
        else pd.Timestamp(value).tz_convert("UTC")


@dataclass(frozen=True)
class FeatureSnapshot:
    league: str
    entity: str
    feature: str
    value: float | str | None
    available_at: str
    source: str
    game_id: str | None = None
    kickoff: str | None = None

    def __post_init__(self):
        if self.kickoff is not None and _utc(self.available_at) >= _utc(self.kickoff):
            raise ValueError("feature available_at must be strictly before kickoff")


def merge_asof_features(
    games: pd.DataFrame,
    features: pd.DataFrame,
    *,
    entity_column: str,
    cutoff_column: str = "data_cutoff",
    required_features: "list[str] | tuple[str, ...]" = (),
) -> pd.DataFrame:
    """Wide-join the latest feature known by each row's data cutoff.

    Missing values stay missing.  Imputation is a model decision and cannot occur in the
    source layer, where it would make unavailable information look observed.
    """
    if entity_column not in games or cutoff_column not in games:
        raise ValueError(f"games require {entity_column!r} and {cutoff_column!r}")
    expected = {entity_column, "feature", "value", "available_at", "source"}
    missing = expected - set(features.columns)
    if missing:
        raise ValueError(f"feature snapshots missing columns: {sorted(missing)}")
    out = games.copy().reset_index(drop=True)
    out[cutoff_column] = pd.to_datetime(out[cutoff_column], utc=True, errors="coerce")
    if out[cutoff_column].isna().any():
        raise ValueError("every game requires a valid data cutoff")
    feats = features.copy()
    feats["available_at"] = pd.to_datetime(feats["available_at"], utc=True, errors="coerce")
    feats = feats.dropna(subset=["available_at"])
    feature_names = sorted(set(required_features) | set(feats["feature"].astype(str)))
    for feature in feature_names:
        values, sources = [], []
        candidate = feats[feats["feature"].eq(feature)]
        for row in out.itertuples(index=False):
            entity = getattr(row, entity_column)
            cutoff = getattr(row, cutoff_column)
            eligible = candidate[
                candidate[entity_column].eq(entity) & (candidate["available_at"] <= cutoff)
            ].sort_values("available_at")
            if eligible.empty:
                values.append(pd.NA); sources.append(pd.NA)
            else:
                latest = eligible.iloc[-1]
                values.append(latest["value"]); sources.append(latest["source"])
        out[feature] = values
        out[f"{feature}__source"] = sources
    out["unavailable_features"] = [
        tuple(feature for feature in required_features if pd.isna(row[feature]))
        for _, row in out.iterrows()
    ]
    return out


__all__ = ["FeatureSnapshot", "merge_asof_features"]
