"""Validated NCAA point estimates for live publication."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ._shared import validated_model
from .features import maturity_phase


@dataclass(frozen=True)
class LiveMeanModel:
    spread: validated_model.ValidatedRidge
    total: validated_model.ValidatedRidge
    phase: str
    validation_season: int

    def predict(self, frame: pd.DataFrame) -> tuple[float, float]:
        if len(frame) != 1:
            raise ValueError("live mean prediction requires exactly one game")
        return float(self.spread.predict(frame)[0]), float(self.total.predict(frame)[0])


def _candidates(frame: pd.DataFrame, phase: str, kind: str) -> list[str]:
    phased = [column for column in frame if column.endswith(f"_{kind}_{phase}")]
    candidates = phased or [column for column in frame if column.endswith(f"_{kind}")]
    return [column for column in candidates if frame[column].notna().any()]


def fit_live_mean(frame: pd.DataFrame, week: int) -> LiveMeanModel:
    phase = maturity_phase(week)
    eligible = frame[frame["week"].map(maturity_phase).eq(phase)].copy()
    seasons = sorted(eligible["season"].dropna().unique())
    if len(seasons) < 2:
        raise ValueError(f"insufficient historical seasons for {phase} live mean")
    common = dict(season_column="season", alpha=45.0, minimum_fit_rows=150,
                  minimum_improvement=0.05)
    spread = validated_model.fit_validated_ridge(
        eligible, base_features=["net_diff", "is_home"],
        candidate_features=_candidates(eligible, phase, "diff"),
        target="actual_margin", **common,
    )
    total = validated_model.fit_validated_ridge(
        eligible, base_features=["pace_sum", "eff_sum"],
        candidate_features=_candidates(eligible, phase, "sum"),
        target="actual_total", **common,
    )
    return LiveMeanModel(spread, total, phase, int(seasons[-1]))


__all__ = ["LiveMeanModel", "fit_live_mean"]
