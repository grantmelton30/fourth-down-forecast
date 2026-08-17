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
        spread = float(self.spread.predict(frame)[0])
        total = float(self.total.predict(frame)[0])
        # PHYSICAL FLOOR. spread and total are two separate ridge fits (fit_live_mean)
        # with nothing constraining their combination -- for an ordinary game that never
        # matters, real spread/total pairs are always internally consistent. Real FCS
        # ratings (2026-08-17) can now be extreme enough to break it: Georgia vs
        # Tennessee State predicted spread +47.6, total +47.0, implying a losing score of
        # (47.0-47.6)/2 = -0.3. `total` can never be less than `|spread|` -- the losing
        # team scoring exactly 0 is the most lopsided a game can physically be -- so clip
        # here rather than let an impossible pair reach `SimResult.recentered().retotaled()`,
        # which correctly raises on it (shared/sim_core.py's `_calibration_weights`).
        # Adjusting total is the smaller, more defensible correction than adjusting
        # spread: total's own model has no richer, phase-promoted feature set of its own
        # in these extreme cases (`total_preseason_promoted` is False for both games this
        # was found on), so it is the less-trusted of the two numbers to begin with.
        if total < abs(spread):
            total = abs(spread)
        return spread, total


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
