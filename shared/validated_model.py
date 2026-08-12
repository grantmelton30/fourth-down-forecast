"""Small ridge challenger promoted only by earlier-season validation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidatedRidge:
    features: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    medians: np.ndarray
    beta: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    challenger_promoted: bool
    validation_base_rmse: float
    validation_challenger_rmse: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.reindex(columns=self.features).to_numpy(float)
        values = np.where(np.isnan(values), self.medians, values)
        values = np.clip(values, self.lower, self.upper)
        design = np.column_stack([np.ones(len(values)), (values - self.center) / self.scale])
        return design @ self.beta


def _fit(train: pd.DataFrame, features: tuple[str, ...], target: str, alpha: float):
    values = train.loc[:, features].to_numpy(float)
    medians = np.nanmedian(values, axis=0)
    values = np.where(np.isnan(values), medians, values)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    X = np.column_stack([np.ones(len(values)), (values - center) / scale])
    penalty = np.eye(X.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ train[target].to_numpy(float))
    return center, scale, medians, beta


def _predict(frame, features, fit):
    center, scale, medians, beta = fit
    values = frame.loc[:, features].to_numpy(float)
    values = np.where(np.isnan(values), medians, values)
    return np.column_stack([np.ones(len(values)), (values - center) / scale]) @ beta


def fit_validated_ridge(
    frame: pd.DataFrame,
    *,
    base_features: list[str],
    candidate_features: list[str],
    target: str,
    season_column: str = "season",
    alpha: float = 25.0,
    minimum_fit_rows: int = 300,
    minimum_improvement: float = 0.0,
) -> ValidatedRidge:
    clean = frame.dropna(subset=[target, season_column]).copy()
    seasons = sorted(clean[season_column].unique())
    if len(seasons) < 2:
        raise ValueError("validated ridge requires at least two seasons")
    validation_season = seasons[-1]
    train, validation = clean[clean[season_column] < validation_season], clean[
        clean[season_column] == validation_season
    ]
    if len(train) < minimum_fit_rows or validation.empty:
        raise ValueError("insufficient earlier-season rows for challenger validation")
    base = tuple(base_features)
    challenger = tuple(dict.fromkeys([*base_features, *candidate_features]))
    base_fit, challenger_fit = _fit(train, base, target, alpha), _fit(
        train, challenger, target, alpha
    )
    actual = validation[target].to_numpy(float)
    base_rmse = float(np.sqrt(np.mean((_predict(validation, base, base_fit) - actual) ** 2)))
    challenger_rmse = float(np.sqrt(np.mean(
        (_predict(validation, challenger, challenger_fit) - actual) ** 2
    )))
    promoted = challenger_rmse <= base_rmse - minimum_improvement
    selected = challenger if promoted else base
    final_fit = _fit(clean, selected, target, alpha)
    bounds = clean.loc[:, selected].to_numpy(float)
    bounds = np.where(np.isnan(bounds), final_fit[2], bounds)
    return ValidatedRidge(
        features=selected, center=final_fit[0], scale=final_fit[1],
        medians=final_fit[2], beta=final_fit[3],
        lower=np.nanmin(bounds, axis=0), upper=np.nanmax(bounds, axis=0),
        challenger_promoted=promoted,
        validation_base_rmse=base_rmse,
        validation_challenger_rmse=challenger_rmse,
    )


__all__ = ["ValidatedRidge", "fit_validated_ridge"]
