from __future__ import annotations

import numpy as np
import pandas as pd

from validated_model import fit_validated_ridge


def _frame(useful: bool) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for season in range(2019, 2025):
        for _ in range(180):
            base = rng.normal()
            candidate = rng.normal()
            target = 2.0 * base + (1.5 * candidate if useful else 0.0) + rng.normal(0, .4)
            rows.append({"season": season, "base": base, "candidate": candidate,
                         "actual": target})
    return pd.DataFrame(rows)


def test_challenger_is_promoted_only_after_prior_season_improvement():
    model = fit_validated_ridge(
        _frame(True), base_features=["base"], candidate_features=["candidate"],
        target="actual", season_column="season", minimum_fit_rows=300,
    )
    assert model.challenger_promoted is True
    assert "candidate" in model.features
    assert model.validation_challenger_rmse < model.validation_base_rmse


def test_noise_candidate_stays_out_of_published_model():
    model = fit_validated_ridge(
        _frame(False), base_features=["base"], candidate_features=["candidate"],
        target="actual", season_column="season", minimum_fit_rows=300,
        minimum_improvement=0.02,
    )
    assert model.challenger_promoted is False
    assert model.features == ("base",)


def test_missing_candidate_values_are_imputed_from_training_only():
    frame = _frame(True)
    frame.loc[frame.index[::7], "candidate"] = np.nan
    model = fit_validated_ridge(
        frame, base_features=["base"], candidate_features=["candidate"],
        target="actual", season_column="season", minimum_fit_rows=300,
    )
    pred = model.predict(pd.DataFrame({"base": [0.2], "candidate": [np.nan]}))
    assert np.isfinite(pred[0])


def test_predictions_clip_features_to_historical_support():
    frame = _frame(True)
    model = fit_validated_ridge(
        frame, base_features=["base"], candidate_features=["candidate"],
        target="actual", season_column="season", minimum_fit_rows=300,
    )
    extreme = model.predict(pd.DataFrame({"base": [1e9], "candidate": [-1e9]}))[0]
    boundary = model.predict(pd.DataFrame({
        "base": [model.upper[model.features.index("base")]],
        "candidate": [model.lower[model.features.index("candidate")]],
    }))[0]
    assert extreme == boundary
