"""Multinomial drive-outcome model (§6b).

Five predictors, four classes. This is a calibration layer that turns team strength and
field position into drive-outcome probabilities -- it is deliberately not the place to
add features. With ~285 games a season, anything richer backtests beautifully and loses
money.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ._shared import sim_core
from .config import CACHE_DIR, Config
from .drives import DRIVE_CLASSES
from .ratings import net_epa_vec

FEATURES = sim_core.FEATURES
MultinomialModel = sim_core.MultinomialModel
EndgameTable = sim_core.EndgameTable
ENDGAME_BUCKETS = sim_core.ENDGAME_BUCKETS


def build_training_features(
    drives: pd.DataFrame, walkforward: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Attach `net_epa` to each historical drive using the ratings that were current in
    that drive's own game week.

    Using a single end-of-history ratings snapshot would leak: a team that was good in
    2020 and bad in 2024 would have its 2020 drives labelled with its 2024 strength.
    """
    wf = walkforward[["season", "week", "team", "off_rating", "def_rating"]]
    off = wf.rename(
        columns={"team": "offense", "off_rating": "off_o", "def_rating": "def_o"}
    )
    dfn = wf.rename(
        columns={"team": "defense", "off_rating": "off_d", "def_rating": "def_d"}
    )

    df = drives.merge(off, on=["season", "week", "offense"], how="inner")
    df = df.merge(dfn, on=["season", "week", "defense"], how="inner")
    df["net_epa"] = net_epa_vec(df["off_o"].to_numpy(), df["def_d"].to_numpy(), cfg)
    return df


def fit_drive_model(
    drives: pd.DataFrame,
    walkforward: pd.DataFrame,
    cfg: Config,
    as_of_season: int,
    C: float = 1.0,
    cache: bool = True,
) -> MultinomialModel:
    """Fit on seasons train_start..as_of_season-1 (strictly earlier than the season being
    predicted).

    Refit once per season rather than weekly: this is a slowly-varying calibration layer,
    and refitting it every week is wasted compute for no measurable gain (§8, step 2).
    """
    path = CACHE_DIR / f"drive_model_{cfg.seasons.train_start}_{as_of_season}.json"
    if cache and path.exists():
        model = MultinomialModel.from_json(path.read_text())
        # Schema guard, ported from ncaa-model/src/drive_model.py's fit_drive_model
        # 2026-08-17 cache-consistency audit -- this side never had it. A cache written
        # before the feature list changed would otherwise be served against a design
        # matrix of a different width and fail somewhere far from here.
        if model.classes == list(DRIVE_CLASSES) and model.coef.shape == (
            len(DRIVE_CLASSES), len(FEATURES)
        ):
            return model

    train = drives[
        (drives["season"] >= cfg.seasons.train_start)
        & (drives["season"] < as_of_season)
    ]
    feat = build_training_features(train, walkforward, cfg)
    if len(feat) < 5000:
        raise ValueError(
            f"fit_drive_model: only {len(feat)} training drives before {as_of_season}; "
            "refusing to fit a calibration layer on that little data."
        )

    # Field position is centred and scaled so the squared term does not dominate the
    # optimizer's step size; the scaling is stored and reapplied at predict time.
    fp_mean, fp_scale = 50.0, 25.0
    fp = (feat["start_yardline_100"].to_numpy(float) - fp_mean) / fp_scale
    net = feat["net_epa"].to_numpy(float)
    X = np.column_stack([
        net, fp, fp ** 2, feat["is_home_offense"].to_numpy(float), net * fp
    ])
    y = feat["result"].to_numpy()

    clf = LogisticRegression(solver="lbfgs", C=C, max_iter=2000)
    clf.fit(X, y)

    # Reorder rows into the canonical DRIVE_CLASSES order so the simulator can index by
    # position without consulting the label list.
    order = [list(clf.classes_).index(c) for c in DRIVE_CLASSES]
    model = MultinomialModel(
        coef=clf.coef_[order],
        intercept=clf.intercept_[order],
        classes=list(DRIVE_CLASSES),
        as_of_season=as_of_season,
        n_drives_trained=len(feat),
        fp_mean=fp_mean,
        fp_scale=fp_scale,
    )
    if cache:
        path.write_text(model.to_json())
    return model


# --------------------------------------------------------------------------------------
# Endgame layer
# --------------------------------------------------------------------------------------
def fit_endgame_table(
    drives: pd.DataFrame,
    cfg: Config,
    as_of_season: int,
    late_seconds: float = 300.0,
    cache: bool = True,
) -> EndgameTable:
    """NFL binding for the shared endgame fit -- supplies this repo's cache path."""
    path = CACHE_DIR / f"endgame_{cfg.seasons.train_start}_{as_of_season}.json"
    return sim_core.fit_endgame_table(
        drives,
        train_start=cfg.seasons.train_start,
        as_of_season=as_of_season,
        cache_path=path if cache else None,
        late_seconds=late_seconds,
    )


def summarize(model: MultinomialModel) -> pd.DataFrame:
    """Coefficient table, for the README and for eyeballing signs."""
    return pd.DataFrame(
        model.coef, index=model.classes, columns=FEATURES
    ).assign(intercept=model.intercept)
