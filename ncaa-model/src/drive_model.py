"""NCAA binding for the shared multinomial drive-outcome model.

The model itself -- five predictors, four classes, and the endgame table -- lives in
`../shared/sim_core.py` and is the same one nfl-model fits. What is NCAA-specific, and so
lives here, is only: which ratings table to merge onto each drive, how to turn those
ratings into `net_epa`, and where this repo keeps its caches.

This is a calibration layer, not a place to add features. The NFL note applies with more
force here, not less: college has more games but far more teams, so per-team information is
thinner, and anything richer than these five predictors backtests beautifully and loses
money.
"""

from __future__ import annotations

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

# Below this many training drives the fit is not trustworthy. Set higher than the NFL's
# 5,000 because college runs roughly 33,000 drives a season against the NFL's 6,000 -- a
# season and a half of college data is the comparable floor, and anything less means the
# caller has silently lost most of its history.
_MIN_TRAINING_DRIVES = 20_000


def build_training_features(
    drives: pd.DataFrame, walkforward: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Attach `net_epa` to each historical drive using the ratings that were current in
    that drive's own game week.

    NO LOOKAHEAD (non-negotiable #1). The merge is on (season, week, team), not on team
    alone. Using a single end-of-history ratings snapshot would label a 2021 drive with a
    team's 2025 strength, and in college that is a far worse error than in the NFL: the
    portal turns rosters over annually, which is why `offseason_regression` here is 0.42
    against the NFL's 0.28.

    THE FCS BUCKET. CFBD's drives endpoint is not filtered by division, so the raw table
    is 28.8% FCS-vs-FCS games that say nothing about FBS teams, plus 9.9% FBS-vs-FCS games
    that say a great deal. A plain inner join throws away both -- it keeps only the 61.3%
    where both teams carry a fitted rating. Instead, every team with no rating is rewritten
    to `cfg.teams.fcs_bucket_name`, matching what `ingest.build_game_offense` does, and the
    drive is kept as long as at least ONE side is a rated FBS team. That recovers the
    body-bag games while still discarding FCS-vs-FCS.

    Rewriting on "absent from the ratings table" rather than on a classification column is
    safe here because the ridge fits every FBS team plus the bucket -- 137 teams -- so an
    absent name is an FCS name.
    """
    wf = walkforward[["season", "week", "team", "off_rating", "def_rating"]]

    drives = drives.copy()
    rated = set(wf["team"].unique())
    bucket = cfg.teams.fcs_bucket_name
    offense_rated = drives["offense"].isin(rated)
    defense_rated = drives["defense"].isin(rated)
    drives = drives[offense_rated | defense_rated]
    drives["offense"] = drives["offense"].where(offense_rated, bucket)
    drives["defense"] = drives["defense"].where(defense_rated, bucket)

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
    """Fit on seasons train_start..as_of_season-1, strictly earlier than the season being
    predicted.

    Refit once per season rather than weekly: this is a slowly-varying calibration layer,
    and refitting it every week is wasted compute for no measurable gain.
    """
    path = CACHE_DIR / f"drive_model_{cfg.seasons.train_start}_{as_of_season}.json"
    if cache and path.exists():
        model = MultinomialModel.from_json(path.read_text())
        # Schema guard, the JSON analogue of `config.read_cached_frame`. A cache written
        # before the feature list changed would otherwise be served against a design matrix
        # of a different width and fail somewhere far from here.
        if model.classes == list(DRIVE_CLASSES) and model.coef.shape == (
            len(DRIVE_CLASSES), len(FEATURES)
        ):
            return model

    train = drives[
        (drives["season"] >= cfg.seasons.train_start)
        & (drives["season"] < as_of_season)
    ]
    feat = build_training_features(train, walkforward, cfg)
    if len(feat) < _MIN_TRAINING_DRIVES:
        raise ValueError(
            f"fit_drive_model: only {len(feat):,} training drives before {as_of_season} "
            f"(floor {_MIN_TRAINING_DRIVES:,}); refusing to fit a calibration layer on "
            "that little data."
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
    """NCAA binding for the shared endgame fit -- supplies this repo's cache path.

    `late_seconds` keeps the NFL's 5:00 window. That is a real choice and not merely
    inherited: college teams average 1.4243 drives inside it against the NFL's 1.3408, and
    `cfg.simulation.endgame_second_drive_prob` carries that difference. Widening the window
    would change what "endgame" means and would need the prob re-measured with it.
    """
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
