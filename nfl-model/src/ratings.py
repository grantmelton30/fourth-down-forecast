"""Team strength (§5). Two weighted ridge regressions solved from the normal equations.

NON-NEGOTIABLE #1 lives here. Every fit takes an `as_of` timestamp and uses only games
that kicked off strictly before it. `_assert_no_lookahead` runs on every fit, not just in
tests.

SIGN CONVENTION -- read this before touching anything downstream:
    off_rating  higher = BETTER offense
    def_rating  higher = WORSE  defense   (it is the coefficient on the opposing
                                           defense in a model of offensive EPA, so a
                                           defense that allows more carries a larger
                                           value)
`tests/test_ratings.py` asserts the league's best defense carries the lowest def_rating.
"""

from __future__ import annotations

from dataclasses import asdict,dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .config import CACHE_DIR,Config,build_cache_signature,frame_signature,read_cached_frame,write_cached_frame
from .ingest import CANONICAL_TEAMS

TEAM_INDEX = {t: i for i, t in enumerate(CANONICAL_TEAMS)}
N_TEAMS = len(CANONICAL_TEAMS)

# Column layout of the efficiency design matrix.
OFF_SLICE = slice(0, N_TEAMS)
DEF_SLICE = slice(N_TEAMS, 2 * N_TEAMS)
COL_HOME_OFFENSE = 2 * N_TEAMS
COL_INTERCEPT = 2 * N_TEAMS + 1
N_COLS = 2 * N_TEAMS + 2

# Games whose decay weight falls below this contribute nothing but cost; dropping them is
# a pure speed optimization, not a modelling choice.
_MIN_DECAY_WEIGHT = 1e-3


@dataclass(frozen=True)
class RatingsFit:
    """Result of one efficiency fit at a point in time."""

    table: pd.DataFrame        # index=team, columns=[off_rating, def_rating, n_games]
    hfa_epa: float             # fitted home-offense coefficient, EPA/play
    intercept: float
    as_of: pd.Timestamp
    n_games: int
    lambda_off: float
    lambda_def: float

    def off(self, team: str) -> float:
        return float(self.table["off_rating"].get(team, 0.0))

    def dfn(self, team: str) -> float:
        return float(self.table["def_rating"].get(team, 0.0))


# --------------------------------------------------------------------------------------
# Pre-aggregation: PBP -> one row per (game, offensive unit)
# --------------------------------------------------------------------------------------

def build_game_offense_table(
    pbp: pd.DataFrame,
    schedules: pd.DataFrame,
    cache_key: "str | None" = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """One row per (game, offensive unit); a 16-game week produces 32 rows.

    Computed once for the whole history and reused by every walk-forward fit. This is
    safe: each row aggregates only that game's own plays, and the no-lookahead property
    comes from *which rows a fit is allowed to see*, enforced in `fit_ratings`.

    Response is mean EPA per play on competitive, non-special-teams plays, excluding
    plays nullified by penalty where no yardage occurred.

    Caching this ~5.5k-row aggregate is what makes the weekly refresh fast: nothing on
    that path has to touch the 480k-row play-by-play frame at all.
    """
    path = CACHE_DIR / f"game_offense_{cache_key}.parquet" if cache_key else None
    signature=build_cache_signature(builder=Path(__file__),config={"teams":CANONICAL_TEAMS},inputs={"pbp":frame_signature(pbp,["game_id","posteam","defteam","epa","competitive","special","fixed_drive","yards_gained","penalty"]),"schedules":frame_signature(schedules,["game_id","season","week","kickoff","home_team","away_team","location"])},artifact_version=2)
    if path is not None and not refresh:
        cached=read_cached_frame(path,["game_id","offense","defense","epa_per_play","drives","kickoff"],signature)
        if cached is not None:return cached

    plays = pbp[
        pbp["competitive"]
        & (pbp["special"] == 0)
        & pbp["epa"].notna()
        & pbp["posteam"].notna()
        & ~((pbp["penalty"] == 1) & (pbp["yards_gained"] == 0))
    ]

    agg = plays.groupby(["game_id", "posteam"], as_index=False).agg(
        epa_per_play=("epa", "mean"),
        n_competitive_plays=("epa", "size"),
    )

    # Drives are counted over all drives in the game, not just competitive plays -- pace
    # is about how many possessions a game produces, and garbage time still produces them.
    drive_counts = (
        pbp[pbp["fixed_drive"].notna() & pbp["posteam"].notna()]
        .groupby(["game_id", "posteam"], as_index=False)["fixed_drive"]
        .nunique()
        .rename(columns={"fixed_drive": "drives"})
    )
    agg = agg.merge(drive_counts, on=["game_id", "posteam"], how="left")

    sched = schedules[[
        "game_id", "season", "week", "game_type", "kickoff", "home_team", "away_team",
        "location",
    ]]
    df = agg.merge(sched, on="game_id", how="inner")
    df = df.rename(columns={"posteam": "offense"})
    df["defense"] = np.where(
        df["offense"] == df["home_team"], df["away_team"], df["home_team"]
    )
    # Neutral-site games get 0.5 on both rows rather than a home/away split.
    df["home_offense"] = np.where(
        df["location"].eq("Neutral"),
        0.5,
        (df["offense"] == df["home_team"]).astype(float),
    )
    df = df[df["offense"].isin(TEAM_INDEX) & df["defense"].isin(TEAM_INDEX)]
    df = df.sort_values("kickoff").reset_index(drop=True)
    if path is not None:
        write_cached_frame(df,path,signature)
    return df


# --------------------------------------------------------------------------------------
# Shared machinery
# --------------------------------------------------------------------------------------

def _assert_no_lookahead(rows: pd.DataFrame, as_of: pd.Timestamp, what: str) -> None:
    """NON-NEGOTIABLE #1. Not a test-only check -- it runs on every fit."""
    if len(rows) and rows["kickoff"].max() >= as_of:
        offenders = rows.loc[rows["kickoff"] >= as_of, "game_id"].unique()[:5]
        raise AssertionError(
            f"LOOKAHEAD in {what}: games at or after as_of={as_of} entered the fit, "
            f"e.g. {list(offenders)}"
        )


def _decay_weights(rows: pd.DataFrame, half_life_games: float) -> np.ndarray:
    """0.5 ** (games_ago / half_life), where games_ago counts each team's *own* games
    back from the cutoff."""
    games_ago = (
        rows.groupby("offense")["kickoff"].rank(method="first", ascending=False) - 1.0
    )
    return np.power(0.5, games_ago.to_numpy() / float(half_life_games))


def _solve_weighted_ridge(
    X: sparse.csr_matrix, y: np.ndarray, w: np.ndarray, penalty: np.ndarray
) -> np.ndarray:
    """minimize  sum_i w_i (y_i - x_i.beta)^2  +  sum_j penalty_j beta_j^2

    Solved directly rather than through sklearn.Ridge, which applies one penalty to all
    columns; here offense, defense, home-field and intercept need different penalties
    (the last two are unpenalized).
    """
    W = sparse.diags(w)
    A = (X.T @ W @ X).toarray() + np.diag(penalty)
    b = X.T @ (w * y)
    return np.linalg.solve(A, b)


# --------------------------------------------------------------------------------------
# 5a. Efficiency ridge
# --------------------------------------------------------------------------------------

def fit_ratings(
    game_off: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: Config,
    priors: "pd.DataFrame | None" = None,
    lambdas: "tuple[float, float] | None" = None,
) -> RatingsFit:
    """Offense/defense strength in EPA/play as of `as_of`.

    `priors` is an optional frame indexed by team with `prior_off` / `prior_def`,
    injected as synthetic observations (see `_prior_rows`).
    """
    lambda_off, lambda_def = lambdas if lambdas else cfg.effective_lambdas
    rows = game_off[game_off["kickoff"] < as_of]
    _assert_no_lookahead(rows, as_of, "fit_ratings")

    w_decay = _decay_weights(rows, cfg.ratings.half_life_games)
    keep = w_decay >= _MIN_DECAY_WEIGHT
    rows, w_decay = rows[keep], w_decay[keep]

    n = len(rows)
    if n == 0:
        empty = pd.DataFrame(
            0.0, index=CANONICAL_TEAMS, columns=["off_rating", "def_rating"]
        ).assign(n_games=0)
        empty.index.name = "team"
        return RatingsFit(empty, 0.0, 0.0, as_of, 0, lambda_off, lambda_def)

    off_idx = rows["offense"].map(TEAM_INDEX).to_numpy()
    def_idx = rows["defense"].map(TEAM_INDEX).to_numpy()

    data, ri, ci = [], [], []
    r = np.arange(n)
    for cols, vals in (
        (off_idx, np.ones(n)),
        (N_TEAMS + def_idx, np.ones(n)),
        (np.full(n, COL_HOME_OFFENSE), rows["home_offense"].to_numpy(float)),
        (np.full(n, COL_INTERCEPT), np.ones(n)),
    ):
        ri.append(r)
        ci.append(cols)
        data.append(vals)

    y = rows["epa_per_play"].to_numpy(float)
    w = rows["n_competitive_plays"].to_numpy(float) * w_decay

    if priors is not None and len(priors):
        pr_rows, pr_y, pr_w = _prior_rows(priors, cfg)
        for cols, vals, rr in pr_rows:
            ri.append(rr + n)
            ci.append(cols)
            data.append(vals)
        y = np.concatenate([y, pr_y])
        w = np.concatenate([w, pr_w])

    X = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(ri), np.concatenate(ci))),
        shape=(len(y), N_COLS),
    )

    penalty = np.zeros(N_COLS)
    penalty[OFF_SLICE] = lambda_off
    penalty[DEF_SLICE] = lambda_def
    beta = _solve_weighted_ridge(X, y, w, penalty)

    off = beta[OFF_SLICE].copy()
    dfn = beta[DEF_SLICE].copy()
    intercept = float(beta[COL_INTERCEPT])
    # Center both units on zero; the level lives in the intercept.
    intercept += float(off.mean() + dfn.mean())
    off -= off.mean()
    dfn -= dfn.mean()

    counts = rows["offense"].value_counts()
    table = pd.DataFrame(
        {
            "off_rating": off,
            "def_rating": dfn,
            "n_games": [int(counts.get(t, 0)) for t in CANONICAL_TEAMS],
        },
        index=CANONICAL_TEAMS,
    )
    table.index.name = "team"
    return RatingsFit(
        table=table,
        hfa_epa=float(beta[COL_HOME_OFFENSE]),
        intercept=intercept,
        as_of=as_of,
        n_games=int(rows["game_id"].nunique()),
        lambda_off=lambda_off,
        lambda_def=lambda_def,
    )


def _prior_rows(priors: pd.DataFrame, cfg: Config):
    """Synthetic preseason observations (§5a, "prior injection").

    For each team, one row asserting `off_t = prior_off_t` and one asserting
    `def_t = prior_def_t`, weighted as `prior_weight_games` games' worth of plays. The
    rows carry a 1 in exactly one team column -- intercept and home_offense are zero -- so
    they act as a direct Gaussian prior on that coefficient rather than dragging the level
    around. This is what keeps September output sane; the prior's share of total weight
    decays automatically as real games accumulate.
    """
    teams = [t for t in priors.index if t in TEAM_INDEX]
    w_prior = cfg.ratings.prior_weight_games * cfg.ratings.league_mean_plays_per_game

    off_cols = np.array([TEAM_INDEX[t] for t in teams])
    def_cols = np.array([N_TEAMS + TEAM_INDEX[t] for t in teams])
    k = len(teams)
    rows = [
        (off_cols, np.ones(k), np.arange(k)),
        (def_cols, np.ones(k), np.arange(k) + k),
    ]
    y = np.concatenate([
        priors.loc[teams, "prior_off"].to_numpy(float),
        priors.loc[teams, "prior_def"].to_numpy(float),
    ])
    w = np.full(2 * k, w_prior)
    return rows, y, w


def season_priors(prev_fit: "RatingsFit | None", cfg: Config) -> "pd.DataFrame | None":
    """Last season's final ratings regressed toward the league mean (zero) by
    `offseason_regression`. Missing teams get 0."""
    if prev_fit is None:
        return None
    keep = 1.0 - cfg.ratings.offseason_regression
    return pd.DataFrame(
        {
            "prior_off": prev_fit.table["off_rating"] * keep,
            "prior_def": prev_fit.table["def_rating"] * keep,
        }
    )


# --------------------------------------------------------------------------------------
# 5b. Pace ridge
# --------------------------------------------------------------------------------------

def fit_pace(game_off: pd.DataFrame, as_of: pd.Timestamp, cfg: Config) -> pd.Series:
    """Per-team pace such that E[drives per team] = league_mean + pace_A + pace_B.

    Both teams in a game contribute to the same possession count, so a row carries a 1
    for the offense *and* a 1 for its opponent against a single 32-column block, rather
    than the separate offense/defense blocks the efficiency ridge uses.
    """
    rows = game_off[game_off["kickoff"] < as_of]
    _assert_no_lookahead(rows, as_of, "fit_pace")

    w = _decay_weights(rows, cfg.pace.half_life_games)
    keep = (w >= _MIN_DECAY_WEIGHT) & rows["drives"].notna().to_numpy()
    rows, w = rows[keep], w[keep]
    if len(rows) == 0:
        return pd.Series(0.0, index=CANONICAL_TEAMS, name="pace_rating")

    n = len(rows)
    off_idx = rows["offense"].map(TEAM_INDEX).to_numpy()
    def_idx = rows["defense"].map(TEAM_INDEX).to_numpy()
    r = np.arange(n)
    X = sparse.csr_matrix(
        (
            np.ones(3 * n),
            (
                np.concatenate([r, r, r]),
                np.concatenate([off_idx, def_idx, np.full(n, N_TEAMS)]),
            ),
        ),
        shape=(n, N_TEAMS + 1),
    )
    penalty = np.full(N_TEAMS + 1, float(cfg.pace.lambda_))
    penalty[N_TEAMS] = 0.0                      # intercept unpenalized
    beta = _solve_weighted_ridge(X, rows["drives"].to_numpy(float), w, penalty)

    pace = beta[:N_TEAMS] - beta[:N_TEAMS].mean()
    return pd.Series(pace, index=CANONICAL_TEAMS, name="pace_rating")


# --------------------------------------------------------------------------------------
# Matchup net strength
# --------------------------------------------------------------------------------------

def net_epa(off_rating: float, def_rating_opponent: float, cfg: Config) -> float:
    """Expected EPA/play for this offense against this defense.

    NOTE ON THE PLAYBOOK'S FORMULA. §5a writes
        net_A = off_weight * off_A - def_weight * def_B
    while the same section defines a *higher* def_rating as a *worse* defense and
    requires a unit test that the best defense carries the lowest def_rating. Those two
    statements contradict each other: under that convention, subtracting def_B would make
    an offense look stronger the better the defense it faces. The stated convention and
    its test are kept (they are the load-bearing half, and §5a itself flags this sign flip
    as the most common bug in this kind of build), so the operator here is `+`.
    Logged in DECISIONS.md.

    The 1.6 / 1.0 asymmetry is not arbitrary: offensive efficiency is meaningfully
    stickier week to week than defensive efficiency.
    """
    return (
        cfg.ratings.off_weight * off_rating
        + cfg.ratings.def_weight * def_rating_opponent
    )


def net_epa_vec(
    off_rating: np.ndarray, def_rating_opponent: np.ndarray, cfg: Config
) -> np.ndarray:
    return (
        cfg.ratings.off_weight * np.asarray(off_rating, dtype=float)
        + cfg.ratings.def_weight * np.asarray(def_rating_opponent, dtype=float)
    )


# --------------------------------------------------------------------------------------
# Walk-forward ratings history
# --------------------------------------------------------------------------------------

def week_cutoffs(schedules: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """(season, week) -> kickoff of that week's first game, the `as_of` for that week."""
    reg = schedules[
        schedules["season"].isin(seasons)
        & schedules["game_type"].eq("REG")
        & schedules["kickoff"].notna()
    ]
    return (
        reg.groupby(["season", "week"], as_index=False)["kickoff"]
        .min()
        .rename(columns={"kickoff": "as_of"})
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )


def build_walkforward_ratings(
    game_off: pd.DataFrame,
    schedules: pd.DataFrame,
    cfg: Config,
    seasons: "list[int] | None" = None,
    lambdas: "tuple[float, float] | None" = None,
    cache_key: "str | None" = "default",
    verbose: bool = False,
) -> pd.DataFrame:
    """Fit ratings, pace and HFA once per (season, week), walking forward.

    Returns a long frame: season, week, as_of, team, off_rating, def_rating, pace_rating,
    hfa_epa, n_games. Every downstream consumer -- the drive model's training features,
    the simulator, the backtest -- reads this one artifact, so the expensive walk happens
    exactly once and no consumer can accidentally reach a rating from the future.
    """
    seasons = seasons or cfg.train_seasons
    # The ridge penalties and the pace penalty are baked into every number in this frame,
    # so they belong in the cache key. Without them, `run_backtest.py --tune-lambdas`
    # writes a new `tuned:` block, reloads the config, and then silently reuses ratings fit
    # with the OLD penalties -- the grid search has no effect on anything downstream. Same
    # failure mode as the stale drive cache; see config.read_cached_frame.
    lo, ld = lambdas if lambdas is not None else cfg.effective_lambdas
    tag = f"{cache_key}_o{lo:g}_d{ld:g}_p{cfg.pace.lambda_:g}"
    path = (
        CACHE_DIR / f"walkforward_ratings_{tag}_{min(seasons)}_{max(seasons)}.parquet"
    )
    signature=build_cache_signature(builder=Path(__file__),config={"config":asdict(cfg),"seasons":seasons,"lambdas":[lo,ld]},inputs={"game_offense":frame_signature(game_off,["game_id","kickoff","offense","defense","epa_per_play","n_competitive_plays","drives"]),"schedules":frame_signature(schedules,["game_id","season","week","kickoff","result","total"])},artifact_version=2)
    if cache_key:
        cached=read_cached_frame(path,["season","week","as_of","team","off_rating","def_rating","pace_rating"],signature)
        if cached is not None:return cached

    cutoffs = week_cutoffs(schedules, seasons)
    out = []
    prev_season_fit: "RatingsFit | None" = None
    priors: "pd.DataFrame | None" = None
    current_season: "int | None" = None

    for row in cutoffs.itertuples(index=False):
        if row.season != current_season:
            # Roll the offseason: last season's final ratings, regressed, become priors.
            if current_season is not None:
                prev_season_fit = fit_ratings(
                    game_off, row.as_of, cfg, priors=priors, lambdas=lambdas
                )
            priors = season_priors(prev_season_fit, cfg)
            current_season = row.season

        fit = fit_ratings(game_off, row.as_of, cfg, priors=priors, lambdas=lambdas)
        pace = fit_pace(game_off, row.as_of, cfg)
        frame = fit.table.copy()
        frame["pace_rating"] = pace
        frame["season"] = row.season
        frame["week"] = row.week
        frame["as_of"] = row.as_of
        frame["hfa_epa"] = fit.hfa_epa
        frame["intercept"] = fit.intercept
        out.append(frame.reset_index())
        if verbose:
            print(
                f"  ratings {row.season} wk{row.week:>2}  games={fit.n_games:>4}  "
                f"hfa_epa={fit.hfa_epa:+.4f}"
            )

    result = pd.concat(out, ignore_index=True)
    if cache_key:
        write_cached_frame(result,path,signature)
    return result


def ratings_at(walkforward: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """The team table for one (season, week), indexed by team."""
    sub = walkforward[(walkforward["season"] == season) & (walkforward["week"] == week)]
    if sub.empty:
        raise KeyError(f"ratings_at: no walk-forward ratings for {season} week {week}")
    return sub.set_index("team")


# --------------------------------------------------------------------------------------
# 5c. Lambda tuning
# --------------------------------------------------------------------------------------

def tune_lambdas(
    game_off: pd.DataFrame,
    schedules: pd.DataFrame,
    cfg: Config,
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk-forward grid search over `lambda_grid` x `lambda_grid`.

    Scored by out-of-sample RMSE of predicted margin vs actual margin across
    backtest_start..current-1. Margin is predicted with a points-per-net-EPA scale fit on
    an expanding window of *earlier* seasons only, so the scoring is walk-forward too.
    """
    graded = schedules[
        schedules["season"].isin(cfg.backtest_seasons)
        & schedules["game_type"].eq("REG")
        & schedules["result"].notna()
        & (schedules["week"] >= 4)
    ][["game_id", "season", "week", "home_team", "away_team", "result", "location"]]

    grid = [float(x) for x in cfg.ratings.lambda_grid]
    records = []
    for lo in grid:
        for ld in grid:
            wf = build_walkforward_ratings(
                game_off, schedules, cfg, seasons=cfg.train_seasons,
                lambdas=(lo, ld), cache_key=None,
            )
            preds = _predict_margins_from_ratings(wf, graded, cfg)
            err = preds["pred_margin"] - preds["result"]
            records.append({
                "lambda_off": lo,
                "lambda_def": ld,
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
                "n": len(preds),
            })
            if verbose:
                r = records[-1]
                print(
                    f"  lambda_off={lo:>6.0f} lambda_def={ld:>6.0f}  "
                    f"RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}"
                )

    return pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)


def _predict_margins_from_ratings(
    walkforward: pd.DataFrame, games: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Linear points-per-net-EPA map, calibrated on strictly earlier seasons.

    Used only for lambda selection -- the real margin projection comes from the drive
    simulator. A cheap monotone map suffices because the grid search only needs to *rank*
    lambda pairs.
    """
    wf = walkforward.set_index(["season", "week", "team"])[
        ["off_rating", "def_rating"]
    ]
    lookup = wf.to_dict("index")
    rows = []
    for g in games.itertuples(index=False):
        h = lookup.get((g.season, g.week, g.home_team))
        a = lookup.get((g.season, g.week, g.away_team))
        if h is None or a is None:
            continue
        net_h = net_epa(h["off_rating"], a["def_rating"], cfg)
        net_a = net_epa(a["off_rating"], h["def_rating"], cfg)
        rows.append({
            "season": g.season,
            "net_diff": net_h - net_a,
            "is_home": 0.0 if g.location == "Neutral" else 1.0,
            "result": g.result,
        })
    df = pd.DataFrame(rows)

    preds = []
    for season in sorted(df["season"].unique()):
        train, test = df[df["season"] < season], df[df["season"] == season]
        if len(train) < 200:
            # Not enough earlier data to calibrate: fall back to an HFA-only prediction
            # rather than fitting on the season being graded.
            scale, hfa = 0.0, cfg.context.hfa_league_mean
        else:
            X = np.column_stack([train["net_diff"], train["is_home"]])
            scale, hfa = np.linalg.lstsq(
                X, train["result"].to_numpy(float), rcond=None
            )[0]
        out = test.copy()
        out["pred_margin"] = scale * out["net_diff"] + hfa * out["is_home"]
        preds.append(out)
    return pd.concat(preds, ignore_index=True)
