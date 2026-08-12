"""Team strength: ridge on offensive efficiency, plus a pace ridge (§6).

Same machinery as the NFL build, with three college-specific differences that all trace
back to §0: heavier penalties because the schedule graph is barely connected, conference
shrinkage because the global mean is the wrong target for a team in a weak league, and an
FCS bucket so non-FBS games still inform the FBS side.

NO LOOKAHEAD. Every fit takes an `as_of` timestamp and sees only games that kicked off
strictly before it. Asserted on every fit, not just in tests.

SIGN CONVENTION, identical to the NFL build:
    off_rating  higher = BETTER offense
    def_rating  higher = WORSE  defense
"""

from __future__ import annotations

from dataclasses import asdict,dataclass

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .config import CACHE_DIR,Config,build_cache_signature,frame_signature,read_cached_frame,write_cached_frame

_MIN_DECAY_WEIGHT = 1e-3


@dataclass(frozen=True)
class RatingsFit:
    table: pd.DataFrame          # index=team, cols off_rating/def_rating/n_games
    hfa_ppa: float
    intercept: float
    as_of: object
    n_games: int


def team_universe(game_off: pd.DataFrame) -> list:
    return sorted(set(game_off["offense_norm"]) | set(game_off["defense_norm"]))


def _assert_no_lookahead(rows: pd.DataFrame, as_of, what: str) -> None:
    if len(rows) and rows["kickoff"].max() >= as_of:
        bad = rows.loc[rows["kickoff"] >= as_of, "game_id"].unique()[:5]
        raise AssertionError(
            f"LOOKAHEAD in {what}: games at or after as_of={as_of} entered the fit, "
            f"e.g. {list(bad)}"
        )


def _decay(rows: pd.DataFrame, half_life: float) -> np.ndarray:
    games_ago = (
        rows.groupby("offense_norm")["kickoff"].rank(method="first", ascending=False) - 1.0
    )
    return np.power(0.5, games_ago.to_numpy() / float(half_life))


def _solve(X, y, w, penalty):
    W = sparse.diags(w)
    A = (X.T @ W @ X).toarray() + np.diag(penalty)
    return np.linalg.solve(A, X.T @ (w * y))


def fit_ratings(
    game_off: pd.DataFrame,
    as_of,
    cfg: Config,
    teams: list,
    priors: "pd.DataFrame | None" = None,
) -> RatingsFit:
    """Offense/defense efficiency in PPA-per-play units, as of `as_of`."""
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    col_home, col_int = 2 * n_teams, 2 * n_teams + 1
    n_cols = 2 * n_teams + 2

    rows = game_off[game_off["kickoff"] < as_of]
    _assert_no_lookahead(rows, as_of, "fit_ratings")
    w_decay = _decay(rows, cfg.ratings.half_life_games)
    keep = w_decay >= _MIN_DECAY_WEIGHT
    rows, w_decay = rows[keep], w_decay[keep]

    if len(rows) == 0:
        empty = pd.DataFrame(
            {"off_rating": 0.0, "def_rating": 0.0, "n_games": 0}, index=teams
        )
        empty.index.name = "team"
        empty["conference"] = None
        return RatingsFit(empty, 0.0, 0.0, as_of, 0)

    n = len(rows)
    off_i = rows["offense_norm"].map(idx).to_numpy()
    def_i = rows["defense_norm"].map(idx).to_numpy()
    r = np.arange(n)
    data, ri, ci = [], [], []
    for cols, vals in (
        (off_i, np.ones(n)),
        (n_teams + def_i, np.ones(n)),
        (np.full(n, col_home), rows["is_home_offense"].to_numpy(float)),
        (np.full(n, col_int), np.ones(n)),
    ):
        ri.append(r)
        ci.append(cols)
        data.append(vals)

    y = rows["ppa_per_play"].to_numpy(float)
    w = rows["n_plays"].to_numpy(float) * w_decay * rows["game_weight"].to_numpy(float)

    if priors is not None and len(priors):
        pteams = [t for t in priors.index if t in idx]
        k = len(pteams)
        w_prior = cfg.ratings.prior_weight_games * cfg.ratings.league_mean_plays_per_game
        ri.append(np.arange(k) + n)
        ci.append(np.array([idx[t] for t in pteams]))
        data.append(np.ones(k))
        ri.append(np.arange(k) + n + k)
        ci.append(np.array([n_teams + idx[t] for t in pteams]))
        data.append(np.ones(k))
        y = np.concatenate([
            y, priors.loc[pteams, "prior_off"].to_numpy(float),
            priors.loc[pteams, "prior_def"].to_numpy(float),
        ])
        w = np.concatenate([w, np.full(2 * k, w_prior)])

    X = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(ri), np.concatenate(ci))),
        shape=(len(y), n_cols),
    )
    penalty = np.zeros(n_cols)
    penalty[:n_teams] = cfg.ratings.lambda_off
    penalty[n_teams:2 * n_teams] = cfg.ratings.lambda_def
    beta = _solve(X, y, w, penalty)

    off, dfn = beta[:n_teams].copy(), beta[n_teams:2 * n_teams].copy()
    intercept = float(beta[col_int]) + float(off.mean() + dfn.mean())
    off -= off.mean()
    dfn -= dfn.mean()

    counts = rows["offense_norm"].value_counts()
    table = pd.DataFrame(
        {"off_rating": off, "def_rating": dfn,
         "n_games": [int(counts.get(t, 0)) for t in teams]},
        index=teams,
    )
    table.index.name = "team"
    table = _conference_shrink(table, rows, cfg)
    return RatingsFit(
        table, float(beta[col_home]), intercept, as_of, int(rows["game_id"].nunique())
    )


def _team_conference(rows: pd.DataFrame, teams: list) -> pd.Series:
    """Conference per team, taken from whichever side of the game they appeared on."""
    home = rows[["homeTeam", "homeConference"]].rename(
        columns={"homeTeam": "team", "homeConference": "conference"})
    away = rows[["awayTeam", "awayConference"]].rename(
        columns={"awayTeam": "team", "awayConference": "conference"})
    both = pd.concat([home, away], ignore_index=True).dropna()
    lookup = both.drop_duplicates("team").set_index("team")["conference"]
    return lookup.reindex(teams)


def _conference_shrink(
    table: pd.DataFrame, rows: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Shrink each team toward its conference mean (§6c).

    The ridge penalty shrinks toward the *global* mean, which is the wrong target for a
    team in a weak conference -- it makes a mediocre SEC team and a mediocre MAC team look
    more alike than they are. `n_games` stands in for effective sample size, so a team with
    three data points is pulled hard and a team with a full season barely moves.
    """
    table = table.copy()
    table["conference"] = _team_conference(rows, list(table.index)).values
    n_eff = table["n_games"].astype(float)
    k = cfg.ratings.conference_shrink_weight

    for col in ("off_rating", "def_rating"):
        conf_mean = table.groupby("conference")[col].transform("mean")
        conf_mean = conf_mean.fillna(table[col].mean())
        table[col] = (n_eff * table[col] + k * conf_mean) / (n_eff + k)
        table[col] = table[col] - table[col].mean()
    return table


def fit_pace(game_off: pd.DataFrame, as_of, cfg: Config, teams: list) -> pd.Series:
    """Per-team pace such that E[drives per team] = league_mean + pace_A + pace_B.

    College pace genuinely ranges from sub-10 to 15-plus drives, and it is where a
    disproportionate share of the available totals edge lives -- so this is a first-class
    target, not a correction.
    """
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    rows = game_off[(game_off["kickoff"] < as_of) & game_off["drives"].notna()]
    _assert_no_lookahead(rows, as_of, "fit_pace")
    w = _decay(rows, cfg.pace.half_life_games)
    keep = w >= _MIN_DECAY_WEIGHT
    rows, w = rows[keep], w[keep]
    if len(rows) == 0:
        return pd.Series(0.0, index=teams, name="pace_rating")

    n = len(rows)
    off_i = rows["offense_norm"].map(idx).to_numpy()
    def_i = rows["defense_norm"].map(idx).to_numpy()
    r = np.arange(n)
    X = sparse.csr_matrix(
        (np.ones(3 * n),
         (np.concatenate([r, r, r]),
          np.concatenate([off_i, def_i, np.full(n, n_teams)]))),
        shape=(n, n_teams + 1),
    )
    penalty = np.full(n_teams + 1, float(cfg.pace.lambda_))
    penalty[n_teams] = 0.0
    beta = _solve(X, rows["drives"].to_numpy(float), w, penalty)
    pace = beta[:n_teams] - beta[:n_teams].mean()
    return pd.Series(pace, index=teams, name="pace_rating")


def season_priors(prev: "RatingsFit | None", cfg: Config):
    """Last season's final ratings regressed toward the mean.

    College regression is heavier than the NFL's (0.42 vs 0.28) because the portal turns
    rosters over annually rather than gradually.
    """
    if prev is None:
        return None
    keep = 1.0 - cfg.ratings.offseason_regression
    return pd.DataFrame({
        "prior_off": prev.table["off_rating"] * keep,
        "prior_def": prev.table["def_rating"] * keep,
    })


# --------------------------------------------------------------------------------------
# Ratings -> net_epa, the one number the shared simulator takes per offense
# --------------------------------------------------------------------------------------

def net_epa(off_rating: float, def_rating_opponent: float, cfg: Config) -> float:
    """Expected EPA/play for this offense against this defense.

    Identical in form and sign to `nfl-model/src/ratings.py::net_epa`, and identical to the
    combination `backtest.build_features` has always formed inline -- this is that
    expression given a name so the simulator and the linear projection cannot drift apart.

    The operator is `+`, not the `-` a naive reading of the playbook suggests, because a
    HIGHER def_rating means a WORSE defense (the convention asserted at the top of this
    module). Facing a worse defense must raise the offense's expectation.

    The off/def weight asymmetry (1.45 / 1.0) is heavier than the NFL's for the same reason
    the ridge penalties are: offensive efficiency is the stickier of the two week to week,
    and college schedules are barely connected enough to identify defense as sharply.
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


def resolve_team_ratings(ratings: pd.DataFrame, team: str, cfg: Config) -> pd.Series:
    """One team's ratings row, falling back to the FCS bucket for non-FBS opponents.

    `ingest.build_game_offense` rewrites every non-FBS team to `cfg.teams.fcs_bucket_name`,
    so the ridge fits one pooled rating for all of them and that is what a projection
    against an FCS opponent should use. This function exists because a plain
    `ratings.loc[team]` -- which is what the NFL binding can afford to do -- raises KeyError
    the first time someone projects a week-one body-bag game.

    The configured `fcs_off_rating` / `fcs_def_rating` are the last resort, used only when
    the bucket itself has no fitted rating yet.
    """
    if team in ratings.index:
        return ratings.loc[team]
    bucket = cfg.teams.fcs_bucket_name
    if bucket in ratings.index:
        return ratings.loc[bucket]
    return pd.Series({
        "off_rating": cfg.teams.fcs_off_rating,
        "def_rating": cfg.teams.fcs_def_rating,
        "pace_rating": 0.0,
    })


def week_cutoffs(games: pd.DataFrame, seasons: list) -> pd.DataFrame:
    reg = games[games["season"].isin(seasons) & games["kickoff"].notna()]
    return (
        reg.groupby(["season", "week"], as_index=False)["kickoff"].min()
        .rename(columns={"kickoff": "as_of"})
        .sort_values(["season", "week"]).reset_index(drop=True)
    )


def live_tag(games: pd.DataFrame, cfg: Config) -> str:
    """How far the live season has actually progressed, as a cache-key fragment.

    The hyperparameter tag below stops being sufficient once `all_seasons` includes the
    season being played. Every ingredient of the fit can be identical week to week while
    the DATA changes underneath it, so an artifact built in week 0 would be served in week
    8 under a filename that looks perfectly correct. Encoding the live season's
    completed-game count makes new results re-key the artifact and force a rebuild, exactly
    as a retuned lambda does.

    Returns "" when the live season is not being ingested, so historical-only builds keep
    their existing filenames and nothing already on disk is invalidated.
    """
    live = cfg.live_season
    if live not in cfg.all_seasons or games is None or not len(games):
        return ""
    g = games[games["season"] == live]
    if "completed" in g.columns:
        g = g[g["completed"].fillna(False).astype(bool)]
    weeks = pd.to_numeric(g["week"], errors="coerce").dropna() if len(g) else []
    return f"_live{live}w{int(max(weeks)) if len(weeks) else 0}g{len(g)}"


def walkforward_path(cfg: Config, cache_key: "str | None" = "default",
                     live_tag: "str | None" = None) -> Path:
    """Where the walk-forward artifact for THIS configuration lives.

    THE HYPERPARAMETERS ARE IN THE FILENAME ON PURPOSE. The cache key used to be just
    "default", so `build_walkforward` returned whatever parquet was on disk no matter what
    the config said. That is not hypothetical: `pace.lambda` was retuned from 900 to 15 and
    the artifact silently kept serving the lambda-900 fit for five days. The visible
    symptom was in the simulator -- fitted pace spanned 0.14 drives of team-to-team
    variation against a real 4.4, so it projected Army and Air Force, the two slowest
    offenses in the sport, as league-average pace and over-projected their totals by ~15
    points.

    Mirrors `nfl-model/src/ratings.py::build_walkforward_ratings`, which already tags its
    artifacts this way. Encoding the config in the key rather than validating it on read
    also means two configurations can coexist -- which is what the `sweep{i}` keys want.

    Anything the fit reads belongs in the tag. Over-tagging costs a 12-second rebuild;
    under-tagging costs a silently wrong model.
    """
    r, pc = cfg.ratings, cfg.pace
    tag = (
        f"o{r.lambda_off:g}_d{r.lambda_def:g}_hl{r.half_life_games:g}"
        f"_pw{r.prior_weight_games:g}_or{r.offseason_regression:g}"
        f"_cs{r.conference_shrink_weight:g}"
        f"_p{pc.lambda_:g}_phl{pc.half_life_games:g}"
        f"_{min(cfg.all_seasons)}_{max(cfg.all_seasons)}"
    )
    stem = f"walkforward_ratings_{cache_key}_{tag}"
    if live_tag is not None:
        return CACHE_DIR / f"{stem}{live_tag}.parquet"
    # No live tag supplied: this is a READER (project_game, shared/sport.py) that wants
    # whatever the last build produced. The writer knows how far the season had progressed;
    # a reader does not, and should not have to re-ingest the schedule to find out. Newest
    # mtime wins -- the hyperparameters are already in `stem`, so the only thing varying
    # across these candidates is how much of the live season had been played.
    candidates = sorted(
        CACHE_DIR.glob(f"{stem}*.parquet"), key=lambda p: p.stat().st_mtime
    )
    return candidates[-1] if candidates else CACHE_DIR / f"{stem}.parquet"


def build_walkforward(
    game_off: pd.DataFrame,
    games: pd.DataFrame,
    cfg: Config,
    cache_key: "str | None" = "default",
    verbose: bool = False,
) -> pd.DataFrame:
    """Fit ratings, pace and HFA once per (season, week), walking forward.

    Every downstream consumer reads this one artifact, so the expensive walk happens once
    and nothing can accidentally reach a rating from the future.
    """
    # The writer names the artifact for the data it is about to fit on, so a new week of
    # live results produces a new filename instead of silently reusing last week's.
    path = walkforward_path(cfg, cache_key, live_tag=live_tag(games, cfg))
    signature=build_cache_signature(builder=Path(__file__),config=asdict(cfg),inputs={"game_offense":frame_signature(game_off,["game_id","kickoff","offense_norm","defense_norm","ppa_per_play","n_plays","drives"]),"games":frame_signature(games,["game_id","season","week","kickoff","completed","homePoints","awayPoints"])},artifact_version=2)
    if cache_key:
        cached=read_cached_frame(path,["season","week","as_of","team","off_rating","def_rating","pace_rating"],signature)
        if cached is not None:return cached

    teams = team_universe(game_off)
    cutoffs = week_cutoffs(games, cfg.all_seasons)
    out, prev_fit, priors, current_season = [], None, None, None

    for row in cutoffs.itertuples(index=False):
        if row.season != current_season:
            if current_season is not None:
                prev_fit = fit_ratings(game_off, row.as_of, cfg, teams, priors)
            priors = season_priors(prev_fit, cfg)
            current_season = row.season

        fit = fit_ratings(game_off, row.as_of, cfg, teams, priors)
        pace = fit_pace(game_off, row.as_of, cfg, teams)
        frame = fit.table.copy()
        frame["pace_rating"] = pace
        frame["season"], frame["week"] = row.season, row.week
        frame["as_of"] = row.as_of
        frame["hfa_ppa"] = fit.hfa_ppa
        out.append(frame.reset_index())
        if verbose:
            print(f"  {row.season} wk{row.week:>2}  games={fit.n_games:>5}", flush=True)

    result = pd.concat(out, ignore_index=True)
    if cache_key:
        write_cached_frame(result,path,signature)
    return result


def ratings_at(walkforward: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    sub = walkforward[(walkforward["season"] == season) & (walkforward["week"] == week)]
    if sub.empty:
        raise KeyError(f"no walk-forward ratings for {season} week {week}")
    return sub.set_index("team")
