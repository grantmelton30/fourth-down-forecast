"""NCAA quarterback adjustment (DECISIONS.md D17).

The NFL build treats a quarterback change as its own explicit term because it is "the
largest single source of NFL model error" (`nfl-model/src/qb.py`). This module is the NCAA
equivalent. It did not exist before 2026-08-19: the college model had no notion of who was
playing quarterback at all.

WHAT WAS MEASURED, AND WHAT IS AND IS NOT CLAIMED. On 2,985 graded games, a team starting a
different quarterback than its previous game underperformed the closing line by 3.21 points
(t = -5.56). That splits in two, and the split is the whole design:

  * the previous starter ABSENT entirely -- announced, knowable before kickoff:
    -0.25 vs the market, t = -0.37. The market prices this correctly; there is no edge in it.
  * the previous starter PLAYED and lost the job mid-game: -6.46, t = -7.56. Nobody can know
    this in advance, and it is a consequence of the game rather than a cause of it.

Only the first is used. The second is deliberately excluded -- see `incumbent_absent`.

Against OUR model rather than the market, known absences cost 1.37 points against a +0.61
baseline. That ~2-point gap on roughly 9% of games is what this module closes. It is an
accuracy fix, not an edge; the t = -0.37 above is direct evidence against an edge.

IT IS A BINARY FLAG, AND THAT IS A MEASURED RETREAT. This module was built first to value
the quarterback INDIVIDUALLY, mirroring the NFL one, on the reasoning that a flat scalar
wrongly prices an elite starter's absence and a fringe starter's the same. That reasoning is
sound and the implementation still lost:

    quality-scaled  slope +5.12 per unit,  t = +0.89   RMSE +0.020% (WORSE)
    binary          slope -1.758 per out,  t = -3.05   RMSE -0.142% overall, -0.533% affected

A college backup has almost no prior attempts, so shrinkage pulls his rating to league
average and the "quality delta" turns out to be the incumbent's own rating in disguise
rather than any read on the drop-off. `nfl-model` escapes this only because nfeloqb supplies
per-quarterback point values maintained outside the model; CFBD has no equivalent. The
elaborate version was the better idea and the worse feature, so the simple one ships.

The 1.75 points is corroborated twice from sources that share no input: the outcome-fitted
slope is -1.758, and the market's own implied adjustment (how much further it moves than our
pre-QB projection) is +1.82 home-out / -1.62 away-out, about 1.72.

THE EMBEDDED QUARTERBACK IS STILL THE LOAD-BEARING IDEA. A team's `off_rating` already
reflects whoever has been taking the snaps, so the incumbent is identified from PRIOR games
and the adjustment fires only when that specific passer is missing. If the usual starter
plays, the rating already describes him and the adjustment is exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PASSER_COLUMNS = ["season", "week", "team", "qb_id", "qb_name", "attempts", "ppa"]

QB_TABLE_COLUMNS = [
    "game_id", "season", "week", "team", "incumbent_id", "incumbent_rating",
    "replacement_id", "replacement_rating", "incumbent_absent", "qb_delta",
]


@dataclass(frozen=True)
class QBAdjustment:
    """Points of spread adjustment, home perspective, already capped."""

    points: float
    source: str          # measured | manual | none
    note: str = ""


def build_passer_games(
    attempts: pd.DataFrame, ppa: pd.DataFrame,
) -> pd.DataFrame:
    """Tidy one row per (season, week, team, passer) with attempts and efficiency.

    Two sources because neither carries both halves: CFBD's `games/players` gives C/ATT (the
    sample size that drives shrinkage) and `ppa/players/games` gives the efficiency. They
    join on (season, week, team, player id) -- `ppa/players/games` does not return a
    `game_id`, which is why the join is not on one.
    """
    if attempts is None or attempts.empty:
        return pd.DataFrame(columns=[*PASSER_COLUMNS, "game_id"])
    a = attempts.copy()
    a["qb_id"] = a["qb_id"].astype(str)
    if ppa is None or ppa.empty:
        a["ppa"] = np.nan
        return a.reindex(columns=[*PASSER_COLUMNS, "game_id"])
    p = ppa.copy()
    p["qb_id"] = p["qb_id"].astype(str)
    merged = a.merge(
        p[["season", "week", "team", "qb_id", "ppa"]],
        on=["season", "week", "team", "qb_id"], how="left",
    )
    return merged.reindex(columns=[*PASSER_COLUMNS, "game_id"])


def qb_ratings_table(passers: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per (game_id, team): the incumbent, the replacement, and their prior-form ratings.

    NO LOOKAHEAD, and the mechanism is a shift rather than a filter: every rating is built
    from a cumulative sum that is shifted one game back within (season, team, passer), so a
    passer's rating for week N uses weeks 1..N-1 and never week N itself. The one value read
    from the game being predicted is whether the incumbent threw a pass at all -- which is
    announced before kickoff and is the only reason this is usable prospectively.

    Shrinkage mirrors the NFL module's `dropbacks / (dropbacks + 320)`, for the same reason:
    a freshman with 40 good attempts is not an elite quarterback. Ratings are centred so
    that 0 is an average passer, and the centre itself is an expanding mean over strictly
    earlier games so it cannot import the future either.
    """
    empty = pd.DataFrame(columns=QB_TABLE_COLUMNS)
    if passers is None or passers.empty:
        return empty

    raw = passers.dropna(subset=["season", "week", "team", "qb_id"]).copy()
    if raw.empty:
        return empty
    raw["attempts"] = pd.to_numeric(raw["attempts"], errors="coerce").fillna(0.0)
    raw["ppa"] = pd.to_numeric(raw["ppa"], errors="coerce")

    # THE ABSENT STARTER HAS NO ROW. CFBD's `games/players` only returns players who
    # actually appeared, so a quarterback who misses a game is simply missing from it --
    # not present with zero attempts. Ranking within the rows a game happens to contain
    # therefore can never detect an absence, which made the first version of this function
    # a silent no-op across 11,441 team-games (caught by the "0 with the incumbent absent"
    # line in run_backtest.py, which is why that count is printed).
    #
    # So the roster is reconstructed explicitly: every passer who has ever thrown for a team
    # is carried into every one of that team's later weeks, with zero attempts when he did
    # not appear. Absence then becomes an observable value rather than an absent row.
    team_weeks = raw[["season", "team", "week", "game_id"]].drop_duplicates()
    team_passers = raw[["season", "team", "qb_id", "qb_name"]].drop_duplicates(
        ["season", "team", "qb_id"])
    df = team_weeks.merge(team_passers, on=["season", "team"], how="left")
    df = df.merge(
        raw[["season", "team", "week", "qb_id", "attempts", "ppa"]],
        on=["season", "team", "week", "qb_id"], how="left",
    )
    # Attempts fill to zero (he was there and threw nothing, or was not there at all);
    # efficiency stays missing, because there is no performance to average.
    df["attempts"] = df["attempts"].fillna(0.0)
    df = df.sort_values(["season", "team", "qb_id", "week"])

    # Cumulative, strictly-prior totals per passer within a season.
    df["_att_x_ppa"] = df["attempts"] * df["ppa"].fillna(0.0)
    df["_att_rated"] = np.where(df["ppa"].notna(), df["attempts"], 0.0)
    grp = df.groupby(["season", "team", "qb_id"], sort=False)
    df["prior_att"] = grp["attempts"].cumsum() - df["attempts"]
    df["prior_att_rated"] = grp["_att_rated"].cumsum() - df["_att_rated"]
    df["prior_att_x_ppa"] = grp["_att_x_ppa"].cumsum() - df["_att_x_ppa"]

    # Expanding league centre, aggregated to the WEEK and then shifted a whole week back.
    # Doing this row-wise would let other games from the SAME week into the centre -- a
    # tiny leak, since it is an average over hundreds of passers, but a leak, and this
    # repo's whole no-lookahead contract is that "tiny" is not a category that exists.
    wk = df.groupby(["season", "week"], as_index=False)[["_att_x_ppa", "_att_rated"]].sum()
    wk = wk.sort_values(["season", "week"])
    cum_num = wk["_att_x_ppa"].cumsum() - wk["_att_x_ppa"]
    cum_den = wk["_att_rated"].cumsum() - wk["_att_rated"]
    wk["league_ppa"] = (cum_num / cum_den.replace(0.0, np.nan)).fillna(0.0)
    df = df.merge(wk[["season", "week", "league_ppa"]], on=["season", "week"], how="left")

    raw = df["prior_att_x_ppa"] / df["prior_att_rated"].replace(0.0, np.nan)
    shrink = df["prior_att"] / (df["prior_att"] + float(cfg.qb.regression_attempts))
    df["rating"] = ((raw - df["league_ppa"]) * shrink).fillna(0.0)
    # A passer with no prior workload at all is not "average", he is unknown. Rating 0 says
    # exactly that, and `usable` records the distinction for the caller.
    df["usable"] = df["prior_att"] >= float(cfg.qb.min_attempts)

    # Rank passers within each team-game by PRIOR workload: rank 1 is the quarterback the
    # team's offensive rating is really describing.
    df["_rank"] = df.groupby(["game_id", "team"], sort=False)["prior_att"].rank(
        method="first", ascending=False)

    inc = df[df["_rank"] == 1].rename(columns={
        "qb_id": "incumbent_id", "rating": "incumbent_rating"})
    rep = df[df["_rank"] == 2].rename(columns={
        "qb_id": "replacement_id", "rating": "replacement_rating"})
    keys = ["game_id", "season", "week", "team"]
    out = inc[[*keys, "incumbent_id", "incumbent_rating", "attempts"]].rename(
        columns={"attempts": "incumbent_attempts_now"})
    out = out.merge(rep[[*keys, "replacement_id", "replacement_rating"]],
                    on=keys, how="left")

    # THE ONLY VALUE TAKEN FROM THE GAME BEING PREDICTED. Zero passes from the incumbent is
    # a full absence, which is announced pre-kickoff. A partial workload means he played and
    # lost the job mid-game -- worth -6.5 points and knowable to nobody in advance -- so it
    # is explicitly NOT treated as an absence.
    out["incumbent_absent"] = out["incumbent_attempts_now"].fillna(0.0).eq(0.0)

    # BINARY, and this is a measured retreat from a more elaborate design that did not work.
    #
    # The first version scaled the adjustment by a quarterback QUALITY delta
    # (`replacement_rating - incumbent_rating`), mirroring `nfl-model/src/qb.py`. Measured
    # on 2,985 graded games it carried nothing: slope +5.12 points per unit, t = +0.89, and
    # applying it made spread RMSE WORSE (+0.020% overall, +0.071% on affected games).
    #
    # The reason is visible in the ratings themselves. A college backup has almost no prior
    # attempts, so shrinkage pulls his rating to ~0 -- league average -- and the "quality
    # delta" is therefore not a read on the drop-off at all, it is the incumbent's own
    # rating wearing a disguise. The NFL module escapes this because nfeloqb supplies real
    # per-quarterback point values maintained outside the model; college has no equivalent.
    #
    # The binary form -- he is out, or he is not -- is significant where the quality form is
    # not: slope -1.758 points per starter-out, t = -3.05, RMSE -0.142% overall and -0.533%
    # on affected games. The market agrees independently (+1.82 home-out, -1.62 away-out,
    # implying ~-1.72), which is a cross-check from a source that never saw an outcome.
    #
    # The ratings above are retained because they identify the incumbent, and as the record
    # of what was tried; they no longer scale the adjustment.
    out["qb_delta"] = np.where(out["incumbent_absent"], -1.0, 0.0)
    return out.reindex(columns=QB_TABLE_COLUMNS)


def apply_manual_status(
    table: pd.DataFrame, manual: "pd.DataFrame | None",
) -> pd.DataFrame:
    """Override availability from `data/manual/qb_status.csv`, which always wins.

    Mirrors `nfl-model`'s `resting_starters.csv` convention. The live path has no box score
    to read an absence from, so a human (or an ESPN pull) supplies it. Columns:
    `season, week, team, starter_out` plus `observed_at` and `source` for the same
    leakage-audit reason `features.normalize_preseason_sources` requires them.
    """
    if manual is None or manual.empty or table.empty:
        return table
    required = {"season", "week", "team", "starter_out"}
    missing = required - set(manual.columns)
    if missing:
        raise ValueError(f"qb_status manual file missing columns: {sorted(missing)}")
    m = manual.copy()
    m["season"] = pd.to_numeric(m["season"], errors="coerce").astype("Int64")
    m["week"] = pd.to_numeric(m["week"], errors="coerce").astype("Int64")
    m["starter_out"] = m["starter_out"].astype(bool)
    out = table.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    out = out.merge(m[["season", "week", "team", "starter_out"]],
                    on=["season", "week", "team"], how="left")
    override = out["starter_out"].notna()
    out.loc[override, "incumbent_absent"] = out.loc[override, "starter_out"].astype(bool)
    out["qb_delta"] = np.where(out["incumbent_absent"].fillna(False), -1.0, 0.0)
    return out.drop(columns=["starter_out"]).reindex(columns=QB_TABLE_COLUMNS)


def game_qb_delta(table: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per game with `qb_delta_gap` = home delta minus away delta."""
    if table is None or table.empty:
        return pd.DataFrame(columns=["game_id", "qb_delta_gap"])
    t = table[["game_id", "team", "qb_delta"]]
    g = games[["game_id", "homeTeam", "awayTeam"]].copy()
    g = g.merge(t.rename(columns={"team": "homeTeam", "qb_delta": "home_qb_delta"}),
                on=["game_id", "homeTeam"], how="left")
    g = g.merge(t.rename(columns={"team": "awayTeam", "qb_delta": "away_qb_delta"}),
                on=["game_id", "awayTeam"], how="left")
    g["qb_delta_gap"] = (g["home_qb_delta"].fillna(0.0)
                          - g["away_qb_delta"].fillna(0.0))
    return g[["game_id", "qb_delta_gap"]]


def qb_points(delta_diff, cfg) -> np.ndarray:
    """Convert the starter-out indicator into points of spread, capped.

    The cap exists for the reason the NFL module states: a bad depth-chart read must not be
    able to wreck a line. A missing delta is zero, never a penalty -- so an absent feed
    degrades to today's behaviour rather than quietly moving a number.
    """
    if not cfg.qb.enabled:
        return np.zeros(len(delta_diff))
    d = pd.to_numeric(pd.Series(delta_diff), errors="coerce").fillna(0.0).to_numpy(float)
    return np.clip(d * float(cfg.qb.points_per_starter_out),
                   -float(cfg.qb.max_adjustment_points),
                   float(cfg.qb.max_adjustment_points))


__all__ = ["QBAdjustment", "PASSER_COLUMNS", "QB_TABLE_COLUMNS", "apply_manual_status",
           "build_passer_games", "game_qb_delta", "qb_points", "qb_ratings_table"]
