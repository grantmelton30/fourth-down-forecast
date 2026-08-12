"""Measure the `simulation:` block of config/ncaa.yaml from CFBD data.

WHY THIS FILE EXISTS. The NFL model's constants are not transferable. College kicking is
materially worse, two-point attempts are far more common, and the endgame is longer. Every
number the shared simulator reads for NCAA is produced here, from the cached CFBD drives
payload, so that a future reader can rerun it rather than trust it.

Run it:

    NCAA_MODEL_CACHE_DIR=~/.cache/ncaa-model \\
      python -m analysis.measure_simulation_constants

It reads only the permanent JSON cache and spends no API calls.

HOW THE CONVERSION CONSTANTS ARE IDENTIFIED. The CFBD /plays endpoint does not return PAT
plays -- across 1.44M cached plays there are 27 conversion rows -- so extra points and
two-point tries cannot be counted directly. What the drives payload does give exactly is
the points the offense scored on a touchdown drive: 6 (conversion failed), 7 (kick good),
or 8 (two-point good). That is three outcome probabilities, i.e. two free numbers, against
three unknowns (attempt rate, two-point success, kick success) -- underdetermined on its
own.

It becomes identified by splitting on the post-touchdown score differential. The attempt
rate swings by an order of magnitude across margins (that is what a two-point chart *is*),
while the kick-success and two-point-success probabilities do not depend on the scoreboard.
So we fit one attempt rate per margin bucket and a single shared `xp_make_prob` and
`two_point_success_prob` across all of them by maximum likelihood. With B buckets that is
2B free cell probabilities against B + 2 parameters, leaving B - 2 degrees of freedom, and
the fit is checked against them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from src.config import CACHE_DIR

# 2020 was structurally abnormal -- partial conference schedules and empty stadiums -- and
# the repo already excludes it from scoring (config `exclude_from_scoring`). Excluded here
# too, for the same reason.
EXCLUDE_SEASONS = (2020,)

# driveResult values where the OFFENSE scored the touchdown. The 2021 payload uses
# "RUSHING TD" / "PASSING TD" for a handful of drives; see `_RESULT_MAP` in src/drives.py.
OFFENSIVE_TD = {"TD", "RUSHING TD", "PASSING TD"}

# driveResult values where the DEFENSE returned the ball for a touchdown. These are a
# turnover for the offense; the points belong to the opponent and the simulator generates
# them from `defensive_td_lambda`.
DEFENSIVE_TD = {
    "INT TD", "FUMBLE TD", "FUMBLE RETURN TD", "DOWNS TD", "INT RETURN TOUCH",
    "END OF HALF TD", "END OF GAME TD",
}

# Return touchdowns on a kicking play. NO_SCORE for the offense; the simulator generates
# them from `special_teams_td_lambda`.
SPECIAL_TEAMS_TD = {"PUNT TD", "PUNT RETURN TD", "MISSED FG TD", "FG TD"}

SAFETY = {"SF"}


def load_raw_drives(seasons: "list[int] | None" = None) -> pd.DataFrame:
    """Concatenate the permanently cached per-season CFBD drives payloads."""
    seasons = seasons or list(range(2019, 2026))
    frames = []
    for year in seasons:
        path = CACHE_DIR / f"drives_{year}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Set NCAA_MODEL_CACHE_DIR to the permanent cache."
            )
        d = pd.json_normalize(json.loads(path.read_text()))
        d["season"] = year
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["offense_points"] = df["endOffenseScore"] - df["startOffenseScore"]
    df["score_diff"] = df["startOffenseScore"] - df["startDefenseScore"]
    return df[~df["season"].isin(EXCLUDE_SEASONS)].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Conversions: xp_make_prob, two_point_attempt_rate, two_point_success_prob
# --------------------------------------------------------------------------------------

def _conversion_buckets(td: pd.DataFrame, min_count: int = 300) -> pd.DataFrame:
    """Counts of (6, 7, 8)-point touchdown drives per post-touchdown margin.

    Bucketing on the margin *after* the touchdown is what makes the three constants
    separately identifiable -- it is the quantity a two-point chart is indexed by.
    """
    post_td = (td["score_diff"] + 6).clip(-21, 21).astype(int)
    # `pts` must be integer-typed: the pivot puts its *values* in the column names, so a
    # float column yields "n6.0" and the n6/n7/n8 selection below silently reads zeros.
    grp = (
        pd.DataFrame({"post_td": post_td, "pts": td["offense_points"].astype(int)})
        .pivot_table(index="post_td", columns="pts", aggfunc=len, fill_value=0)
    )
    grp.columns = [f"n{c}" for c in grp.columns]
    for col in ("n6", "n7", "n8"):
        if col not in grp:
            grp[col] = 0
    grp = grp[["n6", "n7", "n8"]]
    # Margins too thin to support their own attempt rate are pooled into one bucket rather
    # than dropped, so the fitted league-average attempt rate stays a full-sample number.
    thin = grp.sum(axis=1) < min_count
    pooled = grp[thin].sum()
    out = grp[~thin].copy()
    out.index = out.index.astype(str)
    if thin.any():
        out.loc["other"] = pooled
    return out


def fit_conversions(td: pd.DataFrame) -> dict:
    """Maximum-likelihood xp_make_prob and two_point_success_prob, with one attempt rate
    per post-touchdown margin bucket.

    Cell probabilities in bucket i:  P(8) = a_i * s,  P(7) = (1 - a_i) * x,
    P(6) = 1 - P(7) - P(8), the residual covering both a missed kick and a failed try.
    """
    buckets = _conversion_buckets(td)
    n6 = buckets["n6"].to_numpy(float)
    n7 = buckets["n7"].to_numpy(float)
    n8 = buckets["n8"].to_numpy(float)
    n_buckets = len(buckets)
    if (n6 + n7 + n8).sum() == 0:
        raise ValueError(
            "fit_conversions: no touchdown drives after bucketing. Check that "
            "`offense_points` holds 6/7/8 and `score_diff` is populated."
        )

    def unpack(theta):
        x = 1.0 / (1.0 + np.exp(-theta[0]))
        s = 1.0 / (1.0 + np.exp(-theta[1]))
        a = 1.0 / (1.0 + np.exp(-theta[2:]))
        return x, s, a

    def neg_loglik(theta):
        x, s, a = unpack(theta)
        p8 = np.clip(a * s, 1e-12, 1 - 1e-12)
        p7 = np.clip((1 - a) * x, 1e-12, 1 - 1e-12)
        p6 = np.clip(1 - p7 - p8, 1e-12, 1 - 1e-12)
        return -float(n6 @ np.log(p6) + n7 @ np.log(p7) + n8 @ np.log(p8))

    theta0 = np.concatenate([[2.8, 0.0], np.full(n_buckets, -3.0)])
    res = minimize(neg_loglik, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"conversion MLE failed to converge: {res.message}")
    x, s, a = unpack(res.x)

    # League-average attempt rate: the fitted rates weighted by how often each margin
    # actually occurs. This is the number the simulator draws against off the chart.
    totals = n6 + n7 + n8
    attempt_rate = float((a * totals).sum() / totals.sum())

    # Goodness of fit against the buckets' own cell counts.
    p8, p7 = a * s, (1 - a) * x
    p6 = 1 - p7 - p8
    obs = np.column_stack([n6, n7, n8])
    exp = np.column_stack([p6, p7, p8]) * totals[:, None]
    g_stat = 2.0 * float(np.sum(obs * np.log(np.where(obs > 0, obs / exp, 1.0))))
    dof = n_buckets - 2

    # What the simulator actually consumes is the pooled conversion distribution -- the
    # per-bucket split only exists to identify the three parameters. A per-bucket misfit
    # matters only insofar as it distorts this, so report both.
    fitted_pooled = (exp.sum(axis=0) / totals.sum())
    observed_pooled = (obs.sum(axis=0) / totals.sum())

    return {
        "xp_make_prob": float(x),
        "two_point_success_prob": float(s),
        "two_point_attempt_rate": attempt_rate,
        "_n_td_drives": int(totals.sum()),
        "_n_buckets": n_buckets,
        "_gof_G": g_stat,
        "_gof_dof": dof,
        "_gof_p": float(chi2.sf(g_stat, dof)),
        "_pooled_fitted": fitted_pooled,
        "_pooled_observed": observed_pooled,
        "_attempt_by_margin": dict(zip(list(buckets.index), a.round(4))),
    }


# --------------------------------------------------------------------------------------
# Rare non-offensive scores, as per-team-per-game rates
# --------------------------------------------------------------------------------------

def measure_lambdas_from_drives(drives: pd.DataFrame) -> dict:
    """Per-team-per-game rates counted off `driveResult`, for comparison only.

    Kept because it is the obvious way to do this and it is subtly wrong: see
    `measure_lambdas`, which is the one the config is built from.
    """
    team_games = 2.0 * drives["gameId"].nunique()
    counts = drives["driveResult"].value_counts()

    def rate(names) -> float:
        return float(sum(int(counts.get(n, 0)) for n in names) / team_games)

    return {
        "defensive_td_lambda": rate(DEFENSIVE_TD),
        "special_teams_td_lambda": rate(SPECIAL_TEAMS_TD),
        "safety_lambda": rate(SAFETY),
        "_team_games": int(team_games),
    }


# Scoring plays the drive multinomial does not produce, named as CFBD's /plays `playType`.
DEFENSIVE_TD_PLAYS = ["Interception Return Touchdown", "Fumble Return Touchdown"]
SPECIAL_TEAMS_TD_PLAYS = [
    "Kickoff Return Touchdown", "Punt Return Touchdown", "Blocked Punt Touchdown",
    "Blocked Field Goal Touchdown", "Missed Field Goal Return Touchdown",
]


def measure_lambdas(plays: pd.DataFrame) -> dict:
    """Poisson rates for scores the drive multinomial does not produce.

    Each is a per-team-per-game rate, which is how the simulator draws them: one Poisson
    per team per game. A game contributes two team-games.

    MEASURED FROM /plays, NOT /drives. Counting `driveResult` values is the obvious
    approach and it structurally cannot see kickoff-return touchdowns: a kickoff is not a
    scrimmage drive, so CFBD emits no drive row for one and `src/drives.py` drops the
    handful it does emit. That silently omits 0.0201 of 0.0377 -- more than half of all
    special-teams touchdowns. The /plays payload enumerates every scoring play by type, so
    it is the complete source and the one used here.

    On the identical 7,705-game sample the two disagree by at most 0.008 per team-game
    (drives: 0.1095 / 0.0326 / 0.0256), i.e. under 0.06 points per team per game on
    everything except special teams, where the drives figure is simply short.
    """
    team_games = 2.0 * plays["game_id"].nunique()
    counts = plays["playType"].value_counts()

    def rate(names) -> float:
        return float(sum(int(counts.get(n, 0)) for n in names) / team_games)

    return {
        "defensive_td_lambda": rate(DEFENSIVE_TD_PLAYS),
        "special_teams_td_lambda": rate(SPECIAL_TEAMS_TD_PLAYS),
        "safety_lambda": rate(["Safety"]),
        "_team_games": int(team_games),
        "_kickoff_return_td_share": rate(["Kickoff Return Touchdown"]),
    }


def load_plays() -> pd.DataFrame:
    """The cached play table, restricted to the seasons the constants are fit on."""
    path = CACHE_DIR / "plays_2019_2025.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Set NCAA_MODEL_CACHE_DIR to the permanent cache."
        )
    plays = pd.read_parquet(path)
    return plays[~plays["season"].isin(EXCLUDE_SEASONS)].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# How many of a team's drives are played from the scoreboard
# --------------------------------------------------------------------------------------

_PERIOD_SECONDS = 900.0


def measure_endgame_second_drive_prob(
    drives: pd.DataFrame, late_seconds: float = 300.0
) -> dict:
    """The probability that a team's *second-to-last* drive is also an endgame drive.

    The shared simulator always plays a team's final drive off the endgame table and its
    second-to-last drive with this probability, which is what reproduces the measured mean
    number of endgame drives per team.

    CONDITION ON HAVING AT LEAST ONE. The simulator gives every team a final drive by
    construction, so the empirical quantity it must match is the mean number of endgame
    drives among team-games that had one, not the unconditional mean. The difference is
    not cosmetic: on the NFL drive table the unconditional mean is 1.165 while the
    conditional mean is 1.3408 -- and 1.34, with its 68/30/2 split over one, two and three
    drives, is exactly the figure sim_core's `_ENDGAME_SECOND_DRIVE_PROB` was set from.
    Measuring this unconditionally would have set NCAA's rate less than half what it
    should be.
    """
    within = (
        drives["startTime.minutes"].astype(float).fillna(0) * 60.0
        + drives["startTime.seconds"].astype(float).fillna(0)
    )
    periods_left = np.clip(4 - drives["startPeriod"].astype(float), 0, None)
    gsr = periods_left * _PERIOD_SECONDS + within

    late = (
        (drives["startPeriod"].astype(float) == 4)
        & (gsr <= late_seconds)
        & drives["score_diff"].notna()
    )
    per_team = (
        pd.DataFrame(
            {"gameId": drives["gameId"], "offense": drives["offense"], "late": late}
        )
        .groupby(["gameId", "offense"])["late"].sum()
    )
    having_one = per_team[per_team >= 1]
    mean_late = float(having_one.mean())
    return {
        "endgame_second_drive_prob": round(max(0.0, mean_late - 1.0), 4),
        "_mean_endgame_drives_per_team": mean_late,
        "_unconditional_mean": float(per_team.mean()),
        "_share_no_endgame_drive": float((per_team == 0).mean()),
        "_conditional_dist": (
            having_one.value_counts(normalize=True).sort_index().round(4).to_dict()
        ),
        "_n_team_games": int(len(having_one)),
    }


def main() -> None:
    drives = load_raw_drives()
    # Regulation only. College overtime forces a two-point try from the third period on,
    # so OT drives convert on 8 points 2.6x as often and fail outright 5x as often as
    # regulation ones. They are 0.4% of touchdown drives -- small, but they belong to a
    # different rule set than the constants being fit here.
    td = drives[
        drives["driveResult"].isin(OFFENSIVE_TD)
        & drives["offense_points"].isin([6, 7, 8])
        & (drives["startPeriod"].astype(float) <= 4)
    ]

    conv = fit_conversions(td)
    lam = measure_lambdas(load_plays())
    lam_drv = measure_lambdas_from_drives(drives)
    end = measure_endgame_second_drive_prob(drives)

    seasons = sorted(drives["season"].unique())
    print(f"seasons {seasons[0]}-{seasons[-1]} (2020 excluded), "
          f"{len(drives):,} drives, {lam['_team_games']:,} team-games\n")

    print("conversions (regulation only)")
    print(f"  touchdown drives fit      {conv['_n_td_drives']:,} "
          f"in {conv['_n_buckets']} margin buckets")
    print(f"  per-bucket fit            G={conv['_gof_G']:.1f} "
          f"dof={conv['_gof_dof']} p={conv['_gof_p']:.4f}")
    fp, op = conv["_pooled_fitted"], conv["_pooled_observed"]
    print(f"  pooled P(6/7/8) observed  {op[0]:.4f} / {op[1]:.4f} / {op[2]:.4f}")
    print(f"  pooled P(6/7/8) fitted    {fp[0]:.4f} / {fp[1]:.4f} / {fp[2]:.4f}")
    print(f"  xp_make_prob              {conv['xp_make_prob']:.4f}")
    print(f"  two_point_attempt_rate    {conv['two_point_attempt_rate']:.4f}")
    print(f"  two_point_success_prob    {conv['two_point_success_prob']:.4f}\n")

    print("  attempt rate by post-TD margin")
    for margin, rate in conv["_attempt_by_margin"].items():
        print(f"    {margin:>5s}  {rate:.4f}")
    print()

    print("non-offensive scores (per team per game)")
    print(f"  {'':25s} {'/plays':>8s} {'/drives':>9s}")
    for key in ("defensive_td_lambda", "special_teams_td_lambda", "safety_lambda"):
        print(f"  {key:25s} {lam[key]:8.4f} {lam_drv[key]:9.4f}")
    print(f"  ...of which kickoff returns, invisible to /drives: "
          f"{lam['_kickoff_return_td_share']:.4f}\n")

    print("endgame (last 5:00 of Q4)")
    print(f"  team-games with none      {end['_share_no_endgame_drive']:.4f} "
          f"(excluded; the simulator always gives a final drive)")
    print(f"  mean among those with one {end['_mean_endgame_drives_per_team']:.4f} "
          f"over {end['_n_team_games']:,} team-games")
    print(f"  distribution              {end['_conditional_dist']}")
    print(f"  endgame_second_drive_prob {end['endgame_second_drive_prob']:.4f}")


if __name__ == "__main__":
    main()
