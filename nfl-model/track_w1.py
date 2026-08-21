#!/usr/bin/env python
"""W1 forward tracking -- the pre-registered NFL wind rule (PREREG.md W1).

    python track_w1.py record            # log this week's qualifying games
    python track_w1.py record --week 5
    python track_w1.py grade             # settle logged bets that have finished
    python track_w1.py report            # running record + forecast-error measurement

THE RULE, quoted from PREREG.md W1: an OUTDOOR game whose FORECAST wind at kickoff, taken
at the moment of recording, is at least 10 mph, is bet UNDER the total at flat stakes.
Wind alone triggers it. The model is not consulted -- deliberately, because this model's
disagreements with the NFL market are measurably worse than useless (GATES.md), while wind
is the one effect the market appears not to price.

WHY THIS EXISTS. W1 was declared before the 2026 season on the reasoning that forward record
is the only evidence source this build has not exhausted. A rule nobody logs produces no
record, and in December the temptation to reconstruct one favourably would be enormous. This
writes the bet down when it qualifies, at the number and the forecast available then, and
never edits it afterwards.

THE LOG IS APPEND-ONLY, AND THAT IS THE POINT. `record` refuses to rewrite a game already
present. The bet-time line is the whole experiment -- a line "remembered" later is the same
class of error as the opener anchoring that produced this repo's one false positive.

THE FORECAST IS THE EXPERIMENT, NOT THE OBSERVATION. `weather.forecast_at_bet_time` fetches
live and never touches the weather cache, which is keyed without an issue time and would
silently serve a days-old reading. Every row stores `forecast_issued_at` and
`hours_before_kickoff`, so the horizon is measured rather than assumed. At grading the
OBSERVED wind is stored beside the forecast -- across a season that is a direct measurement
of real forecast error, which is the untested assumption the whole edge rests on. See
PREREG.md W1, "The byproduct that may matter more than the record".

NOT AN AUTHORISATION TO STAKE MONEY. `bets_allowed()` is False and six gates fail. This
records what the rule would have done.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import ingest
from src.config import CACHE_DIR, load_config
from src.weather import forecast_at_bet_time

LOG_PATH = CACHE_DIR.parent / "w1_log.csv"

# The rule, from PREREG.md W1. These are frozen -- changing one voids the test and starts a
# new registration under a new id, which is why they are constants and not CLI flags.
# tests/test_track_w1.py asserts they still match the registration.
MIN_WIND_MPH = 10.0
MAX_WIND_MPH = 40.0          # above this the forecast is rejected, never treated as calm
SIDE = "UNDER"               # one-sided by construction; wind suppresses scoring
UNIVERSE = "outdoor"

LOG_COLUMNS = [
    "game_id", "season", "week", "away_team", "home_team", "kickoff",
    "forecast_wind_mph", "forecast_issued_at", "hours_before_kickoff",
    "market_total_at_bet", "side", "recorded_at",
    "actual_total", "market_total_close", "observed_wind_mph", "result", "graded_at",
]


def _slate(cfg) -> pd.DataFrame:
    """Every game of the current season, played or not.

    Deliberately NOT `walk_forward`, which keeps only graded games and so cannot answer
    "what do we think about Sunday". W1 consults no projection at all, so schedules plus a
    live forecast is the entire input.
    """
    return ingest.load_schedules([int(cfg.seasons.current)])


def _qualifying(slate: pd.DataFrame, season: int, week: int, *,
                verbose: bool = True) -> pd.DataFrame:
    """Games meeting W1's trigger. Every filter here is quoted from the registration."""
    sub = slate[(slate["season"] == season) & (slate["week"] == week)].copy()
    # Ungraded only. A finished game has an observed wind and cannot be forecast.
    sub = sub[sub["result"].isna()]
    sub = sub.dropna(subset=["total_line"])
    if sub.empty:
        return sub.assign(forecast_wind_mph=[], forecast_issued_at=[],
                          hours_before_kickoff=[], side=[])

    rows = []
    for _, game in sub.iterrows():
        fc = forecast_at_bet_time(game)
        if fc.indoor or fc.source != "forecast" or fc.wind_mph is None:
            if verbose and not fc.indoor:
                print(f"  skip {game['away_team']} at {game['home_team']}: "
                      f"no forecast ({fc.source})")
            continue
        rows.append({
            **game.to_dict(),
            "forecast_wind_mph": fc.wind_mph,
            "forecast_issued_at": fc.issued_at,
            "hours_before_kickoff": (
                round(fc.hours_before_kickoff, 1)
                if fc.hours_before_kickoff is not None else np.nan),
        })
    if not rows:
        return sub.iloc[0:0].assign(forecast_wind_mph=[], forecast_issued_at=[],
                                    hours_before_kickoff=[], side=[])

    out = pd.DataFrame(rows)
    out = out[(out["forecast_wind_mph"] >= MIN_WIND_MPH)
              & (out["forecast_wind_mph"] <= MAX_WIND_MPH)]
    out["side"] = SIDE
    return out


def _read_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def _write_log(frame: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)


def cmd_record(cfg, args) -> int:
    slate = _slate(cfg)
    season = args.season or int(cfg.seasons.current)
    if args.week:
        week = int(args.week)
    else:
        upcoming = slate[(slate["season"] == season) & slate["result"].isna()]
        if upcoming.empty:
            print(f"no ungraded {season} games found -- nothing to record")
            return 0
        week = int(upcoming["week"].min())

    picks = _qualifying(slate, season, week)
    log = _read_log()
    already = set(log["game_id"].astype(str)) if len(log) else set()
    fresh = picks[~picks["game_id"].astype(str).isin(already)] if len(picks) else picks

    print(f"W1 {season} week {week}: {len(picks)} qualifying "
          f"(forecast wind >= {MIN_WIND_MPH:.0f} mph), {len(fresh)} new")
    if len(picks) and not len(fresh):
        print("  (all already logged -- the log is append-only and will not be rewritten)")
    if not len(fresh):
        return 0

    rows = pd.DataFrame({
        "game_id": fresh["game_id"], "season": season, "week": week,
        "away_team": fresh["away_team"], "home_team": fresh["home_team"],
        "kickoff": fresh["kickoff"],
        "forecast_wind_mph": fresh["forecast_wind_mph"].round(1),
        "forecast_issued_at": fresh["forecast_issued_at"],
        "hours_before_kickoff": fresh["hours_before_kickoff"],
        "market_total_at_bet": fresh["total_line"],
        "side": fresh["side"],
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actual_total": np.nan, "market_total_close": np.nan,
        "observed_wind_mph": np.nan, "result": "", "graded_at": "",
    })
    for r in rows.itertuples(index=False):
        print(f"  {r.away_team} at {r.home_team}: wind {r.forecast_wind_mph:.0f} mph "
              f"({r.hours_before_kickoff:.0f}h out)  ->  UNDER {r.market_total_at_bet}")
    _write_log(pd.concat([log, rows], ignore_index=True))
    print(f"appended {len(rows)} to {LOG_PATH}")
    return 0


def cmd_grade(cfg, args) -> int:
    log = _read_log()
    if log.empty:
        print("no bets logged yet")
        return 0
    open_bets = log[log["result"].fillna("").eq("")]
    if open_bets.empty:
        print("every logged bet is already graded")
        return 0

    slate = _slate(cfg)
    done = slate[slate["result"].notna()].copy()
    done.index = done["game_id"].astype(str)

    graded = 0
    for idx, row in open_bets.iterrows():
        hit = done[done.index == str(row["game_id"])]
        if hit.empty:
            continue
        g = hit.iloc[0]
        if pd.isna(g.get("total")):
            continue
        actual = float(g["total"])
        close = float(g["total_line"]) if pd.notna(g.get("total_line")) else np.nan
        observed_wind = float(g["wind"]) if pd.notna(g.get("wind")) else np.nan
        # Settled at the number the bet was RECORDED at, which is the money question.
        line = float(row["market_total_at_bet"])
        if actual == line:
            result = "PUSH"
        else:
            # W1 is UNDER-only; a win is the game staying below the recorded total.
            result = "WIN" if actual < line else "LOSS"
        log.loc[idx, ["actual_total", "market_total_close", "observed_wind_mph",
                      "result", "graded_at"]] = [
            actual, close, observed_wind, result,
            datetime.now(timezone.utc).isoformat(timespec="seconds")]
        graded += 1

    _write_log(log)
    print(f"graded {graded} bet(s)")
    return cmd_report(cfg, args)


def _forecast_error_report(log: pd.DataFrame) -> None:
    """Real forecast error, which PREREG W1 calls the primary output of season one.

    The backtest's whole wind result assumed error was unbiased Gaussian noise. This is the
    first measurement of what it actually is, at the horizons bets were really placed at.
    """
    paired = log.dropna(subset=["forecast_wind_mph", "observed_wind_mph"])
    print("-" * 68)
    print("  FORECAST ERROR (the untested assumption behind the edge)")
    if len(paired) < 2:
        print(f"    {len(paired)} paired reading(s) -- need more before this means anything")
        return
    err = paired["observed_wind_mph"].astype(float) - paired["forecast_wind_mph"].astype(float)
    bias, rmse = float(err.mean()), float(np.sqrt((err ** 2).mean()))
    print(f"    n={len(paired)}   bias {bias:+.2f} mph   RMSE {rmse:.2f} mph")
    print(f"    (backtest assumed UNBIASED noise; 2-4 mph RMSE was the realistic band)")
    if bias < -1.0:
        print("    NOTE: forecasts are running HIGH -- the rule is firing on games that")
        print("          turned out calmer than predicted, which inflates the bet count")
        print("          and dilutes the edge. This is the failure mode PREREG W1 named.")
    horizons = paired["hours_before_kickoff"].dropna().astype(float)
    if len(horizons):
        print(f"    horizon: median {horizons.median():.0f}h, "
              f"range {horizons.min():.0f}-{horizons.max():.0f}h")


def cmd_report(cfg, args) -> int:
    log = _read_log()
    print("=" * 68)
    print(f"W1 FORWARD RECORD -- pre-registered 2026-08-21, "
          f"UNDER on forecast wind >= {MIN_WIND_MPH:.0f} mph")
    print("=" * 68)
    if not len(log):
        print("  no bets logged yet")
        return 0
    settled = log[log["result"].isin(["WIN", "LOSS"])]
    pushes = int((log["result"] == "PUSH").sum())
    pending = int(log["result"].fillna("").eq("").sum())
    n = len(settled)
    print(f"  logged {len(log)}   settled {n}   pushes {pushes}   pending {pending}")
    if not n:
        _forecast_error_report(log)
        return 0
    wins = int((settled["result"] == "WIN").sum())
    rate = wins / n
    units = wins * 1.0 - (n - wins) * 1.1
    se = np.sqrt(rate * (1 - rate) / n) if n > 1 else float("nan")
    print(f"  record {wins}-{n - wins}   win rate {100 * rate:.1f}%   "
          f"units @ -110 {units:+.1f}")
    if n > 1:
        print(f"  95% CI [{100 * (rate - 1.96 * se):.1f}, {100 * (rate + 1.96 * se):.1f}]"
              f"   breakeven 52.4%")
    # The pre-committed kill condition, evaluated rather than remembered.
    if n >= 60:
        verdict = "KILL -- below 50% at the checkpoint" if rate < 0.50 else (
            "continue (NOT a claim that it works -- see PREREG W1)")
        print(f"  checkpoint reached (n>=60): {verdict}")
    else:
        print(f"  checkpoint at 60 settled bets ({60 - n} to go); "
              "kill condition is below 50%")
    _forecast_error_report(log)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="log this week's qualifying games")
    rec.add_argument("--season", type=int)
    rec.add_argument("--week", type=int)
    sub.add_parser("grade", help="settle logged bets that have finished")
    sub.add_parser("report", help="running record + forecast-error measurement")
    args = p.parse_args()

    cfg = load_config()
    return {"record": cmd_record, "grade": cmd_grade, "report": cmd_report}[args.cmd](
        cfg, args)


if __name__ == "__main__":
    sys.exit(main())
