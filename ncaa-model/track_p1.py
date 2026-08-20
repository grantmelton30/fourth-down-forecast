#!/usr/bin/env python
"""P1 forward tracking -- the pre-registered totals rule (PREREG.md P1).

    python track_p1.py record            # log this week's qualifying games
    python track_p1.py record --week 5
    python track_p1.py grade             # settle logged bets that have finished
    python track_p1.py report            # running out-of-sample record

WHY THIS EXISTS. P1 was declared before the 2026 season on the explicit reasoning that
forward record is the only evidence source this build has not exhausted. A rule nobody
logs produces no record, and in December the temptation to reconstruct one favourably
would be enormous. This writes the bet down at the time it qualifies, at the number
available then, and never edits it afterwards.

THE LOG IS APPEND-ONLY, AND THAT IS THE POINT. `record` refuses to rewrite a game already
present. The bet-time line is the whole experiment -- a line "remembered" later is the
same class of error as the opener anchoring that produced this repo's one false positive.

WHAT IS BEING PROJECTED. `model_total` here is the SAME quantity P1 was measured on: the
walk-forward OLS totals projection with the weather adjustment applied, produced by
`project_walkforward`. Deliberately NOT the raw simulator mean, which is a different and
measurably worse estimate (16.412 vs 16.396 RMSE, DECISIONS D17) and was never what the
rule was declared against.

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
from src import qb as QB
from src.backtest import build_features, project_walkforward
from src.cfbd_client import BudgetedCFBD
from src.config import CACHE_DIR, MANUAL_DIR, load_config
from src.features import (load_free_preseason, matchup_feature_table,
                          normalize_preseason_sources)
from src.ratings import build_walkforward
from src.venues import attach_venues, load_venues
from src.weather import bulk_game_weather

LOG_PATH = CACHE_DIR.parent / "p1_log.csv"

# The rule, from PREREG.md P1. These are frozen -- changing one voids the test and starts a
# new registration under a new id, which is why they are constants and not CLI flags.
MIN_EDGE = 0.5
MAX_EDGE = 6.0
UNIVERSE = "restricted"

LOG_COLUMNS = [
    "game_id", "season", "week", "away_team", "home_team", "kickoff",
    "model_total", "market_total_at_bet", "edge", "side", "recorded_at",
    "actual_total", "market_total_close", "result", "graded_at",
]


def _projected_totals(cfg, client) -> pd.DataFrame:
    """Every game with a projection, completed or not, mirroring run_backtest's features.

    `walk_forward` deliberately keeps only graded games, so it cannot answer "what do we
    think about Saturday". This runs the same projection over the unfiltered market so the
    current week is present, and takes `model_total` from it.
    """
    games = ingest.load_games(client, cfg.all_seasons)
    drives = ingest.load_drives(client, cfg.all_seasons)
    plays = ingest.load_plays(client, games, cfg.all_seasons)
    lines = ingest.load_lines(client, cfg, cfg.all_seasons)
    game_off = ingest.build_game_offense(plays, drives, games, cfg)
    market = ingest.build_market(lines, games, cfg)

    wf = build_walkforward(game_off, games, cfg)
    preseason = normalize_preseason_sources(**load_free_preseason(client, cfg.all_seasons))
    challenger = matchup_feature_table(plays, games, preseason)

    games_v = attach_venues(games, load_venues(client))
    weather = bulk_game_weather(games_v, allow_network=False)
    challenger = challenger.merge(
        weather[["game_id", "indoor", "wind_mph"]], on="game_id", how="left")

    feats = build_features(market, wf, cfg, challenger)
    return project_walkforward(feats, cfg)


def _qualifying(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Games meeting P1's trigger. Every filter here is quoted from the registration."""
    sub = frame[(frame["season"] == season) & (frame["week"] == week)].copy()
    if UNIVERSE == "restricted" and "restricted" in sub.columns:
        sub = sub[sub["restricted"].fillna(False).astype(bool)]
    sub = sub.dropna(subset=["model_total", "total_open"])
    # The number available now. `total_open` is what this repo carries for an ungraded
    # game; the close does not exist yet and must never be used to select a bet.
    sub["market_total_at_bet"] = sub["total_open"]
    sub["edge"] = sub["model_total"] - sub["market_total_at_bet"]
    sub = sub[(sub["edge"].abs() >= MIN_EDGE) & (sub["edge"].abs() < MAX_EDGE)]
    sub["side"] = np.where(sub["edge"] > 0, "OVER", "UNDER")
    return sub


def _read_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def _write_log(frame: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)


def cmd_record(cfg, client, args) -> int:
    frame = _projected_totals(cfg, client)
    season = args.season or int(cfg.seasons.current)
    if args.week:
        week = int(args.week)
    else:
        upcoming = frame[(frame["season"] == season) & frame["actual_total"].isna()]
        if upcoming.empty:
            print(f"no ungraded {season} games found -- nothing to record")
            return 0
        week = int(upcoming["week"].min())

    picks = _qualifying(frame, season, week)
    log = _read_log()
    already = set(log["game_id"].astype(str)) if len(log) else set()
    fresh = picks[~picks["game_id"].astype(str).isin(already)]

    print(f"P1 {season} week {week}: {len(picks)} qualifying, {len(fresh)} new")
    if len(picks) and not len(fresh):
        print("  (all already logged -- the log is append-only and will not be rewritten)")
    if not len(fresh):
        return 0

    rows = pd.DataFrame({
        "game_id": fresh["game_id"], "season": season, "week": week,
        "away_team": fresh["away_team"], "home_team": fresh["home_team"],
        "kickoff": fresh.get("kickoff"),
        "model_total": fresh["model_total"].round(2),
        "market_total_at_bet": fresh["market_total_at_bet"],
        "edge": fresh["edge"].round(2), "side": fresh["side"],
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actual_total": np.nan, "market_total_close": np.nan,
        "result": "", "graded_at": "",
    })
    for r in rows.itertuples(index=False):
        print(f"  {r.away_team} at {r.home_team}: model {r.model_total:.1f} vs "
              f"market {r.market_total_at_bet:.1f}  ->  {r.side} ({r.edge:+.1f})")
    _write_log(pd.concat([log, rows], ignore_index=True))
    print(f"appended {len(rows)} to {LOG_PATH}")
    return 0


def cmd_grade(cfg, client, args) -> int:
    log = _read_log()
    if log.empty:
        print("no bets logged yet")
        return 0
    open_bets = log[log["result"].fillna("").eq("")]
    if open_bets.empty:
        print("every logged bet is already graded")
        return 0

    frame = _projected_totals(cfg, client)
    done = frame.dropna(subset=["actual_total"]).set_index(
        frame.dropna(subset=["actual_total"])["game_id"].astype(str))

    graded = 0
    for idx, row in open_bets.iterrows():
        hit = done[done.index == str(row["game_id"])]
        if hit.empty:
            continue
        g = hit.iloc[0]
        actual = float(g["actual_total"])
        close = float(g["total_close"]) if pd.notna(g.get("total_close")) else np.nan
        # Settled at the number the bet was RECORDED at, which is the money question.
        line = float(row["market_total_at_bet"])
        if actual == line:
            result = "PUSH"
        elif (actual > line) == (row["side"] == "OVER"):
            result = "WIN"
        else:
            result = "LOSS"
        log.loc[idx, ["actual_total", "market_total_close", "result", "graded_at"]] = [
            actual, close, result,
            datetime.now(timezone.utc).isoformat(timespec="seconds")]
        graded += 1

    _write_log(log)
    print(f"graded {graded} bet(s)")
    return cmd_report(cfg, client, args)


def cmd_report(cfg, client, args) -> int:
    log = _read_log()
    settled = log[log["result"].isin(["WIN", "LOSS"])] if len(log) else log
    print("=" * 68)
    print("P1 FORWARD RECORD -- pre-registered 2026-08-19, totals, edge 0.5-6.0")
    print("=" * 68)
    if not len(log):
        print("  no bets logged yet")
        return 0
    pushes = int((log["result"] == "PUSH").sum())
    pending = int(log["result"].fillna("").eq("").sum())
    n = len(settled)
    print(f"  logged {len(log)}   settled {n}   pushes {pushes}   pending {pending}")
    if not n:
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
    if n >= 200:
        verdict = "KILL -- below 50% at the checkpoint" if rate < 0.50 else (
            "continue (NOT a claim that it works -- see PREREG P1)")
        print(f"  checkpoint reached (n>=200): {verdict}")
    else:
        print(f"  checkpoint at 200 settled bets ({200 - n} to go); "
              "kill condition is below 50%")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="log this week's qualifying games")
    rec.add_argument("--season", type=int)
    rec.add_argument("--week", type=int)
    sub.add_parser("grade", help="settle logged bets that have finished")
    sub.add_parser("report", help="running out-of-sample record")
    args = p.parse_args()

    cfg = load_config()
    client = BudgetedCFBD(cfg)
    return {"record": cmd_record, "grade": cmd_grade, "report": cmd_report}[args.cmd](
        cfg, client, args)


if __name__ == "__main__":
    sys.exit(main())
