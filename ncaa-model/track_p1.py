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
import hashlib
import json
import sys
from pathlib import Path
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
REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "P1-current-v2"

# The rule, from PREREG.md P1. These are frozen -- changing one voids the test and starts a
# new registration under a new id, which is why they are constants and not CLI flags.
MIN_EDGE = 0.5
MAX_EDGE = 6.0
UNIVERSE = "restricted"

LOG_COLUMNS = [
    "game_id", "season", "week", "away_team", "home_team", "kickoff",
    "model_total", "market_total_open", "market_total_at_bet", "line_basis", "edge",
    "side", "recorded_at",
    "actual_total", "market_total_close", "result", "graded_at",
    "protocol_version", "book", "quote_observed_at", "quote_source", "quote_id",
    "price", "price_status", "model_version", "quote_payload",
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


def _qualifying(frame: pd.DataFrame, season: int, week: int, *, now=None) -> pd.DataFrame:
    """Games meeting P1's trigger. Every filter here is quoted from the registration."""
    sub = frame[(frame["season"] == season) & (frame["week"] == week)].copy()
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("recording time must be timezone-aware")
    required = {"kickoff", "restricted", "actual_total", "total_close"}
    if required - set(sub.columns):
        return sub.iloc[0:0]
    kickoff = pd.to_datetime(sub["kickoff"], utc=True, errors="coerce", format="mixed")
    sub = sub[(kickoff > now) & sub["actual_total"].isna()
              & sub["restricted"].eq(True)].copy()
    if "completed" in sub:
        sub = sub[~sub["completed"].fillna(False).astype(bool)]
    sub = sub.dropna(subset=["model_total"])
    # THE NUMBER AVAILABLE NOW IS `total_close`, NOT `total_open`.
    #
    # Fixed 2026-08-21. This previously used `total_open`, which is CFBD's `overUnderOpen`
    # -- the HISTORICAL OPENER, set whenever the book first hung the game, often weeks
    # earlier. `total_close` is `overUnder`, the currently displayed number, and for an
    # unplayed game that is exactly the quote you could bet right now; it only becomes "the
    # close" once the game kicks off. On the live 2026 week 1 slate the two differ on 31 of
    # 51 priced games, mean 0.70 points and up to 4.
    #
    # This is a BUG FIX, not a rule change. PREREG P1 says "graded at the number actually
    # available when the bet is recorded, which is the real money question", and the code
    # was not doing that. It also made the forward record test a materially different and
    # historically LOSING strategy: selected and priced at the opener the same rule reads
    # 52.07% and -7.1 units across 2021-2025, against 54.13% and +39.1 at the current line.
    #
    # The 26 bets logged before this fix keep their recorded numbers -- the log is
    # append-only -- and carry `line_basis="opener"` so they can never be silently pooled
    # with what follows.
    sub["market_total_at_bet"] = pd.to_numeric(sub["total_close"], errors="coerce")
    # The OPENER is recorded alongside, and is never what the bet is graded at. Storing it
    # makes the full arc visible on the row itself -- open, the number actually taken, and
    # the close filled in at grading -- so "did we bet before or after the market moved, and
    # did that help" is answerable from the log without joining an external archive.
    # Purely additive: no trigger, side, universe or stake changes, so PREREG P1 is untouched.
    sub["market_total_open"] = sub["total_open"]
    sub["line_basis"] = "current"
    sub = sub.dropna(subset=["market_total_at_bet"])
    sub["edge"] = sub["model_total"] - sub["market_total_at_bet"]
    sub = sub[(sub["edge"].abs() >= MIN_EDGE) & (sub["edge"].abs() < MAX_EDGE)]
    sub["side"] = np.where(sub["edge"] > 0, "OVER", "UNDER")
    return sub


def _fresh_quotes(frame, cfg, client, season):
    """Refresh the exact provider's reference immediately before recording.

    CFBD does not supply side prices here. Preserve that absence instead of
    inventing -110 or describing the reference as an executable quote.
    """
    payload = client.call("lines", f"lines_{season}", ttl_hours=0.0,
                          year=season, seasonType="regular")
    observed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    quotes = {}
    for game in payload:
        chosen = ingest._pick_provider(game.get("lines") or [], list(cfg.market.provider_priority))
        raw = json.dumps({"game_id": game["id"], "quote": chosen, "observed_at": observed},
                         sort_keys=True, separators=(",", ":"))
        quotes[str(game["id"])] = {
            "total_close": chosen.get("overUnder"), "total_open": chosen.get("overUnderOpen"),
            "book": chosen.get("provider"), "quote_observed_at": observed,
            "quote_source": "cfbd/lines", "quote_id": hashlib.sha256(raw.encode()).hexdigest(),
            "quote_payload": raw,
            "price": np.nan, "price_status": "unavailable_reference_only",
        }
    out = frame.copy()
    for col in ("total_close", "total_open", "book", "quote_observed_at", "quote_source",
                "quote_id", "price", "price_status", "quote_payload"):
        out[col] = out["game_id"].astype(str).map(lambda gid: quotes.get(gid, {}).get(col))
    return out


def _read_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        # Rows written before the 2026-08-21 line-basis fix were priced at the OPENER.
        # Label them rather than repair them: the numbers recorded are what they are, and
        # rewriting a logged bet is the one thing an append-only ledger may never do.
        if "line_basis" not in log.columns:
            log["line_basis"] = "opener"
        else:
            log["line_basis"] = log["line_basis"].fillna("opener")
        return log
    return pd.DataFrame(columns=LOG_COLUMNS)


def _write_log(frame: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys([*LOG_COLUMNS, *frame.columns]))
    temporary = LOG_PATH.with_suffix(".csv.tmp")
    frame.reindex(columns=columns).to_csv(temporary, index=False)
    temporary.replace(LOG_PATH)


def cmd_record(cfg, client, args) -> int:
    frame = _projected_totals(cfg, client)
    season = args.season or int(cfg.seasons.current)
    if args.week:
        week = int(args.week)
    else:
        kickoff = pd.to_datetime(frame["kickoff"], utc=True, errors="coerce", format="mixed")
        upcoming = frame[(frame["season"] == season) & frame["actual_total"].isna()
                         & (kickoff > pd.Timestamp.now(tz="UTC"))]
        if upcoming.empty:
            print(f"no ungraded {season} games found -- nothing to record")
            return 0
        week = int(upcoming["week"].min())

    frame = _fresh_quotes(frame, cfg, client, season)
    recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    picks = _qualifying(frame, season, week, now=recorded_at)
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
        "market_total_open": fresh["market_total_open"],
        "market_total_at_bet": fresh["market_total_at_bet"],
        "edge": fresh["edge"].round(2), "side": fresh["side"],
        "line_basis": fresh["line_basis"],
        "recorded_at": recorded_at,
        "actual_total": np.nan, "market_total_close": np.nan,
        "result": "", "graded_at": "",
    })
    from model_identity import build_model_version
    rows["protocol_version"] = PROTOCOL_VERSION
    rows["model_version"] = build_model_version(
        "ncaa", Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "config" / "ncaa.yaml", frame)
    for col in ("book", "quote_observed_at", "quote_source", "quote_id", "price", "price_status", "quote_payload"):
        rows[col] = fresh[col]
    from execution_quotes import matching_quote
    for idx, row in rows.iterrows():
        quote = matching_quote(REPO_ROOT / "data" / "execution_quotes.jsonl",
            league="ncaa", game_id=row["game_id"], kickoff=row["kickoff"],
            book=row["book"], line=row["market_total_at_bet"], now=recorded_at)
        if quote is not None:
            rows.loc[idx,"price"] = quote["over_price" if row["side"] == "OVER" else "under_price"]
            rows.loc[idx,"price_status"] = "quoted_execution_unverified"
            rows.loc[idx,"quote_id"] = quote["quote_id"]
            rows.loc[idx,"quote_payload"] = json.dumps(quote,sort_keys=True)
            rows.loc[idx,"quote_observed_at"] = quote["observed_at"]
            rows.loc[idx,"quote_source"] = quote["source"]
    for r in rows.itertuples(index=False):
        print(f"  {r.away_team} at {r.home_team}: model {r.model_total:.1f} vs "
              f"market {r.market_total_at_bet:.1f}  ->  {r.side} ({r.edge:+.1f})")
    _write_log(rows if log.empty else pd.concat([log, rows], ignore_index=True))
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
    total = len(log)
    protocol = log.get("protocol_version", pd.Series("", index=log.index))
    log = log[protocol.eq(PROTOCOL_VERSION)].copy()
    print(f"Active protocol: {PROTOCOL_VERSION}; {total-len(log)} historical rows excluded")
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
    z = 1.96
    center = (rate + z*z/(2*n)) / (1 + z*z/n)
    radius = z*np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
    print(f"  record {wins}-{n - wins}   win rate {100 * rate:.1f}%   "
          f"assumed units @ -110 {units:+.1f}")
    if n > 1:
        print(f"  95% CI [{100 * (center - radius):.1f}, {100 * (center + radius):.1f}]"
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
