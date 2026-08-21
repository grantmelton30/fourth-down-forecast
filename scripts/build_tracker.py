#!/usr/bin/env python3
"""Publish the forward record of both pre-registered rules to web/api/v1/tracker.json.

    python scripts/build_tracker.py

READS THE LOGS, NEVER RECOMPUTES. `ncaa-model/data/p1_log.csv` and `nfl-model/data/w1_log.csv`
are append-only records written at bet time by `track_p1.py` / `track_w1.py`. This file
reports what those logs say and nothing else. Re-deriving "what would have qualified" from
today's model would silently replace the bet-time number with a better one and turn a
forward record into a backtest -- the exact failure both registrations exist to prevent.

A MISSING LOG IS AN EMPTY RECORD, NOT AN ERROR. Neither rule has graded bets before its
season starts, and the site must render that honestly rather than fail the publish.

NOT A PROFIT STATEMENT. `bets_allowed()` is False for both leagues; these are logs of what
declared rules would have done, at one flat unit, before any vig beyond the -110 assumed
in the unit maths.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "api" / "v1" / "tracker.json"
BREAKEVEN = 52.38   # -110 both sides

RULES = {
    "P1": {
        "league": "ncaa",
        "log": ROOT / "ncaa-model" / "data" / "p1_log.csv",
        "headline": "NCAA totals, restricted universe, model-market gap 0.5-6.0",
        "prereg": "ncaa-model/PREREG.md P1, declared 2026-08-19",
        "checkpoint_n": 200,
        "kill_below": 50.0,
    },
    "W1": {
        "league": "nfl",
        "log": ROOT / "nfl-model" / "data" / "w1_log.csv",
        "headline": "NFL totals, outdoor, forecast wind 10+ mph, UNDER only",
        "prereg": "nfl-model/PREREG.md W1, declared 2026-08-21",
        "checkpoint_n": 60,
        "kill_below": 50.0,
    },
}

COMMON = ["game_id", "season", "week", "away_team", "home_team", "kickoff",
          "market_total_at_bet", "side", "recorded_at", "actual_total",
          "market_total_close", "result", "graded_at"]
EXTRA = {"P1": ["model_total", "edge"],
         "W1": ["forecast_wind_mph", "observed_wind_mph", "hours_before_kickoff"]}


def _clean(value):
    """JSON has no NaN. Anything not finite becomes null rather than a literal NaN token."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarise(rule: str, spec: dict) -> tuple[dict, list]:
    if not spec["log"].exists():
        return ({"rule": rule, "league": spec["league"], "headline": spec["headline"],
                 "prereg": spec["prereg"], "logged": 0, "settled": 0, "pending": 0,
                 "pushes": 0, "wins": 0, "losses": 0, "win_pct": None, "units": 0.0,
                 "breakeven": BREAKEVEN, "ci_low": None, "ci_high": None,
                 "checkpoint_n": spec["checkpoint_n"], "kill_below": spec["kill_below"],
                 "status": "no bets logged yet"}, [])

    log = pd.read_csv(spec["log"])
    result = log["result"].fillna("") if "result" in log else pd.Series([""] * len(log))
    settled = log[result.isin(["WIN", "LOSS"])]
    wins = int((settled["result"] == "WIN").sum()) if len(settled) else 0
    n = len(settled)
    losses = n - wins
    pct = (100.0 * wins / n) if n else None
    units = wins - losses * 1.1
    ci_low = ci_high = None
    if n > 1 and pct is not None:
        r = wins / n
        se = math.sqrt(r * (1 - r) / n)
        ci_low, ci_high = 100 * (r - 1.96 * se), 100 * (r + 1.96 * se)

    if n >= spec["checkpoint_n"]:
        status = ("KILL — below the pre-committed floor at the checkpoint"
                  if pct is not None and pct < spec["kill_below"]
                  else "continuing — not a claim that it works")
    else:
        status = f"{spec['checkpoint_n'] - n} more settled bets to the first checkpoint"

    cols = [c for c in COMMON + EXTRA[rule] if c in log.columns]
    rows = []
    for rec in log[cols].to_dict("records"):
        rows.append({"rule": rule, "league": spec["league"],
                     **{k: _clean(v) for k, v in rec.items()}})
    # Newest first: graded games at the top, then pending, by kickoff.
    rows.sort(key=lambda r: str(r.get("kickoff") or ""), reverse=True)

    return ({"rule": rule, "league": spec["league"], "headline": spec["headline"],
             "prereg": spec["prereg"], "logged": int(len(log)), "settled": n,
             "pending": int((result == "").sum()), "pushes": int((result == "PUSH").sum()),
             "wins": wins, "losses": losses,
             "win_pct": round(pct, 1) if pct is not None else None,
             "units": round(float(units), 1), "breakeven": BREAKEVEN,
             "ci_low": round(ci_low, 1) if ci_low is not None else None,
             "ci_high": round(ci_high, 1) if ci_high is not None else None,
             "checkpoint_n": spec["checkpoint_n"], "kill_below": spec["kill_below"],
             "status": status}, rows)


def main() -> int:
    rules, bets = {}, []
    for rule, spec in RULES.items():
        summary, rows = summarise(rule, spec)
        rules[rule] = summary
        bets.extend(rows)
        print(f"  {rule}: {summary['logged']} logged, {summary['settled']} settled, "
              f"{summary['win_pct'] if summary['win_pct'] is not None else '—'}%")

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "breakeven": BREAKEVEN,
        "disclaimer": ("Forward record of two pre-registered tracking rules. "
                       "bets_allowed() is False for both leagues and neither rule is an "
                       "authorisation to stake money. Units assume one flat unit at -110."),
        "rules": rules,
        "bets": bets,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
    print(f"wrote {len(bets)} logged bets -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
