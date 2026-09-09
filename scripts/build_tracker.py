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
          "market_total_at_bet", "price_under", "line_basis", "side", "recorded_at",
          "protocol_version", "book", "quote_observed_at", "quote_source", "quote_id",
          "price", "price_status", "model_version", "actual_total",
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




def _loss_units(odds) -> float:
    """What a LOSS costs, in units, on the "stake to win one unit" convention.

    This repo states every result as `wins - losses * 1.1`, i.e. you risk 1.1 to win 1 at
    -110. Everything in GATES.md and both registrations is on that basis, so the priced
    version must be too -- mixing it with the risk-one-unit convention silently changes
    every figure by about 10% and makes the site disagree with its own documentation.

    At -110 a loss costs 1.10, at -105 it costs 1.05, at -120 it costs 1.20. That spread is
    not cosmetic: break-even runs 51.2% at -105 and 54.5% at -120, so a rule measured at 54%
    is profitable at one price and losing at the other.
    """
    if odds is None or (isinstance(odds, float) and not math.isfinite(odds)):
        odds = -110.0
    odds = float(odds)
    return abs(odds) / 100.0 if odds < 0 else 100.0 / odds


def _breakeven(odds) -> float:
    """Win rate needed to break even at this price. -110 when the price was not recorded."""
    if odds is None or (isinstance(odds, float) and not math.isfinite(odds)):
        odds = -110.0
    odds = float(odds)
    return abs(odds) / (abs(odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def _stats(wins: int, losses: int) -> dict:
    """Win rate, units and interval from a W-L pair. One place, so the combined view and
    the per-rule cards cannot drift apart on the arithmetic."""
    n = wins + losses
    if not n:
        return {"win_pct": None, "units": 0.0, "ci_low": None, "ci_high": None}
    r = wins / n
    # Wilson interval stays inside [0, 1], including all-win/all-loss samples.
    z = 1.96
    denom = 1 + z*z/n
    center = (r + z*z/(2*n))/denom
    half = z*math.sqrt(r*(1-r)/n + z*z/(4*n*n))/denom
    return {"win_pct": round(100 * r, 1),
            "units": round(wins - losses * 1.1, 1),
            "ci_low": round(100 * (center-half), 1) if n > 1 else None,
            "ci_high": round(100 * (center+half), 1) if n > 1 else None}


ACTIVE_PROTOCOL = {"P1": "P1-current-v2", "W1": "W1-reference-v2"}


def cohort_for(row, rule):
    protocol = row.get("protocol_version")
    if isinstance(protocol, str) and protocol.strip():
        return protocol
    if rule == "W1":
        return "W1-legacy-reference"
    basis = row.get("line_basis")
    return "P1-legacy-current" if basis == "current" else "P1-legacy-opener"


def _valid_price(value):
    try:
        price = float(value)
        return math.isfinite(price) and abs(price) >= 100
    except (TypeError, ValueError):
        return False


def _cohort_stats(log, rule, spec, cohort):
    results = log.get("result", pd.Series(index=log.index, dtype=object)).fillna("")
    settled = log[results.isin(["WIN", "LOSS"])]
    wins = int((settled["result"] == "WIN").sum()) if len(settled) else 0
    losses = len(settled)-wins
    priced = []
    for row in settled.to_dict("records"):
        price = row.get("price")
        if not _valid_price(price) and rule == "W1":
            price = row.get("price_under")
        if _valid_price(price):
            priced.append((row["result"], float(price)))
    n = len(settled)
    pct = 100*wins/n if n else None
    checkpoint = spec["checkpoint_n"]
    status = (f"{max(0, checkpoint-n)} more settled selections to this cohort's monitoring checkpoint"
              if n < checkpoint else "monitoring checkpoint reached; no profitability claim")
    if n >= checkpoint and pct < spec["kill_below"]:
        status = "KILL — below the pre-committed floor at this cohort's checkpoint"
    return {"rule": rule, "league": spec["league"], "headline": spec["headline"],
            "prereg": spec["prereg"], "cohort": cohort,
            "logged": len(log), "settled": n, "pending": int((results == "").sum()),
            "pushes": int((results == "PUSH").sum()), "wins": wins, "losses": losses,
            **_stats(wins, losses), "breakeven": BREAKEVEN,
            "priced_units": round(sum(1 if r == "WIN" else -_loss_units(p) for r,p in priced), 3) if priced else None,
            "priced_bets": len(priced),
            "real_breakeven": round(100*sum(_loss_units(p) for _,p in priced)/sum(1+_loss_units(p) for _,p in priced), 2) if priced else None,
            "checkpoint_n": checkpoint, "kill_below": spec["kill_below"],
            "status": status, "evidence_status": "reference tracking; execution is not verified"}


def summarise(rule: str, spec: dict) -> tuple[dict, list]:
    log = pd.read_csv(spec["log"]) if spec["log"].exists() else pd.DataFrame()
    log["cohort"] = [cohort_for(r, rule) for r in log.to_dict("records")]
    active = ACTIVE_PROTOCOL[rule]
    summary = _cohort_stats(log[log.cohort == active], rule, spec, active)
    summary["cohorts"] = [_cohort_stats(g, rule, spec, str(c))
                          for c,g in log.groupby("cohort", sort=True)]
    cols = [c for c in [*COMMON, *EXTRA[rule], "cohort"] if c in log.columns]
    rows = [{"rule": rule, "league": spec["league"],
             **{k: _clean(v) for k,v in rec.items()}}
            for rec in log[cols].to_dict("records")]
    rows.sort(key=lambda r: str(r.get("kickoff") or ""), reverse=True)
    return summary, rows


def main() -> int:
    rules, bets = {}, []
    for rule, spec in RULES.items():
        summary, rows = summarise(rule, spec)
        rules[rule] = summary
        bets.extend(rows)
        print(f"  {rule}: {summary['logged']} logged, {summary['settled']} settled, "
              f"{summary['win_pct'] if summary['win_pct'] is not None else '—'}%")

    # Combined is a BANKROLL view, not a rule: two different rules pooled, which is what a
    # person actually wants to know about their own record. Labelled as such in the UI so it
    # is never mistaken for evidence about either rule on its own.
    tw = sum(r["wins"] for r in rules.values())
    tl = sum(r["losses"] for r in rules.values())
    combined = {
        "rule": "ALL", "league": "all",
        "headline": "Active tracking protocols only; legacy cohorts listed separately",
        "prereg": "NCAA P1 + NFL W1",
        "logged": sum(r["logged"] for r in rules.values()),
        "settled": tw + tl, "pending": sum(r["pending"] for r in rules.values()),
        "pushes": sum(r["pushes"] for r in rules.values()),
        "wins": tw, "losses": tl, **_stats(tw, tl), "breakeven": BREAKEVEN,
        "priced_units": round(sum(r["priced_units"] or 0 for r in rules.values()), 3) if any(r["priced_bets"] for r in rules.values()) else None,
        "priced_bets": sum(r["priced_bets"] for r in rules.values()),
        "real_breakeven": None,
        "checkpoint_n": None, "kill_below": None,
        "status": ("no bets logged yet" if tw + tl == 0 else
                   "Active protocols only; each cohort has its own monitoring checkpoint"),
    }
    print(f"  ALL: {combined['logged']} logged, {combined['settled']} settled, "
          f"{combined['win_pct'] if combined['win_pct'] is not None else '--'}%")

    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "breakeven": BREAKEVEN,
        "disclaimer": ("Reference tracking separated by protocol and line basis. "
                       "bets_allowed() is False for both leagues and neither rule is an "
                       "authorisation to stake money. Assumed units risk 1.1 to win 1 at -110. "
                       "Priced units include only rows with recorded odds; execution is unverified."),
        "combined": combined,
        "rules": rules,
        "bets": bets,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
    print(f"wrote {len(bets)} logged bets -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
