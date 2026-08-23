#!/usr/bin/env python
"""Pull everything useful from the balldontlie NFL API while a GOAT key is live.

    BDL_API_KEY=... uv run python prop-model/scrape_bdl.py --what all

WHY THIS EXISTS AND WHY IT RUNS ONCE. A GOAT key is a short-lived, paid resource. What it
can reach is not recoverable afterwards, so this pulls broadly, saves incrementally, and is
resumable -- a rate-limit stall or a dropped connection must never cost work already done.

WHAT RECONNAISSANCE ESTABLISHED, before any of this was written:

  player props     2026 ONLY. 2024 and 2025 return zero rows on every week probed, so the
                   documented "most recently completed season" is not populated for the NFL.
                   THERE IS NO PROP BACKTEST HERE. Props are a forward feed, same as the
                   free Odds API tier -- which is worth knowing before paying anyone.

  OPENING game odds  2022-2025 ARE populated, with spreads, totals, moneylines AND PRICES,
                   per vendor. This is the genuinely valuable find: nflverse publishes
                   CLOSING lines only, and the Sportsbook Reviews scrape in scripts/ stops
                   at 2021-22. This fills 2022-2025 -- precisely the post-2022 era GATES.md
                   identifies as a regime change, and the years the wind rule's
                   opener-versus-close question could not previously reach.

  player injuries  populated. nfl-model currently has NO live injury feed at all: nflverse
                   clamps at 2024 and data/manual/availability.csv does not exist.

RATE LIMITING IS CONSERVATIVE ON PURPOSE. The advertised GOAT limit is 600/min but 429s
appeared within a handful of calls, so this paces itself and backs off rather than racing a
limit it does not control. A slow complete pull beats a fast truncated one.

NOTHING HERE TOUCHES THE EXISTING MODELS. Output lands in prop-model/data/bdl/ as JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = "https://api.balldontlie.io/nfl/v1"
OUT = Path(__file__).resolve().parent / "data" / "bdl"
# Paced for the limit OBSERVED, not the one advertised. GOAT documents 600/min but 429s
# appear within a handful of calls and persist for minutes, which means the real constraint
# is a longer window than per-minute. A slow complete pull beats a fast truncated one, and
# the first attempt was truncated at 2022 week 10 by exactly this.
PAUSE = 2.5          # seconds between calls
BACKOFF = 45.0       # seconds after a 429


def key() -> str:
    k = os.environ.get("BDL_API_KEY")
    if not k:
        sys.exit("Set BDL_API_KEY. This is a paid key -- never commit it.")
    return k


def get(path: str, params: dict, k: str, tries: int = 6):
    for attempt in range(tries):
        try:
            r = requests.get(f"{BASE}/{path}", params=params,
                             headers={"Authorization": k}, timeout=30)
        except requests.RequestException as exc:
            print(f"    network error ({exc.__class__.__name__}), retrying")
            time.sleep(BACKOFF)
            continue
        if r.status_code == 429:
            time.sleep(BACKOFF)
            continue
        return r
    return r


def write(name: str, rows: list) -> int:
    """Append-only JSONL. Resumability depends on never rewriting what is already saved."""
    if not rows:
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def done_keys(name: str, field: str) -> set:
    """What has already been pulled, so a resumed run skips it."""
    path = OUT / f"{name}.jsonl"
    if not path.exists():
        return set()
    seen = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                seen.add(json.loads(line).get(field))
            except json.JSONDecodeError:
                continue
    return seen


def paged(path: str, params: dict, k: str, name: str, cap: int = 200) -> int:
    """Walk a cursor-paginated endpoint to exhaustion."""
    total, cursor, pages = 0, None, 0
    while pages < cap:
        p = dict(params, per_page=100)
        if cursor:
            p["cursor"] = cursor
        r = get(path, p, k)
        if r.status_code != 200:
            print(f"    stopped: HTTP {r.status_code} {r.text[:120]}")
            break
        body = r.json()
        total += write(name, body.get("data", []))
        cursor = (body.get("meta") or {}).get("next_cursor")
        pages += 1
        if not cursor:
            break
        time.sleep(PAUSE)
    return total


def games(k: str, seasons) -> list:
    out = []
    for s in seasons:
        n = paged("games", {"seasons[]": s}, k, "games")
        print(f"  games {s}: {n}", flush=True)
        time.sleep(PAUSE)
    path = OUT / "games.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            out = [json.loads(l) for l in fh if l.strip()]
    return out


def _odds_window(path: str, name: str, k: str, seasons, weeks) -> int:
    """Season/week odds, PAGINATED and RESUMABLE.

    The first version took only the first page: the endpoint returns 25 rows by default and
    an NFL week is 16 games across several vendors, so roughly two thirds of every week was
    being silently dropped. Cursor is followed to exhaustion now.

    Already-captured (season, week) pairs are skipped, so a run killed part-way resumes
    rather than restarting -- which matters when the key is short-lived.
    """
    have = set()
    fp = OUT / f"{name}.jsonl"
    if fp.exists():
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    have.add((row.get("_season"), row.get("_week")))
                except json.JSONDecodeError:
                    continue

    total = 0
    for s_ in seasons:
        for w in weeks:
            if (s_, w) in have:
                continue
            cursor, got = None, 0
            while True:
                params = {"season": s_, "week": w, "per_page": 100}
                if cursor:
                    params["cursor"] = cursor
                r = get(path, params, k)
                if r.status_code != 200:
                    break
                body = r.json()
                rows = body.get("data", [])
                for row in rows:
                    row["_season"], row["_week"] = s_, w
                got += write(name, rows)
                cursor = (body.get("meta") or {}).get("next_cursor")
                time.sleep(PAUSE)
                if not cursor or not rows:
                    break
            total += got
            if got:
                print(f"  {name} {s_} wk{w}: {got}", flush=True)
    return total


def opening_odds(k: str, seasons, weeks=range(1, 23)) -> int:
    """The valuable one: opening spreads/totals/moneylines WITH prices, 2022-2025."""
    return _odds_window("odds/opening", "opening_odds", k, seasons, weeks)


def closing_odds(k: str, seasons, weeks=range(1, 23)) -> int:
    return _odds_window("odds", "closing_odds", k, seasons, weeks)


def player_props(k: str, game_ids, name="player_props", path="odds/player_props") -> int:
    already = done_keys(name, "game_id")
    total = 0
    for i, gid in enumerate(game_ids, 1):
        if gid in already:
            continue
        r = get(path, {"game_id": gid}, k)
        if r.status_code != 200:
            print(f"    game {gid}: HTTP {r.status_code}", flush=True)
            time.sleep(PAUSE)
            continue
        rows = r.json().get("data", [])
        total += write(name, rows)
        if rows:
            print(f"  [{i}/{len(game_ids)}] game {gid}: {len(rows)} props", flush=True)
        time.sleep(PAUSE)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="all",
                    choices=["all", "games", "opening", "closing", "props", "injuries"])
    ap.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    a = ap.parse_args()
    k = key()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.what in ("all", "games"):
        print("GAMES")
        games(k, a.seasons)
    if a.what in ("all", "injuries"):
        print("PLAYER INJURIES")
        print(f"  {paged('player_injuries', {}, k, 'player_injuries')} rows")
    if a.what in ("all", "opening"):
        print("OPENING ODDS (the find: 2022-2025 with prices)")
        print(f"  total {opening_odds(k, a.seasons)}")
    if a.what in ("all", "closing"):
        print("CLOSING ODDS")
        print(f"  total {closing_odds(k, a.seasons)}")
    if a.what in ("all", "props"):
        print("PLAYER PROPS (2026 only -- reconnaissance found nothing earlier)")
        gs = games(k, [2026])
        ids = sorted({g["id"] for g in gs if g.get("season") == 2026})
        print(f"  {len(ids)} candidate games")
        print(f"  total {player_props(k, ids)}")
        print(f"  opening props {player_props(k, ids, 'opening_player_props', 'odds/player_props/opening')}")
    print(f"\nsaved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
