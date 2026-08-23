#!/usr/bin/env python
"""Finish the 2026 prop pull, unattended, over however long it takes.

    BDL_API_KEY=... python prop-model/pull_props_overnight.py

WHY A SEPARATE SCRIPT. `scrape_bdl.py` records progress by which game_ids appear in the
output file, so a game that legitimately returns ZERO props is never marked done and gets
re-attempted on every resume. With 272 games and most of them empty this season, that turns
a resume into a near-restart -- and at the observed rate (16 games in 78 minutes, almost all
of it rate-limit backoff) a restart is a night wasted.

This keeps a separate `_attempted.json` of every game_id tried, empty or not, so a resume
picks up exactly where it stopped. It also pulls the OPENING props endpoint, which the
earlier run died before reaching.

PATIENCE OVER SPEED. The advertised GOAT limit is 600/min; 429s appear within a handful of
calls and persist for minutes. Nothing here races that. It sleeps, retries, and writes after
every single game so a kill at any moment costs at most one request.

STATE OF PLAY WHEN THIS WAS WRITTEN: 16 of 272 games attempted, 4 with props (the week 1
openers, 399 and 363 rows, plus two mid-September games at 38 and 30). Games 3-14 returned
zero, which is expected -- books post props close to kickoff and the season starts in
September. The remaining 256 are most likely empty too, and this exists to replace that
guess with a fact.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.balldontlie.io/nfl/v1"
OUT = Path(__file__).resolve().parent / "data" / "bdl"
STATE = OUT / "_attempted.json"
LOG = OUT / "_overnight.log"
PAUSE = 3.0
BACKOFF = 60.0
MAX_TRIES = 20          # ~20 minutes of patience per request before giving up on it


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"props": [], "opening": []}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st), encoding="utf-8")


def get(path: str, params: dict, key: str):
    for i in range(MAX_TRIES):
        try:
            r = requests.get(f"{BASE}/{path}", params=params,
                             headers={"Authorization": key}, timeout=30)
        except requests.RequestException as exc:
            log(f"    network {type(exc).__name__}; sleeping")
            time.sleep(BACKOFF)
            continue
        if r.status_code == 429:
            time.sleep(BACKOFF)
            continue
        return r
    return None


def append(name: str, rows: list) -> int:
    if not rows:
        return 0
    with (OUT / f"{name}.jsonl").open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)



def probe_history(games: list, key: str, st: dict) -> dict:
    """Settle whether 2024/2025 props exist ANYWHERE, before assuming they do not.

    The earlier conclusion -- "there is no historical prop data" -- rested on four probes:
    2024 week 1 and 2025 weeks 1, 10 and 18, all zero. That is consistent with the docs but
    it is not a proof, and writing off the key's most valuable content on four data points
    would be exactly the kind of confident-and-wrong call this project keeps catching.

    So: one game from EVERY week of 2024 and 2025, 44 requests. If any week returns rows,
    that season gets swept in full and the whole picture changes. If none do, the absence is
    established rather than assumed.
    """
    if st.get("history_probe_done"):
        log(f"history probe already done: {st.get('history_hits', [])}")
        return st
    hits = []
    for season in (2024, 2025):
        weeks = sorted({g.get("week") for g in games
                        if g.get("season") == season and g.get("week")})
        for wk in weeks:
            cand = [g for g in games if g.get("season") == season and g.get("week") == wk]
            if not cand:
                continue
            gid = sorted(g["id"] for g in cand)[0]
            r = get("odds/player_props", {"game_id": gid}, key)
            n = len(r.json().get("data", [])) if (r is not None and r.status_code == 200) else -1
            if n > 0:
                hits.append((season, wk, gid, n))
                append("player_props", r.json()["data"])
                log(f"  *** {season} wk{wk}: {n} PROPS FOUND -- history exists ***")
            time.sleep(PAUSE)
        log(f"  probed {season}: {len(weeks)} weeks")
    st["history_probe_done"] = True
    st["history_hits"] = hits
    save_state(st)
    if hits:
        log(f"HISTORY EXISTS in {len(hits)} probed weeks -- sweeping those seasons in full")
    else:
        log("NO historical props in any week of 2024 or 2025. Absence established.")
    return st


def run(kind: str, path: str, name: str, ids: list, key: str, st: dict) -> None:
    done = set(st[kind])
    todo = [g for g in ids if g not in done]
    log(f"{kind}: {len(done)} already attempted, {len(todo)} to go")
    for n, gid in enumerate(todo, 1):
        r = get(path, {"game_id": gid}, key)
        if r is None or r.status_code != 200:
            log(f"  game {gid}: giving up (HTTP {getattr(r, 'status_code', 'none')})")
            time.sleep(BACKOFF)
            continue
        rows = r.json().get("data", [])
        got = append(name, rows)
        # Recorded whether or not it returned anything -- that is the entire point.
        st[kind].append(gid)
        save_state(st)
        if got:
            log(f"  [{n}/{len(todo)}] game {gid}: {got} rows")
        elif n % 25 == 0:
            log(f"  [{n}/{len(todo)}] ...{n} attempted, still writing")
        time.sleep(PAUSE)
    log(f"{kind}: COMPLETE ({len(st[kind])} games attempted)")


def main() -> int:
    key = os.environ.get("BDL_API_KEY")
    if not key:
        sys.exit("Set BDL_API_KEY")
    OUT.mkdir(parents=True, exist_ok=True)
    games = [json.loads(l) for l in (OUT / "games.jsonl").open(encoding="utf-8")]
    ids = sorted({g["id"] for g in games if g.get("season") == 2026})
    log(f"=== overnight prop pull: {len(ids)} games in 2026 ===")

    st = load_state()
    # Seed from what the earlier run already wrote, so those are not repeated.
    pp = OUT / "player_props.jsonl"
    if pp.exists() and not st["props"]:
        have = {json.loads(l)["game_id"] for l in pp.open(encoding="utf-8")}
        st["props"] = sorted(have)
        save_state(st)
        log(f"seeded {len(have)} games already known to have props")

    st = probe_history(games, key, st)

    # A hit anywhere means the season is worth sweeping in full, not just probing.
    sweep = list(ids)
    for season, *_ in st.get("history_hits", []):
        sweep += [g["id"] for g in games if g.get("season") == season]
    sweep = sorted(set(sweep))
    if len(sweep) > len(ids):
        log(f"sweeping {len(sweep)} games (2026 plus seasons where history was found)")

    run("props", "odds/player_props", "player_props", sweep, key, st)
    run("opening", "odds/player_props/opening", "opening_player_props", sweep, key, st)
    log("=== ALL DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
