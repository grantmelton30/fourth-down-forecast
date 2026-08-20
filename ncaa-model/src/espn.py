"""Quarterback availability from ESPN. Free, no key, no registration.

WHY THIS EXISTS AT ALL. The quarterback adjustment (`src/qb.py`, DECISIONS D17) reads an
absence off the box score, which works for history and is useless for the thing it was built
for -- an upcoming game has no box score. Without a feed it depends on someone remembering
to edit `data/manual/qb_status.csv`, and an optional file nobody fills in is a silent no-op:
exactly the failure class that made the QB adjustment, and then the weather adjustment,
change nothing anyone could see.

WHY ESPN AND NOT CFBD. CFBD has no injury or depth-chart endpoint -- `/injuries`, `/depth`,
`/depthchart`, `/player/injuries` and `/teams/depth` all 404 (verified 2026-08-19). ESPN's
core API is undocumented but public and keyless.

THE JOIN IS FREE, WHICH IS THE WHOLE REASON THIS IS PRACTICAL. CFBD's player ids ARE ESPN
athlete ids -- verified on the four highest-volume 2025 passers, all resolving to the right
person at the right position. So the incumbent identified from CFBD box scores can be looked
up on ESPN directly, with no name matching and none of the collision risk that comes with it.

WHAT CANNOT BE VERIFIED YET, STATED PLAINLY. Every roster reads `Active` in August because
the season has not started, so the code path that turns a status into an absence has never
been exercised against real data. That is why an UNRECOGNISED status is reported loudly and
treated as available: on the first weekend of games the unseen codes will appear in the run
output instead of being silently swallowed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from .config import CACHE_DIR

ATHLETE_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/athletes"
)
_TIMEOUT = 20

STATUS_COLUMNS = ["athlete_id", "status", "available", "fetched_at"]

# Conservative, and deliberately mirrors `nfl-model/src/injuries.py`'s decision to exclude
# "Questionable": most questionable players play, and a false absence moves a line by the
# full 1.75 points in the wrong direction. Only unambiguous absences count.
UNAVAILABLE_STATUSES = {
    "out", "injured", "injured reserve", "suspended", "ineligible",
    "medical redshirt", "season-ending injury",
}
# "Inactive" is DELIBERATELY NOT UNAVAILABLE, and this was found the hard way. Every
# graduated passer reads `Inactive` -- it means "not on an active roster", which for a
# former player is permanent and has nothing to do with this week. Treating it as an
# absence produced 66 spurious overrides on the first live test against real incumbents.
# It is ambiguous rather than informative, so it is scored as available and the caller
# restricts status checks to games that have not been played anyway.
AVAILABLE_STATUSES = {"active", "probable", "questionable", "day-to-day", "inactive"}


def _status_is_available(status: str) -> "bool | None":
    """True/False for a known status, None for one this code has never seen.

    None is not a failure -- it is the honest answer, and the caller reports it rather than
    guessing. Guessing is how a feed silently starts moving lines on a code nobody checked.
    """
    key = str(status or "").strip().lower()
    if key in AVAILABLE_STATUSES:
        return True
    if key in UNAVAILABLE_STATUSES:
        return False
    return None


def fetch_athlete_status(
    athlete_ids, allow_network: bool = False, ttl_hours: float = 6.0,
    cache_key: str = "default",
) -> pd.DataFrame:
    """One row per athlete: raw ESPN status plus a resolved availability flag.

    TTL-cached, not permanent. A quarterback's status is the one thing here that genuinely
    changes hour to hour, and a permanent cache would freeze the first reading taken and
    still look like it was working -- the failure `cfbd_client.call`'s live TTL and
    `bulk_game_weather`'s forecast handling both exist to prevent.

    `allow_network=False` returns whatever is cached and fetches nothing, so a backtest can
    never start making hundreds of HTTP calls.
    """
    ids = [str(a) for a in pd.Series(list(athlete_ids)).dropna().unique() if str(a)]
    path = CACHE_DIR / f"espn_qb_status_{cache_key}.parquet"
    cached = pd.DataFrame(columns=STATUS_COLUMNS)
    if path.exists():
        try:
            disk = pd.read_parquet(path)
            if not set(STATUS_COLUMNS) - set(disk.columns):
                cached = disk
        except Exception:  # noqa: BLE001 - a bad cache must not stop a projection
            cached = pd.DataFrame(columns=STATUS_COLUMNS)

    now = time.time()
    fresh_ids: set = set()
    if len(cached):
        age = now - pd.to_numeric(cached["fetched_at"], errors="coerce").fillna(0.0)
        fresh_ids = set(cached.loc[age < float(ttl_hours) * 3600.0, "athlete_id"].astype(str))

    need = [a for a in ids if a not in fresh_ids]
    rows = []
    if need and allow_network:
        for athlete_id in need:
            try:
                resp = requests.get(f"{ATHLETE_URL}/{athlete_id}", timeout=_TIMEOUT)
                if resp.status_code != 200:
                    continue
                status = ((resp.json().get("status") or {}).get("name")) or ""
            except Exception:  # noqa: BLE001 - one bad athlete must not stop the slate
                continue
            rows.append({
                "athlete_id": str(athlete_id), "status": str(status),
                "available": _status_is_available(status), "fetched_at": now,
            })

    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached = cached.drop_duplicates("athlete_id", keep="last")[STATUS_COLUMNS]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(path, index=False)

    return cached[cached["athlete_id"].astype(str).isin(ids)].copy()


def qb_status_overrides(qb_table: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """Turn ESPN statuses into `qb_status.csv`-shaped rows for the INCUMBENT of each game.

    Only unambiguous absences produce a row. An `Active` incumbent produces nothing (the
    default is already "he plays"), and an unrecognised status produces nothing either --
    `unrecognised_statuses` reports those so they surface instead of being swallowed.
    """
    cols = ["season", "week", "team", "starter_out", "observed_at", "source"]
    if qb_table is None or qb_table.empty or status is None or status.empty:
        return pd.DataFrame(columns=cols)
    s = status.copy()
    s["athlete_id"] = s["athlete_id"].astype(str)
    out = qb_table.copy()
    out["incumbent_id"] = out["incumbent_id"].astype(str)
    merged = out.merge(s, left_on="incumbent_id", right_on="athlete_id", how="inner")
    unavailable = merged[merged["available"] == False]  # noqa: E712 - None must not match
    if unavailable.empty:
        return pd.DataFrame(columns=cols)
    stamp = pd.Timestamp.utcnow().isoformat()
    return pd.DataFrame({
        "season": unavailable["season"], "week": unavailable["week"],
        "team": unavailable["team"], "starter_out": True,
        "observed_at": stamp, "source": "espn:" + unavailable["status"].astype(str),
    })[cols].drop_duplicates(["season", "week", "team"])


def unrecognised_statuses(status: pd.DataFrame) -> list:
    """Status strings this module has no rule for. Printed by callers on every run.

    The point of surfacing these: nothing has been observed except `Active` (the season has
    not started), so the first real injury week will produce codes this file has never seen,
    and they must appear in the output rather than being quietly treated as available.
    """
    if status is None or status.empty or "available" not in status.columns:
        return []
    unknown = status[status["available"].isna()]
    return sorted({str(v) for v in unknown.get("status", []) if str(v).strip()})


__all__ = ["ATHLETE_URL", "AVAILABLE_STATUSES", "STATUS_COLUMNS", "UNAVAILABLE_STATUSES",
           "fetch_athlete_status", "qb_status_overrides", "unrecognised_statuses"]
