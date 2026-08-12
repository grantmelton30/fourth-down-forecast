"""Venue coordinates, for weather.

The NFL build can hardcode a 32-row stadium table (`nfl-model/src/stadiums.py`) because
those are public facts about 32 rows that change once a decade. College cannot: there are
844 venues, FBS teams share and rename them, and neutral sites move every year. So this
reads CFBD's `/venues` endpoint instead -- ONE call, permanently cached -- and joins it to
each game on the `venueId` the games payload already carries.

WHAT CFBD GIVES AND WHAT IT DOES NOT. `latitude`, `longitude`, `dome` and `elevation` are
populated for essentially every venue that hosts an FBS game. `timezone` is NOT: it is null
for a long tail of small venues. That matters because Open-Meteo returns hourly series in
whatever timezone it is asked for, and asking for the wrong one shifts the reading by hours
-- so a missing timezone is resolved from longitude rather than silently defaulted to UTC.
"""

from __future__ import annotations

import pandas as pd

from .config import CACHE_DIR

VENUE_COLUMNS = ["venue_id", "name", "lat", "lon", "tz", "dome", "elev_ft"]

# Longitude midpoints of the US zones, used only when CFBD's `timezone` is null. Crude, but
# it is never worse than an hour and it beats defaulting a Pacific venue to UTC.
_TZ_BY_LONGITUDE = [
    (-127.0, -115.0, "America/Los_Angeles"),
    (-115.0, -101.0, "America/Denver"),
    (-101.0, -87.0, "America/Chicago"),
    (-87.0, -60.0, "America/New_York"),
]
_FALLBACK_TZ = "America/New_York"


def _timezone_from_longitude(lon: float) -> str:
    for lo, hi, tz in _TZ_BY_LONGITUDE:
        if lo <= lon < hi:
            return tz
    return _FALLBACK_TZ


def load_venues(client, refresh: bool = False) -> pd.DataFrame:
    """The CFBD venue table, normalised to VENUE_COLUMNS. One API call, then cached."""
    path = CACHE_DIR / "venues.parquet"
    if not refresh and path.exists():
        cached = pd.read_parquet(path)
        if not set(VENUE_COLUMNS) - set(cached.columns):
            return cached

    raw = client.frame("venues", "venues")
    df = pd.DataFrame({
        "venue_id": raw["id"].astype("Int64"),
        "name": raw["name"],
        "lat": pd.to_numeric(raw["latitude"], errors="coerce"),
        "lon": pd.to_numeric(raw["longitude"], errors="coerce"),
        "tz": raw.get("timezone"),
        "dome": raw["dome"].fillna(False).astype(bool),
        "elev_ft": pd.to_numeric(raw.get("elevation"), errors="coerce") * 3.28084,
    })
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    missing_tz = df["tz"].isna() | (df["tz"] == "")
    df.loc[missing_tz, "tz"] = df.loc[missing_tz, "lon"].map(_timezone_from_longitude)
    df = df[VENUE_COLUMNS].reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def attach_venues(games: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """Left-join venue coordinates onto games on `venueId`.

    Left, not inner: a game whose venue is unknown must still appear in the slate. It ends
    up with a null `lat`, which `weather.weather_for_game` reports as unavailable rather
    than treating as calm -- the distinction the NFL build got wrong for its whole life.
    """
    if "venueId" not in games.columns:
        raise KeyError(
            "attach_venues: games has no `venueId`. The cached games parquet predates it; "
            "delete it or call ingest.load_games again to rebuild from the JSON cache."
        )
    out = games.copy()
    out["venue_id"] = out["venueId"].astype("Int64")
    return out.merge(
        venues.rename(columns={"name": "venue_name"}), on="venue_id", how="left"
    )


def coverage(games_with_venues: pd.DataFrame) -> dict:
    """How much of a slate can actually be given weather. Call this before trusting it."""
    n = len(games_with_venues)
    has_coords = games_with_venues["lat"].notna()
    return {
        "games": n,
        "with_coordinates": int(has_coords.sum()),
        "pct_with_coordinates": round(100.0 * float(has_coords.mean()), 2) if n else 0.0,
        "indoor": int(games_with_venues["dome"].fillna(False).sum()),
    }
