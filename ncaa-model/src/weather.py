"""Weather via Open-Meteo. Free, no API key, no registration, CC-BY 4.0.

WHY OPEN-METEO AND NOT CFBD. CFBD does expose a weather endpoint, but it is metered against
the same 900-call monthly budget the rest of this repo lives inside, and weather is
per-game rather than per-season -- a single backtest would cost more calls than everything
else combined. Open-Meteo is unmetered for this volume and is already what nfl-model uses,
so the two sports read the same source and their wind numbers are comparable.

Two endpoints, chosen by whether the game has kicked off:
  * archive  -- completed games. Observed reanalysis, ~5 day lag.
  * forecast -- upcoming games. Only useful inside its horizon; beyond that Open-Meteo
                returns nothing and this reports unavailable rather than guessing.

Unlike the NFL, there is no third path: nflverse ships observed `temp` / `wind` columns on
its schedule, and CFBD ships nothing equivalent, so every college reading is a fetch.

UNITS ARE SET EXPLICITLY ON EVERY REQUEST. Open-Meteo defaults to Celsius and km/h. A 15
"mph" threshold compared against km/h suppresses the total on every calm afternoon and you
do not notice for a month. This is the same note as nfl-model/src/weather.py; it is
repeated rather than referenced because getting it wrong is silent.

THE READING IS TAKEN AT KICKOFF IN THE VENUE'S LOCAL TIME, not UTC and not a daily mean. A
7pm kickoff in Pullman and a 7pm kickoff in Miami are seven hours apart in UTC, and a daily
mean averages away exactly the evening wind the total cares about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from .config import CACHE_DIR

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_HOURLY = "temperature_2m,wind_speed_10m,precipitation"
_TIMEOUT = 30

WEATHER_COLUMNS = [
    "lat", "lon", "date", "hour", "temp_f", "wind_mph", "precip_in", "source",
]


@dataclass(frozen=True)
class GameWeather:
    """Conditions at kickoff. `source` records provenance so a run can be audited."""

    temp_f: "float | None"
    wind_mph: "float | None"
    precip_in: "float | None"
    indoor: bool
    source: str          # indoor | archive | forecast | cache | unavailable

    @property
    def available(self) -> bool:
        """True when the game can be adjusted. Indoors counts: there is nothing to fetch
        and nothing to be uncertain about."""
        return self.indoor or self.wind_mph is not None


_GAME_CACHE: "dict | None" = None


def _game_cache_lookup(game_id, cache_key: str = "default") -> "GameWeather | None":
    """Resolve one game from the game-keyed cache written by `bulk_game_weather`.

    Held in a module-level dict rather than re-read per call: the backtest asks for this
    once per graded game and the viewer once per rendered game, and re-reading a parquet
    thousands of times is how a cheap lookup turns into a visibly slow page.

    Returns None for "not in the cache", which is distinct from "cached as having no
    reading" -- the caller falls through to its own hourly cache and, if allowed, the
    network.
    """
    global _GAME_CACHE
    if game_id is None or pd.isna(game_id):
        return None
    if _GAME_CACHE is None:
        path = CACHE_DIR / f"game_weather_{cache_key}.parquet"
        rows: dict = {}
        if path.exists():
            try:
                frame = pd.read_parquet(path)
                for r in frame.itertuples(index=False):
                    rows[str(getattr(r, "game_id"))] = r
            except Exception:  # noqa: BLE001 - a bad cache must not stop a projection
                rows = {}
        _GAME_CACHE = rows
    hit = _GAME_CACHE.get(str(game_id))
    if hit is None:
        return None
    if bool(getattr(hit, "indoor", False)):
        return GameWeather(None, None, None, True, "indoor")
    wind = getattr(hit, "wind_mph", None)
    if wind is None or pd.isna(wind):
        return None
    def _num(name):
        v = getattr(hit, name, None)
        return None if v is None or pd.isna(v) else float(v)
    return GameWeather(_num("temp_f"), float(wind), _num("precip_in"), False,
                       str(getattr(hit, "source", "cache")))


def reset_game_weather_cache() -> None:
    """Drop the memo. For tests, and for a long-lived process that refetches mid-run."""
    global _GAME_CACHE
    _GAME_CACHE = None


def _cache_read() -> pd.DataFrame:
    path = CACHE_DIR / "weather.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        if not set(WEATHER_COLUMNS) - set(cached.columns):
            return cached
    return pd.DataFrame(columns=WEATHER_COLUMNS)


def _cache_write(df: pd.DataFrame) -> None:
    df.to_parquet(CACHE_DIR / "weather.parquet", index=False)


def _fetch(url: str, lat: float, lon: float, tz: str, date: str) -> pd.DataFrame:
    params = {
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "hourly": _HOURLY,
        "temperature_unit": "fahrenheit",   # never rely on the Celsius default
        "wind_speed_unit": "mph",           # never rely on the km/h default
        "precipitation_unit": "inch",
        "timezone": tz,
        "start_date": date,
        "end_date": date,
    }
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    if not hourly.get("time"):
        raise RuntimeError(f"Open-Meteo returned no hourly data for {date} at {lat},{lon}")
    return pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temp_f": hourly.get("temperature_2m"),
        "wind_mph": hourly.get("wind_speed_10m"),
        "precip_in": hourly.get("precipitation"),
    })


def fetch_archive(lat, lon, tz, date) -> pd.DataFrame:
    return _fetch(ARCHIVE_URL, lat, lon, tz, date)


def fetch_forecast(lat, lon, tz, date) -> pd.DataFrame:
    return _fetch(FORECAST_URL, lat, lon, tz, date)


def _local_kickoff(kickoff, tz: str) -> "pd.Timestamp | None":
    """Kickoff in the venue's own timezone. CFBD's `startDate` is UTC."""
    if pd.isna(kickoff):
        return None
    ts = pd.Timestamp(kickoff)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    try:
        return ts.tz_convert(tz)
    except Exception:  # noqa: BLE001 - an unknown tz string must not stop the run
        return ts


def weather_for_game(game: pd.Series, allow_network: bool = True) -> GameWeather:
    """Resolve conditions for one game. `game` needs lat, lon, tz, dome, kickoff.

    Order: indoor short-circuit, then the on-disk cache, then the network. The dome check
    comes first because a weather call for an indoor game is pure waste, and college plays
    a meaningful share of its season indoors.
    """
    if bool(game.get("dome", False)):
        return GameWeather(None, None, None, True, "indoor")

    # THE GAME-KEYED CACHE COMES FIRST, and it is the only reason weather reaches anything
    # a user sees. `bulk_game_weather` writes `game_weather_{key}.parquet` keyed on
    # `game_id`; this function's own cache below is keyed on lat/lon/date/hour and lives in
    # a different file that the bulk path never writes. The shared viewer calls
    # `build_context(..., allow_network=False)`, so before this lookup existed it found an
    # empty hourly cache, returned "unavailable", and every projection on the page was
    # silently computed as though wind did not exist (DECISIONS.md D16/D17).
    cached_game = _game_cache_lookup(game.get("game_id"))
    if cached_game is not None:
        return cached_game

    lat, lon = game.get("lat"), game.get("lon")
    if pd.isna(lat) or pd.isna(lon):
        return GameWeather(None, None, None, False, "unavailable")

    tz = game.get("tz") or "America/New_York"
    local = _local_kickoff(game.get("kickoff"), tz)
    if local is None:
        return GameWeather(None, None, None, False, "unavailable")

    key_lat, key_lon = round(float(lat), 4), round(float(lon), 4)
    date = local.strftime("%Y-%m-%d")

    cache = _cache_read()
    hit = cache[
        (cache["lat"] == key_lat) & (cache["lon"] == key_lon) & (cache["date"] == date)
    ]
    if len(hit):
        return _nearest(hit, local.hour, "cache")

    if not allow_network:
        return GameWeather(None, None, None, False, "unavailable")

    past = local.tz_localize(None) < pd.Timestamp.now()
    source = "archive" if past else "forecast"
    try:
        hourly = (fetch_archive if past else fetch_forecast)(lat, lon, tz, date)
    except Exception as exc:  # noqa: BLE001 - a missing reading must not stop the slate
        print(f"  weather: fetch failed for game {game.get('game_id')}: {exc}")
        return GameWeather(None, None, None, False, "unavailable")

    fresh = pd.DataFrame({
        "lat": key_lat, "lon": key_lon, "date": date,
        "hour": hourly["time"].dt.hour,
        "temp_f": hourly["temp_f"], "wind_mph": hourly["wind_mph"],
        "precip_in": hourly["precip_in"], "source": source,
    })
    _cache_write(pd.concat([cache, fresh], ignore_index=True)[WEATHER_COLUMNS])
    return _nearest(fresh, local.hour, source)


GAME_WEATHER_COLUMNS = ["game_id", "indoor", "wind_mph", "temp_f", "precip_in", "source"]


def bulk_game_weather(
    games: pd.DataFrame, allow_network: bool = False, cache_key: str = "default",
) -> pd.DataFrame:
    """Kickoff conditions for MANY games, one row per `game_id`.

    `weather_for_game` is per-game and re-reads the whole parquet on every call, which is
    fine for a slate and unusable for a backtest of several thousand games. This batches by
    (venue, season) instead: Open-Meteo's archive accepts a date RANGE, so one request
    covers a venue's entire football season -- ~700 requests for 2021-2025 rather than
    ~2,900. Only games missing from the cache are fetched, so the in-progress season costs
    a handful of requests a week and finished seasons cost nothing.

    Indoor games short-circuit with no request, exactly as `weather_for_game` does.
    `allow_network=False` returns whatever is already cached and nothing more, so a
    backtest can never silently start making thousands of HTTP calls.

    `games` needs `game_id`, `season`, `lat`, `lon`, `tz`, `dome`, `kickoff`.
    """
    path = CACHE_DIR / f"game_weather_{cache_key}.parquet"
    cached = pd.DataFrame(columns=GAME_WEATHER_COLUMNS)
    if path.exists():
        disk = pd.read_parquet(path)
        if not set(GAME_WEATHER_COLUMNS) - set(disk.columns):
            cached = disk

    # A FORECAST GOES STALE; AN ARCHIVE READING DOES NOT. Caching by game_id alone would
    # freeze a forecast taken two weeks out and serve it at kickoff, which is the same
    # "cached forever, still looks like it works" failure the live-season TTL in
    # `cfbd_client.call` exists to prevent. Only `archive` and `indoor` rows are permanent;
    # forecasts are refetched every time a refresh is allowed.
    permanent = cached[cached["source"].isin(("archive", "indoor"))] if len(cached) else cached
    need = games[~games["game_id"].isin(permanent["game_id"])].copy()
    indoor = need[need["dome"].fillna(False).astype(bool)]
    fresh = [pd.DataFrame({
        "game_id": indoor["game_id"], "indoor": True, "wind_mph": np.nan,
        "temp_f": np.nan, "precip_in": np.nan, "source": "indoor",
    })] if len(indoor) else []

    outdoor = need[~need["dome"].fillna(False).astype(bool)].dropna(subset=["lat", "lon"])
    outdoor = outdoor[outdoor["kickoff"].notna()]
    if len(outdoor) and allow_network:
        local = [_local_kickoff(k, tz if isinstance(tz, str) and tz else "America/New_York")
                 for k, tz in zip(outdoor["kickoff"], outdoor["tz"])]
        outdoor = outdoor.assign(
            _date=[t.strftime("%Y-%m-%d") if t is not None else None for t in local],
            _hour_key=[t.strftime("%Y-%m-%dT%H:00") if t is not None else None
                       for t in local],
        ).dropna(subset=["_date"])
        for (_, _), sub in outdoor.groupby(["lat", "lon"], sort=False):
            for _, season_sub in sub.groupby("season", sort=False):
                lat, lon = float(season_sub["lat"].iloc[0]), float(season_sub["lon"].iloc[0])
                tz = season_sub["tz"].iloc[0]
                tz = tz if isinstance(tz, str) and tz else "America/New_York"
                start, end = season_sub["_date"].min(), season_sub["_date"].max()
                past = pd.Timestamp(end) < pd.Timestamp.now().normalize()
                try:
                    hourly = _fetch_range(
                        ARCHIVE_URL if past else FORECAST_URL, lat, lon, tz, start, end)
                except Exception as exc:  # noqa: BLE001 - a missing venue must not stop it
                    print(f"  weather: {lat},{lon} {start}..{end} failed: {exc}")
                    continue
                merged = season_sub[["game_id", "_hour_key"]].merge(
                    hourly, left_on="_hour_key", right_on="hour_key", how="left")
                fresh.append(pd.DataFrame({
                    "game_id": merged["game_id"], "indoor": False,
                    "wind_mph": merged["wind_mph"], "temp_f": merged["temp_f"],
                    "precip_in": merged["precip_in"],
                    "source": "archive" if past else "forecast",
                }))

    if fresh:
        cached = pd.concat([cached, *fresh], ignore_index=True)
        cached = cached.drop_duplicates("game_id", keep="last")[GAME_WEATHER_COLUMNS]
        cached.to_parquet(path, index=False)
    return games[["game_id"]].merge(cached, on="game_id", how="left")


def _fetch_range(url, lat, lon, tz, start_date, end_date) -> pd.DataFrame:
    """One request covering a whole date range, keyed by local `YYYY-MM-DDTHH:00`."""
    resp = requests.get(url, params={
        "latitude": round(float(lat), 4), "longitude": round(float(lon), 4),
        "hourly": _HOURLY,
        "temperature_unit": "fahrenheit",   # never rely on the Celsius default
        "wind_speed_unit": "mph",           # never rely on the km/h default
        "precipitation_unit": "inch",
        "timezone": tz, "start_date": start_date, "end_date": end_date,
    }, timeout=_TIMEOUT)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    if not hourly.get("time"):
        raise RuntimeError(f"no hourly data for {start_date}..{end_date} at {lat},{lon}")
    return pd.DataFrame({
        "hour_key": hourly["time"],
        "temp_f": hourly.get("temperature_2m"),
        "wind_mph": hourly.get("wind_speed_10m"),
        "precip_in": hourly.get("precipitation"),
    })


def _nearest(rows: pd.DataFrame, hour: int, source: str) -> GameWeather:
    """The hourly reading closest to kickoff, never the daily mean."""
    row = rows.iloc[(rows["hour"] - hour).abs().argmin()]
    if pd.isna(row["wind_mph"]):
        return GameWeather(None, None, None, False, "unavailable")
    return GameWeather(
        float(row["temp_f"]) if pd.notna(row["temp_f"]) else None,
        float(row["wind_mph"]),
        float(row["precip_in"]) if pd.notna(row["precip_in"]) else None,
        False,
        source,
    )


def weather_for_slate(games: pd.DataFrame, allow_network: bool = True) -> pd.DataFrame:
    """Resolve a whole slate, one row per game. Reports provenance per game so a caller
    can see how much of the slate is real and how much is missing."""
    out = []
    for _, g in games.iterrows():
        wx = weather_for_game(g, allow_network=allow_network)
        out.append({
            "game_id": g.get("game_id"),
            "temp_f": wx.temp_f, "wind_mph": wx.wind_mph, "precip_in": wx.precip_in,
            "indoor": wx.indoor, "source": wx.source, "available": wx.available,
        })
    return pd.DataFrame(out)
