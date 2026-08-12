"""Weather via Open-Meteo (§13, DATA_SOURCES.md §4). Free, no API key, CC-BY 4.0.

Two paths, both required:
  * Historical (backtest): `schedules.temp` / `schedules.wind` are already populated for
    completed outdoor games. Use them; no network call.
  * Upcoming games: those columns are null until the game is played, so fetch the
    forecast.

UNITS ARE SET EXPLICITLY on every request. Open-Meteo defaults to Celsius and km/h, and a
15 "mph" threshold silently compared against km/h suppresses the total on every calm
afternoon -- and you do not notice for a month.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from .config import CACHE_DIR
from .stadiums import venue_for_game

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_HOURLY = "temperature_2m,wind_speed_10m,precipitation"
_TIMEOUT = 20


@dataclass(frozen=True)
class GameWeather:
    temp_f: "float | None"
    wind_mph: "float | None"
    precip_in: "float | None"
    indoor: bool
    source: str          # schedules | forecast | archive | indoor | unavailable

    @property
    def available(self) -> bool:
        return self.indoor or self.wind_mph is not None


def _cache_file() -> pd.DataFrame:
    path = CACHE_DIR / "weather.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(
        columns=["lat", "lon", "date", "hour", "temp_f", "wind_mph", "precip_in", "source"]
    )


def _cache_write(df: pd.DataFrame) -> None:
    df.to_parquet(CACHE_DIR / "weather.parquet", index=False)


def _fetch(url: str, lat: float, lon: float, date: str, extra: dict) -> pd.DataFrame:
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": _HOURLY,
        "temperature_unit": "fahrenheit",   # never rely on the Celsius default
        "wind_speed_unit": "mph",           # never rely on the km/h default
        "precipitation_unit": "inch",
        **extra,
    }
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    if not hourly.get("time"):
        raise RuntimeError(f"weather: Open-Meteo returned no hourly data for {date}")
    return pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temp_f": hourly.get("temperature_2m"),
        "wind_mph": hourly.get("wind_speed_10m"),
        "precip_in": hourly.get("precipitation"),
    })


def fetch_forecast(
    lat: float, lon: float, tz: str, kickoff: pd.Timestamp
) -> pd.DataFrame:
    date = kickoff.strftime("%Y-%m-%d")
    return _fetch(FORECAST_URL, lat, lon, date, {
        "timezone": tz, "start_date": date, "end_date": date,
    })


def fetch_archive(
    lat: float, lon: float, tz: str, kickoff: pd.Timestamp
) -> pd.DataFrame:
    date = kickoff.strftime("%Y-%m-%d")
    return _fetch(ARCHIVE_URL, lat, lon, date, {
        "timezone": tz, "start_date": date, "end_date": date,
    })


def weather_for_game(game: pd.Series, allow_network: bool = True) -> GameWeather:
    """Resolve weather for one scheduled game.

    Order of preference: indoor short-circuit, the values nflverse already carries, then
    a network call. Roughly a third of the slate is indoors and a weather call for those
    is pure waste, so the roof check comes first.
    """
    roof = game.get("roof")
    venue = venue_for_game(
        game.get("home_team"), game.get("location", "Home"), game.get("stadium")
    )

    indoor = roof in ("dome", "closed") or (
        not isinstance(roof, str) and venue.get("roof") == "dome"
    )
    if indoor:
        return GameWeather(None, None, None, True, "indoor")

    temp, wind = game.get("temp"), game.get("wind")
    if pd.notna(temp) and pd.notna(wind):
        return GameWeather(float(temp), float(wind), None, False, "schedules")

    kickoff = game.get("kickoff")
    if not allow_network or pd.isna(kickoff):
        return GameWeather(
            float(temp) if pd.notna(temp) else None,
            float(wind) if pd.notna(wind) else None,
            None, False, "unavailable",
        )

    kickoff = pd.Timestamp(kickoff)
    past = kickoff < pd.Timestamp.now()
    cache = _cache_file()
    key = (round(venue["lat"], 4), round(venue["lon"], 4), kickoff.strftime("%Y-%m-%d"))
    hit = cache[
        (cache["lat"] == key[0]) & (cache["lon"] == key[1]) & (cache["date"] == key[2])
    ]
    if len(hit):
        row = hit.iloc[(hit["hour"] - kickoff.hour).abs().argmin()]
        return GameWeather(
            float(row["temp_f"]), float(row["wind_mph"]),
            float(row["precip_in"]) if pd.notna(row["precip_in"]) else None,
            False, str(row["source"]),
        )

    try:
        hourly = (fetch_archive if past else fetch_forecast)(
            venue["lat"], venue["lon"], venue["tz"], kickoff
        )
    except Exception as exc:  # noqa: BLE001 - a missing forecast must not stop the run
        print(f"  weather: fetch failed for {game.get('game_id')}: {exc}")
        return GameWeather(None, None, None, False, "unavailable")

    source = "archive" if past else "forecast"
    fresh = pd.DataFrame({
        "lat": key[0], "lon": key[1], "date": key[2],
        "hour": hourly["time"].dt.hour,
        "temp_f": hourly["temp_f"], "wind_mph": hourly["wind_mph"],
        "precip_in": hourly["precip_in"], "source": source,
    })
    _cache_write(pd.concat([cache, fresh], ignore_index=True))

    # The hourly reading nearest kickoff, never the daily mean.
    nearest = fresh.iloc[(fresh["hour"] - kickoff.hour).abs().argmin()]
    return GameWeather(
        float(nearest["temp_f"]), float(nearest["wind_mph"]),
        float(nearest["precip_in"]) if pd.notna(nearest["precip_in"]) else None,
        False, source,
    )


def backfill_from_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    """Report which completed outdoor games are missing observed weather.

    The backtest reads temp/wind straight off schedules; this says how much of the sample
    has no reading at all, so a silent gap is not mistaken for calm weather.
    """
    outdoor_done = schedules[
        schedules["result"].notna() & ~schedules["roof"].isin(["dome", "closed"])
    ]
    return pd.DataFrame([{
        "outdoor_completed_games": len(outdoor_done),
        "missing_wind": int(outdoor_done["wind"].isna().sum()),
        "missing_temp": int(outdoor_done["temp"].isna().sum()),
        "pct_missing_wind": round(float(outdoor_done["wind"].isna().mean()) * 100, 2),
    }])
