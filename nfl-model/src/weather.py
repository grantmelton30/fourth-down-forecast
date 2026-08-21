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
ET = "America/New_York"   # nflverse kickoffs are naive US/Eastern (ingest)
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


# --- Bet-time forecast (PREREG W1) ---------------------------------------------------

@dataclass(frozen=True)
class ForecastReading:
    """A forecast captured at a known moment, for a game that has not been played.

    Distinct from `GameWeather` on purpose. `GameWeather` answers "what was the weather",
    preferring the observed kickoff reading and falling back to a forecast; this answers
    "what did the forecast say WHEN THE BET WAS PLACED", which is a different question and
    the only one a prospective rule may ask.
    """
    wind_mph: "float | None"
    temp_f: "float | None"
    precip_in: "float | None"
    indoor: bool
    issued_at: str            # UTC ISO-8601, when this forecast was retrieved
    hours_before_kickoff: "float | None"
    source: str               # forecast | indoor | unavailable


def forecast_at_bet_time(
    game: pd.Series, *, now: "pd.Timestamp | None" = None
) -> ForecastReading:
    """The live forecast for an upcoming game, fetched fresh and stamped with the time.

    NEVER READS THE CACHE, AND NEVER WRITES IT. `weather.parquet` is keyed on
    (lat, lon, date) with no column recording when a forecast was issued, so a cache hit
    would silently substitute a reading fetched days earlier for the one available now --
    turning a 3-hour-horizon forecast into a 6-day one with no visible difference. That is
    the "cache that ignores its own inputs" bug class NEXT_SESSION.md records hitting three
    times in a single day. A prospective bet log is exactly where it would do the most
    damage, because the error would look like an edge.

    NEVER FALLS BACK TO THE OBSERVED READING either. `schedules.wind` is the game-time
    observation and does not exist before kickoff; if it is somehow populated, using it
    would be lookahead of the plainest kind.

    Returns `source="unavailable"` rather than raising, so one failed venue does not stop a
    slate from being recorded. An unavailable forecast means the game does not qualify -- it
    is never treated as calm.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    issued_at = now.isoformat(timespec="seconds")

    roof = game.get("roof")
    venue = venue_for_game(
        game.get("home_team"), game.get("location", "Home"), game.get("stadium")
    )
    indoor = roof in ("dome", "closed") or (
        not isinstance(roof, str) and venue.get("roof") == "dome"
    )
    if indoor:
        return ForecastReading(None, None, None, True, issued_at, None, "indoor")

    # A RETRACTABLE ROOF WHOSE GAME-DAY STATE IS NOT PUBLISHED YET IS UNKNOWN, NOT OPEN.
    # `schedules.roof` carries the per-game state (`open`/`closed`) but is null for games
    # that have not been played, which is every game this function is ever called on. Five
    # venues are retractable (ARI, ATL, DAL, HOU, IND), ~15% of the slate. Betting an under on wind at a stadium that may be sealed shut
    # is betting on nothing, so an unresolved roof disqualifies the game -- the same
    # "absence of a report is unknown, never healthy" rule src/availability.py runs on.
    if not isinstance(roof, str) and venue.get("roof") == "retractable":
        return ForecastReading(None, None, None, False, issued_at, None, "roof_unknown")

    kickoff = game.get("kickoff")
    if pd.isna(kickoff):
        return ForecastReading(None, None, None, False, issued_at, None, "unavailable")
    # nflverse kickoffs are NAIVE US/EASTERN (ingest._kickoff_timestamp), not UTC. Reading
    # them as UTC would misplace every game by 4-5 hours, which silently shifts the hourly
    # forecast picked for a night game and quietly corrupts the horizon.
    kickoff = pd.Timestamp(kickoff)
    ko_et = (kickoff.tz_localize(ET, ambiguous=True, nonexistent="shift_forward")
             if kickoff.tzinfo is None else kickoff)
    horizon = (ko_et.tz_convert("UTC") - now).total_seconds() / 3600.0

    try:
        hourly = fetch_forecast(venue["lat"], venue["lon"], venue["tz"], kickoff)
    except Exception as exc:  # noqa: BLE001 - a missing forecast must not stop the run
        # Open-Meteo reaches ~16 days ahead and answers 400 past that. That is the ordinary
        # case for a slate recorded early, not an error worth a stack of URLs, so it is
        # named rather than dumped. Either way the game does not qualify.
        beyond = "400" in str(exc)
        why = "beyond forecast horizon" if beyond else str(exc)[:120]
        print(f"  weather: no forecast for {game.get('game_id')} "
              f"({horizon:.0f}h out): {why}")
        return ForecastReading(None, None, None, False, issued_at, horizon, "unavailable")

    # The hourly reading nearest kickoff in the VENUE's local time, because `fetch_forecast`
    # requests the venue timezone and returns naive local timestamps. A 4:05pm ET kickoff in
    # Seattle is 1:05pm local, and matching hour-to-hour without converting would read the
    # forecast three hours late -- on the west coast that is the sea-breeze ramp.
    local_hour = ko_et.tz_convert(venue["tz"]).hour
    nearest = hourly.iloc[(hourly["time"].dt.hour - local_hour).abs().argmin()]

    def _f(value):
        return float(value) if pd.notna(value) else None

    return ForecastReading(
        _f(nearest["wind_mph"]), _f(nearest["temp_f"]), _f(nearest["precip_in"]),
        False, issued_at, horizon, "forecast",
    )
