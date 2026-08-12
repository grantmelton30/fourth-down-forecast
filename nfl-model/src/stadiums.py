"""32-team stadium table: lat, lon, timezone, roof, elevation.

A literal dict, per DATA_SOURCES.md §6. These are public facts about 32 rows that change
once a decade -- scraping them or adding a geocoding dependency would be strictly worse.
Coordinates are stadium centroids; elevations are field level in feet.

`roof` here is the *structural* type. nflverse's `schedules.roof` records the per-game
state of a retractable roof (`closed` / `open`), so game-level code prefers the schedule
value and falls back to this table. `crosscheck_roofs()` asserts the two agree, which
catches the franchise relocations that make this table go stale (LV, LA x2, BUF, TEN).
"""

from __future__ import annotations

import math

import pandas as pd

OUTDOORS = "outdoors"
DOME = "dome"
RETRACTABLE = "retractable"

STADIUMS = {
    "ARI": {"lat": 33.5276, "lon": -112.2626, "tz": "America/Phoenix", "roof": RETRACTABLE, "elev_ft": 1070, "name": "State Farm Stadium"},
    "ATL": {"lat": 33.7554, "lon": -84.4008, "tz": "America/New_York", "roof": RETRACTABLE, "elev_ft": 1050, "name": "Mercedes-Benz Stadium"},
    "BAL": {"lat": 39.2780, "lon": -76.6227, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 33, "name": "M&T Bank Stadium"},
    "BUF": {"lat": 42.7738, "lon": -78.7870, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 600, "name": "Highmark Stadium"},
    "CAR": {"lat": 35.2258, "lon": -80.8528, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 751, "name": "Bank of America Stadium"},
    "CHI": {"lat": 41.8623, "lon": -87.6167, "tz": "America/Chicago", "roof": OUTDOORS, "elev_ft": 597, "name": "Soldier Field"},
    "CIN": {"lat": 39.0955, "lon": -84.5161, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 490, "name": "Paycor Stadium"},
    "CLE": {"lat": 41.5061, "lon": -81.6995, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 581, "name": "Huntington Bank Field"},
    "DAL": {"lat": 32.7473, "lon": -97.0945, "tz": "America/Chicago", "roof": RETRACTABLE, "elev_ft": 595, "name": "AT&T Stadium"},
    "DEN": {"lat": 39.7439, "lon": -105.0201, "tz": "America/Denver", "roof": OUTDOORS, "elev_ft": 5280, "name": "Empower Field at Mile High"},
    "DET": {"lat": 42.3400, "lon": -83.0456, "tz": "America/Detroit", "roof": DOME, "elev_ft": 600, "name": "Ford Field"},
    "GB": {"lat": 44.5013, "lon": -88.0622, "tz": "America/Chicago", "roof": OUTDOORS, "elev_ft": 640, "name": "Lambeau Field"},
    "HOU": {"lat": 29.6847, "lon": -95.4107, "tz": "America/Chicago", "roof": RETRACTABLE, "elev_ft": 50, "name": "NRG Stadium"},
    "IND": {"lat": 39.7601, "lon": -86.1639, "tz": "America/Indiana/Indianapolis", "roof": RETRACTABLE, "elev_ft": 715, "name": "Lucas Oil Stadium"},
    "JAX": {"lat": 30.3239, "lon": -81.6373, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 16, "name": "EverBank Stadium"},
    "KC": {"lat": 39.0489, "lon": -94.4839, "tz": "America/Chicago", "roof": OUTDOORS, "elev_ft": 889, "name": "GEHA Field at Arrowhead Stadium"},
    "LA": {"lat": 33.9535, "lon": -118.3392, "tz": "America/Los_Angeles", "roof": DOME, "elev_ft": 105, "name": "SoFi Stadium"},
    "LAC": {"lat": 33.9535, "lon": -118.3392, "tz": "America/Los_Angeles", "roof": DOME, "elev_ft": 105, "name": "SoFi Stadium"},
    "LV": {"lat": 36.0909, "lon": -115.1833, "tz": "America/Los_Angeles", "roof": DOME, "elev_ft": 2030, "name": "Allegiant Stadium"},
    "MIA": {"lat": 25.9580, "lon": -80.2389, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 8, "name": "Hard Rock Stadium"},
    "MIN": {"lat": 44.9736, "lon": -93.2575, "tz": "America/Chicago", "roof": DOME, "elev_ft": 830, "name": "U.S. Bank Stadium"},
    "NE": {"lat": 42.0909, "lon": -71.2643, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 289, "name": "Gillette Stadium"},
    "NO": {"lat": 29.9511, "lon": -90.0812, "tz": "America/Chicago", "roof": DOME, "elev_ft": 3, "name": "Caesars Superdome"},
    "NYG": {"lat": 40.8135, "lon": -74.0745, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 7, "name": "MetLife Stadium"},
    "NYJ": {"lat": 40.8135, "lon": -74.0745, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 7, "name": "MetLife Stadium"},
    "PHI": {"lat": 39.9008, "lon": -75.1675, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 39, "name": "Lincoln Financial Field"},
    "PIT": {"lat": 40.4468, "lon": -80.0158, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 728, "name": "Acrisure Stadium"},
    "SEA": {"lat": 47.5952, "lon": -122.3316, "tz": "America/Los_Angeles", "roof": OUTDOORS, "elev_ft": 20, "name": "Lumen Field"},
    "SF": {"lat": 37.4033, "lon": -121.9694, "tz": "America/Los_Angeles", "roof": OUTDOORS, "elev_ft": 30, "name": "Levi's Stadium"},
    "TB": {"lat": 27.9759, "lon": -82.5033, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 26, "name": "Raymond James Stadium"},
    "TEN": {"lat": 36.1665, "lon": -86.7713, "tz": "America/Chicago", "roof": OUTDOORS, "elev_ft": 431, "name": "Nissan Stadium"},
    "WAS": {"lat": 38.9077, "lon": -76.8645, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 203, "name": "Northwest Stadium"},
}

# International venues, identified from schedules.location == "Neutral". Treated as
# neutral sites with zero HFA (§13) -- we do not attempt to model them. Coordinates are
# carried only so weather still resolves for the outdoor ones.
INTERNATIONAL_VENUES = {
    "Tottenham Hotspur Stadium": {"lat": 51.6043, "lon": -0.0665, "tz": "Europe/London", "roof": OUTDOORS, "elev_ft": 105},
    "Wembley Stadium": {"lat": 51.5560, "lon": -0.2795, "tz": "Europe/London", "roof": OUTDOORS, "elev_ft": 148},
    "Allianz Arena": {"lat": 48.2188, "lon": 11.6247, "tz": "Europe/Berlin", "roof": OUTDOORS, "elev_ft": 1614},
    "Estadio Azteca": {"lat": 19.3029, "lon": -99.1505, "tz": "America/Mexico_City", "roof": OUTDOORS, "elev_ft": 7200},
    "Neo Quimica Arena": {"lat": -23.5453, "lon": -46.4742, "tz": "America/Sao_Paulo", "roof": OUTDOORS, "elev_ft": 2493},
    "Deutsche Bank Park": {"lat": 50.0685, "lon": 8.6455, "tz": "Europe/Berlin", "roof": RETRACTABLE, "elev_ft": 348},
    "Croke Park": {"lat": 53.3607, "lon": -6.2511, "tz": "Europe/Dublin", "roof": OUTDOORS, "elev_ft": 33},
    "Santiago Bernabeu": {"lat": 40.4531, "lon": -3.6883, "tz": "Europe/Madrid", "roof": RETRACTABLE, "elev_ft": 2188},
}

# Approximate UTC offsets, used only for the "west coast team, early eastern kickoff"
# penalty. Whole hours are sufficient; DST shifts every US zone together.
_TZ_OFFSET_HOURS = {
    "America/New_York": -5,
    "America/Detroit": -5,
    "America/Indiana/Indianapolis": -5,
    "America/Chicago": -6,
    "America/Denver": -7,
    "America/Phoenix": -7,
    "America/Los_Angeles": -8,
}

EARTH_RADIUS_MI = 3958.7613


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in statute miles."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def travel_distance_miles(away_team: str, home_team: str) -> float:
    """Distance the away team travels to the home team's venue."""
    a, h = STADIUMS.get(away_team), STADIUMS.get(home_team)
    if a is None or h is None:
        return 0.0
    return haversine_miles(a["lat"], a["lon"], h["lat"], h["lon"])


def tz_offset_hours(team: str) -> int:
    s = STADIUMS.get(team)
    return _TZ_OFFSET_HOURS.get(s["tz"], -5) if s else -5


def timezone_gap(away_team: str, home_team: str) -> int:
    """Positive when the away team travels east (loses hours), the direction that costs
    them in an early kickoff."""
    return tz_offset_hours(home_team) - tz_offset_hours(away_team)


def is_indoor(team: str, schedule_roof: "str | None" = None) -> bool:
    """Prefer the per-game roof state from schedules; fall back to the structural type.

    nflverse roof values: outdoors / dome / closed / open. `closed` and `open` describe a
    retractable roof's state for that game, so `open` counts as outdoors.
    """
    if isinstance(schedule_roof, str) and schedule_roof:
        return schedule_roof in ("dome", "closed")
    s = STADIUMS.get(team)
    return bool(s) and s["roof"] == DOME


def crosscheck_roofs(schedules: pd.DataFrame, since_season: int = 2023) -> pd.DataFrame:
    """Compare this table's roof against schedules.roof for recent home games (§13).

    A mismatch means a franchise moved venues and this dict is stale. Returns one row per
    disagreeing team; an empty frame is the pass condition.
    """
    recent = schedules[
        (schedules["season"] >= since_season)
        & (schedules["location"] == "Home")
        & schedules["roof"].notna()
    ]
    expected_by_structure = {
        OUTDOORS: {"outdoors"},
        DOME: {"dome", "closed"},
        RETRACTABLE: {"closed", "open", "outdoors", "dome"},
    }
    rows = []
    for team, grp in recent.groupby("home_team"):
        observed = set(grp["roof"].unique())
        if team not in STADIUMS:
            rows.append({
                "team": team,
                "table_roof": None,
                "schedule_roof": "/".join(sorted(observed)),
                "issue": "team missing from STADIUMS",
            })
            continue
        structural = STADIUMS[team]["roof"]
        if not observed <= expected_by_structure[structural]:
            rows.append({
                "team": team,
                "table_roof": structural,
                "schedule_roof": "/".join(sorted(observed)),
                "issue": "roof disagreement",
            })
    return pd.DataFrame(rows, columns=["team", "table_roof", "schedule_roof", "issue"])


def venue_for_game(
    home_team: str, location: str, stadium: "str | None" = None
) -> dict:
    """Resolve the venue record for a game, handling neutral-site internationals."""
    if location == "Neutral" and stadium:
        for name, rec in INTERNATIONAL_VENUES.items():
            if name.lower() in str(stadium).lower():
                return rec
    return STADIUMS.get(
        home_team,
        {"lat": 0.0, "lon": 0.0, "tz": "America/New_York", "roof": OUTDOORS, "elev_ft": 0},
    )
