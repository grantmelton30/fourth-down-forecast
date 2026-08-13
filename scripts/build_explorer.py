#!/usr/bin/env python3
"""Build the static league/team explorer artifact from completed model caches."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from sport import load_adapter  # noqa: E402


NFL_DIVISIONS = {
    "AFC East": ("BUF", "MIA", "NE", "NYJ"),
    "AFC North": ("BAL", "CIN", "CLE", "PIT"),
    "AFC South": ("HOU", "IND", "JAX", "TEN"),
    "AFC West": ("DEN", "KC", "LAC", "LV"),
    "NFC East": ("DAL", "NYG", "PHI", "WAS"),
    "NFC North": ("CHI", "DET", "GB", "MIN"),
    "NFC South": ("ATL", "CAR", "NO", "TB"),
    "NFC West": ("ARI", "LA", "SEA", "SF"),
}
NFL_TEAM_DIVISION = {
    team: division for division, teams in NFL_DIVISIONS.items() for team in teams
}


def competition_group(league: str, team: str, conference) -> str | None:
    """Public conference/division label used for browsing, never model fitting."""
    if league == "nfl":
        return NFL_TEAM_DIVISION.get(str(team))
    return None if pd.isna(conference) else str(conference)


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def build_league(adapter) -> dict:
    period = adapter.default_period()
    if period is None:
        return {"season": None, "week": None, "summary": {"rated_teams": 0},
                "ratings": [], "schedule": []}
    season, week = map(int, period)
    ratings = adapter.ratings(season, week).reset_index()
    # Internal pooled fallback buckets support unseen opponents in the simulator;
    # they are not teams and must never appear in the public selector/rankings.
    if "team" in ratings:
        ratings = ratings[~ratings["team"].astype(str).str.match(r"^__.*__$")].copy()
        league = str(adapter.profile.key).lower()
        ratings["group"] = [
            competition_group(league, team, conference)
            for team, conference in zip(
                ratings["team"], ratings.get("conference", pd.Series(None, index=ratings.index))
            )
        ]
    rating_columns = [
        column for column in (
            "team", "conference", "group", "off_rating", "def_rating", "pace_rating",
            "net_rating", "off_rank", "def_rank", "net_rank", "pace_rank",
            "net_change", "n_games",
        ) if column in ratings
    ]
    rating_rows = [
        {key: _json_value(value) for key, value in row.items()}
        for row in ratings[rating_columns].to_dict("records")
    ]
    schedule = adapter.season_schedule(season)
    renames = {
        "homeTeam": "home_team", "awayTeam": "away_team",
        "homePoints": "home_points", "awayPoints": "away_points",
        "neutralSite": "neutral_site",
    }
    schedule = schedule.rename(columns=renames)
    schedule_columns = [
        column for column in (
            "game_id", "season", "week", "kickoff", "home_team", "away_team",
            "home_points", "away_points", "completed", "neutral_site",
            "spread_line", "total_line",
        ) if column in schedule
    ]
    schedule_rows = [
        {key: _json_value(value) for key, value in row.items()}
        for row in schedule[schedule_columns].to_dict("records")
    ]
    return {
        "season": season, "week": week,
        "summary": {
            "rated_teams": len(rating_rows), "scheduled_games": len(schedule_rows),
            "completed_games": sum(bool(row.get("completed")) for row in schedule_rows),
        },
        "ratings": rating_rows, "schedule": schedule_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "web/api/v1/explorer.json")
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "leagues": {},
    }
    for league in ("nfl", "ncaa"):
        adapter = load_adapter(league, ROOT)
        payload["leagues"][league] = build_league(adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"published explorer -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
