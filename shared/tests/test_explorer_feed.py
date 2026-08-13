from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_explorer", ROOT / "scripts/build_explorer.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Adapter:
    profile = SimpleNamespace(key="ncaa")

    def default_period(self):
        return 2026, 1

    def ratings(self, season, week):
        return pd.DataFrame({
            "off_rating": [.2, -.1, 0], "def_rating": [-.1, .1, 0],
            "pace_rating": [1., -1., 0], "net_rating": [.3, -.2, 0],
            "off_rank": [1, 2, 3], "def_rank": [1, 2, 3], "net_rank": [1, 2, 3],
            "n_games": [12, 12, 200], "conference": ["A", "B", "FCS"],
        }, index=pd.Index(["Alpha", "Beta", "__FCS__"], name="team"))

    def season_schedule(self, season):
        return pd.DataFrame([{
            "game_id": 1, "season": 2026, "week": 1,
            "kickoff": "2026-08-29T16:00:00Z", "homeTeam": "Alpha",
            "awayTeam": "Beta", "homePoints": None, "awayPoints": None,
            "completed": False,
        }])


def test_explorer_feed_contains_rankings_and_team_schedule():
    feed = MODULE.build_league(Adapter())
    assert feed["season"] == 2026
    assert feed["ratings"][0]["team"] == "Alpha"
    assert feed["ratings"][0]["group"] == "A"
    assert feed["ratings"][0]["net_rank"] == 1
    assert feed["schedule"][0]["home_team"] == "Alpha"
    assert feed["summary"]["rated_teams"] == 2
    assert all(row["team"] != "__FCS__" for row in feed["ratings"])


def test_nfl_team_alignment_exposes_division_for_projection_filters():
    assert MODULE.competition_group("nfl", "DAL", None) == "NFC East"
    assert MODULE.competition_group("nfl", "BUF", None) == "AFC East"
