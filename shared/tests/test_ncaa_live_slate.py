from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from sport import NCAAAdapter


def _adapter(tmp_path):
    repo = tmp_path / "ncaa-model"
    (repo / "src").mkdir(parents=True)
    (repo / "src/__init__.py").write_text("")
    adapter = NCAAAdapter(repo)
    adapter._cache_dir = lambda: tmp_path / "cache"
    adapter._cfg = lambda: SimpleNamespace(
        seasons=SimpleNamespace(train_start=2019, current=2026),
        all_seasons=list(range(2019, 2027)),
        market=SimpleNamespace(provider_priority=["Bovada", "DraftKings"]),
        is_p5=lambda conference, team: conference in {"ACC", "Big Ten", "Big 12", "SEC"},
    )
    return adapter


def test_ncaa_market_adds_live_raw_cache_to_stale_history(tmp_path):
    adapter = _adapter(tmp_path)
    cache = adapter._cache_dir()
    cache.mkdir()
    pd.DataFrame([{
        "game_id": 1, "season": 2025, "week": 1, "kickoff": "2025-08-28T00:00:00Z",
        "spread_open": 2.5, "total_open": 48.5,
    }]).to_parquet(cache / "market.parquet", index=False)
    (cache / "games_2026.json").write_text(json.dumps([{
        "id": 2, "season": 2026, "week": 1, "startDate": "2026-08-29T23:00:00Z",
        "completed": False, "neutralSite": False, "homeTeam": "Home",
        "awayTeam": "Away", "homeConference": "ACC", "awayConference": "Sun Belt",
        "homeClassification": "fbs", "awayClassification": "fbs",
        "homePoints": None, "awayPoints": None,
    }]))
    (cache / "lines_2026.json").write_text(json.dumps([{
        "id": 2, "season": 2026, "week": 1, "homeTeam": "Home", "awayTeam": "Away",
        "homeConference": "ACC", "awayConference": "Sun Belt",
        "homeClassification": "fbs", "awayClassification": "fbs",
        "homeScore": None, "awayScore": None,
        "lines": [{"provider": "Bovada", "spreadOpen": -4.5,
                   "spread": -5.0, "overUnderOpen": 51.5, "overUnder": 52.0}],
    }]))

    market = adapter._market()

    live = market.loc[market["season"].eq(2026)].iloc[0]
    assert live.game_id == 2
    assert live.spread_open == 4.5
    assert live.total_open == 51.5
    assert live.kickoff == pd.Timestamp("2026-08-29T23:00:00Z")


def test_ncaa_schedule_requests_configured_live_season(tmp_path):
    adapter = _adapter(tmp_path)
    seen = {}
    adapter._module = lambda name: (
        SimpleNamespace(load_games=lambda client, seasons: seen.setdefault("seasons", seasons) or pd.DataFrame())
        if name == "ingest" else
        SimpleNamespace(load_venues=lambda client: pd.DataFrame(),
                        attach_venues=lambda games, venues: games)
        if name == "venues" else
        SimpleNamespace(BudgetedCFBD=lambda cfg: object())
    )

    adapter._games_with_venues()

    assert seen["seasons"][-1] == 2026
