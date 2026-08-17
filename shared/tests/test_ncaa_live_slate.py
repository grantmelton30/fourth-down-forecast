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


def test_ncaa_schedule_falls_back_to_cached_live_payload_without_api_key(tmp_path):
    adapter = _adapter(tmp_path)
    cache = adapter._cache_dir()
    cache.mkdir()
    (cache / "games_2026.json").write_text(json.dumps([{
        "id": 2, "season": 2026, "week": 1, "startDate": "2026-08-29T23:00:00Z",
        "completed": False, "neutralSite": False, "homeTeam": "Home",
        "awayTeam": "Away", "homeConference": "ACC", "awayConference": "Sun Belt",
        "homeClassification": "fbs", "awayClassification": "fbs",
        "homePoints": None, "awayPoints": None, "venueId": 10,
    }]))
    adapter._module = lambda name: (
        SimpleNamespace(load_games=lambda client, seasons: (_ for _ in ()).throw(
            RuntimeError("CFBD_API_KEY not found")))
        if name == "ingest" else
        SimpleNamespace(load_venues=lambda client: pd.DataFrame(),
                        attach_venues=lambda games, venues: games)
        if name == "venues" else
        SimpleNamespace(BudgetedCFBD=lambda cfg: object())
    )

    games = adapter._games_with_venues()

    assert games.loc[0, "game_id"] == 2
    assert games.loc[0, "kickoff"] == pd.Timestamp("2026-08-29T23:00:00Z")


def test_ncaa_raw_books_remain_distinct_consensus_inputs(tmp_path):
    adapter = _adapter(tmp_path)
    cache = adapter._cache_dir()
    cache.mkdir()
    path = cache / "lines_2026.json"
    path.write_text(json.dumps([{
        "id": 2, "season": 2026, "week": 1,
        "lines": [
            {"provider": "Bovada", "spreadOpen": -4.5, "overUnderOpen": 51.5},
            {"provider": "DraftKings", "spreadOpen": -3.5, "overUnderOpen": 52.5},
        ],
    }]))
    rows = adapter.market_snapshots()
    assert list(rows.provider) == ["Bovada", "DraftKings"]
    assert list(rows.spread) == [4.5, 3.5]
    assert rows.observed_at.notna().all()


def test_ncaa_public_market_uses_latest_quote_not_stale_opener(tmp_path):
    adapter = _adapter(tmp_path)
    cache = adapter._cache_dir()
    cache.mkdir()
    (cache / "games_2026.json").write_text(json.dumps([{
        "id": 2, "season": 2026, "week": 1, "startDate": "2026-08-29T23:00:00Z",
        "completed": False, "neutralSite": False, "homeTeam": "Home",
        "awayTeam": "Away", "homeConference": "ACC", "awayConference": "Sun Belt",
        "homeClassification": "fbs", "awayClassification": "fbs",
    }]))
    (cache / "lines_2026.json").write_text(json.dumps([{
        "id": 2, "season": 2026, "week": 1, "homeTeam": "Home", "awayTeam": "Away",
        "homeConference": "ACC", "awayConference": "Sun Belt",
        "homeClassification": "fbs", "awayClassification": "fbs",
        "lines": [{"provider": "Bovada", "spreadOpen": -14.5, "spread": -11.5,
                   "overUnderOpen": 51.5, "overUnder": 49.5}],
    }]))
    game = adapter.games(2026, 1).iloc[0]
    assert game.spread_line == 11.5
    assert game.total_line == 49.5
    assert game.spread_open == 14.5
    assert game.total_open == 51.5


def test_market_without_a_history_file_has_no_completed_games_to_train_on(tmp_path):
    """Regression guard for the 2026-08-17 bug: market.parquet was never written by any
    production code path (only a test fixture wrote it), so NCAAAdapter._market() was
    silently scoped to the live season alone. projection_mean()'s training_market filters
    for completed games with a real actual_margin/actual_total -- which the live season
    never has yet -- so it was permanently empty, fit_live_mean's "insufficient historical
    seasons" check always tripped, and projection_mean() returned None unconditionally for
    every NCAA prediction ever published, not just this test's synthetic scenario.

    This does not re-derive the whole validated-ridge machinery (that needs hundreds of
    realistic training rows); it asserts the specific precondition that machinery depends
    on, which is exactly the thing that silently broke: does _market() ever expose a
    completed, graded game to train on. Without a history file, the honest answer must be
    "no completed games at all" -- not an empty DataFrame masquerading as "not yet
    computed," and not a crash.
    """
    adapter = _adapter(tmp_path)
    cache = adapter._cache_dir()
    cache.mkdir()
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

    completed = market[market.get("completed", False).fillna(False).astype(bool)]
    assert completed.empty, (
        "no market.parquet on disk, so there is no legitimate source of completed "
        "historical games -- any non-empty result here means something is inventing "
        "training data that was never actually validated"
    )


def test_market_with_a_history_file_exposes_completed_games_to_train_on(tmp_path):
    """The other half of the guard above: once market.parquet DOES carry completed,
    graded seasons (what ingest.build_market produces and run_backtest.py now persists,
    per the 2026-08-17 fix), _market() must actually expose them -- this is the
    precondition projection_mean()'s training_market filter depends on."""
    adapter = _adapter(tmp_path)
    cache = adapter._cache_dir()
    cache.mkdir()
    pd.DataFrame([
        {"game_id": 101, "season": 2024, "week": 1, "kickoff": "2024-08-29T00:00:00Z",
         "completed": True, "actual_margin": 14.0, "actual_total": 55.0,
         "spread_open": 3.0, "total_open": 50.0},
        {"game_id": 102, "season": 2025, "week": 1, "kickoff": "2025-08-28T00:00:00Z",
         "completed": True, "actual_margin": -7.0, "actual_total": 44.0,
         "spread_open": -2.5, "total_open": 47.5},
    ]).to_parquet(cache / "market.parquet", index=False)
    (cache / "games_2026.json").write_text("[]")
    (cache / "lines_2026.json").write_text("[]")

    market = adapter._market()

    completed = market[market.get("completed", False).fillna(False).astype(bool)]
    assert sorted(completed["season"].tolist()) == [2024, 2025]
    assert completed["actual_margin"].notna().all()


def test_missing_projection_row_has_consistent_uncertainty_disclosure_shape(tmp_path):
    adapter = _adapter(tmp_path)
    adapter._live_features = lambda: pd.DataFrame({"game_id": []})
    assert adapter.projection_uncertainty("missing") == {
        "multiplier": 1.0, "reasons": (),
    }
