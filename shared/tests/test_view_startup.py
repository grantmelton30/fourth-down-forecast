from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

from sport import NFLAdapter, _select_default_period


def test_default_period_uses_next_upcoming_slate_not_farthest_future_week():
    games = pd.DataFrame({"season": [2026, 2026, 2026], "week": [1, 3, 4],
        "kickoff": ["2026-09-09", "2026-09-24", "2026-10-01"],
        "spread_line": [3.5, 7.5, -2.5]})
    assert _select_default_period(games, "spread_line", now="2026-08-11T12:00:00Z") == (2026, 1)


def test_default_period_falls_back_to_most_recent_completed_slate():
    games = pd.DataFrame({"season": [2025, 2025], "week": [17, 18],
        "kickoff": ["2025-12-28", "2026-01-04"], "spread_line": [3.5, -1.5]})
    assert _select_default_period(games, "spread_line", now="2026-08-11T12:00:00Z") == (2025, 18)


def test_nfl_viewer_reads_completed_ratings_artifact_without_rebuilding(tmp_path):
    expected = pd.DataFrame({"season": [2026], "week": [1], "team": ["CHI"],
        "off_rating": [0.1], "def_rating": [-0.1], "pace_rating": [11.0]})
    path = tmp_path / "walkforward_ratings_default_o750_d550_p10_2016_2026.parquet"
    expected.to_parquet(path, index=False)
    adapter = object.__new__(NFLAdapter); adapter._cache = {}
    adapter._cache_dir = lambda: Path(tmp_path)
    adapter._module = lambda name: (_ for _ in ()).throw(AssertionError(f"viewer attempted to import/build {name}"))
    pd.testing.assert_frame_equal(adapter._walkforward(), expected)


def test_rendering_one_section_does_not_execute_hidden_sections(monkeypatch):
    monkeypatch.setenv("VIEWER_EMBEDDED", "1")
    view = importlib.import_module("view"); called = []
    monkeypatch.setattr(view, "tab_ratings", lambda *args: called.append("Ratings"))
    monkeypatch.setattr(view, "tab_team", lambda *args: called.append("Team"))
    monkeypatch.setattr(view, "tab_game_explorer", lambda *args: called.append("Game explorer"))
    monkeypatch.setattr(view, "tab_diagnostics", lambda *args: called.append("Diagnostics"))
    view._render_section("Ratings", object(), 2026, 1)
    assert called == ["Ratings"]
