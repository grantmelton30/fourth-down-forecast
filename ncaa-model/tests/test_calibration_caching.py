"""Per-season caching for `attach_calibration` (GATES.md 2026-08-18).

`_season_calibration`'s simulation loop itself needs real ratings and a real drive table
to run at all -- like `pooled_margin_pmf`/`attach_calibration`'s own docstrings say, that
half is validated by a real `run_backtest.py` rebuild, not a synthetic fixture here. What
*is* fully unit-testable without touching the simulator is the actual bug being fixed: the
per-season cache signature must depend only on data at or before that season, so a change
to a later season can never invalidate an earlier one's cache, while a change to the
season's own data still correctly does.

Tested by pre-populating a season's cache file with a known signature, monkeypatching the
simulator-driving calls to raise if invoked, and confirming `_season_calibration` returns
the cached result (proving it took the cache-hit path and never touched the simulator) --
or correctly recomputes (raises, since the mocks explode) when it shouldn't have hit cache.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from src import backtest
from src.config import (build_cache_signature, frame_signature, load_config,
                        write_cached_frame)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "CACHE_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _explode_if_simulated(monkeypatch):
    """Any test that reaches the real simulation path (a cache miss) must say so
    explicitly by NOT using this fixture's guards -- by default, every simulator-driving
    call raises, so a passing test proves the cache-hit path was taken."""
    def _boom(*a, **k):
        raise AssertionError("simulator-driving call reached on what should have been a cache hit")
    monkeypatch.setattr(backtest, "fit_start_field_position", _boom)
    monkeypatch.setattr(backtest, "fit_drive_model", _boom)
    monkeypatch.setattr(backtest, "fit_endgame_table", _boom)
    monkeypatch.setattr(backtest, "estimate_venue_hfa", _boom)
    monkeypatch.setattr(backtest, "simulate_game", _boom)


def _games(seasons):
    return pd.DataFrame([{
        "game_id": f"g{s}", "season": s, "week": 5, "homeTeam": "A", "awayTeam": "B",
        "neutralSite": False, "homeClassification": "fbs", "awayClassification": "fbs",
        "kickoff": pd.Timestamp(f"{s}-09-01", tz="UTC"),
    } for s in seasons])


def _drive_table(seasons):
    return pd.DataFrame([{"season": s, "game_id": f"g{s}", "offense": "A"} for s in seasons])


def _walkforward(seasons):
    return pd.DataFrame([{
        "season": s, "week": 5, "as_of": pd.Timestamp(f"{s}-09-01", tz="UTC"),
        "team": "A", "off_rating": 0.01 * s, "def_rating": -0.01 * s, "pace_rating": 0.0,
    } for s in seasons])


def _targets(season):
    return pd.DataFrame([{
        "game_id": f"g{season}", "season": season, "week": 5,
        "model_spread": 3.0, "model_total": 45.0,
        "spread_close": 2.5, "total_close": 44.5,
        "actual_margin": 4.0, "actual_total": 46.0,
    }])


def _prepopulate(tmp_path, cfg, season, games, drive_table, walkforward, cache_key="default",
                 season_targets=None):
    """Write a cache file with the exact signature `_season_calibration` would compute,
    so a real call against identical inputs is guaranteed to hit it. `season_targets`
    defaults to a freshly-built one-row frame (fine when the caller also calls
    `_season_calibration` directly with that same fresh frame), but callers going through
    `attach_calibration`'s real groupby pipeline must pass the actual group slice --
    `frame_signature` hashes the row index too, and a groupby slice does not reset it."""
    season_targets = _targets(season) if season_targets is None else season_targets
    at_or_before = games["season"] <= season
    signature = build_cache_signature(builder=Path(backtest.__file__), config=asdict(cfg), inputs={
        "season_targets": frame_signature(season_targets, [
            "game_id", "season", "week", "model_spread", "model_total",
            "spread_close", "total_close", "actual_margin", "actual_total"]),
        "games": frame_signature(games[at_or_before], [
            "game_id", "season", "week", "homeTeam", "awayTeam", "neutralSite",
            "homeClassification", "awayClassification"]),
        "drive_table": frame_signature(drive_table[drive_table["season"] <= season]),
        "walkforward": frame_signature(walkforward[walkforward["season"] <= season], [
            "season", "week", "as_of", "team", "off_rating", "def_rating", "pace_rating"]),
    }, artifact_version=1)
    path = tmp_path / f"backtest_calibration_{cache_key}_s{season}.parquet"
    fake = pd.DataFrame([{"game_id": f"g{season}", "cover_prob_home": 0.5, "over_prob": 0.5}])
    write_cached_frame(fake, path, signature)
    return fake


def test_cache_path_is_named_per_season(tmp_path):
    cfg = load_config()
    seasons = [2023, 2024]
    games, drives, wf = _games(seasons), _drive_table(seasons), _walkforward(seasons)
    _prepopulate(tmp_path, cfg, 2023, games, drives, wf)
    assert (tmp_path / "backtest_calibration_default_s2023.parquet").exists()


def test_a_completed_seasons_cache_is_unaffected_by_a_later_seasons_data(tmp_path):
    """The actual bug: a change to season 2024's data must never invalidate 2023's cache."""
    cfg = load_config()
    seasons = [2023, 2024]
    games, drives, wf = _games(seasons), _drive_table(seasons), _walkforward(seasons)
    _prepopulate(tmp_path, cfg, 2023, games, drives, wf)

    # Mutate ONLY the later season's rows -- a new week graded, a rating recomputed.
    games_changed = games.copy()
    games_changed.loc[games_changed["season"] == 2024, "homeTeam"] = "Z"
    wf_changed = wf.copy()
    wf_changed.loc[wf_changed["season"] == 2024, "off_rating"] = 9.99
    drives_changed = drives.copy()

    games_idx = games_changed.set_index("game_id")
    result = backtest._season_calibration(
        2023, _targets(2023), cfg, games_changed, drives_changed, games_idx,
        wf_changed, "default",
    )
    assert result["game_id"].tolist() == ["g2023"]
    assert result["cover_prob_home"].iloc[0] == pytest.approx(0.5)


def test_a_seasons_own_data_change_correctly_invalidates_its_cache(tmp_path, monkeypatch):
    """Safeguard: this must not degrade into never re-simulating anything."""
    cfg = load_config()
    seasons = [2023, 2024]
    games, drives, wf = _games(seasons), _drive_table(seasons), _walkforward(seasons)
    _prepopulate(tmp_path, cfg, 2023, games, drives, wf)

    wf_changed = wf.copy()
    wf_changed.loc[wf_changed["season"] == 2023, "off_rating"] = 9.99
    games_idx = games.set_index("game_id")

    with pytest.raises(AssertionError, match="simulator-driving call"):
        backtest._season_calibration(
            2023, _targets(2023), cfg, games, drives, games_idx, wf_changed, "default",
        )


def test_attach_calibration_hits_per_season_caches_without_touching_the_simulator(
    tmp_path, monkeypatch,
):
    """End-to-end through the real `attach_calibration` groupby pipeline, not
    `_season_calibration` called directly. `frame_signature` hashes with the row index
    included (`pd.util.hash_pandas_object(..., index=True)`), and `targets.groupby("season")`
    preserves each row's position from the concatenated frame rather than resetting to a
    fresh 0-based index per group -- so the pre-populated cache must be built from a
    `season_targets` slice taken the same way, not a freshly-constructed one-row frame,
    or the two will only happen to match for whichever season lands first."""
    cfg = load_config()
    seasons = [2023, 2024]
    games, drives, wf = _games(seasons), _drive_table(seasons), _walkforward(seasons)
    frame = pd.concat([_targets(s) for s in seasons], ignore_index=True)
    targets = frame.dropna(subset=["model_spread", "model_total"])
    for season, season_targets in targets.groupby("season"):
        _prepopulate(tmp_path, cfg, int(season), games, drives, wf,
                     season_targets=season_targets)
    # `drives_raw` here isn't real drive-shaped data -- attach_calibration's own
    # build_drive_table transform isn't what this test is about, so hand it the
    # already-built table directly.
    monkeypatch.setattr(backtest, "build_drive_table", lambda drives_raw, games_: drives)

    result = backtest.attach_calibration(frame, cfg, games, drives, wf, cache_key="default")

    assert set(result["game_id"]) == {"g2023", "g2024"}
    assert (result["cover_prob_home"] == 0.5).all()
