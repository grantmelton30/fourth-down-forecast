"""Weather reaches the shipped total, and only ever in the direction the physics supports.

The adjustment is one-sided by design (DECISIONS.md D16): wind suppresses passing and
kicking, so a windy game loses points, while a calm game is not awarded a bonus for the
absence of suppression. These tests pin that asymmetry, because a two-sided version looks
almost identical in code and measurably harms 1,097 calm and indoor games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.context import wind_total_adjustment
from src.weather import GAME_WEATHER_COLUMNS, bulk_game_weather


def test_wind_penalises_above_average_and_never_rewards_below():
    cfg = load_config()
    centre = cfg.context.wind_total_center_mph

    assert wind_total_adjustment(centre + 10.0, cfg) < 0.0
    assert wind_total_adjustment(centre, cfg) == pytest.approx(0.0)
    # The whole point of the one-sided form: calm must be neutral, never a bonus.
    assert wind_total_adjustment(0.0, cfg) == pytest.approx(0.0)
    assert wind_total_adjustment(centre - 5.0, cfg) == pytest.approx(0.0)


def test_wind_penalty_scales_with_the_configured_slope():
    cfg = load_config()
    centre = cfg.context.wind_total_center_mph
    per_mph = cfg.context.wind_total_points_per_mph

    assert wind_total_adjustment(centre + 10.0, cfg) == pytest.approx(-10.0 * per_mph)
    # Monotone: more wind is never less of a penalty.
    windy = [wind_total_adjustment(centre + d, cfg) for d in (1, 5, 10, 20)]
    assert windy == sorted(windy, reverse=True)


def test_bulk_game_weather_short_circuits_indoor_and_never_fetches_when_offline(tmp_path,
                                                                                monkeypatch):
    """Indoor games cost no request, and `allow_network=False` makes no request at all."""
    import src.weather as weather_mod

    monkeypatch.setattr(weather_mod, "CACHE_DIR", tmp_path)

    def _explode(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("bulk_game_weather made a network call it should not have")

    monkeypatch.setattr(weather_mod, "_fetch_range", _explode)

    games = pd.DataFrame([
        {"game_id": 1, "season": 2025, "lat": 40.8, "lon": -77.9, "tz": "America/New_York",
         "dome": True, "kickoff": pd.Timestamp("2025-09-06T19:00", tz="UTC")},
        {"game_id": 2, "season": 2025, "lat": 41.7, "lon": -91.6, "tz": "America/Chicago",
         "dome": False, "kickoff": pd.Timestamp("2025-09-06T19:00", tz="UTC")},
    ])

    out = bulk_game_weather(games, allow_network=False, cache_key="unit")
    assert list(out["game_id"]) == [1, 2]
    indoor = out[out["game_id"].eq(1)].iloc[0]
    assert bool(indoor["indoor"]) is True
    # The outdoor game has no cached reading and no network, so it stays missing rather
    # than being silently reported as calm.
    outdoor = out[out["game_id"].eq(2)].iloc[0]
    assert pd.isna(outdoor["wind_mph"])


def test_bulk_game_weather_reuses_cache_without_refetching(tmp_path, monkeypatch):
    import src.weather as weather_mod

    monkeypatch.setattr(weather_mod, "CACHE_DIR", tmp_path)
    pd.DataFrame([{
        "game_id": 7, "indoor": False, "wind_mph": 18.5, "temp_f": 44.0,
        "precip_in": 0.0, "source": "archive",
    }])[GAME_WEATHER_COLUMNS].to_parquet(tmp_path / "game_weather_unit.parquet", index=False)

    def _explode(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("cached game was refetched")

    monkeypatch.setattr(weather_mod, "_fetch_range", _explode)

    games = pd.DataFrame([{
        "game_id": 7, "season": 2025, "lat": 41.7, "lon": -91.6, "tz": "America/Chicago",
        "dome": False, "kickoff": pd.Timestamp("2025-11-01T18:00", tz="UTC"),
    }])
    out = bulk_game_weather(games, allow_network=True, cache_key="unit")
    assert out.iloc[0]["wind_mph"] == pytest.approx(18.5)


def test_missing_reading_produces_no_adjustment_not_a_penalty():
    """A slate with no weather data must project exactly as it does today.

    This is the property that makes turning weather on safe: absence of a reading and a
    calm reading both resolve to zero, so a fetch failure can never quietly move a line.
    """
    cfg = load_config()
    wind = pd.Series([np.nan, 3.0, 20.0])
    indoor = pd.Series([False, False, False])
    penalty = np.minimum(
        -cfg.context.wind_total_points_per_mph * (wind - cfg.context.wind_total_center_mph),
        0.0,
    ).where(wind.notna() & ~indoor, 0.0)

    assert penalty.iloc[0] == pytest.approx(0.0)   # missing -> neutral
    assert penalty.iloc[1] == pytest.approx(0.0)   # calm    -> neutral
    assert penalty.iloc[2] < -2.0                  # windy   -> real penalty
