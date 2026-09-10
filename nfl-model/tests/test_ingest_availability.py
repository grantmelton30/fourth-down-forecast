from __future__ import annotations

import pandas as pd
import pytest

from src import ingest


def test_current_season_404_falls_back_to_completed_seasons(monkeypatch):
    calls = []

    def cached(name, seasons, refresh, fetch, expected_cols=None):
        calls.append((name, list(seasons)))
        if max(seasons) == 2026:
            raise RuntimeError(
                "404 Client Error: Not Found for url: "
                "https://example.test/snap_counts_2026.parquet"
            )
        return pd.DataFrame({"season": [2025]})

    monkeypatch.setattr(ingest, "_cached", cached)
    monkeypatch.setattr("nflreadpy.get_current_season", lambda: 2026)
    result = ingest._cached_with_current_fallback(
        "snap_counts", [2024, 2025, 2026], False, lambda seasons: None)

    assert calls == [
        ("snap_counts", [2024, 2025, 2026]),
        ("snap_counts", [2024, 2025]),
    ]
    assert result["season"].tolist() == [2025]


@pytest.mark.parametrize("message", [
    "503 Server Error for snap_counts_2026.parquet",
    "404 Client Error for snap_counts_2025.parquet",
    "source returned zero rows",
])
def test_fallback_does_not_hide_other_source_failures(monkeypatch, message):
    monkeypatch.setattr(
        ingest, "_cached",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(message)),
    )
    monkeypatch.setattr("nflreadpy.get_current_season", lambda: 2026)

    with pytest.raises(RuntimeError, match=message):
        ingest._cached_with_current_fallback(
            "snap_counts", [2025, 2026], False, lambda seasons: None)
