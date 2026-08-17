"""Regression guard for the 2026-08-17 dead-endgame-table bug.

fit_endgame_table's whole job is to produce the score-conditioned drive-outcome
probabilities that give the simulator its 3/7-point margin spikes. When it was handed a
drives frame with zero rows in the "late game" window, it silently built and cached a
table with counts.sum() == 0 and every probability NaN -- and _play_drive's
`(u[:, None] > cum).sum(axis=1)` resolves an all-NaN row to index 0, which is TD, so every
endgame-masked drive (a team's final 1-2 drives of every simulated game) was resolving to
a guaranteed touchdown instead of the scoreboard-conditioned outcome this table exists to
produce. No test caught this: it takes real, cached, multi-season drive data to exercise
in the ordinary course of things, and a broken table looks exactly like a working one
until you inspect its counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim_core import DRIVE_CLASSES, fit_endgame_table


def _late_drives(n_per_bucket: int = 30) -> pd.DataFrame:
    """A minimal but real drives frame: several score buckets, each with enough late-Q4
    drives to clear fit_endgame_table's own <100-observations pooled-fallback threshold
    is not required here -- this exercises the "does it produce a live table at all" path,
    not the per-bucket confidence path."""
    rng = np.random.default_rng(0)
    rows = []
    for season in (2019, 2020, 2021, 2022):
        for score_diff in (-15, -6, -2, 0, 2, 6, 15):
            for _ in range(n_per_bucket):
                rows.append({
                    "season": season,
                    "start_qtr": 4,
                    "start_gsr": float(rng.integers(0, 300)),
                    "start_score_diff": float(score_diff),
                    "result": rng.choice(DRIVE_CLASSES),
                })
    return pd.DataFrame(rows)


def test_fit_endgame_table_raises_rather_than_caching_a_degenerate_table(tmp_path):
    empty_of_late_drives = pd.DataFrame({
        "season": [2020, 2021],
        "start_qtr": [1, 2],  # first half only -- never matches the "late game" filter
        "start_gsr": [1200.0, 900.0],
        "start_score_diff": [0.0, 3.0],
        "result": ["TD", "FG"],
    })
    cache_path = tmp_path / "endgame_test.json"

    with pytest.raises(ValueError, match="0 late-game drives"):
        fit_endgame_table(
            empty_of_late_drives, train_start=2019, as_of_season=2022,
            cache_path=cache_path,
        )

    assert not cache_path.exists(), (
        "a degenerate fit must never reach the cache -- writing it here is exactly what "
        "let a NaN table get served to every simulated game for the rest of this "
        "codebase's life"
    )


def test_fit_endgame_table_produces_a_real_table_from_real_late_drives(tmp_path):
    cache_path = tmp_path / "endgame_test.json"

    table = fit_endgame_table(
        _late_drives(), train_start=2019, as_of_season=2022, cache_path=cache_path,
    )

    assert table.counts.sum() > 0
    assert np.isfinite(table.cum).all()
    assert cache_path.exists()


def test_a_corrupted_cache_on_disk_self_heals_instead_of_being_trusted_forever(tmp_path):
    """The other half of the fix: `path.exists()` was the only freshness check before
    2026-08-17, so a table corrupted once (by any means -- not just this exact bug) would
    be served forever. Simulates that corrupted state directly and confirms a validated
    fetch route refits instead of trusting it."""
    cache_path = tmp_path / "endgame_test.json"
    # Hand-write a corrupted cache exactly like the one this bug produced: valid JSON,
    # present on disk, but zero total drives and every probability NaN.
    import json
    cache_path.write_text(json.dumps({
        "cum": [[float("nan")] * len(DRIVE_CLASSES) for _ in range(7)],
        "counts": [0] * 7,
        "classes": list(DRIVE_CLASSES),
    }))

    table = fit_endgame_table(
        _late_drives(), train_start=2019, as_of_season=2022, cache_path=cache_path,
    )

    assert table.counts.sum() > 0, (
        "a corrupted on-disk cache must be treated as absent and rebuilt, not trusted "
        "just because the file exists"
    )
