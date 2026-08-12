from __future__ import annotations

import pandas as pd
import pytest

from market_consensus import aggregate_market_consensus


def test_consensus_is_median_and_reports_dispersion_and_book_count():
    lines = pd.DataFrame(
        {
            "game_id": ["g"] * 4,
            "provider": ["A", "B", "C", "C"],
            "spread": [3.5, 4.0, 4.5, 99.0],
            "total": [47.0, 47.5, 48.0, 99.0],
            "observed_at": ["2026-09-01T12:00:00Z"] * 3
            + ["2026-09-01T11:00:00Z"],
        }
    )
    out = aggregate_market_consensus(lines).iloc[0]

    assert out.spread == pytest.approx(4.0)
    assert out.total == pytest.approx(47.5)
    assert out.book_count_spread == 3
    assert out.book_count_total == 3
    assert bool(out.is_consensus) is True
    assert out.spread_dispersion > 0


def test_single_free_line_is_baseline_not_false_consensus():
    lines = pd.DataFrame(
        {"game_id": ["g"], "provider": ["nflverse"], "spread": [3.5],
         "total": [47.0], "observed_at": ["2026-09-01T12:00:00Z"]}
    )
    out = aggregate_market_consensus(lines).iloc[0]
    assert bool(out.is_consensus) is False
    assert out.book_count_spread == 1
