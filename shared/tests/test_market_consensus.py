"""Single-book market selection (shared/market_consensus.py).

This used to take a median across books. With two books that is the midpoint, and the
midpoint of 54 and 53.5 is 53.75 -- a line no book offers and nobody can bet. It produced
synthetic quarter-point totals on 21 of 95 priced college games. These tests exist to make
sure the quoted number is always one a book actually posted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_consensus import DEFAULT_PROVIDER_PRIORITY, aggregate_market_consensus

WHEN = "2026-09-01T12:00:00Z"


def _lines(rows):
    return pd.DataFrame(
        [{"game_id": "g", "provider": p, "spread": s, "total": t,
          "observed_at": WHEN} for p, s, t in rows])


def test_the_quoted_number_is_a_real_book_quote_never_a_blend():
    """The whole point. Bovada 53.5 and DraftKings 54.0 must yield 53.5 or 54.0 -- never
    53.75, which is what the median produced and which nobody can bet."""
    out = aggregate_market_consensus(
        _lines([("DraftKings", 4.0, 54.0), ("Bovada", 3.5, 53.5)])).iloc[0]
    assert out.total in (53.5, 54.0)
    assert out.spread in (3.5, 4.0)
    assert out.total != pytest.approx(53.75)


def test_bovada_is_preferred_over_draftkings():
    out = aggregate_market_consensus(
        _lines([("DraftKings", 4.0, 54.0), ("Bovada", 3.5, 53.5)])).iloc[0]
    assert out.chosen_provider == "Bovada"
    assert (out.spread, out.total) == (3.5, 53.5)


def test_it_falls_down_the_priority_list_when_the_top_book_is_absent():
    out = aggregate_market_consensus(
        _lines([("DraftKings", 4.0, 54.0), ("ESPN Bet", 3.0, 52.0)])).iloc[0]
    assert out.chosen_provider == "ESPN Bet", "second in priority beats third"

    only_dk = aggregate_market_consensus(_lines([("DraftKings", 4.0, 54.0)])).iloc[0]
    assert only_dk.chosen_provider == "DraftKings"


def test_spread_and_total_always_come_from_the_SAME_book():
    """Taking the spread from one book and the total from another rebuilds the merged line
    this module exists to stop producing. A book quoting only half a game is skipped."""
    out = aggregate_market_consensus(
        _lines([("Bovada", 3.5, np.nan), ("DraftKings", 4.0, 54.0)])).iloc[0]
    assert out.chosen_provider == "DraftKings"
    assert (out.spread, out.total) == (4.0, 54.0), "not Bovada's spread with DK's total"


def test_an_unlisted_book_is_used_only_when_nothing_listed_is_available():
    out = aggregate_market_consensus(
        _lines([("MysteryBook", 7.0, 60.0), ("Bovada", 3.5, 53.5)])).iloc[0]
    assert out.chosen_provider == "Bovada"
    solo = aggregate_market_consensus(_lines([("MysteryBook", 7.0, 60.0)])).iloc[0]
    assert solo.chosen_provider == "MysteryBook"


def test_only_the_latest_snapshot_per_book_counts():
    lines = pd.DataFrame([
        {"game_id": "g", "provider": "Bovada", "spread": 3.5, "total": 53.5,
         "observed_at": "2026-09-01T12:00:00Z"},
        {"game_id": "g", "provider": "Bovada", "spread": 9.9, "total": 99.0,
         "observed_at": "2026-09-01T11:00:00Z"},
    ])
    out = aggregate_market_consensus(lines).iloc[0]
    assert (out.spread, out.total) == (3.5, 53.5), "the stale 11:00 quote must not win"


def test_dispersion_still_reports_disagreement_without_moving_the_number():
    """Knowing the books were 0.5 apart is useful. Letting that change the quote is not."""
    out = aggregate_market_consensus(
        _lines([("Bovada", 3.5, 53.5), ("DraftKings", 4.0, 54.0)])).iloc[0]
    assert out.total_dispersion == pytest.approx(0.5)
    assert out.book_count_total == 2
    assert out.total == 53.5, "dispersion is a diagnostic, not an input"


def test_a_single_book_is_not_a_false_consensus():
    out = aggregate_market_consensus(_lines([("nflverse", 3.5, 47.0)])).iloc[0]
    assert bool(out.is_consensus) is False
    assert out.book_count_spread == 1


def test_no_priced_book_yields_no_number_rather_than_a_guess():
    out = aggregate_market_consensus(
        _lines([("Bovada", np.nan, np.nan)])).iloc[0]
    assert out.chosen_provider is None
    assert np.isnan(out.spread) and np.isnan(out.total)


def test_the_priority_matches_what_the_ncaa_config_declares():
    """ncaa-model/config/ncaa.yaml sets provider_priority and ingest.build_market -- which
    the P1 tracker grades against -- already uses it. If these drift, the site would show a
    different book's line than the one the tracker bet."""
    assert DEFAULT_PROVIDER_PRIORITY == ("Bovada", "ESPN Bet", "DraftKings")
