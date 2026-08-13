"""Objective 1: a durable append-only ledger of timestamped market observations.

Timestamped quotes are the one artifact in this project that cannot be reconstructed
after the fact. A model can be refitted from cached data whenever we like; a line that
moved on a Tuesday and was never written down is gone. So capture is separated from
interpretation here: the ledger records exactly what a provider showed and when it was
seen, and every derived view -- opener, close, median, dispersion, movement -- is computed
from those rows later, because derivation is repeatable and observation is not.

Three distinctions the ledger must never blur:
  * a quote that exists, versus no line posted yet, versus the source failing;
  * a spread-only quote, which stays usable, versus a missing observation;
  * the latest quote we happen to hold, versus a genuine closing quote.
"""
from __future__ import annotations

import pandas as pd
import pytest

from market_ledger import (MarketQuote, MarketQuoteLedger, QuotePhase, QuoteStatus,
                           derive_snapshots)


def _quote(**overrides):
    base = dict(
        league="ncaa", season=2026, week=1, game_id="401856766",
        home_team="TCU", away_team="North Carolina",
        kickoff="2026-08-29T16:00:00Z", provider="Bovada",
        spread=-7.5, total=50.5,
        observed_at="2026-08-13T18:00:00Z", source_id="cfbd/lines",
        collector_version="market-collector/1",
    )
    base.update(overrides)
    return MarketQuote(**base)


# -- append-only capture -----------------------------------------------------------------

def test_two_observations_of_a_changed_line_are_both_retained(tmp_path):
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    assert ledger.append(_quote()) is True
    assert ledger.append(_quote(spread=-8.0,
                                observed_at="2026-08-14T18:00:00Z")) is True

    rows = ledger.records()
    assert [r.spread for r in rows] == [-7.5, -8.0]


def test_replaying_the_same_payload_creates_no_duplicate(tmp_path):
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    assert ledger.append(_quote()) is True
    assert ledger.append(_quote()) is False
    assert len(ledger.records()) == 1


def test_provider_disagreement_is_preserved_not_collapsed(tmp_path):
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    ledger.append(_quote(provider="Bovada", spread=-7.5))
    ledger.append(_quote(provider="DraftKings", spread=-7.0))
    ledger.append(_quote(provider="Pinnacle", spread=-8.0))

    assert sorted(r.spread for r in ledger.records()) == [-8.0, -7.5, -7.0]


def test_an_earlier_quote_is_never_overwritten(tmp_path):
    """Same provider, same observation time, contradictory number: keep both."""
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    ledger.append(_quote(spread=-7.5))
    ledger.append(_quote(spread=-9.5))

    assert [r.spread for r in ledger.records()] == [-7.5, -9.5]


def test_an_unchanged_price_seen_again_is_not_re_recorded(tmp_path):
    """Polling daily must not write "still -7.5" hundreds of thousands of times."""
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    assert ledger.append_if_changed(_quote(observed_at="2026-08-13T18:00:00Z")) is True
    assert ledger.append_if_changed(_quote(observed_at="2026-08-14T18:00:00Z")) is False
    assert len(ledger.records()) == 1


def test_a_price_that_returns_to_an_earlier_value_is_recorded(tmp_path):
    """Comparison is against the LAST quote for that game and provider, not all history.

    Hashing on price alone would silently discard a line that moved away and came back,
    which is precisely the movement a closing-line study cares about.
    """
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    ledger.append_if_changed(_quote(spread=-7.5, observed_at="2026-08-13T18:00:00Z"))
    ledger.append_if_changed(_quote(spread=-8.0, observed_at="2026-08-14T18:00:00Z"))
    ledger.append_if_changed(_quote(spread=-7.5, observed_at="2026-08-15T18:00:00Z"))

    assert [r.spread for r in ledger.records()] == [-7.5, -8.0, -7.5]


def test_change_detection_is_per_provider_and_per_phase(tmp_path):
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    ledger.append_if_changed(_quote(provider="Bovada", spread=-7.5))
    # A different book showing the same number is a separate observation.
    assert ledger.append_if_changed(_quote(provider="Pinnacle", spread=-7.5)) is True
    # As is the opener, which is a different claim from the current price.
    assert ledger.append_if_changed(
        _quote(provider="Bovada", spread=-7.5, quote_phase=QuotePhase.OPENING)) is True


# -- partial quotes ----------------------------------------------------------------------

def test_a_spread_only_quote_is_retained(tmp_path):
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    assert ledger.append(_quote(total=None)) is True
    row = ledger.records()[0]
    assert row.spread == -7.5 and row.total is None
    assert row.status is QuoteStatus.QUOTED


def test_a_totally_empty_quote_is_rejected_as_meaningless(tmp_path):
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    with pytest.raises(ValueError, match="no price"):
        ledger.append(_quote(spread=None, total=None))


# -- absence is not failure --------------------------------------------------------------

def test_no_line_posted_is_distinct_from_a_source_failure(tmp_path):
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    ledger.append(MarketQuote.no_line_posted(
        league="ncaa", season=2026, week=0, game_id="g0",
        home_team="A", away_team="B", kickoff="2026-08-23T16:00:00Z",
        provider="Bovada", observed_at="2026-08-13T18:00:00Z",
        source_id="cfbd/lines", collector_version="market-collector/1"))
    ledger.append(MarketQuote.source_failure(
        league="ncaa", season=2026, week=0, game_id="g0",
        home_team="A", away_team="B", kickoff="2026-08-23T16:00:00Z",
        provider="Bovada", observed_at="2026-08-13T19:00:00Z",
        source_id="cfbd/lines", collector_version="market-collector/1",
        detail="HTTP 503"))

    statuses = [r.status for r in ledger.records()]
    assert statuses == [QuoteStatus.NO_LINE_POSTED, QuoteStatus.SOURCE_FAILURE]
    # Neither is a price, so neither may ever be read as one.
    assert all(r.spread is None and r.total is None for r in ledger.records())


# -- normalization -----------------------------------------------------------------------

def test_week_zero_is_a_real_week_not_a_missing_value(tmp_path):
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    ledger.append(_quote(week=0))
    assert ledger.records()[0].week == 0


def test_spread_is_stored_from_the_home_perspective(tmp_path):
    """A home favourite is negative, matching the documented convention."""
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    ledger.append(_quote(spread=-7.5))
    assert ledger.records()[0].spread == -7.5

    flipped = MarketQuote.from_away_perspective(_quote(spread=None, total=48.0),
                                                away_spread=+3.5)
    assert flipped.spread == -3.5


def test_neutral_site_games_are_marked_and_keep_a_home_label(tmp_path):
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    ledger.append(_quote(neutral_site=True))
    row = ledger.records()[0]
    assert row.neutral_site is True
    assert row.home_team == "TCU"


# -- the close must be earned ------------------------------------------------------------

def test_the_latest_quote_is_not_assumed_to_be_the_close(tmp_path):
    """A quote is only 'closing' if it was observed inside the documented cutoff."""
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    ledger.append(_quote(observed_at="2026-08-27T18:00:00Z"))   # two days out
    ledger.append(_quote(spread=-8.0, observed_at="2026-08-29T15:00:00Z"))  # 1h out

    snap = derive_snapshots(ledger.records(), closing_cutoff_hours=3.0)
    assert snap["first"].spread == -7.5
    assert snap["latest"].spread == -8.0
    assert snap["closing"].spread == -8.0

    early = derive_snapshots([_quote(observed_at="2026-08-27T18:00:00Z")],
                             closing_cutoff_hours=3.0)
    assert early["latest"] is not None
    assert early["closing"] is None, "no quote inside the cutoff is not a close"


def test_derived_movement_is_measured_from_the_first_observation():
    quotes = [_quote(spread=-7.5, total=50.5),
              _quote(spread=-9.0, total=52.0, observed_at="2026-08-28T18:00:00Z")]
    snap = derive_snapshots(quotes, closing_cutoff_hours=3.0)
    assert snap["spread_movement"] == pytest.approx(-1.5)
    assert snap["total_movement"] == pytest.approx(1.5)


def test_absence_rows_never_contribute_to_derived_prices():
    quotes = [
        MarketQuote.no_line_posted(
            league="ncaa", season=2026, week=1, game_id="g", home_team="A",
            away_team="B", kickoff="2026-08-29T16:00:00Z", provider="Bovada",
            observed_at="2026-08-13T18:00:00Z", source_id="s",
            collector_version="v"),
        _quote(spread=-7.5),
    ]
    snap = derive_snapshots(quotes, closing_cutoff_hours=3.0)
    assert snap["first"].spread == -7.5
    assert snap["provider_count_spread"] == 1


def test_partial_quotes_do_not_suppress_the_side_that_exists():
    """One missing total must never hide an available spread."""
    quotes = [_quote(provider="Bovada", spread=-7.5, total=None),
              _quote(provider="Pinnacle", spread=-7.0, total=51.0)]
    snap = derive_snapshots(quotes, closing_cutoff_hours=3.0)
    assert snap["provider_count_spread"] == 2
    assert snap["provider_count_total"] == 1
    assert snap["median_spread"] == pytest.approx(-7.25)
    assert snap["median_total"] == pytest.approx(51.0)


# -- provenance --------------------------------------------------------------------------

def test_every_row_carries_ingestion_provenance(tmp_path):
    ledger = MarketQuoteLedger(tmp_path / "quotes.jsonl")
    ledger.append(_quote())
    row = ledger.records()[0]
    assert row.source_id == "cfbd/lines"
    assert row.collector_version == "market-collector/1"
    assert row.raw_hash and len(row.raw_hash) == 64
    assert pd.notna(pd.to_datetime(row.ingested_at, utc=True))
    assert row.quote_phase is QuotePhase.UNKNOWN
