"""Turning free feeds into ledger rows, without inventing anything (Objective 1).

The two free sources disagree about what a spread sign means, which is verified here
rather than trusted to a comment. CFBD quotes a home favourite negative; nflverse quotes
a home favourite positive -- measured over 400 recent games, `spread_line > 0` sees the
home team win 70% of the time and `spread_line < 0` sees 33.5%. One of them has to be
flipped, and a silent mistake would invert every NFL line in the ledger.

The other job here is refusing to flatten. CFBD returns every book that priced a game;
`ingest.load_lines` deliberately collapses that to one provider for modelling, which is
the opposite of what a market archive needs.
"""
from __future__ import annotations

import pandas as pd

from market_ledger import QuotePhase, QuoteStatus
from market_sources import cfbd_quotes, nflverse_quotes

OBSERVED = "2026-08-13T18:00:00Z"


def _cfbd_game(lines, **overrides):
    game = {
        "id": 401856766, "season": 2026, "week": 1, "seasonType": "regular",
        "startDate": "2026-08-29T16:00:00.000Z",
        "homeTeam": "TCU", "awayTeam": "North Carolina",
        "homeClassification": "fbs", "awayClassification": "fbs",
        "lines": lines,
    }
    game.update(overrides)
    return game


def test_every_book_becomes_its_own_row():
    game = _cfbd_game([
        {"provider": "Bovada", "spread": -7.5, "spreadOpen": -7.0,
         "overUnder": 50.5, "overUnderOpen": 50.0,
         "homeMoneyline": -300, "awayMoneyline": 240},
        {"provider": "DraftKings", "spread": -7.0, "spreadOpen": -7.0,
         "overUnder": 51.0, "overUnderOpen": 50.5,
         "homeMoneyline": None, "awayMoneyline": None},
    ])
    quotes = cfbd_quotes([game], observed_at=OBSERVED, collector_version="v1")

    current = [q for q in quotes if q.quote_phase is QuotePhase.CURRENT]
    assert sorted(q.provider for q in current) == ["Bovada", "DraftKings"]
    assert sorted(q.spread for q in current) == [-7.5, -7.0]


def test_the_reported_opener_is_captured_as_its_own_phase():
    """CFBD states the opener explicitly, so it is recorded rather than inferred."""
    game = _cfbd_game([{"provider": "Bovada", "spread": -7.5, "spreadOpen": -7.0,
                        "overUnder": 50.5, "overUnderOpen": 50.0}])
    quotes = cfbd_quotes([game], observed_at=OBSERVED, collector_version="v1")

    opening = [q for q in quotes if q.quote_phase is QuotePhase.OPENING]
    assert len(opening) == 1
    assert opening[0].spread == -7.0 and opening[0].total == 50.0


def test_a_game_no_book_has_priced_records_absence_not_silence():
    quotes = cfbd_quotes([_cfbd_game([])], observed_at=OBSERVED, collector_version="v1")
    assert len(quotes) == 1
    assert quotes[0].status is QuoteStatus.NO_LINE_POSTED
    assert quotes[0].spread is None and quotes[0].total is None


def test_a_book_with_only_a_spread_still_produces_a_quote():
    game = _cfbd_game([{"provider": "Bovada", "spread": -7.5, "spreadOpen": None,
                        "overUnder": None, "overUnderOpen": None}])
    quotes = cfbd_quotes([game], observed_at=OBSERVED, collector_version="v1")
    current = [q for q in quotes if q.quote_phase is QuotePhase.CURRENT]
    assert len(current) == 1
    assert current[0].spread == -7.5 and current[0].total is None
    # An opener with neither price is not an observation at all.
    assert not [q for q in quotes if q.quote_phase is QuotePhase.OPENING]


def test_week_zero_survives_ingestion():
    game = _cfbd_game([{"provider": "Bovada", "spread": -3.0, "overUnder": 44.0}],
                      week=0)
    quotes = cfbd_quotes([game], observed_at=OBSERVED, collector_version="v1")
    assert quotes[0].week == 0


def test_cross_tier_and_neutral_site_context_is_retained():
    game = _cfbd_game([{"provider": "Bovada", "spread": -38.0, "overUnder": 55.0}],
                      awayClassification="fcs", neutralSite=True)
    quotes = cfbd_quotes([game], observed_at=OBSERVED, collector_version="v1")
    assert quotes[0].neutral_site is True
    assert "fcs" in quotes[0].detail


# -- nflverse ----------------------------------------------------------------------------

def _nfl_schedule(**overrides):
    row = {
        "game_id": "2026_01_KC_BAL", "season": 2026, "week": 1,
        "home_team": "BAL", "away_team": "KC",
        "gameday": "2026-09-10", "gametime": "20:20",
        "spread_line": 3.5, "total_line": 46.5,
        "home_moneyline": -180, "away_moneyline": 155, "location": "Home",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_nflverse_home_favourite_is_stored_negative():
    """nflverse is positive-when-home-favoured; the ledger convention is negative."""
    quotes = nflverse_quotes(_nfl_schedule(spread_line=3.5),
                             observed_at=OBSERVED, collector_version="v1")
    assert quotes[0].spread == -3.5
    assert quotes[0].total == 46.5


def test_nflverse_home_underdog_is_stored_positive():
    quotes = nflverse_quotes(_nfl_schedule(spread_line=-6.0),
                             observed_at=OBSERVED, collector_version="v1")
    assert quotes[0].spread == 6.0


def test_nflverse_missing_spread_keeps_the_total():
    quotes = nflverse_quotes(_nfl_schedule(spread_line=None),
                             observed_at=OBSERVED, collector_version="v1")
    assert quotes[0].spread is None and quotes[0].total == 46.5


def test_nflverse_game_with_no_prices_records_absence():
    quotes = nflverse_quotes(_nfl_schedule(spread_line=None, total_line=None),
                             observed_at=OBSERVED, collector_version="v1")
    assert quotes[0].status is QuoteStatus.NO_LINE_POSTED


def test_nflverse_neutral_site_is_flagged():
    quotes = nflverse_quotes(_nfl_schedule(location="Neutral"),
                             observed_at=OBSERVED, collector_version="v1")
    assert quotes[0].neutral_site is True
