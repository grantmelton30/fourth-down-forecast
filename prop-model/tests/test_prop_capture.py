"""Prop quote capture and ledger (src/odds_api.py, src/prop_ledger.py).

The ledger is the only record a prop backtest will ever have, and no free historical archive
exists to rebuild it from — the vendor's own history is paid and starts 2023-05-03. So these
tests are about the properties that make an archive trustworthy: every quote keeps its price,
no quote is invented by averaging, a logged row is never rewritten, and a failed capture is
loud rather than silent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.odds_api import OddsApiError, Usage, api_key, flatten_props
from src.prop_ledger import PropLedger


def _payload():
    """Two books quoting the same player at DIFFERENT numbers -- the case a median would
    destroy by inventing a line neither offered."""
    return {
        "id": "evt1", "commence_time": "2026-09-13T17:00:00Z",
        "home_team": "KC", "away_team": "BUF",
        "bookmakers": [
            {"key": "draftkings", "last_update": "2026-09-11T12:00:00Z", "markets": [
                {"key": "player_receptions", "outcomes": [
                    {"name": "Over", "description": "Travis Kelce", "point": 5.5, "price": -115},
                    {"name": "Under", "description": "Travis Kelce", "point": 5.5, "price": -105},
                ]}]},
            {"key": "fanduel", "last_update": "2026-09-11T12:01:00Z", "markets": [
                {"key": "player_receptions", "outcomes": [
                    {"name": "Over", "description": "Travis Kelce", "point": 6.5, "price": +100},
                    {"name": "Under", "description": "Travis Kelce", "point": 6.5, "price": -122},
                ]}]},
        ],
    }


def test_every_quote_keeps_its_price_and_its_book():
    """A line without a price cannot become expected value. Break-even is 53.5% at -115 and
    55.6% at -125 -- wide enough to flip the sign of any edge measured in this project."""
    rows = flatten_props(_payload(), captured_at="2026-09-11T12:05:00Z")
    assert len(rows) == 4
    assert all(r["price"] is not None for r in rows)
    assert all(r["book"] for r in rows)
    assert {r["side"] for r in rows} == {"Over", "Under"}


def test_disagreeing_books_are_kept_apart_not_averaged():
    """DK says 5.5, FD says 6.5. A median would produce 6.0 -- a line neither book offers.
    That is exactly the defect the audits found in shared/market_consensus.py, where 21 of
    95 live NCAA totals are synthetic quarter-points."""
    rows = flatten_props(_payload(), captured_at="t")
    lines = {(r["book"], r["line"]) for r in rows}
    assert ("draftkings", 5.5) in lines and ("fanduel", 6.5) in lines
    assert not any(r["line"] == 6.0 for r in rows), "no invented midpoint may appear"


def test_recapturing_an_unchanged_board_adds_nothing(tmp_path):
    """Otherwise the archive would measure how often the script ran, not how often the
    market moved."""
    led = PropLedger(tmp_path / "q.jsonl")
    rows = flatten_props(_payload(), captured_at="2026-09-11T12:05:00Z")
    assert led.append(rows) == 4
    later = flatten_props(_payload(), captured_at="2026-09-11T18:00:00Z")
    assert led.append(later) == 0, "same board, later clock -- nothing new happened"


def test_a_moved_line_is_a_new_row_and_the_old_one_survives(tmp_path):
    """The archive's whole value is that it records what was showing at the time. A move
    must append, never overwrite -- including when the earlier number looks wrong later."""
    led = PropLedger(tmp_path / "q.jsonl")
    led.append(flatten_props(_payload(), captured_at="t1"))
    moved = _payload()
    moved["bookmakers"][0]["markets"][0]["outcomes"][0]["point"] = 6.5
    moved["bookmakers"][0]["last_update"] = "2026-09-12T09:00:00Z"
    assert led.append(flatten_props(moved, captured_at="t2")) > 0
    lines = [r["line"] for r in led.read()
             if r["book"] == "draftkings" and r["side"] == "Over"]
    assert 5.5 in lines and 6.5 in lines, "the superseded quote must still be there"


def test_a_malformed_ledger_refuses_to_be_appended_to(tmp_path):
    """Appending past a line that cannot be parsed would silently drop history."""
    path = tmp_path / "q.jsonl"
    path.write_text('{"event_id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        PropLedger(path).append([{"event_id": "b"}])


def test_summary_separates_two_sided_from_one_sided(tmp_path):
    """Only a two-sided quote lets you compute the no-vig probability. PrizePicks showed why
    this matters: 79% of its board is over-only, so the usable subset is far smaller than
    the headline count."""
    led = PropLedger(tmp_path / "q.jsonl")
    rows = flatten_props(_payload(), captured_at="t")
    rows = [r for r in rows if not (r["book"] == "fanduel" and r["side"] == "Under")]
    led.append(rows)
    s = led.summary()
    assert s["two_sided_props"] == 1 and s["one_sided_props"] == 1
    assert s["priced_pct"] == 100.0


def test_a_missing_key_names_the_domain_confusion_that_already_cost_us(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(OddsApiError, match="the-odds-api.com"):
        api_key()


def test_usage_is_read_from_headers_not_estimated():
    """Estimating credits is how a free tier gets silently exhausted."""
    u = Usage()
    u.update({"x-requests-used": "12", "x-requests-remaining": "488", "x-requests-last": "3"})
    assert (u.used, u.remaining, u.last_cost, u.calls) == (12, 488, 3, 1)
