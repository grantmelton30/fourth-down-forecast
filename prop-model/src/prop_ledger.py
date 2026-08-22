"""Append-only ledger of player prop quotes.

ONE ROW = ONE QUOTE FROM ONE BOOK, WITH ITS PRICE. Not a consensus, not a median, not a
line without odds. Both audits of the existing build landed on the same two defects and
this file exists to not repeat them:

  * a line recorded without its price cannot be converted to expected value, and the
    break-even moves from 53.5% at -115 to 55.6% at -125 -- wide enough to flip the sign of
    any edge this project has measured;
  * `shared/market_consensus.py` takes a median across books, which with two books
    manufactures quarter-point numbers no book offers. 21 of 95 NCAA totals on the live
    slate are such synthetic quotes.

APPEND-ONLY, AND DEDUPED ON THE QUOTE ITSELF. A quote is identified by everything that makes
it that quote: event, book, market, player, side, line, price, and the book's own
last-update timestamp. Re-running a capture five minutes later appends nothing if the board
has not moved, and appends a genuinely new row the moment it has. A previously written row
is never rewritten -- the whole value of the archive is that it records what was showing at
the time, including when that later turns out to have been wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

FIELDS = ["event_id", "commence_time", "home_team", "away_team", "book",
          "book_last_update", "market", "player", "side", "line", "price", "captured_at"]

# What makes a quote distinct. `captured_at` is deliberately EXCLUDED: capturing the same
# unchanged board twice should not create two rows, or the archive would measure how often
# the script ran rather than how often the market moved.
IDENTITY = ("event_id", "book", "market", "player", "side", "line", "price",
            "book_last_update")


class PropLedger:
    def __init__(self, path: "str | Path"):
        self.path = Path(path)

    def _existing(self) -> set:
        seen = set()
        if not self.path.exists():
            return seen
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A truncated final line is recoverable; a silently skipped one is not.
                    raise ValueError(
                        f"{self.path} contains a malformed line. Refusing to append to a "
                        f"ledger that cannot be fully read -- fix or truncate it first.")
                seen.add(tuple(row.get(k) for k in IDENTITY))
        return seen

    def append(self, rows: list) -> int:
        """Append quotes not already present. Returns how many were genuinely new."""
        if not rows:
            return 0
        seen = self._existing()
        fresh = []
        for row in rows:
            key = tuple(row.get(k) for k in IDENTITY)
            if key in seen:
                continue
            seen.add(key)
            fresh.append({k: row.get(k) for k in FIELDS})
        if not fresh:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for row in fresh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        return len(fresh)

    def read(self) -> list:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def summary(self) -> dict:
        rows = self.read()
        if not rows:
            return {"quotes": 0, "note": "nothing captured yet"}
        books, markets, players, events = set(), {}, set(), set()
        two_sided, priced = 0, 0
        sides = {}
        for r in rows:
            books.add(r.get("book"))
            markets[r.get("market")] = markets.get(r.get("market"), 0) + 1
            players.add(r.get("player"))
            events.add(r.get("event_id"))
            if r.get("price") is not None:
                priced += 1
            sides.setdefault(
                (r.get("event_id"), r.get("book"), r.get("market"), r.get("player")),
                set()).add(r.get("side"))
        two_sided = sum(1 for v in sides.values() if len(v) >= 2)
        return {
            "quotes": len(rows),
            "events": len(events),
            "books": sorted(b for b in books if b),
            "markets": markets,
            "players": len(players),
            "priced_pct": round(100 * priced / len(rows), 1),
            "two_sided_props": two_sided,
            "one_sided_props": len(sides) - two_sided,
            "first_capture": min(r.get("captured_at", "") for r in rows),
            "last_capture": max(r.get("captured_at", "") for r in rows),
        }


__all__ = ["PropLedger", "FIELDS", "IDENTITY"]
