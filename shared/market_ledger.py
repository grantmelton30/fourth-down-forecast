"""Append-only capture of timestamped sportsbook observations (Objective 1).

Everything else in this project can be rebuilt. A model refits from cached play data on
demand; a line that moved on a Tuesday and was never recorded is gone permanently. That
asymmetry sets the design: this module records what a provider showed and when it was
seen, and derives nothing at write time.

Capture and interpretation are kept apart on purpose.  `derive_snapshots` computes the
opener, the close, provider medians, dispersion and movement from the stored rows, so
those definitions can be revised later -- a closing-cutoff rule can be re-argued and
re-applied to history, whereas a missed observation cannot be recovered by any argument.

Three distinctions the ledger refuses to blur:

* **A price, no line posted, and a source failure are different facts.**  Collapsing the
  last two into "no data" would let an outage masquerade as a market that never opened,
  and would quietly bias any study of when lines appear.
* **A partial quote is a real quote.**  A provider showing a spread but no total is
  recorded, and the missing total never suppresses the spread that exists.
* **The latest quote we hold is not the close.**  A quote earns the closing label only by
  falling inside a documented cutoff before kickoff; otherwise the close is `None` and
  says so, rather than promoting whatever happened to be collected last.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


class QuoteStatus(str, Enum):
    """Why a row exists. Only QUOTED ever carries a price."""

    QUOTED = "quoted"
    NO_LINE_POSTED = "no_line_posted"
    SOURCE_FAILURE = "source_failure"


class QuotePhase(str, Enum):
    """Phase is asserted only when known; it is never inferred from arrival order."""

    OPENING = "opening"
    CURRENT = "current"
    CLOSING = "closing"
    UNKNOWN = "unknown"


def _utc(value) -> str:
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        raise ValueError(f"market quote has an unusable timestamp: {value!r}")
    return stamp.isoformat()


@dataclass(frozen=True)
class MarketQuote:
    """One provider's observation of one game at one moment.

    `spread` is always **home perspective**: negative when the home team is favoured,
    matching `result = home_score - away_score`. Away-perspective feeds must be converted
    at the boundary via `from_away_perspective`, never silently.
    """

    league: str
    season: int
    week: int
    game_id: str
    home_team: str
    away_team: str
    kickoff: str
    provider: str
    observed_at: str
    source_id: str
    collector_version: str
    spread: "float | None" = None
    total: "float | None" = None
    moneyline_home: "float | None" = None
    moneyline_away: "float | None" = None
    neutral_site: bool = False
    quote_phase: QuotePhase = QuotePhase.UNKNOWN
    status: QuoteStatus = QuoteStatus.QUOTED
    detail: str = ""
    ingested_at: str = ""
    raw: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(self, "kickoff", _utc(self.kickoff))
        object.__setattr__(self, "observed_at", _utc(self.observed_at))
        object.__setattr__(self, "ingested_at",
                           self.ingested_at or datetime.now(timezone.utc).isoformat())
        object.__setattr__(self, "week", int(self.week))  # Week 0 is a real week.
        object.__setattr__(self, "status", QuoteStatus(self.status))
        object.__setattr__(self, "quote_phase", QuotePhase(self.quote_phase))
        if self.status is not QuoteStatus.QUOTED and (
                self.spread is not None or self.total is not None):
            raise ValueError("only a QUOTED row may carry a price")

    @property
    def priced(self) -> bool:
        return self.status is QuoteStatus.QUOTED

    @property
    def raw_hash(self) -> str:
        """Stable identity of the observation, used to reject exact replays.

        Deliberately excludes `ingested_at`: re-running a collector over an unchanged
        payload is the same observation seen twice, not a new one.
        """
        payload = {k: v for k, v in asdict(self).items()
                   if k not in ("ingested_at", "raw")}
        payload = {k: (v.value if isinstance(v, Enum) else v)
                   for k, v in payload.items()}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def from_away_perspective(quote: "MarketQuote", *, away_spread: float) -> "MarketQuote":
        """Convert an away-quoted spread to the stored home convention."""
        return MarketQuote(**{**_fields(quote), "spread": -float(away_spread)})

    @staticmethod
    def _absence(status: QuoteStatus, **kwargs) -> "MarketQuote":
        return MarketQuote(status=status, spread=None, total=None, **kwargs)

    @staticmethod
    def no_line_posted(**kwargs) -> "MarketQuote":
        """The provider answered and offered nothing on this game."""
        return MarketQuote._absence(QuoteStatus.NO_LINE_POSTED, **kwargs)

    @staticmethod
    def source_failure(**kwargs) -> "MarketQuote":
        """We could not ask. Never to be read as 'no line'."""
        return MarketQuote._absence(QuoteStatus.SOURCE_FAILURE, **kwargs)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["status"] = self.status.value
        out["quote_phase"] = self.quote_phase.value
        out["raw_hash"] = self.raw_hash
        return out

    @staticmethod
    def from_dict(data: dict) -> "MarketQuote":
        known = set(MarketQuote.__dataclass_fields__)
        return MarketQuote(**{k: v for k, v in data.items() if k in known})


def _fields(quote: MarketQuote) -> dict:
    out = asdict(quote)
    out["status"] = quote.status
    out["quote_phase"] = quote.quote_phase
    return out


class MarketQuoteLedger:
    """Append-only JSONL. Rows are added, never edited or removed.

    Duplicate detection is by `raw_hash` held in memory, so appending stays O(1) rather
    than re-reading the whole file per write the way `PredictionLedger` does -- quotes
    are collected far too often for a quadratic write path.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._seen: "set[str] | None" = None

    def _index(self) -> set:
        if self._seen is None:
            self._seen = {r.raw_hash for r in self.records()}
        return self._seen

    def records(self) -> list[MarketQuote]:
        if not self.path.exists():
            return []
        return [MarketQuote.from_dict(json.loads(line))
                for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, quote: MarketQuote) -> bool:
        """Store one observation. Returns False when it is an exact replay."""
        if quote.priced and quote.spread is None and quote.total is None:
            raise ValueError("a quoted row carries no price; record an absence instead")
        digest = quote.raw_hash
        if digest in self._index():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(quote.to_dict(), sort_keys=True,
                                    separators=(",", ":")) + "\n")
        self._index().add(digest)
        return True

    def extend(self, quotes: Iterable[MarketQuote]) -> int:
        return sum(int(self.append(q)) for q in quotes)

    # -- change detection ----------------------------------------------------------

    @staticmethod
    def _series_key(quote: MarketQuote) -> tuple:
        """What counts as "the same line". A book's opener and its current price are
        separate claims about the same game, so phase is part of the key."""
        return (quote.league, quote.game_id, quote.provider, quote.quote_phase.value)

    @staticmethod
    def _price(quote: MarketQuote) -> tuple:
        return (quote.status.value, quote.spread, quote.total,
                quote.moneyline_home, quote.moneyline_away)

    def last_by_series(self) -> dict:
        latest: dict = {}
        for row in self.records():  # file order is chronological by construction
            latest[self._series_key(row)] = row
        return latest

    def append_if_changed(self, quote: MarketQuote,
                          _latest: "dict | None" = None) -> bool:
        """Record only when this differs from the last quote for the same series.

        Polling a market several times a day would otherwise write one row per game per
        provider per run saying nothing changed. Comparing against the LAST stored quote
        rather than against all history matters: a line that moves away and returns is a
        real movement, and content-only deduplication would silently discard it.
        """
        latest = self.last_by_series() if _latest is None else _latest
        previous = latest.get(self._series_key(quote))
        if previous is not None and self._price(previous) == self._price(quote):
            return False
        if not self.append(quote):
            return False
        latest[self._series_key(quote)] = quote
        return True

    def extend_if_changed(self, quotes: Iterable[MarketQuote]) -> int:
        """Bulk form that indexes the file once rather than per quote."""
        latest = self.last_by_series()
        return sum(int(self.append_if_changed(q, _latest=latest)) for q in quotes)


def _median(values: Sequence[float]) -> "float | None":
    return float(np.median(values)) if len(values) else None


def derive_snapshots(
    quotes: Sequence[MarketQuote], *, closing_cutoff_hours: float = 3.0,
    published_at: "str | None" = None,
) -> dict:
    """Everything downstream reads, computed from stored rows rather than stored itself.

    `closing_cutoff_hours` is the documented rule: a quote counts as closing only if it
    was observed within that window before kickoff. With no such quote the close is None,
    because the last thing collected is not evidence of where the market settled.
    """
    priced = sorted((q for q in quotes if q.priced), key=lambda q: q.observed_at)
    out: dict = {
        "first": None, "latest": None, "closing": None, "at_publication": None,
        "median_spread": None, "median_total": None,
        "provider_count_spread": 0, "provider_count_total": 0,
        "spread_dispersion": None, "total_dispersion": None,
        "spread_movement": None, "total_movement": None,
        "spread_movement_after_publication": None,
        "total_movement_after_publication": None,
    }
    if not priced:
        return out

    out["first"], out["latest"] = priced[0], priced[-1]

    kickoff = pd.to_datetime(priced[0].kickoff, utc=True)
    cutoff = kickoff - pd.Timedelta(hours=float(closing_cutoff_hours))
    inside = [q for q in priced
              if cutoff <= pd.to_datetime(q.observed_at, utc=True) <= kickoff]
    out["closing"] = inside[-1] if inside else None

    if published_at is not None:
        stamp = pd.to_datetime(published_at, utc=True)
        at_or_before = [q for q in priced
                        if pd.to_datetime(q.observed_at, utc=True) <= stamp]
        out["at_publication"] = at_or_before[-1] if at_or_before else None

    # One provider contributes only its most recent observation to a cross-sectional read.
    latest_by_provider = {q.provider: q for q in priced}
    spreads = [q.spread for q in latest_by_provider.values() if q.spread is not None]
    totals = [q.total for q in latest_by_provider.values() if q.total is not None]
    out["provider_count_spread"], out["provider_count_total"] = len(spreads), len(totals)
    out["median_spread"], out["median_total"] = _median(spreads), _median(totals)
    if spreads:
        out["spread_dispersion"] = float(max(spreads) - min(spreads))
    if totals:
        out["total_dispersion"] = float(max(totals) - min(totals))

    def movement(attr: str, since) -> "float | None":
        series = [q for q in priced if getattr(q, attr) is not None]
        if since is not None:
            series = [q for q in series
                      if pd.to_datetime(q.observed_at, utc=True) >= since]
        if len(series) < 2:
            return None
        return float(getattr(series[-1], attr) - getattr(series[0], attr))

    out["spread_movement"] = movement("spread", None)
    out["total_movement"] = movement("total", None)
    if published_at is not None:
        stamp = pd.to_datetime(published_at, utc=True)
        out["spread_movement_after_publication"] = movement("spread", stamp)
        out["total_movement_after_publication"] = movement("total", stamp)
    return out


__all__ = ["MarketQuote", "MarketQuoteLedger", "QuotePhase", "QuoteStatus",
           "SCHEMA_VERSION", "derive_snapshots"]
