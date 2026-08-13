"""Free feeds to ledger rows (Objective 1).

Two rules govern everything here.

**Preserve every book.** CFBD returns each provider that priced a game.
`ncaa-model/src/ingest.load_lines` deliberately reduces that to a single provider,
because a model wants one number -- but a market archive wants the disagreement, since
provider spread is itself evidence about how settled a line is. Nothing is collapsed on
the way in.

**Normalize the sign once, at the boundary.** The two sources disagree, and the
disagreement is empirical rather than documented: CFBD prints a home favourite negative
(`homeTeam TCU, spread -7.5`), while nflverse prints one positive -- across 400 recent
games, `spread_line > 0` has the home team winning 70% of the time against 33.5% when
negative. The ledger stores home-negative throughout, so nflverse is flipped here and
nowhere else.

Nothing in this module infers. A book that priced only a spread yields a spread; a game
no book priced yields an explicit absence; a source that could not be reached is the
caller's job to record, because this module never learns that it failed.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from market_ledger import MarketQuote, QuotePhase

CFBD_SOURCE = "cfbd/lines"
NFLVERSE_SOURCE = "nflverse/schedules"
NFLVERSE_PROVIDER = "nflverse"


def _number(value) -> "float | None":
    if value is None:
        return None
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _text(value) -> str:
    return "" if value is None else str(value)


def cfbd_quotes(
    games: Iterable[dict], *, observed_at: str, collector_version: str,
    source_id: str = CFBD_SOURCE,
) -> list[MarketQuote]:
    """One row per book per game, plus the book's reported opener as its own phase.

    CFBD states `spreadOpen`/`overUnderOpen` explicitly, so the opening quote is captured
    as fact rather than reconstructed from whichever observation we happened to see first.
    """
    out: list[MarketQuote] = []
    for game in games:
        classifications = {_text(game.get("homeClassification")).lower(),
                           _text(game.get("awayClassification")).lower()}
        present = sorted(c for c in classifications if c)
        detail = "cross-tier: " + "/".join(present) if len(present) > 1 else ""
        common = dict(
            league="ncaa",
            season=int(game["season"]),
            week=int(game.get("week", 0)),
            game_id=str(game["id"]),
            home_team=_text(game.get("homeTeam")),
            away_team=_text(game.get("awayTeam")),
            kickoff=game["startDate"],
            neutral_site=bool(game.get("neutralSite", False)),
            observed_at=observed_at,
            source_id=source_id,
            collector_version=collector_version,
            detail=detail,
        )
        lines = game.get("lines") or []
        if not lines:
            # The provider answered and offered nothing. Recording this separates a
            # market that has not opened from a collector that never ran.
            out.append(MarketQuote.no_line_posted(provider="*", **common))
            continue
        for line in lines:
            provider = _text(line.get("provider")) or "unknown"
            for phase, spread_key, total_key in (
                (QuotePhase.CURRENT, "spread", "overUnder"),
                (QuotePhase.OPENING, "spreadOpen", "overUnderOpen"),
            ):
                spread, total = _number(line.get(spread_key)), _number(line.get(total_key))
                if spread is None and total is None:
                    continue  # Not an observation; do not manufacture one.
                out.append(MarketQuote(
                    provider=provider, spread=spread, total=total,
                    moneyline_home=_number(line.get("homeMoneyline")),
                    moneyline_away=_number(line.get("awayMoneyline")),
                    quote_phase=phase, **common,
                ))
    return out


def nflverse_quotes(
    schedules: pd.DataFrame, *, observed_at: str, collector_version: str,
    source_id: str = NFLVERSE_SOURCE,
) -> list[MarketQuote]:
    """nflverse publishes one consensus line per game; it is stored as one provider.

    Labelled `nflverse` rather than a book name on purpose: it is an aggregate, and
    pretending otherwise would let a single series masquerade as market consensus in
    `provider_count`.
    """
    out: list[MarketQuote] = []
    if schedules is None or schedules.empty:
        return out
    for _, row in schedules.iterrows():
        kickoff = pd.to_datetime(
            f"{row.get('gameday')} {row.get('gametime') or '00:00'}",
            utc=True, errors="coerce",
        )
        if pd.isna(kickoff):
            continue
        spread = _number(row.get("spread_line"))
        total = _number(row.get("total_line"))
        common = dict(
            league="nfl",
            season=int(row["season"]),
            week=int(row["week"]),
            game_id=str(row["game_id"]),
            home_team=_text(row.get("home_team")),
            away_team=_text(row.get("away_team")),
            kickoff=kickoff.isoformat(),
            neutral_site=_text(row.get("location")).lower() == "neutral",
            observed_at=observed_at,
            source_id=source_id,
            collector_version=collector_version,
            provider=NFLVERSE_PROVIDER,
        )
        if spread is None and total is None:
            out.append(MarketQuote.no_line_posted(**common))
            continue
        out.append(MarketQuote(
            # nflverse is positive-when-home-favoured; the ledger is home-negative.
            spread=None if spread is None else -spread,
            total=total,
            moneyline_home=_number(row.get("home_moneyline")),
            moneyline_away=_number(row.get("away_moneyline")),
            quote_phase=QuotePhase.CURRENT, **common,
        ))
    return out


def source_failures(
    games: Sequence[dict], *, league: str, observed_at: str, collector_version: str,
    source_id: str, detail: str,
) -> list[MarketQuote]:
    """Mark an outage explicitly so it is never mistaken for an unpriced market."""
    return [MarketQuote.source_failure(
        league=league, season=int(g["season"]), week=int(g.get("week", 0)),
        game_id=str(g["game_id"]), home_team=_text(g.get("home_team")),
        away_team=_text(g.get("away_team")), kickoff=g["kickoff"], provider="*",
        observed_at=observed_at, source_id=source_id,
        collector_version=collector_version, detail=detail,
    ) for g in games]


__all__ = ["CFBD_SOURCE", "NFLVERSE_PROVIDER", "NFLVERSE_SOURCE", "cfbd_quotes",
           "nflverse_quotes", "source_failures"]
