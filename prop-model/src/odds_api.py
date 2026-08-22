"""The Odds API v4 client, scoped to NFL player props.

WHY A DEDICATED CLIENT. Props are per-event: you list events, then request odds for each one
individually. That makes credit accounting the central concern rather than an afterthought —
the free tier is 500 credits a month and a careless loop spends it in one run.

    live event odds        cost = markets x regions        PER EVENT   -> 1 credit/game/market
    historical event odds  cost = 10 x markets x regions   PER EVENT   -> 10 credits/game/market

An NFL week is ~16 games, so one market costs ~16 credits live: about 64 a month against a
free allowance of 500. Three markets still fit. The same sweep against the historical
endpoint costs 160 a week and would exhaust the free tier in three weeks — which is why
`fetch_event_odds` refuses to run historically without an explicit acknowledgement.

VERIFY THE KEY BEFORE TRUSTING IT. `nfl-model/DATA_SOURCES.md` warns that more than one
product markets itself under this name with different tiers, and this session already made
that exact error once — quoting $99/month for props that the real vendor sells from $0.
`verify_access()` exists so the answer comes from the API rather than from a pricing page.

PRICES ARE THE POINT. Every quote carries both sides. A line without its price cannot be
turned into expected value, and at -115 vs -125 the break-even moves from 53.5% to 55.6% —
wide enough to flip the sign of any edge this project has ever measured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
TIMEOUT = 30

# Markets worth capturing first, in the order the plan argues for: counting stats before
# yardage, because discrete outcomes tie directly to volume and allocation.
COUNT_MARKETS = ("player_receptions", "player_rush_attempts", "player_pass_attempts")
YARD_MARKETS = ("player_reception_yds", "player_rush_yds", "player_pass_yds")


class OddsApiError(RuntimeError):
    """Raised loudly. A capture that fails silently loses a quote that cannot be recovered."""


@dataclass
class Usage:
    """Credit accounting, read from the response headers rather than estimated."""
    used: "int | None" = None
    remaining: "int | None" = None
    last_cost: "int | None" = None
    calls: int = 0
    history: list = field(default_factory=list)

    def update(self, headers) -> None:
        def _int(name):
            v = headers.get(name)
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        self.used = _int("x-requests-used") or self.used
        self.remaining = _int("x-requests-remaining") or self.remaining
        self.last_cost = _int("x-requests-last")
        self.calls += 1
        self.history.append(self.last_cost)

    def __str__(self) -> str:
        return (f"{self.calls} call(s), last cost {self.last_cost}, "
                f"used {self.used}, remaining {self.remaining}")


def api_key(explicit: "str | None" = None) -> str:
    key = explicit or os.environ.get("ODDS_API_KEY")
    if not key:
        raise OddsApiError(
            "No API key. Set ODDS_API_KEY, or pass one explicitly.\n"
            "A free key (500 credits/month, player props included) comes from "
            "https://the-odds-api.com/#get-access — note that a DIFFERENT product markets "
            "itself under a near-identical name with a far more limited free tier, so "
            "confirm the domain is the-odds-api.com."
        )
    return key


def _get(path: str, params: dict, usage: "Usage | None" = None) -> object:
    resp = requests.get(f"{BASE}{path}", params=params, timeout=TIMEOUT)
    if usage is not None:
        usage.update(resp.headers)
    if resp.status_code == 401:
        raise OddsApiError("401: the API key was rejected.")
    if resp.status_code == 422:
        raise OddsApiError(
            f"422 for {path}: the API rejected these parameters. On a free key this most "
            f"often means the market is not available to that plan. Response: {resp.text[:300]}")
    if resp.status_code == 429:
        raise OddsApiError("429: out of credits or rate limited. Check remaining quota.")
    if not resp.ok:
        raise OddsApiError(f"{resp.status_code} for {path}: {resp.text[:300]}")
    return resp.json()


def verify_access(key: "str | None" = None) -> dict:
    """Ask the API what this key can actually do, rather than believing a pricing page.

    Returns a report rather than raising on a missing market: "props are not available on
    this key" is an answer worth printing clearly, not an exception to swallow.
    """
    key = api_key(key)
    usage = Usage()
    sports = _get("/sports", {"apiKey": key}, usage)
    nfl = [s for s in sports if s.get("key") == SPORT]
    report = {"nfl_listed": bool(nfl), "nfl_active": bool(nfl and nfl[0].get("active")),
              "usage": str(usage), "markets": {}}
    if not nfl:
        report["error"] = f"{SPORT} not returned by /sports on this key"
        return report

    events = _get(f"/sports/{SPORT}/events", {"apiKey": key}, usage)
    report["upcoming_events"] = len(events)
    if not events:
        report["error"] = "no upcoming NFL events; cannot test a props market"
        return report

    # One event, one market: the cheapest possible test of whether props are included.
    event_id = events[0]["id"]
    for market in ("player_receptions",):
        try:
            odds = _get(f"/sports/{SPORT}/events/{event_id}/odds",
                        {"apiKey": key, "regions": "us", "markets": market,
                         "oddsFormat": "american"}, usage)
            books = odds.get("bookmakers", [])
            report["markets"][market] = {
                "available": True, "bookmakers": len(books),
                "outcomes": sum(len(m.get("outcomes", []))
                                for b in books for m in b.get("markets", [])),
            }
        except OddsApiError as exc:
            report["markets"][market] = {"available": False, "error": str(exc)[:200]}
    report["usage"] = str(usage)
    return report


def list_events(key: "str | None" = None, usage: "Usage | None" = None) -> list:
    """Upcoming NFL events. Costs nothing — the events endpoint is free of quota."""
    return _get(f"/sports/{SPORT}/events", {"apiKey": api_key(key)}, usage)


def fetch_event_odds(
    event_id: str, markets, *, key: "str | None" = None, regions: str = "us",
    usage: "Usage | None" = None,
) -> dict:
    """Player props for one event, all books, both sides, with prices.

    Deliberately does NOT accept a `date` parameter. The historical variant costs ten times
    as much per call and would drain a free key in weeks; pulling history is a separate,
    deliberate, paid operation and should look like one in the code.
    """
    if isinstance(markets, str):
        markets = [markets]
    return _get(
        f"/sports/{SPORT}/events/{event_id}/odds",
        {"apiKey": api_key(key), "regions": regions, "markets": ",".join(markets),
         "oddsFormat": "american"},
        usage,
    )


def flatten_props(payload: dict, *, captured_at: str) -> list:
    """One row per (book, market, player, side) — never a cross-book average.

    The existing build already learned this the expensive way: `shared/market_consensus.py`
    takes a median, and with two books that manufactures quarter-point totals no book
    offers. A prop record has to be a quote somebody would actually have accepted.
    """
    rows = []
    home, away = payload.get("home_team"), payload.get("away_team")
    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            for outcome in market.get("outcomes", []):
                rows.append({
                    "event_id": payload.get("id"),
                    "commence_time": payload.get("commence_time"),
                    "home_team": home, "away_team": away,
                    "book": book.get("key"), "book_last_update": book.get("last_update"),
                    "market": market.get("key"),
                    "player": outcome.get("description"),
                    "side": outcome.get("name"),          # Over / Under
                    "line": outcome.get("point"),
                    "price": outcome.get("price"),        # American odds
                    "captured_at": captured_at,
                })
    return rows


__all__ = ["BASE", "SPORT", "COUNT_MARKETS", "YARD_MARKETS", "OddsApiError", "Usage",
           "api_key", "verify_access", "list_events", "fetch_event_odds", "flatten_props"]
