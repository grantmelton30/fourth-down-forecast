"""Pick ONE book's line per game, by declared preference. Never merge two.

WHY NOT A MEDIAN. This used to take the median across whatever books had posted. With two
books that is the midpoint, and a midpoint of 54 and 53.5 is 53.75 -- a number no book
offers and nobody can bet. On the live slate that produced synthetic quarter-point totals on
21 of 95 priced college games. Both external audits flagged it, and a line you cannot take
is not a market reference; it is an average of two markets pretending to be one.

So this returns a single book's actual quote, chosen by `provider_priority`. Bovada first,
then ESPN Bet, then DraftKings -- the order `ncaa-model/config/ncaa.yaml` already declares
and that `ingest.build_market` already uses.

THIS ALSO ENDS A DIVERGENCE NOBODY HAD NOTICED. `track_p1.py` grades against
`ingest.build_market`, which has always resolved a single provider by that priority. Only
the website was merging. So the number on the site and the number the tracker actually bet
could differ, and on a two-book game they usually did. They now come from the same rule.

DISPERSION IS STILL REPORTED, and is now more useful than before: it says how far the books
that were NOT chosen sat from the one that was. That is a diagnostic about market
disagreement, and it never moves the quoted number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Bovada first, then ESPN Bet, then DraftKings. Mirrors ncaa-model/config/ncaa.yaml
# `market.provider_priority`; anything not listed sorts after these, alphabetically.
DEFAULT_PROVIDER_PRIORITY = ("Bovada", "ESPN Bet", "DraftKings")


def _rank(provider: str, priority) -> tuple:
    """Lower sorts first. Unlisted books rank after every listed one, alphabetically, so the
    choice is deterministic rather than dependent on row order."""
    name = str(provider)
    lowered = [p.lower() for p in priority]
    try:
        return (0, lowered.index(name.lower()), name)
    except ValueError:
        return (1, 0, name)


def aggregate_market_consensus(
    lines: pd.DataFrame, *, minimum_books: int = 3, provider_priority=None,
) -> pd.DataFrame:
    """One row per game carrying ONE book's spread and total.

    `minimum_books` no longer gates the number -- it only sets `is_consensus`, which callers
    use to describe how well covered a game was. The quote itself is always a real one.
    """
    priority = tuple(provider_priority or DEFAULT_PROVIDER_PRIORITY)
    required = {"game_id", "provider", "spread", "total", "observed_at"}
    missing = required - set(lines.columns)
    if missing:
        raise ValueError(f"market lines missing columns: {sorted(missing)}")
    frame = lines.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    # A provider contributes only its latest known snapshot to a given game.
    frame = (frame.dropna(subset=["observed_at"])
                  .sort_values("observed_at")
                  .drop_duplicates(["game_id", "provider"], keep="last"))

    rows = []
    for game_id, game in frame.groupby("game_id", sort=False):
        spread_all = game["spread"].dropna().astype(float)
        total_all = game["total"].dropna().astype(float)

        # The chosen book must have BOTH numbers; a book quoting only a spread cannot
        # supply the total, and taking the two halves from different books would rebuild
        # the merged line this function exists to stop producing.
        priced = game.dropna(subset=["spread", "total"])
        if len(priced):
            pick = priced.loc[
                min(priced.index, key=lambda i: _rank(priced.at[i, "provider"], priority))]
            spread, total = float(pick["spread"]), float(pick["total"])
            chosen = str(pick["provider"])
        else:
            spread = total = np.nan
            chosen = None

        rows.append({
            "game_id": game_id,
            "spread": spread,
            "total": total,
            "chosen_provider": chosen,
            # How far the books that were NOT chosen sat from each other. Diagnostic only:
            # it never moves the quoted number.
            "spread_dispersion": (float(spread_all.max() - spread_all.min())
                                  if len(spread_all) else np.nan),
            "total_dispersion": (float(total_all.max() - total_all.min())
                                 if len(total_all) else np.nan),
            "book_count_spread": int(len(spread_all)),
            "book_count_total": int(len(total_all)),
            "is_consensus": bool(len(spread_all) >= minimum_books
                                 and len(total_all) >= minimum_books),
            "observed_at": game["observed_at"].max(),
            "providers": tuple(sorted(game["provider"].astype(str).unique())),
        })
    return pd.DataFrame(rows)


__all__ = ["aggregate_market_consensus", "DEFAULT_PROVIDER_PRIORITY"]
