"""Which bets are worth making, following the PLOS framework rather than gut feel.

THE RESULT THAT REFRAMES EVERYTHING. "A statistical theory of optimal decision-making in
sports betting" (PLOS ONE, 2023, doi 10.1371/journal.pone.0287601) proves that if the
sportsbook has accurately captured the median outcome, **wagering always yields a negative
expected profit -- even when consistently betting the side with the higher probability of
winning**. When the book's number is right, your win rate is bounded between 47.6% and 52.4%
and the vig takes the difference.

So the objective is not "predict the player's yards accurately". A perfect prediction of a
correctly-priced line earns nothing. The objective is to find lines the book has placed away
from the true median, and bet only those.

The paper's other half is why the compound distribution was worth building: "knowledge of
the median outcome is a sufficient condition for optimal prediction, but ADDITIONAL
QUANTILES are necessary to optimally select the subset of matches to wager on". A point
estimate cannot tell you whether a bet is positive expected value. The shape can.

WHAT `required_edge` COMPUTES, AND WHY IT DECIDES THE PROJECT. Moving a line away from your
median changes your win probability at a rate set by the density there -- a tight
distribution swings quickly, a diffuse one barely moves. So each market has a minimum
distance the book must be wrong by before any bet clears the vig. On NFL spreads and totals
the paper measures that distance as roughly ONE POINT. If the equivalent figure for a
yardage prop turns out to be larger than books plausibly err, the market is unbeatable
regardless of model quality, and that is worth knowing before paying for data.
"""

from __future__ import annotations

import numpy as np


def breakeven_prob(american_odds: float) -> float:
    """Win rate needed to break even at a price. -110 -> 0.5238, -125 -> 0.5556."""
    o = float(american_odds)
    return abs(o) / (abs(o) + 100.0) if o < 0 else 100.0 / (o + 100.0)


def ev_per_unit(p_win: float, american_odds: float) -> float:
    """Expected profit per unit RISKED. Negative means do not bet, however good it feels."""
    o = float(american_odds)
    payout = 100.0 / abs(o) if o < 0 else o / 100.0
    return p_win * payout - (1.0 - p_win)


def required_edge(sims: np.ndarray, american_odds: float = -110,
                  max_shift: float = 200.0, step: float = 0.5) -> float:
    """How far the book's line must sit from your median before a bet is +EV.

    Walks the line away from the simulated median until the better side clears break-even
    at the offered price. Returns the distance in the market's own units -- yards for a
    yardage prop, receptions for a reception prop.

    This is the number that decides whether a market is worth attacking. It is a property of
    the DISTRIBUTION's shape and the PRICE, and needs no market data at all: a wide outcome
    distribution or a fat vig both push it up, and it can exceed anything a book would
    plausibly get wrong.
    """
    need = breakeven_prob(american_odds)
    med = float(np.median(sims))
    for shift in np.arange(step, max_shift + step, step):
        # The line moves AWAY from the median and the bet is taken on the near side, so the
        # win probability RISES with the shift. Getting this backwards -- taking the far
        # side -- makes every probability fall and every market look unbeatable, which is
        # what it did on the first run.
        #
        #   book too LOW  (line = med - shift)  -> take the OVER,  p = P(sims > line)
        #   book too HIGH (line = med + shift)  -> take the UNDER, p = P(sims < line)
        p_over = float(np.mean(sims > med - shift))
        p_under = float(np.mean(sims < med + shift))
        if max(p_over, p_under) >= need:
            return float(shift)
    return float("inf")


def select_bets(sims_by_row, lines, prices=-110, *, min_ev: float = 0.0):
    """Positive-EV selection: bet a side only when its own probability clears its own price.

    `sims_by_row` is one simulated distribution per candidate bet. Returns, per row, the
    chosen side and its EV, or None where neither side clears -- which the paper argues is
    the majority of the board and the whole point of the exercise.
    """
    lines = np.atleast_1d(np.asarray(lines, dtype=float))
    prices = np.broadcast_to(np.asarray(prices, dtype=float), lines.shape)
    out = []
    for sims, line, price in zip(sims_by_row, lines, prices):
        p_over = float(np.mean(sims > line))
        ev_over = ev_per_unit(p_over, price)
        ev_under = ev_per_unit(1.0 - p_over, price)
        if max(ev_over, ev_under) <= min_ev:
            out.append(None)
        elif ev_over >= ev_under:
            out.append({"side": "OVER", "p": p_over, "ev": ev_over, "line": line})
        else:
            out.append({"side": "UNDER", "p": 1.0 - p_over, "ev": ev_under, "line": line})
    return out


__all__ = ["breakeven_prob", "ev_per_unit", "required_edge", "select_bets"]
