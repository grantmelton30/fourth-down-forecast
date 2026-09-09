"""Price each side using its win/loss/push probabilities and offered odds.

This module evaluates a supplied distribution; it does not establish that the
model is calibrated or that a quoted price can be executed.
"""
from __future__ import annotations
import numpy as np


def _odds(value):
    odds = float(value)
    if not np.isfinite(odds) or abs(odds) < 100:
        raise ValueError("American odds must be finite and at least 100 in absolute value")
    return odds


def _sims(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("simulations must be a nonempty finite one-dimensional array")
    return values


def breakeven_prob(american_odds):
    """Break-even probability conditional on the bet resolving."""
    o = _odds(american_odds)
    return abs(o)/(abs(o)+100) if o < 0 else 100/(o+100)


def ev_per_unit(p_win, american_odds, *, p_push=0.0):
    """Net expectation per unit risked; a push returns the stake."""
    o = _odds(american_odds)
    w, p = float(p_win), float(p_push)
    if not np.isfinite([w,p]).all() or min(w,p) < 0 or w+p > 1+1e-12:
        raise ValueError("win and push probabilities must form a valid distribution")
    payout = 100/abs(o) if o < 0 else o/100
    return w*payout - max(0, 1-w-p)


def required_edge(sims, american_odds=-110, max_shift=200.0, step=0.5):
    sims = _sims(sims)
    _odds(american_odds)
    if not np.isfinite([max_shift,step]).all() or step <= 0 or max_shift <= 0:
        raise ValueError("shift range and step must be positive and finite")
    med = float(np.median(sims))
    for shift in np.arange(step, max_shift+step, step):
        low, high = med-shift, med+shift
        over = ev_per_unit(np.mean(sims>low),american_odds,p_push=np.mean(sims==low))
        under = ev_per_unit(np.mean(sims<high),american_odds,p_push=np.mean(sims==high))
        if max(over,under) > 0:
            return float(shift)
    return float("inf")


def select_bets(sims_by_row, lines, prices=-110, *, min_ev=0.0,
                over_prices=None, under_prices=None):
    """Select only positive EV; supply both side-price arrays for real quotes.

    `prices` remains a symmetric-price research convenience. Missing/invalid
    supplied side prices reject the row rather than borrowing the other price.
    """
    lines = np.atleast_1d(np.asarray(lines,dtype=float))
    if lines.ndim != 1 or not np.isfinite(lines).all() or not np.isfinite(min_ev) or min_ev < 0:
        raise ValueError("finite lines and nonnegative min_ev required")
    boards = list(sims_by_row)
    if len(boards) != len(lines):
        raise ValueError("one simulation array is required per line")
    if (over_prices is None) != (under_prices is None):
        raise ValueError("supply both over_prices and under_prices")
    op = np.broadcast_to(np.asarray(prices if over_prices is None else over_prices,dtype=float),lines.shape)
    up = np.broadcast_to(np.asarray(prices if under_prices is None else under_prices,dtype=float),lines.shape)
    out=[]
    for values,line,over_price,under_price in zip(boards,lines,op,up):
        sims = _sims(values)
        if not np.isfinite([over_price,under_price]).all():
            out.append(None)
            continue
        over,under,push = np.mean(sims>line),np.mean(sims<line),np.mean(sims==line)
        eo = ev_per_unit(over,over_price,p_push=push)
        eu = ev_per_unit(under,under_price,p_push=push)
        if max(eo,eu) <= min_ev:
            out.append(None)
            continue
        side,win,ev,price = (("OVER",over,eo,over_price) if eo>=eu else ("UNDER",under,eu,under_price))
        out.append({"side":side,"p":float(win),"p_push":float(push),
                    "p_loss":float(under if side=="OVER" else over),
                    "ev":float(ev),"line":float(line),"price":float(price)})
    return out


__all__=["breakeven_prob","ev_per_unit","required_edge","select_bets"]
