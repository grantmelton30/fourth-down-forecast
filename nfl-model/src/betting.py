"""Bet selection and staking (§9).

Nothing here re-simulates. The blended line is applied by *shifting* the simulated margin
distribution, which preserves the key-number shape the simulator worked to produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .market import (
    BlendWeights,
    american_to_decimal,
    apply_blend,
    apply_blend_total,
    key_number_value,
)
from .simulate import SimResult

BET_SHEET_COLUMNS = [
    "week", "kickoff", "away", "home", "market_spread", "model_spread",
    "blended_spread", "edge_pts", "market_total", "model_total", "blended_total",
    "edge_total", "pick", "pick_line", "cover_prob", "key_cross", "stake", "kelly_pct",
    "bet_to_line", "confidence", "notes",
]


@dataclass
class GameProjection:
    """Everything the bet sheet needs about one game."""

    game_id: str
    season: int
    week: int
    kickoff: object
    home: str
    away: str
    sim: SimResult
    market_spread: float
    market_total: float
    spread_price_home: int = -110
    spread_price_away: int = -110
    total_price_over: int = -110
    total_price_under: int = -110
    data_incomplete: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def model_spread(self) -> float:
        return self.sim.mean_margin

    @property
    def model_total(self) -> float:
        return self.sim.mean_total


def early_season_thresholds(week: int, cfg: Config) -> tuple:
    """Weeks 1-4 lean on the prior and deserve a higher bar (§9)."""
    edge_spread = cfg.betting.min_edge_points_spread
    edge_total = cfg.betting.min_edge_points_total
    min_prob = cfg.betting.min_cover_prob
    if week <= 4:
        return edge_spread * 1.6, edge_total * 1.6, min_prob + 0.015
    return edge_spread, edge_total, min_prob


def kelly_stake(prob: float, price: int, cfg: Config) -> tuple:
    """Quarter-Kelly, capped, floored at zero."""
    b = american_to_decimal(price) - 1.0
    if b <= 0:
        return 0.0, 0.0
    f = max((prob * b - (1.0 - prob)) / b, 0.0)
    capped = min(cfg.betting.kelly_fraction * f, cfg.betting.max_stake_pct_bankroll)
    return capped * cfg.betting.bankroll, f


def confidence_band(prob: float, cfg: Config) -> str:
    """Driven by cover probability bands, not by vibes."""
    if prob >= cfg.betting.min_cover_prob + 0.03:
        return "HIGH"
    if prob >= cfg.betting.min_cover_prob:
        return "MED"
    return "NONE"


def bet_to_line(
    sim: SimResult,
    blended: float,
    market_line: float,
    side: str,
    min_prob: float,
    min_edge: float,
    is_total: bool = False,
) -> float:
    """The worst number at which this bet is still above threshold.

    "Bet this down to -4.5, no further." This is the column you actually use when shopping
    the number, so it is computed rather than left blank: step the line half a point at a
    time in the direction that hurts, and return the last value where both the probability
    and the edge condition still hold.
    """
    prob_fn = sim.total_prob if is_total else sim.cover_prob
    direction = 1.0 if side in ("home", "over") else -1.0
    worst = None
    for i in range(0, 61):
        candidate = market_line + direction * 0.5 * i
        edge_ok = (blended - candidate) * direction >= min_edge
        if prob_fn(candidate, side) >= min_prob and edge_ok:
            worst = candidate
        else:
            break
    return float(worst) if worst is not None else float("nan")


def build_bet_sheet(
    games: "list[GameProjection]", weights: BlendWeights, cfg: Config
) -> pd.DataFrame:
    """One row per game, with a live pick or a NO BET."""
    rows = []
    for g in games:
        min_edge_spread, min_edge_total, min_prob = early_season_thresholds(g.week, cfg)

        blended_spread = apply_blend(g.market_spread, g.model_spread, weights)
        blended_total = apply_blend_total(g.market_total, g.model_total, weights)

        # Shift, do not re-simulate: this preserves the key-number structure.
        shifted = g.sim.recentered(blended_spread).retotaled(blended_total)
        pmf = shifted.margin_pmf()

        edge_pts = blended_spread - g.market_spread
        edge_total = blended_total - g.market_total

        candidates = []
        for side, price in (("home", g.spread_price_home), ("away", g.spread_price_away)):
            candidates.append({
                "kind": "spread", "side": side, "price": price,
                "prob": shifted.cover_prob(g.market_spread, side),
                "edge": edge_pts if side == "home" else -edge_pts,
                "line": g.market_spread, "min_edge": min_edge_spread,
                "blended": blended_spread,
            })
        for side, price in (("over", g.total_price_over), ("under", g.total_price_under)):
            candidates.append({
                "kind": "total", "side": side, "price": price,
                "prob": shifted.total_prob(g.market_total, side),
                "edge": edge_total if side == "over" else -edge_total,
                "line": g.market_total, "min_edge": min_edge_total,
                "blended": blended_total,
            })

        live = [
            c for c in candidates
            if c["prob"] >= min_prob
            and c["edge"] >= c["min_edge"]
            and not g.data_incomplete
            and ((c["kind"]=="spread" and weights.b_model_raw>0 and weights.b_model>0 and weights.t_model>2) or
                 (c["kind"]=="total" and weights.b_model_total_raw>0 and weights.b_model_total>0 and weights.t_model_total>2))
        ]
        best = max(live, key=lambda c: c["prob"]) if live else None

        row = {
            "week": g.week,
            "kickoff": g.kickoff,
            "away": g.away,
            "home": g.home,
            "market_spread": g.market_spread,
            "model_spread": g.model_spread,
            "blended_spread": blended_spread,
            "edge_pts": edge_pts,
            "market_total": g.market_total,
            "model_total": g.model_total,
            "blended_total": blended_total,
            "edge_total": edge_total,
            "pick": "NO BET",
            "pick_line": np.nan,
            "cover_prob": np.nan,
            "key_cross": key_number_value(blended_spread, g.market_spread, pmf),
            "stake": 0.0,
            "kelly_pct": 0.0,
            "bet_to_line": np.nan,
            "confidence": "NONE",
            "notes": g.notes + (" data_incomplete" if g.data_incomplete else ""),
            "game_id": g.game_id,
            "season": g.season,
            "pick_kind": None,
            "pick_side": None,
        }

        if best is not None:
            stake, f = kelly_stake(best["prob"], best["price"], cfg)
            row.update({
                "pick": _pick_label(best, g),
                "pick_line": best["line"],
                "cover_prob": best["prob"],
                "stake": stake,
                "kelly_pct": f,
                "confidence": confidence_band(best["prob"], cfg),
                "pick_kind": best["kind"],
                "pick_side": best["side"],
                "bet_to_line": bet_to_line(
                    shifted, best["blended"], best["line"], best["side"],
                    min_prob, best["min_edge"], is_total=best["kind"] == "total",
                ),
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[BET_SHEET_COLUMNS + ["game_id", "season", "pick_kind", "pick_side"]]


def _pick_label(best: dict, g: GameProjection) -> str:
    if best["kind"] == "spread":
        team = g.home if best["side"] == "home" else g.away
        line = -best["line"] if best["side"] == "home" else best["line"]
        return f"{team} {line:+.1f}"
    return f"{best['side'].upper()} {best['line']:.1f}"
