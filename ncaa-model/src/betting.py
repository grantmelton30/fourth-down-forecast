"""Bet selection -- TOTALS ONLY. Spreads stay dark.

Spreads are not emitted: `b` is indistinguishable from zero on them. Coefficients are not
quoted here -- they drift with every refit and a stale literal in a docstring is how a
known-false number survives. See the blend artifact and Appendix A for current values.

NO STAKE SIZING. There is no `stake` column, so nothing here can be mistaken for one.
Stakes go live only once paper CLV is positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibrate import ConditionalPMF, calibrated_prob
from .config import Config
from .market import BlendWeights

PAPER_COLUMNS = [
    "season", "week", "kickoff", "away_team", "home_team",
    "total_open", "model_total", "blended_total", "edge_pts",
    "pick", "side", "cover_prob", "total_close", "line_move", "clv_positive",
    "actual_total", "result", "restricted", "cross_tier", "home_conference",
    "away_conference", "provider", "notes",
]


def blended_total(market_total: float, model_total: float, w: BlendWeights) -> float:
    """market + b * (model - market).

    NOTE: this anchors on the OPENER, and `model_total` has itself already been rescaled
    and level-shifted against the opener upstream (backtest.py::_fit_gain and
    ::_fit_week_bias). See DECISIONS "opener-injection points". The NCAA totals result
    measured through this path was shown to be an errors-in-variables artifact -- see
    NCAA_PLAYBOOK.md Appendix A. Retained as the record of what was computed.

    The model's absolute RMSE against actual totals is WORSE than the market line's.
    `rmse_gate` now blocks promotion on exactly that condition; the previous reading of
    it -- that the edge lived in the blend -- was a halt condition mistaken for an
    explanation.
    """
    return market_total + w.b_model_total * (model_total - market_total)


def build_totals_sheet(
    games: pd.DataFrame,
    weights: BlendWeights,
    cfg: Config,
    pmf: "ConditionalPMF | None" = None,
    picks_only: bool = True,
    caps: bool = True,
) -> pd.DataFrame:
    """One row per candidate game; only rows clearing every filter carry a pick.

    Selection is on the OPENER, which is the number that would actually have been bet.
    """
    df = games.copy()
    if cfg.betting.exclude_fcs_games:
        df = df[df["restricted"]]
    # Weeks the by-week bias gate still fails on after correction (DECISIONS D6d).
    excluded = set(cfg.betting.exclude_weeks_total)
    if excluded:
        df = df[~df["week"].isin(excluded)]
    df = df.dropna(subset=["total_open", "model_total"]).copy()

    df["blended_total"] = [
        blended_total(m, mo, weights)
        for m, mo in zip(df["total_open"], df["model_total"])
    ]
    df["edge_pts"] = df["blended_total"] - df["total_open"]

    if pmf is not None:
        raw = np.array([
            calibrated_prob(b, t, pmf)
            for b, t in zip(df["blended_total"], df["total_open"])
        ])
        # Probability of the side we would take, not of the over specifically.
        df["cover_prob"] = np.where(df["edge_pts"] >= 0, raw, 1.0 - raw)
    else:
        df["cover_prob"] = np.nan

    in_range = df["total_open"].between(cfg.betting.min_total, cfg.betting.max_total)
    big_enough = df["edge_pts"].abs() >= cfg.betting.min_edge_points_total
    confident = df["cover_prob"].isna() | (
        df["cover_prob"] >= cfg.betting.min_cover_prob
    )
    live = in_range & big_enough & confident

    df["side"] = np.where(live, np.sign(df["edge_pts"]), 0.0)
    df["pick"] = np.where(
        live,
        np.where(df["edge_pts"] > 0, "OVER ", "UNDER ")
        + df["total_open"].map(lambda v: f"{v:.1f}"),
        "NO BET",
    )

    # CLV: did the close move toward the side we took?
    df["line_move"] = df["total_close"] - df["total_open"]
    df["clv_positive"] = np.where(
        (df["side"] != 0) & df["line_move"].notna() & (df["line_move"] != 0),
        np.sign(df["line_move"]) == np.sign(df["side"]),
        np.nan,
    )

    if "actual_total" in df.columns:
        df["result"] = np.where(
            df["actual_total"].isna() | (df["side"] == 0), "",
            np.where(
                df["actual_total"] == df["total_open"], "PUSH",
                np.where(
                    np.sign(df["actual_total"] - df["total_open"]) == df["side"],
                    "WIN", "LOSS",
                ),
            ),
        )
    else:
        df["result"] = ""

    df["notes"] = np.where(live, "paper only -- no stake sized until CLV is positive", "")
    cols = [c for c in PAPER_COLUMNS if c in df.columns]
    out = df[cols].sort_values(["season", "week", "kickoff"])
    if not picks_only:
        return out
    out = out[out["pick"] != "NO BET"]
    # Caps are ON by default: a sheet that quietly concentrates on a few teams is the
    # failure mode they exist to catch, so it must not be possible to skip them by
    # forgetting a flag.
    return apply_concentration_caps(out, cfg) if caps else out


def apply_concentration_caps(sheet: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Cap how much of a pick set any one team or conference may supply.

    A slate where four of thirteen picks involve the same team is more plausibly a couple
    of rating errors producing repeat disagreements than a distributed edge. The caps make
    that distinction testable: if CLV survives them the edge is spread across the sample,
    and if it evaporates the concentration was carrying it -- which is a defect to fix
    rather than a signal to bet.

    Greedy on descending |edge|, so when a cap binds it is the weakest disagreements that
    are dropped, not an arbitrary subset.
    """
    if sheet.empty:
        return sheet
    max_team = max(1, int(np.floor(cfg.betting.max_team_pick_share * len(sheet))))
    max_conf = max(1, int(np.floor(cfg.betting.max_conference_pick_share * len(sheet))))

    team_n: dict = {}
    conf_n: dict = {}
    keep = []
    for idx, row in sheet.sort_values("edge_pts", key=abs, ascending=False).iterrows():
        teams = [row.get("home_team"), row.get("away_team")]
        confs = [row.get("home_conference"), row.get("away_conference")]
        if any(team_n.get(t, 0) >= max_team for t in teams if t):
            continue
        if any(conf_n.get(c, 0) >= max_conf for c in confs if c):
            continue
        for t in teams:
            if t:
                team_n[t] = team_n.get(t, 0) + 1
        for c in confs:
            if c:
                conf_n[c] = conf_n.get(c, 0) + 1
        keep.append(idx)
    return sheet.loc[sorted(keep)]


def clv_summary(sheet: pd.DataFrame) -> dict:
    """CLV and the blend read -- the two numbers that matter in a paper period.

    Deliberately not headlined by the win record: every filtered record's 5th percentile is
    still below 50%, so four weeks of wins carry almost no information, while CLV does not
    wait on outcomes and is far less noisy.
    """
    live = sheet[sheet["pick"] != "NO BET"] if "pick" in sheet else sheet
    graded = live[live["clv_positive"].notna()]
    moved = live[live["line_move"].fillna(0) != 0]
    out = {
        "picks": len(live),
        "with_close_move": len(graded),
        "clv": float(graded["clv_positive"].mean()) if len(graded) else float("nan"),
        "mean_move_toward_side": (
            float((moved["line_move"] * moved["side"]).mean()) if len(moved) else float("nan")
        ),
        "mean_edge_pts": (
            float(live["edge_pts"].abs().mean()) if len(live) else float("nan")
        ),
    }
    if "result" in live.columns:
        res = live["result"].value_counts()
        out["record"] = f"{res.get('WIN', 0)}-{res.get('LOSS', 0)}-{res.get('PUSH', 0)}"
    return out
