"""Backtest the new player/roster-level candidate features (added 2026-08-12/13)
against MARKET error, not model error: qb_continuity, portal_net_rating,
returning_production, talent_composite, recruiting_rating, head_coach_continuity.

Standalone, not wired into any gate or the live model, and does not touch
model_spread / model_total. Only pulls games + lines + the free preseason feature
sources (`ingest.load_games`, `ingest.load_lines`, `features.load_free_preseason`) --
deliberately skips plays/drives/ratings/the simulator, since this question does not need
them: it is asking whether a team-level attribute the market may under-react to predicts
where the OPENING line missed, not whether the model's own projection beats the market.

Method
------
For each graded, opener-covered game:
  `{feature}_diff`  = home value - away value   (tested against spread error)
  `{feature}_sum`   = home value + away value   (tested against total error)
  `market_error_spread` = actual_margin - spread_open
  `market_error_total`  = actual_total  - total_open

A significant, sign-stable b means the opening line under- or over-weights that
attribute relative to what it should. This is exactly the shape of test that
`_fit_feature_challenger` runs (out-of-sample RMSE improvement) before promoting a
feature into the live mean; this script instead asks the more basic, more legible
question directly against the market with an interpretable regression coefficient
and t-stat, on the full historical window rather than a single held-out season.

Run with a real CFBD_API_KEY set (free tier, ~60-70 calls for this reduced pull --
well under the repo's own 250-call cold-build guard and the 900/month self-tracked
budget). Nothing here is cached permanently beyond the repo's normal parquet cache
under CACHE_DIR.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robust_inference import fit_ols  # noqa: E402

from src import ingest
from src.cfbd_client import BudgetedCFBD
from src.config import load_config
from src.features import load_free_preseason, normalize_preseason_sources

CANDIDATES = (
    "qb_continuity", "portal_net_rating", "returning_production",
    "talent_composite", "recruiting_rating", "head_coach_continuity",
)


def load():
    cfg = load_config()
    client = BudgetedCFBD(cfg)
    games = ingest.load_games(client, cfg.all_seasons)
    lines = ingest.load_lines(client, cfg, cfg.all_seasons)
    market = ingest.build_market(lines, games, cfg)
    preseason_raw = load_free_preseason(client, cfg.all_seasons)
    preseason = normalize_preseason_sources(**preseason_raw)
    print(f"CFBD calls used this run: {client.calls_used}")
    return market, preseason


def build_frame(market: pd.DataFrame, preseason: pd.DataFrame) -> pd.DataFrame:
    games = market[
        market["completed"].fillna(False)
        & market["fbs_only"]
        & market["actual_margin"].notna()
        & market["actual_total"].notna()
    ].copy()

    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        cols = {c: f"{side}_{c}" for c in CANDIDATES}
        p = preseason.rename(columns={"team": team_col, **cols})
        games = games.merge(
            p[["season", team_col, *cols.values()]], on=["season", team_col], how="left"
        )

    for c in CANDIDATES:
        games[f"{c}_diff"] = games[f"home_{c}"] - games[f"away_{c}"]
        games[f"{c}_sum"] = games[f"home_{c}"] + games[f"away_{c}"]

    games["market_error_spread"] = games["actual_margin"] - games["spread_open"]
    games["market_error_total"] = games["actual_total"] - games["total_open"]
    return games


def report(games: pd.DataFrame, predictor: str, response: str, label: str) -> "float | None":
    d = games.dropna(subset=[predictor, response])
    if len(d) < 50:
        print(f"{label:<55} only {len(d)} rows -- too few to fit")
        return None
    fit = fit_ols(
        d[response], d[predictor], covariance="cluster", groups=d["season"].to_numpy()
    )
    b, se_b, t_b = float(fit.beta[1]), float(fit.se[1]), float(fit.t_values[1])
    sig = "  <-- |t|>=1.96" if abs(t_b) >= 1.96 else ""
    print(
        f"{label:<55} n={fit.n:>5}  b={b:+.5f}  se={se_b:.5f}  t={t_b:+.2f}  "
        f"r2={fit.r_squared:.4f}{sig}"
    )
    return t_b


def main() -> int:
    market, preseason = load()
    games = build_frame(market, preseason)
    opener = games[games["has_opener"]]
    print(f"gradeable FBS games: {len(games):,}  with opener coverage: {len(opener):,}")
    print(f"seasons: {sorted(games['season'].unique())}")
    print()

    print("ALL WEEKS")
    for c in CANDIDATES:
        report(opener, f"{c}_diff", "market_error_spread", f"  {c}_diff -> spread error")
        report(opener, f"{c}_sum", "market_error_total", f"  {c}_sum -> total error")

    print()
    print("WEEKS <= 5 (roster uncertainty highest -- where a real effect should live)")
    early = opener[opener["week"] <= 5]
    for c in CANDIDATES:
        report(early, f"{c}_diff", "market_error_spread", f"  {c}_diff -> spread error")
        report(early, f"{c}_sum", "market_error_total", f"  {c}_sum -> total error")

    print()
    print("WEEKS >= 6 (should fade toward zero if the effect is real and roster-driven)")
    late = opener[opener["week"] >= 6]
    for c in CANDIDATES:
        report(late, f"{c}_diff", "market_error_spread", f"  {c}_diff -> spread error")
        report(late, f"{c}_sum", "market_error_total", f"  {c}_sum -> total error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
