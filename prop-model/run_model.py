#!/usr/bin/env python
"""Evaluate the rebuilt prop model against the baselines it has to beat.

    uv run python prop-model/run_model.py --market receptions
    uv run python prop-model/run_model.py --market receptions --holdout 2025

TWO SCOREBOARDS, AND THE SECOND IS THE REAL ONE.

MAE says whether the point estimate is close. It is reported because the first model was
judged on it and the comparison has to be like-for-like -- but a prop is not a point
estimate, and this repo has already proved accuracy and profitability are different
questions (Elo is MORE accurate than its NFL model and still returns 48.9 per 100).

CALIBRATION and LOG SCORE say whether the probabilities are usable. A model whose "70%"
happens 55% of the time will lose money however good its MAE is. GATE_CALIBRATED already
fails on team totals in this repo for precisely that reason.

CHRONOLOGICAL HOLDOUT. `--holdout` fits nothing and tunes nothing on the final season, so
the reported numbers are out of sample in the only direction that matters. Everything the
model needs is walk-forward anyway, but the split makes the claim checkable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
    import pandas as pd
    from src.baselines import add_baselines, score
    from src.player_stats import attach_kickoffs, load_player_weeks
    from src.prop_model import (add_opponent, calibration, defense_effects, fit_dispersion,
                                home_effect, is_count_market, log_score, predict)
except ModuleNotFoundError as exc:
    _v = Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"
    if not _v.exists():
        _v = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    sys.exit(f"{exc}\n\nRun with the repo venv:\n    {_v} run_model.py\n"
             f"    uv run python prop-model/run_model.py   (from the repo root)\n")

# Opportunity denominator and the minimum average opportunity worth predicting.
# Ordered by how much the defence adjustment can actually move the bet, measured
# 2026-08-22: the defence effect runs 5-11% of a typical line in every market, and its
# half-season persistence is highest for rushing yards (0.331) and near zero for pass
# attempts (0.054) -- attempts are driven by game script, not by who you play.
MARKETS = {"receptions": ("targets", 2.0), "carries": ("carries", 3.0),
           "attempts": ("attempts", 10.0), "rushing_yards": ("carries", 3.0),
           "receiving_yards": ("targets", 2.0), "passing_yards": ("attempts", 10.0)}


def build(seasons, market):
    import nflreadpy as nfl

    stats = load_player_weeks(seasons)
    sched = nfl.load_schedules().to_pandas()
    sched = sched[sched["season"].isin(seasons)].copy()
    sched["kickoff"] = pd.to_datetime(
        sched["gameday"] + " " + sched["gametime"].fillna("13:00"))
    f = attach_kickoffs(stats, sched)
    f = add_opponent(f, sched)
    f = add_baselines(f, stat=market)
    # The EWMA is the base -- it is the estimator that beat the first model, so the rebuild
    # corrects it rather than competing with it.
    f = defense_effects(f, stat=market, baseline_col="base_ewma")
    f = home_effect(f, stat=market, baseline_col="base_ewma")
    return predict(f, baseline_col="base_ewma")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    p.add_argument("--market", default="receptions", choices=sorted(MARKETS))
    p.add_argument("--holdout", type=int, help="score only this season (out of sample)")
    p.add_argument("--min-prior", type=int, default=4)
    args = p.parse_args()

    denom, min_opp = MARKETS[args.market]
    f = build(args.seasons, args.market)
    f = f[f.groupby("player_id")[denom].transform("mean") >= min_opp]
    f = f[f["prior_games"] >= args.min_prior]

    scored = f[f["season"] == args.holdout] if args.holdout else f
    if scored.empty:
        return print(f"no rows for holdout {args.holdout}") or 1

    print(f"market   : {args.market}")
    print(f"seasons  : {min(args.seasons)}-{max(args.seasons)}"
          + (f"   scored on {args.holdout} only (out of sample)" if args.holdout else ""))
    print(f"rows     : {len(scored):,}")

    table = score(scored, stat=args.market,
                  estimators=["base_last_n", "base_ewma", "base_season_mean", "pred_mean"],
                  min_prior=args.min_prior)
    print(f"\n{'estimator':<18}{'n':>8}{'MAE':>9}{'RMSE':>9}{'bias':>9}{'vs baseline':>14}")
    for r in table.itertuples(index=False):
        tag = "  <- REBUILT" if r.estimator == "pred_mean" else ""
        print(f"{r.estimator:<18}{r.n:>8,}{r.mae:>9.3f}{r.rmse:>9.3f}{r.bias:>+9.3f}"
              f"{r.vs_best_baseline:>+14.4f}{tag}")

    model = table[table["estimator"] == "pred_mean"].iloc[0]
    best = table[table["estimator"].str.startswith("base_")].iloc[0]
    delta = 100 * (best["mae"] - model["mae"]) / best["mae"]
    print(f"\npoint estimate: model {'BEATS' if delta > 0 else 'LOSES to'} "
          f"{best['estimator']} by {abs(delta):.1f}% MAE")

    # The scoreboard that decides whether this can price a bet.
    if not is_count_market(args.market):
        print(f"\nNO DISTRIBUTION SCORE for {args.market}: a negative binomial cannot")
        print("represent yardage -- point mass at zero, negative rushing yards, heavy tails.")
        print("Fitting one produces -inf log scores. A hurdle or empirical-residual model is")
        print("needed before this market can be priced. MAE above is the only valid number.")
        return 0
    r = fit_dispersion(f[f["season"] != args.holdout] if args.holdout else f,
                       stat=args.market)
    print(f"\ndispersion r = {r:.2f}   "
          f"({'overdispersed vs Poisson' if np.isfinite(r) else 'Poisson limit'})")
    for label, col in [("EWMA baseline", "base_ewma"), ("REBUILT model", "pred_mean")]:
        ls = log_score(scored, stat=args.market, mean_col=col, r=r)
        cal = calibration(scored, stat=args.market, mean_col=col, r=r)
        worst = cal.loc[cal["gap"].abs().idxmax()] if len(cal) else None
        print(f"\n{label}")
        print(f"  log score (higher is better): {ls:+.4f}")
        if worst is not None:
            print(f"  worst calibration bucket: predicted {100*worst['predicted']:.0f}% "
                  f"actual {100*worst['actual']:.0f}%  gap {100*worst['gap']:+.1f}pp "
                  f"(n={int(worst['n'])})")
            print(f"  mean |gap| across buckets: "
                  f"{100*cal['gap'].abs().mean():.1f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
