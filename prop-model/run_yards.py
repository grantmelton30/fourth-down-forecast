#!/usr/bin/env python
"""Price a yardage prop as a compound sum, and check whether the probabilities are honest.

    uv run python prop-model/run_yards.py --market rushing_yards --holdout 2025

WHAT THIS FIXES. The earlier attempt fitted a negative binomial to game yardage and scored
-inf, because 8.8% of real carries lose yards and any distribution on the non-negative
integers gives those probability zero. This models the two layers separately instead:

    touches  ~ negative binomial          counts, where that distribution IS valid
    per touch ~ rounded asymmetric Laplace   skewed, peaked, unbounded below
    game yards = sum of the touches

The per-touch distribution is the one Glazer, Parast and Hooten recommend (The American
Statistician, 2026), and it replicates on this repo's own play-by-play: AIC 261,464 against
292,202 for a normal, with tau = 0.313 confirming the right skew they report.

THE SCOREBOARD IS CALIBRATION, NOT MAE. A yardage prop asks P(over 84.5). A model with a
good mean and a wrong shape prices that wrong, and this repo already has GATE_CALIBRATED
failing on team totals for exactly that reason. MAE is printed for continuity with the
earlier runs; the calibration table is what decides anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
    import pandas as pd
    from src.baselines import add_baselines
    from src.player_stats import attach_kickoffs, load_player_weeks
    from src.prop_model import add_opponent, defense_effects, predict
    from src.yardage import (fit_ald, fit_touch_dispersion, prob_over_from_sims,
                             simulate_game_yards)
except ModuleNotFoundError as exc:
    _v = Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"
    if not _v.exists():
        _v = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    sys.exit(f"{exc}\n\nRun with the repo venv:\n    {_v} run_yards.py\n"
             f"    uv run python prop-model/run_yards.py   (from the repo root)\n")

MARKETS = {
    "rushing_yards": ("carries", "run", "rusher_player_id", 3.0),
    "receiving_yards": ("targets", "pass", "receiver_player_id", 2.0),
}


def per_touch_sample(market: str, seasons: list[int]) -> np.ndarray:
    """Per-play gains, from the EXISTING pbp cache -- no re-pull, no cache invalidation.

    Loads nfl-model's `src` as a uniquely-named PACKAGE rather than importing a single file.
    Both repos have a package called `src`, and ingest.py uses relative imports, so loading
    it as a bare module fails on `from .config import CACHE_DIR`. `shared/sport.py` solves
    this the same way and for the same reason.
    """
    import importlib.util

    repo = Path(__file__).resolve().parents[1] / "nfl-model"
    package = "football_model_nfl_src"
    if package not in sys.modules:
        init = repo / "src" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            package, init, submodule_search_locations=[str(init.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    ing = importlib.import_module(f"{package}.ingest")

    _, play_type, id_col, _ = MARKETS[market]
    pbp = ing.load_pbp(seasons)
    sub = pbp[(pbp["play_type"] == play_type) & pbp["yards_gained"].notna()]

    if id_col in pbp.columns:
        sub = sub[sub[id_col].notna()]
    elif play_type == "pass":
        # `receiver_player_id` is not in PBP_COLUMNS, and widening it would invalidate the
        # working model's pbp cache. But the count being paired with this distribution is
        # TARGETS, so the per-touch sample must be targets too -- SACKS ARE NOT TARGETS.
        # Leaving them in put mu at 0.00 and biased receiving yards by -5.5 per game,
        # because every sack entered the sample as a large negative "reception".
        # An incompletion IS a target and correctly contributes zero.
        sub = sub[sub["sack"] != 1]
    return sub["yards_gained"].to_numpy(dtype=float)


from src.evaluation import eligible_rows, chronological_split

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    p.add_argument("--market", default="rushing_yards", choices=sorted(MARKETS))
    p.add_argument("--holdout", type=int, default=2025)
    p.add_argument("--sims", type=int, default=4000)
    p.add_argument("--min-prior", type=int, default=4)
    args = p.parse_args()

    denom, _, _, min_opp = MARKETS[args.market]
    import nflreadpy as nfl

    stats_ = load_player_weeks(args.seasons)
    sched = nfl.load_schedules().to_pandas()
    sched = sched[sched["season"].isin(args.seasons)].copy()
    sched["kickoff"] = pd.to_datetime(
        sched["gameday"] + " " + sched["gametime"].fillna("13:00"))
    f = add_opponent(attach_kickoffs(stats_, sched), sched)

    # Predict TOUCHES (a count), not yards. The yards come from the compound sum.
    f = add_baselines(f, stat=denom)
    f = defense_effects(f, stat=denom, baseline_col="base_ewma")
    f = predict(f, baseline_col="base_ewma")
    f = eligible_rows(f, denom, min_opp)
    f = f[f["prior_games"] >= args.min_prior].dropna(subset=[args.market, "pred_mean"])

    train, test = chronological_split(f, args.holdout)
    if test.empty:
        print(f"no rows for holdout {args.holdout}")
        return 1

    # Per-touch distribution fitted on TRAINING seasons only.
    touches = per_touch_sample(args.market, [s for s in args.seasons if s < args.holdout])
    ald = fit_ald(touches)
    touch_r = fit_touch_dispersion(train[denom])
    print(f"market        : {args.market}")
    print(f"per-touch ALD : mu={ald[0]:.2f} sigma={ald[1]:.2f} tau={ald[2]:.3f}  "
          f"(n={len(touches):,} plays, {args.seasons[0]}-{max(s for s in args.seasons if s < args.holdout)})")
    print(f"touch count r : {touch_r:.2f}")
    print(f"holdout       : {args.holdout}, {len(test):,} player-weeks")

    rng = np.random.default_rng(17)
    # LINES ACROSS THE WHOLE DISTRIBUTION, not just the median.
    #
    # The first version of this test put every line at the simulated median, so P(over) was
    # ~50% on every row by construction and the calibration table read 0.6pp -- a number
    # that proved only that the median is the median. Real books hang lines a player clears
    # 30% or 70% of the time, and a model can be perfect at the middle and badly wrong in
    # the tails, which is exactly where the prop money is.
    quantiles = (0.20, 0.35, 0.50, 0.65, 0.80)
    p_over, hit, means, actuals = [], [], [], []
    for row in test.itertuples(index=False):
        sims = simulate_game_yards(mean_touches=getattr(row, "pred_mean"),
                                   touch_r=touch_r, ald=ald, n_sims=args.sims, rng=rng)
        actual = float(getattr(row, args.market))
        for q in quantiles:
            line = float(np.floor(np.quantile(sims, q))) + 0.5
            p_over.append(prob_over_from_sims(sims, line))
            hit.append(float(actual > line))
            means.append(float(sims.mean()))
            actuals.append(actual)

    res = pd.DataFrame({"p_over": p_over, "hit": hit, "sim_mean": means,
                        "actual": actuals})
    err = res["sim_mean"] - res["actual"]
    print(f"\npoint estimate: MAE {err.abs().mean():.2f}  bias {err.mean():+.2f}")

    print("\nCALIBRATION -- does a predicted 60% happen 60% of the time?")
    res["bucket"] = pd.cut(res["p_over"], np.linspace(0, 1, 11), include_lowest=True)
    tab = res.groupby("bucket", observed=True).agg(
        n=("hit", "size"), predicted=("p_over", "mean"), actual=("hit", "mean"))
    tab["gap_pp"] = (100 * (tab["actual"] - tab["predicted"])).round(1)
    print(f"  {'bucket':<16}{'n':>7}{'predicted':>11}{'actual':>9}{'gap':>8}")
    for b, r in tab.iterrows():
        if r["n"] < 20:
            continue
        print(f"  {str(b):<16}{int(r['n']):>7}{100*r['predicted']:>10.1f}%"
              f"{100*r['actual']:>8.1f}%{r['gap_pp']:>+8.1f}")
    big = tab[tab["n"] >= 20]
    print(f"\n  weighted mean |gap|: "
          f"{(big['gap_pp'].abs() * big['n']).sum() / big['n'].sum():.1f}pp")
    print("  Under 3pp is usable for pricing. Over ~5pp and the probabilities are not")
    print("  trustworthy enough to bet, whatever the point estimate looks like.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
