#!/usr/bin/env python
"""Free accuracy test: can the usage model beat a rolling average?

    uv run python prop-model/run_accuracy.py --seasons 2021 2022 2023 2024
    uv run python prop-model/run_accuracy.py --market receptions

NO MARKET DATA NEEDED, AND THAT IS THE POINT. Historical prop lines are paid-only
(confirmed against the live API: HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN). But the
question that can kill this project cheapest does not need them: if the model cannot predict
a player's receptions better than "what did he average recently", no market data will rescue
it, and nothing further is worth building or buying.

WHAT A PASS HERE DOES AND DOES NOT MEAN. Beating the baselines earns the right to spend $30
on market history. It is NOT evidence of an edge. This repo has already settled that those
are different questions: Elo is MORE accurate than the NFL model and still returns 48.9 wins
per 100. Accuracy is necessary and nowhere near sufficient.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import pandas as pd
    from src.baselines import add_baselines, add_team_volume, add_usage_model, score
    from src.player_stats import attach_kickoffs, load_player_weeks, usage_shares
except ModuleNotFoundError as exc:
    _venv = Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"
    if not _venv.exists():
        _venv = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    sys.exit(f"{exc}\n\nRun with the repo venv:\n    {_venv} run_accuracy.py\n"
             f"    uv run python prop-model/run_accuracy.py   (from the repo root)\n")

# stat -> (opportunity denominator, minimum prior opportunity to be worth predicting)
MARKETS = {
    "receptions": ("targets", 2.0),
    "carries": ("carries", 3.0),
    "attempts": ("attempts", 10.0),          # pass attempts
    "receiving_yards": ("targets", 2.0),
    "rushing_yards": ("carries", 3.0),
}


def build(seasons: list[int], market: str) -> pd.DataFrame:
    import nflreadpy as nfl

    denom, _ = MARKETS[market]
    stats = load_player_weeks(seasons)
    sched = nfl.load_schedules().to_pandas()
    sched = sched[sched["season"].isin(seasons)].copy()
    sched["kickoff"] = pd.to_datetime(
        sched["gameday"] + " " + sched["gametime"].fillna("13:00"))
    joined = attach_kickoffs(stats, sched)

    usage = usage_shares(joined)
    share_col = {"targets": "receptions_share", "carries": "rushing_attempts_share",
                 "attempts": "pass_attempts_share"}[denom]
    if share_col not in usage.columns:
        raise SystemExit(f"no usage share built for {denom}")

    frame = joined.merge(
        usage[["player_id", "season", "week", share_col]],
        on=["player_id", "season", "week"], how="left").sort_values("kickoff")
    frame = add_baselines(frame, stat=market)
    frame = add_team_volume(frame, denom=denom)
    frame = add_usage_model(frame, share_col=share_col, stat=market, denom=denom)
    return frame


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", type=int, nargs="+",
                   default=[2021, 2022, 2023, 2024, 2025])
    p.add_argument("--market", default="receptions", choices=sorted(MARKETS))
    p.add_argument("--min-prior", type=int, default=4,
                   help="games of history required before a row is scored")
    args = p.parse_args()

    denom, min_opp = MARKETS[args.market]
    frame = build(args.seasons, args.market)

    # Only players with real involvement. Scoring every kicker and lineman at zero would
    # let a predict-zero estimator look excellent and hide any real difference.
    frame = frame[frame.groupby("player_id")[denom].transform("mean") >= min_opp]

    print(f"market       : {args.market}  (opportunity: {denom})")
    print(f"seasons      : {min(args.seasons)}-{max(args.seasons)}")
    print(f"player-weeks : {len(frame):,}")
    table = score(frame, stat=args.market, min_prior=args.min_prior)
    print()
    print(f"{'estimator':<20}{'n':>8}{'MAE':>9}{'RMSE':>9}{'bias':>9}{'vs best baseline':>19}")
    for r in table.itertuples(index=False):
        flag = "  <- MODEL" if r.estimator == "model_pred" else ""
        print(f"{r.estimator:<20}{r.n:>8,}{r.mae:>9.3f}{r.rmse:>9.3f}{r.bias:>+9.3f}"
              f"{r.vs_best_baseline:>+19.4f}{flag}")

    model = table[table["estimator"] == "model_pred"].iloc[0]
    best_base = table[table["estimator"].str.startswith("base_")].iloc[0]
    print()
    if model["mae"] < best_base["mae"]:
        gain = 100 * (best_base["mae"] - model["mae"]) / best_base["mae"]
        print(f"MODEL BEATS the best baseline ({best_base['estimator']}) by {gain:.1f}% MAE.")
        print("That earns the right to buy market history. It is NOT an edge -- Elo is more")
        print("accurate than this repo's NFL model and still loses money.")
    else:
        loss = 100 * (model["mae"] - best_base["mae"]) / best_base["mae"]
        print(f"MODEL LOSES to {best_base['estimator']} by {loss:.1f}% MAE.")
        print("Do not buy market history on this. Fix the model or stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
