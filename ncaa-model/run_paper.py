#!/usr/bin/env python
"""Paper period -- totals only, no stakes.

    python run_paper.py                # last 4 completed weeks
    python run_paper.py --weeks 6

Logs, for every pick: the OPENER it would have been bet at, and the CLOSE it settled at.
Reports CLV and the blend read on the paper sample.

DO NOT TRADE. The result this script was built to produce was shown to be an
errors-in-variables artifact of the opener anchor (NCAA_PLAYBOOK.md Appendix A). It is
retained as the record of what was computed, not as a live signal. Coefficients and CLV
figures are not quoted in this docstring -- they go stale and become known-false.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import ingest
from src.backtest import select_window, walk_forward
from src.betting import build_totals_sheet, clv_summary
from src.calibrate import build_conditional, validate
from src.cfbd_client import BudgetedCFBD
from src.config import OUTPUT_DIR, load_config
from src.market import fit_blend,residual_fit,season_week_groups
from src.report import write_workbook
from src.ratings import build_walkforward

RULE = "=" * 96


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=0,
                    help="0 = the pre-declared window, weeks 4-16 of the last season")
    args = ap.parse_args()

    cfg = load_config()
    client = BudgetedCFBD(cfg)
    pd.set_option("display.width", 240)

    games = ingest.load_games(client, cfg.all_seasons)
    drives = ingest.load_drives(client, cfg.all_seasons)
    plays = ingest.load_plays(client, games, cfg.all_seasons)
    lines = ingest.load_lines(client, cfg, cfg.all_seasons)
    game_off = ingest.build_game_offense(plays, drives, games, cfg)
    market = ingest.build_market(lines, games, cfg)
    wf = build_walkforward(game_off, games, cfg)
    frame = walk_forward(cfg, market, wf)

    # The paper period is the last N completed weeks. The blend is fit on everything
    # BEFORE it, so these picks were made with coefficients that never saw these games.
    graded = select_window(frame, cfg.graded_seasons, restricted=True)
    last_season = int(graded["season"].max())
    weeks = sorted(graded.loc[graded["season"] == last_season, "week"].unique())
    # Pre-declared window: weeks 4-16, all of it. Not the last N weeks -- choosing the
    # window after seeing a result is how a false positive gets manufactured, and the
    # first paper run landed in weeks 13-16 purely by that construction.
    paper_weeks = weeks[-args.weeks:] if args.weeks else [w for w in weeks if 4 <= w <= 16]
    is_paper = (graded["season"] == last_season) & graded["week"].isin(paper_weeks)

    train, paper = graded[~is_paper], graded[is_paper]
    weights = fit_blend(train, cfg, window=f"pre-{last_season}wk{paper_weeks[0]}")
    weights.save()

    print(RULE)
    print(f"PAPER PERIOD -- {last_season} weeks {paper_weeks[0]}-{paper_weeks[-1]}, "
          "TOTALS ONLY, NO STAKES")
    print(RULE)
    print(f"blend fit on {len(train):,} earlier games (these picks were not in it)")
    print(f"  b_total  = {weights.b_model_total:+.4f}  t = {weights.t_model_total:+.2f}  "
          f"a = {weights.a_total:+.3f}")
    print(f"  b_spread = {weights.b_model_spread:+.4f}  t = {weights.t_model_spread:+.2f}"
          "   <- spreads stay dark, no picks emitted")

    pmf = build_conditional(
        train, "total_open", "actual_total", as_of_season=last_season, kind="total"
    )
    print(f"  L3 conditional PMF: {len(pmf.centers)} buckets, "
          f"median bucket n = {int(pd.Series(pmf.n_by_bucket).median())}")

    check = validate(pmf, train, "total_open", "actual_total", last_season - 1)
    if len(check):
        drift = float((check["calibrated"] - check["actual"]).abs().mean())
        print(f"  L3 level check: mean |calibrated - actual| across deciles = "
              f"{drift:.2f} pts")

    sheet = build_totals_sheet(paper, weights, cfg, pmf=pmf, picks_only=True)
    out_path = (
        OUTPUT_DIR
        / f"paper_totals_{last_season}_wk{paper_weeks[0]}-{paper_weeks[-1]}.csv"
    )
    sheet.to_csv(out_path, index=False)

    print()
    print(f"candidate games in paper period: {len(paper):,}   picks emitted: {len(sheet)}")
    if not len(sheet):
        print("no pick cleared the filters -- a model that declines is working correctly")
        return 0

    show = sheet[[
        "week", "away_team", "home_team", "total_open", "model_total",
        "blended_total", "edge_pts", "pick", "cover_prob", "total_close",
        "line_move", "clv_positive", "actual_total", "result",
    ]]
    print()
    print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    s = clv_summary(sheet)
    print()
    print(RULE)
    print("CLV REPORT")
    print(RULE)
    print(f"  picks                      {s['picks']}")
    print(f"  with a close that moved    {s['with_close_move']}")
    print(f"  CLV (close moved to us)    {s['clv']:.4f}   <- the number that matters")
    print(f"  mean move toward our side  {s['mean_move_toward_side']:+.3f} pts")
    print(f"  mean edge at entry         {s['mean_edge_pts']:.2f} pts")
    print(f"  record (context only)      {s.get('record', 'n/a')}")

    paper_fit = residual_fit(
        paper["model_total"], paper["total_open"], paper["actual_total"], "total",groups=season_week_groups(paper)
    )
    print(f"  b on paper sample          {paper_fit.b:+.4f} (t = {paper_fit.t_b:+.2f}, "
          f"n = {paper_fit.n}) -- tiny sample, context only")
    print()
    xl = write_workbook(sheet, weights, cfg, last_season, paper_weeks, frame)
    print(f"  wrote {out_path}")
    print(f"  wrote {xl}")
    print()
    if s["clv"] > 0.5:
        print("  CLV is positive on this sample. Four weeks is not yet grounds to size "
              "stakes.")
    else:
        print("  CLV is NOT positive. Stakes stay off, per the standing rule.")
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
