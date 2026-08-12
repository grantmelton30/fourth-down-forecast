"""Project one game and print a score.

    .venv/bin/python project_game.py DAL PHI
    .venv/bin/python project_game.py DAL PHI --total 47.5 --spread -3
    .venv/bin/python project_game.py --game-id 2025_05_DAL_PHI

Runs the same drive-by-drive Monte Carlo the backtest grades, with the same ratings, the
same context adjustments and the same QB adjustment, and reports the projected score, the
distribution around it, and the edge against a line you supply.

READ THE EDGE SECTION. The backtest's own gates say this model does not beat the market on
either spreads or totals (GATE_BLEND_INFORMATIVE fails, b_model t = -0.60 on spreads and
+0.19 on totals over 1,535 games). The projection below is a real simulation of a real
model; it is not a betting signal, and this script refuses to dress it up as one.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src import ingest, nfelo
from src.backtest import _shared_artifacts
from src.config import load_config
from src.context import build_context, estimate_venue_hfa, load_resting_starters
from src.qb import qb_points_for_game
from src.ratings import ratings_at
from src.simulate import simulate_game


def _pick_game(schedules: pd.DataFrame, home: str, away: str, game_id: "str | None"):
    """Find the game to project: an explicit game_id, else the next unplayed meeting."""
    if game_id:
        hit = schedules[schedules["game_id"] == game_id]
        if hit.empty:
            sys.exit(f"no game with game_id={game_id!r}")
        return hit.iloc[0]

    meetings = schedules[
        (schedules["home_team"] == home) & (schedules["away_team"] == away)
    ].sort_values("kickoff")
    if meetings.empty:
        sys.exit(
            f"no scheduled game with {away} at {home}. Note the order is AWAY at HOME, "
            f"and team codes are nflverse abbreviations (e.g. LA, LAC, WAS)."
        )
    unplayed = meetings[meetings["result"].isna()]
    # Prefer the next unplayed meeting; fall back to the most recent played one so the
    # script is still useful out of season.
    return unplayed.iloc[0] if not unplayed.empty else meetings.iloc[-1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("home", nargs="?", help="home team abbreviation")
    p.add_argument("away", nargs="?", help="away team abbreviation")
    p.add_argument("--game-id", default=None)
    p.add_argument("--total", type=float, default=None,
                   help="Vegas total to compare against")
    p.add_argument("--spread", type=float, default=None,
                   help="Vegas spread, home perspective (-3 = home favored by 3)")
    p.add_argument("--n-sims", type=int, default=None)
    p.add_argument("--weather", action="store_true",
                   help="fetch the forecast from Open-Meteo (free, no key). Completed "
                        "games already carry nflverse's own reading and need no fetch.")
    args = p.parse_args()

    if not args.game_id and not (args.home and args.away):
        p.error("give HOME AWAY, or --game-id")

    cfg = load_config()
    if args.n_sims:
        cfg = cfg.with_sims(args.n_sims)

    schedules = ingest.load_schedules(cfg.train_seasons)
    pbp = ingest.load_pbp(cfg.train_seasons)
    game = _pick_game(schedules, args.home, args.away, args.game_id)

    season, week = int(game["season"]), int(game["week"])
    home, away = game["home_team"], game["away_team"]

    walkforward, start_fp, drive_model, endgame = _shared_artifacts(
        cfg, schedules, pbp, season
    )
    ratings = ratings_at(walkforward, season, week)
    for team in (home, away):
        if team not in ratings.index:
            sys.exit(f"no rating available for {team} in {season} week {week}")

    venue_hfa = estimate_venue_hfa(schedules, walkforward, cfg, before_season=season)
    ctx = build_context(
        game, cfg, venue_hfa=venue_hfa, allow_network=args.weather,
        resting_starters=load_resting_starters(),
    )
    qb_adj = nfelo.qb_adjustments_by_game(nfelo.load_qb_elos(), schedules)
    qb = qb_points_for_game(game["game_id"], qb_adj, cfg)
    ctx = ctx.with_qb(qb.points, qb.note)

    sim = simulate_game(
        home, away, ratings, drive_model, ctx, cfg, start_fp,
        endgame=endgame, rng=np.random.default_rng(cfg.simulation.seed),
        neutral_site=game.get("location") == "Neutral",
    )

    # --- the score --------------------------------------------------------------------
    hs, as_ = sim.home_scores.mean(), sim.away_scores.mean()
    print()
    print("=" * 68)
    print(f"  {away} at {home}     {season} week {week}     "
          f"{pd.to_datetime(game['kickoff']):%a %b %d %H:%M} UTC")
    print("=" * 68)
    print()
    print(f"  PROJECTED SCORE      {away} {as_:.1f}  -  {home} {hs:.1f}")
    print(f"  PROJECTED TOTAL      {sim.mean_total:.1f}")
    fav, dog = (home, away) if sim.mean_margin > 0 else (away, home)
    print(f"  PROJECTED SPREAD     {fav} by {abs(sim.mean_margin):.1f}   "
          f"({dog} +{abs(sim.mean_margin):.1f})")
    print(f"  most likely score    {away} {np.median(sim.away_scores):.0f}  -  "
          f"{home} {np.median(sim.home_scores):.0f}   (medians)")
    print()
    print(f"  distribution         total sd {sim.totals.std():.1f}, "
          f"margin sd {sim.margins.std():.1f}, {sim.n_sims:,} sims")
    print(f"  {home} win prob      {(sim.margins > 0).mean():.1%}")

    # --- what moved it ----------------------------------------------------------------
    print()
    print("  context (points, home perspective):")
    for k, v in ctx.components.items():
        if isinstance(v, (int, float)) and abs(float(v)) > 1e-9:
            print(f"    {k:28s} {float(v):+.2f}")
    if qb.note:
        print(f"    qb_note                      {qb.note}")
    if ctx.data_incomplete:
        print(f"    DATA INCOMPLETE: {', '.join(ctx.incomplete_reasons)}")

    # --- edge, honestly ---------------------------------------------------------------
    market_total = args.total if args.total is not None else game.get("total_line")
    market_spread = args.spread if args.spread is not None else game.get("spread_line")

    print()
    print("  vs the market:")
    if pd.notna(market_total):
        edge = sim.mean_total - float(market_total)
        print(f"    total  line {float(market_total):.1f}   model {sim.mean_total:.1f}   "
              f"edge {edge:+.1f}   "
              f"P(over) {sim.total_prob(float(market_total), 'over'):.1%}")
    else:
        print("    total  no line available -- pass --total to compare")
    if pd.notna(market_spread):
        edge = sim.mean_margin - float(market_spread)
        print(f"    spread line {float(market_spread):+.1f}   "
              f"model {sim.mean_margin:+.1f}   edge {edge:+.1f}   "
              f"P({home} covers) {sim.cover_prob(float(market_spread), 'home'):.1%}")
    else:
        print("    spread no line available -- pass --spread to compare")

    print()
    print("  " + "-" * 64)
    print("  NOT A BET. Over 1,535 walk-forward games this model adds no information")
    print("  beyond the closing line on either market (spreads t=-0.60, totals t=+0.19;")
    print("  both need |t|>2). The market's own total predicts results roughly twice as")
    print("  well as this model's. Treat the number above as a projection, not an edge.")
    print("  " + "-" * 64)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
