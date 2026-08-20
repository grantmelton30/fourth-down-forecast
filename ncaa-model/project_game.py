"""Project one college game and print a score.

    NCAA_MODEL_CACHE_DIR=~/.cache/ncaa-model python project_game.py "Ohio State" Michigan
    ... python project_game.py Alabama Georgia --total 54.5 --spread -2.5
    ... python project_game.py --game-id 401628455
    ... python project_game.py Oregon Washington --season 2024 --week 10 --weather

Runs the same drive-by-drive Monte Carlo the NFL build uses -- literally the same module,
`shared/sim_core.py` -- with this repo's ratings, this repo's measured constants, and this
repo's venue and weather context.

READ THE EDGE SECTION. This repo's own gates say the model does not beat the market. The
one "edge" it ever found on totals was an errors-in-variables artifact of rescaling against
the opener, documented and removed in src/betting.py. The projection below is a real
simulation of a real model; it is not a betting signal, and this script refuses to dress it
up as one.

WEATHER IS OFF BY DEFAULT. `--weather` makes live Open-Meteo calls (free, no key). Without
it the wind adjustment is not applied and the script says so, rather than silently treating
every game as calm -- which is how the NFL build ran for its entire life.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src import drive_model as DM
from src import drives as D
from src import ingest
from src import qb as QB
from src import ratings as R
from src import venues as V
from src.cfbd_client import BudgetedCFBD
from src.config import CACHE_DIR, MANUAL_DIR, load_config
from src.context import NULL_CONTEXT, build_context, estimate_venue_hfa
from src.simulate import simulate_game


def _load_drive_table(games: pd.DataFrame) -> pd.DataFrame:
    """The drive table, from the permanent JSON cache. No API calls."""
    cached = CACHE_DIR / "drive_table.parquet"
    if cached.exists():
        df = pd.read_parquet(cached)
        if not set(D.DRIVE_COLUMNS) - set(df.columns):
            return df
    frames = []
    for path in sorted(CACHE_DIR.glob("drives_[0-9][0-9][0-9][0-9].json")):
        year = int(path.stem.split("_")[1])
        d = pd.json_normalize(json.loads(path.read_text()))
        d["season"] = year
        frames.append(d)
    if not frames:
        sys.exit(
            "no cached drives_<year>.json found. Set NCAA_MODEL_CACHE_DIR to the "
            "permanent cache (usually ~/.cache/ncaa-model)."
        )
    raw = pd.concat(frames, ignore_index=True).rename(columns={"gameId": "game_id"})
    return D.build_drive_table(raw, games[["game_id", "week"]], cache_path=cached)


def _pick_game(games: pd.DataFrame, home, away, game_id, season, week):
    """Find the game to project: an explicit game_id, else the named matchup."""
    if game_id:
        hit = games[games["game_id"].astype(str) == str(game_id)]
        if hit.empty:
            sys.exit(f"no game with game_id={game_id!r}")
        return hit.iloc[0]
    hit = games[(games["homeTeam"] == home) & (games["awayTeam"] == away)]
    if season:
        hit = hit[hit["season"] == season]
    if week:
        hit = hit[hit["week"] == week]
    if hit.empty:
        return None
    return hit.sort_values("kickoff").iloc[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("home", nargs="?", help="home team, as CFBD names it")
    p.add_argument("away", nargs="?", help="away team")
    p.add_argument("--game-id")
    p.add_argument("--season", type=int)
    p.add_argument("--week", type=int)
    p.add_argument("--total", type=float, help="market total, to compare against")
    p.add_argument("--spread", type=float, help="market spread, home perspective")
    p.add_argument("--weather", action="store_true",
                   help="fetch live weather from Open-Meteo (free, no key)")
    p.add_argument("--neutral", action="store_true")
    args = p.parse_args()

    if not args.game_id and not (args.home and args.away):
        p.error("give a home and away team, or --game-id")

    cfg = load_config()
    client = BudgetedCFBD(cfg)

    games = ingest.load_games(
        client, list(range(cfg.seasons.train_start, cfg.seasons.current))
    )
    games = V.attach_venues(games, V.load_venues(client))

    row = _pick_game(games, args.home, args.away, args.game_id, args.season, args.week)
    home = row["homeTeam"] if row is not None else args.home
    away = row["awayTeam"] if row is not None else args.away
    season = int(row["season"]) if row is not None else (
        args.season or cfg.seasons.current - 1)
    week = int(row["week"]) if row is not None and pd.notna(row["week"]) else (
        args.week or 10)
    neutral = args.neutral or (bool(row["neutralSite"]) if row is not None else False)

    # --- model artifacts, all fit strictly before the season being projected ----------
    drive_table = _load_drive_table(games)
    wf = pd.read_parquet(R.walkforward_path(cfg))
    model = DM.fit_drive_model(drive_table, wf, cfg, as_of_season=season)
    endgame = DM.fit_endgame_table(drive_table, cfg, as_of_season=season)
    start_fp = D.fit_start_field_position(drive_table[drive_table["season"] < season])

    try:
        rt = R.ratings_at(wf, season, week)
    except KeyError:
        sys.exit(f"no walk-forward ratings for {season} week {week}")

    # --- context ----------------------------------------------------------------------
    as_of = row["kickoff"] if row is not None and pd.notna(row.get("kickoff")) else None
    venue_hfa = estimate_venue_hfa(games, cfg, walkforward=wf, as_of=as_of)
    ctx = (
        build_context(row, cfg, venue_hfa=venue_hfa, allow_network=args.weather)
        if row is not None else NULL_CONTEXT
    )

    # QUARTERBACK (DECISIONS.md D17). Folded into the CONTEXT, not into `model_spread`:
    # this script reports the raw simulator mean, so the context is the only thing that can
    # move the number printed below. Fails soft to zero -- no passer history, no row for the
    # game, or a disabled config all leave the projection exactly as it was.
    qb_note = ""
    if row is not None:
        try:
            passers = ingest.load_passers(client, games, cfg.all_seasons)
            qb_table = QB.qb_ratings_table(passers, cfg, games=games)
            # `--weather` doubles as the "you may use the network" switch: both feeds are
            # free and keyless, and a projection should not make surprise HTTP calls.
            qb_table, qb_report = QB.resolve_status(
                qb_table, MANUAL_DIR, allow_network=args.weather)
            for status_line in QB.format_status_report(qb_report).splitlines():
                print(f"  {status_line}")
            adj = QB.qb_points_for_game(
                row["game_id"], row["homeTeam"], row["awayTeam"], qb_table, cfg)
            ctx = ctx.with_qb(adj.points, adj.note)
            qb_note = adj.note
        except Exception as exc:  # noqa: BLE001 - a projection must not die on this
            print(f"  (quarterback adjustment unavailable: {exc})")

    sim = simulate_game(home, away, rt, model, cfg, start_fp,
                        context_adj=ctx, endgame=endgame, neutral_site=neutral)

    hs, as_ = sim.home_scores.mean(), sim.away_scores.mean()
    fav = home if sim.mean_margin > 0 else away
    print()
    print("=" * 68)
    print(f"  {away} at {home}     {season} week {week}"
          f"{'     NEUTRAL SITE' if neutral else ''}")
    if row is not None and pd.notna(row.get("venue_name")):
        print(f"  {row['venue_name']}"
              f"{'  (indoor)' if bool(row.get('dome')) else ''}")
    print("=" * 68)
    print()
    print(f"  PROJECTED SCORE      {away} {as_:.1f}  -  {home} {hs:.1f}")
    print(f"  PROJECTED TOTAL      {sim.mean_total:.1f}")
    print(f"  PROJECTED SPREAD     {fav} by {abs(sim.mean_margin):.1f}")
    print("    (raw simulator mean, not recentred onto the gain-corrected linear")
    print("    model_spread -- run_backtest.py's GATE_KEY_NUMBERS analysis found the raw")
    print("    simulator mean under-spreads mismatches; this number runs closer to")
    print("    league-average than the codebase's own calibrated projection would.")
    print("    See GATES.md's GATE_KEY_NUMBERS entry.)")
    print(f"  most likely score    {away} {np.median(sim.away_scores):.0f}  -  "
          f"{home} {np.median(sim.home_scores):.0f}")
    print()
    print(f"  distribution         total sd {sim.totals.std():.1f}, "
          f"margin sd {sim.margins.std():.1f}")
    print(f"  {home} win prob      {(sim.margins > 0).mean():.1%}")
    print()
    print("  context (points, home perspective):")
    for k, v in ctx.components.items():
        if isinstance(v, (int, float)):
            print(f"    {k:28s} {float(v):+.2f}")
        else:
            print(f"    {k:28s} {v}")
    if ctx.data_incomplete:
        print(f"    DATA INCOMPLETE: {', '.join(ctx.incomplete_reasons)}")
        if not args.weather:
            print("    (pass --weather to fetch it from Open-Meteo)")

    if args.total is not None or args.spread is not None:
        print()
        print("  vs the market:")
        if args.total is not None:
            print(f"    total  line {args.total:.1f}   model {sim.mean_total:.1f}   "
                  f"diff {sim.mean_total - args.total:+.1f}")
        if args.spread is not None:
            print(f"    spread line {args.spread:+.1f}   model {sim.mean_margin:+.1f}   "
                  f"diff {sim.mean_margin - args.spread:+.1f}")

    print()
    print("  " + "-" * 64)
    print("  NOT A BET. This repo's gates say the model does not beat the market, and")
    print("  the one totals edge it ever showed was an artifact of rescaling against the")
    print("  opener -- the model reading the market back to itself (src/betting.py).")
    print("  Treat the number above as a projection, not an edge.")
    print("  " + "-" * 64)
    print()


if __name__ == "__main__":
    main()
