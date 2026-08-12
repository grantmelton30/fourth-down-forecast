#!/usr/bin/env python
"""Historical CLV and concentration -- the evidence 13 picks cannot provide.

    python run_clv_history.py
    python run_clv_history.py --no-caps

CFBD carries the opener and the close for every graded game, so CLV can be measured across
the whole walk-forward backtest instead of waiting on live weeks. Same discipline as the
backtest: for each season the blend and the L3 distribution are fit only on EARLIER
seasons, so every pick uses coefficients that existed before its kickoff.

CLV is the score. The win record is reported for context only.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src import ingest
from src.backtest import select_window, walk_forward, zero_skill_projection
from src.betting import apply_concentration_caps, build_totals_sheet, clv_summary
from src.calibrate import build_conditional
from src.cfbd_client import BudgetedCFBD
from src.config import OUTPUT_DIR, load_config
from src.market import fit_blend
from src.ratings import build_walkforward

RULE = "=" * 100


def generate_history(cfg, frame, market, use_caps: bool) -> pd.DataFrame:
    """Walk-forward pick generation across every graded season.

    A season's picks are made with a blend and a conditional distribution fit on strictly
    earlier seasons, so nothing here sees its own outcome.
    """
    graded = select_window(frame, cfg.graded_seasons, restricted=True)
    conf = _conference_lookup(market)
    out = []
    for season in sorted(graded["season"].unique()):
        train = graded[graded["season"] < season]
        test = graded[graded["season"] == season]
        if len(train) < 400 or test.empty:
            continue
        weights = fit_blend(train, cfg, window=f"pre-{season}")
        pmf = build_conditional(
            train, "total_open", "actual_total", as_of_season=season, kind="total"
        )
        sheet = build_totals_sheet(test, weights, cfg, pmf=pmf, picks_only=True,
                                   caps=False)
        if sheet.empty:
            continue
        sheet = _attach_conferences(sheet.assign(b_used=weights.b_model_total), conf)
        if use_caps:
            sheet = apply_concentration_caps(sheet, cfg)
        out.append(sheet)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _conference_lookup(market: pd.DataFrame) -> dict:
    home = market[["home_team", "home_conference"]].rename(
        columns={"home_team": "team", "home_conference": "conference"})
    away = market[["away_team", "away_conference"]].rename(
        columns={"away_team": "team", "away_conference": "conference"})
    both = pd.concat([home, away], ignore_index=True).dropna()
    return both.drop_duplicates("team").set_index("team")["conference"].to_dict()


def _attach_conferences(sheet: pd.DataFrame, conf: dict) -> pd.DataFrame:
    sheet = sheet.copy()
    sheet["home_conference"] = sheet["home_team"].map(conf)
    sheet["away_conference"] = sheet["away_team"].map(conf)
    return sheet


def per_team(sheet: pd.DataFrame) -> pd.DataFrame:
    """Every pick involves two teams; both are counted."""
    rows = [
        sheet[[side, "clv_positive", "result", "edge_pts"]].rename(columns={side: "team"})
        for side in ("home_team", "away_team")
    ]
    long = pd.concat(rows, ignore_index=True)
    agg = long.groupby("team").agg(
        picks=("team", "size"),
        clv=("clv_positive", "mean"),
        clv_n=("clv_positive", "count"),
        mean_edge=("edge_pts", lambda s: s.abs().mean()),
    )
    wins = long[long["result"] == "WIN"].groupby("team").size()
    losses = long[long["result"] == "LOSS"].groupby("team").size()
    agg["W"] = wins.reindex(agg.index).fillna(0).astype(int)
    agg["L"] = losses.reindex(agg.index).fillna(0).astype(int)
    agg["hit_rate"] = agg["W"] / (agg["W"] + agg["L"]).replace(0, np.nan)
    agg["share"] = agg["picks"] / max(len(sheet), 1)
    return agg.sort_values("picks", ascending=False)


def report(sheet: pd.DataFrame, label: str) -> None:
    print()
    print(RULE)
    print(f"{label}  --  {len(sheet)} picks")
    print(RULE)
    s = clv_summary(sheet)
    print(f"  CLV overall              {s['clv']:.4f}   (n={s['with_close_move']} "
          f"movable lines of {s['picks']} picks)")
    print(f"  mean move toward side    {s['mean_move_toward_side']:+.3f} pts")
    print(f"  mean edge at entry       {s['mean_edge_pts']:.2f} pts")
    print(f"  record (context only)    {s.get('record', 'n/a')}")

    print()
    print("  BY SEASON")
    for season, g in sheet.groupby("season"):
        gs = clv_summary(g)
        print(f"    {season}  picks {len(g):>3}  CLV {gs['clv']:.4f} "
              f"(n={gs['with_close_move']:>3})  record {gs.get('record','')}")

    print()
    print("  BY WEEK BUCKET")
    for lo, hi, name in [(4, 6, "4-6"), (7, 9, "7-9"), (10, 12, "10-12"),
                         (13, 16, "13-16")]:
        g = sheet[sheet["week"].between(lo, hi)]
        if g.empty:
            continue
        gs = clv_summary(g)
        print(f"    wk {name:<6} picks {len(g):>3}  CLV {gs['clv']:.4f} "
              f"(n={gs['with_close_move']:>3})  record {gs.get('record','')}")

    graded = sheet[sheet["clv_positive"].notna()]
    if len(graded) >= 10:
        from scipy import stats

        k, n = int(graded["clv_positive"].sum()), len(graded)
        print()
        print(f"  binomial vs 0.50:  {k}/{n}  p = "
              f"{float(stats.binomtest(k, n, 0.5).pvalue):.6f}")


def report_concentration(sheet: pd.DataFrame, cfg) -> None:
    print()
    print(RULE)
    print("CONCENTRATION")
    print(RULE)
    tab = per_team(sheet)
    print("  top 12 teams by pick count:")
    print(tab.head(12).to_string(float_format=lambda v: f"{v:.3f}"))

    over = tab[tab["share"] > cfg.betting.max_team_pick_share]
    print()
    print(f"  teams above the {cfg.betting.max_team_pick_share:.0%} cap: {len(over)}"
          + (f"  -> {list(over.index)}" if len(over) else ""))

    cl = pd.concat(
        [sheet[[s, "clv_positive"]].rename(columns={s: "conf"})
         for s in ("home_conference", "away_conference")],
        ignore_index=True,
    ).dropna(subset=["conf"])
    ct = cl.groupby("conf").agg(picks=("conf", "size"), clv=("clv_positive", "mean"))
    ct["share"] = ct["picks"] / max(len(sheet), 1)
    print()
    print("  by conference:")
    print(ct.sort_values("picks", ascending=False).head(10)
          .to_string(float_format=lambda v: f"{v:.3f}"))
    overc = ct[ct["share"] > cfg.betting.max_conference_pick_share]
    print(f"  conferences above the {cfg.betting.max_conference_pick_share:.0%} cap: "
          f"{len(overc)}" + (f"  -> {list(overc.index)}" if len(overc) else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-caps", action="store_true")
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

    print(RULE)
    print("HISTORICAL CLV -- walk-forward pick generation across every graded season")
    print(RULE)

    uncapped = generate_history(cfg, frame, market, use_caps=False)
    if uncapped.empty:
        print("no picks generated")
        return 1
    report(uncapped, "UNCAPPED")
    report_concentration(uncapped, cfg)
    uncapped.to_csv(OUTPUT_DIR / "historical_picks_uncapped.csv", index=False)

    # ZERO-SKILL CONTROL. Permanent fixture: if this does not read near 0.50, CLV on that
    # anchor is measuring line noise rather than skill.
    print()
    print(RULE)
    print("ZERO-SKILL CONTROL -- projection = anchor + gaussian noise, no information")
    print(RULE)
    for anchor in ("total_open", "total_close"):
        ctrl_frame = zero_skill_projection(frame, anchor=anchor)
        ctrl = generate_history(cfg, ctrl_frame, market, use_caps=False)
        if ctrl.empty:
            print(f"  anchored on {anchor}: no picks generated")
            continue
        cs = clv_summary(ctrl)
        print(f"  anchored on {anchor:<12} picks {cs['picks']:>3}  "
              f"CLV {cs['clv']:.4f}  (n={cs['with_close_move']})")
    print("  A control CLV far from 0.50 means the metric is reading line noise.")
    print(RULE)

    if not args.no_caps:
        capped = generate_history(cfg, frame, market, use_caps=True)
        report(capped, "CAPPED (max team / conference share enforced)")
        report_concentration(capped, cfg)
        capped.to_csv(OUTPUT_DIR / "historical_picks_capped.csv", index=False)

        u, c = clv_summary(uncapped), clv_summary(capped)
        print()
        print(RULE)
        print(f"  CLV uncapped {u['clv']:.4f} ({u['picks']} picks)  ->  "
              f"capped {c['clv']:.4f} ({c['picks']} picks)")
        print("  Edge SURVIVES the cap -- it is distributed." if c["clv"] >= 0.60
              else "  Edge does NOT survive the cap -- concentration was carrying it.")
        print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
