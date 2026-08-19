#!/usr/bin/env python
"""NCAA backtest CLI -- produces the PATCH 04 Step 2 verdict.

    python run_backtest.py
    python run_backtest.py --sweep       # the pre-registered 12-cell grid

The kill criterion, restated: held-out t(b_model) on college TOTALS, restricted universe,
graded against OPENING lines. >= 2.0 means signal; below means stop.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src import ingest
from src.backtest import (
    attach_calibration,
    bootstrap,
    build_features,
    calibration_table,
    filtered_record,
    gate_api_budget,
    gate_blend_informative,
    gate_calibrated,
    gate_garbage,
    gate_key_numbers,
    gate_no_lookahead,
    gate_opener_coverage,
    gate_scale,
    gate_unbiased,
    gate_unbiased_by_week,
    pooled_margin_pmf,
    rmse_gate,
    select_window,
    walk_forward,
)
from dataclasses import asdict

from src.cfbd_client import BudgetedCFBD
from src.config import CACHE_DIR,load_config
from src.market import clv,fit_blend,residual_fit,season_week_groups
from src.ratings import build_walkforward, interaction_matchup, split_net_epa_matchup
from src.venues import attach_venues, load_venues
from src.features import (load_free_preseason, matchup_feature_table,
                          normalize_preseason_sources)
from src._shared import SHARED_DIR
from calibration_evidence import write_calibration_evidence
from gate_artifact import load_gate_artifact,write_gate_artifact
from model_identity import build_model_version

# NCAA's only apparent edge -- a totals signal -- was a false positive that existed
# solely against the opener and vanished against the close (GATES.md). `fit_blend`
# defaults to the opener for `run_paper.py`'s exploratory grading of that same
# phenomenon, but the public "Calibrated" column is a validation claim, and every other
# honest NCAA gate (GATE_RMSE_*) grades against the close. Calibration evidence must
# clear the same bar, not the one that produced the false positive.
CALIBRATION_ANCHOR = "close"


def fit_and_write_calibration(frame, cfg, *, cache_dir, model_version, data_cutoff):
    """Fit NCAA blend weights against the close and persist them as calibration
    evidence bound to this build. Caller supplies an already-windowed frame (the same
    primary evaluation window the other close-anchored gates use)."""
    weights = fit_blend(frame, cfg, window="close-anchored-primary",
                         grade=CALIBRATION_ANCHOR)
    return write_calibration_evidence(
        cache_dir, league="ncaa", model_version=model_version,
        data_cutoff=data_cutoff, weights=asdict(weights),
    )

RULE = "=" * 96

# DECISIONS D3 -- declared before the first sweep ran. 12 cells, no more.
GRID = [
    {"half_life_games": 6, "prior_weight_games": 5, "off_weight": 1.45},
    {"half_life_games": 6, "prior_weight_games": 10, "off_weight": 1.45},
    {"half_life_games": 6, "prior_weight_games": 5, "off_weight": 1.20},
    {"half_life_games": 6, "prior_weight_games": 5, "off_weight": 2.00},
    {"half_life_games": 10, "prior_weight_games": 5, "off_weight": 1.45},
    {"half_life_games": 10, "prior_weight_games": 10, "off_weight": 1.45},
    {"half_life_games": 10, "prior_weight_games": 5, "off_weight": 1.20},
    {"half_life_games": 10, "prior_weight_games": 5, "off_weight": 2.00},
    {"half_life_games": 16, "prior_weight_games": 5, "off_weight": 1.45},
    {"half_life_games": 16, "prior_weight_games": 10, "off_weight": 1.45},
    {"half_life_games": 16, "prior_weight_games": 16, "off_weight": 1.45},
    {"half_life_games": 16, "prior_weight_games": 5, "off_weight": 2.00},
]


def load_everything(cfg, client):
    games = ingest.load_games(client, cfg.all_seasons)
    drives = ingest.load_drives(client, cfg.all_seasons)
    plays = ingest.load_plays(client, games, cfg.all_seasons)
    lines = ingest.load_lines(client, cfg, cfg.all_seasons)
    game_off = ingest.build_game_offense(plays, drives, games, cfg)
    market = ingest.build_market(lines, games, cfg)
    return games, drives, plays, lines, game_off, market


def report_window(frame, label):
    """a, t(a), b, t(b), SD ratio, RMSE vs opener and vs close -- spreads and totals."""
    rows, fits = [], {}
    for kind, model, act in (
        ("spread", "model_spread", "actual_margin"),
        ("total", "model_total", "actual_total"),
    ):
        f = residual_fit(frame[model], frame[f"{kind}_open"], frame[act], kind,groups=season_week_groups(frame))
        fits[kind] = f
        rows.append({
            "window": label, "market": kind, "n": f.n,
            "a": f.a, "t(a)": f.t_a, "b": f.b, "t(b)": f.t_b,
            "sd_ratio": f.sd_ratio, "rmse_model": f.rmse,
            "rmse_open": float(np.sqrt(((frame[f"{kind}_open"] - frame[act]) ** 2).mean())),
            "rmse_close": float(np.sqrt(((frame[f"{kind}_close"] - frame[act]) ** 2).mean())),
        })
    return pd.DataFrame(rows), fits


def run_sweep(cfg, market, games, game_off):
    """The pre-registered grid. Tune on 2021-2023, validate on 2024-2025. All cells shown."""
    tune = [s for s in cfg.graded_seasons if s <= 2023]
    hold = [s for s in cfg.graded_seasons if s >= 2024]
    rows = []
    for i, cell in enumerate(GRID, 1):
        c = cfg.with_ratings(**cell)
        wf = build_walkforward(game_off, games, c, cache_key=f"sweep{i}")
        frame = walk_forward(c, market, wf, cache_key=f"sweep{i}")
        r = {"cell": i, **cell}
        for name, seasons in (("tune", tune), ("hold", hold)):
            sub = select_window(frame, seasons, restricted=True)
            if len(sub) < 100:
                r[f"{name}_t_total"] = np.nan
                r[f"{name}_t_spread"] = np.nan
                r[f"{name}_sd_total"] = np.nan
                continue
            ft = residual_fit(sub["model_total"], sub["total_open"], sub["actual_total"],groups=season_week_groups(sub))
            fs = residual_fit(
                sub["model_spread"], sub["spread_open"], sub["actual_margin"],groups=season_week_groups(sub))
            r[f"{name}_t_total"] = ft.t_b
            r[f"{name}_t_spread"] = fs.t_b
            r[f"{name}_sd_total"] = ft.sd_ratio
        rows.append(r)
        print(f"  cell {i:>2}/12 done", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="run the pre-registered grid")
    ap.add_argument(
        "--diagnostic-exit-zero", action="store_true",
        help="return success after a valid build even when betting gates remain closed",
    )
    args = ap.parse_args()

    cfg = load_config()
    client = BudgetedCFBD(cfg)
    pd.set_option("display.width", 220)

    print(RULE)
    print("NCAA -- WALK-FORWARD BACKTEST, GRADED AGAINST OPENING LINES")
    print(f"ratings history {cfg.seasons.train_start}+   "
          f"primary window {cfg.graded_seasons}   "
          f"secondary {cfg.graded_seasons_secondary}")
    print(RULE)

    games, drives, plays, lines, game_off, market = load_everything(cfg, client)
    garbage = ingest.garbage_share(plays)
    print(f"games {len(games):,}  drives {len(drives):,}  plays {len(plays):,}  "
          f"game_offense {len(game_off):,}  market {len(market):,}")
    print(f"garbage-time plays dropped: {garbage:.1%}")

    # `shared/sport.py::NCAAAdapter._market()` reads this file as the historical half of
    # the live slate's market table -- concatenated with the freshly-pulled current season,
    # season-excluded so the live half always wins. Nothing else in the pipeline ever wrote
    # it (only a test fixture did), so `NCAAAdapter._market()` was silently scoped to the
    # live season alone. That made `projection_mean()`'s `training_market` empty on every
    # call (it filters for `completed`, which the live season doesn't have yet), which made
    # `fit_live_mean` raise on 0-or-1 season, which is caught as `return None` -- and every
    # published NCAA prediction fell back to `forecast_from_sim`'s raw, uncalibrated
    # simulator mean instead of the validated, gain-corrected one. Found and fixed
    # 2026-08-17 while reviewing week-1 spread/total quality. `market` already carries this
    # backtest run's own CFBD data across `cfg.all_seasons`, so persisting it here costs no
    # extra calls and keeps this file only as stale as this script's own last run.
    (CACHE_DIR / "market.parquet").parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(CACHE_DIR / "market.parquet", index=False)

    wf = build_walkforward(game_off, games, cfg)
    preseason_raw = load_free_preseason(client, cfg.all_seasons)
    preseason = normalize_preseason_sources(**preseason_raw)
    challenger = matchup_feature_table(plays, games, preseason)

    # Tests a competitor's claimed methodology (GATES.md 2026-08-17): pass-only EPA as a
    # stronger rating signal than the pooled pass+rush average `game_off` uses today.
    # Candidates only -- promotion through the existing validated-ridge gate
    # (_validated_challenger) decides whether either one actually beats the pooled signal;
    # nothing here changes model_spread/model_total unless it does.
    game_off_pass = ingest.build_game_offense(plays, drives, games, cfg, play_type="pass")
    game_off_rush = ingest.build_game_offense(plays, drives, games, cfg, play_type="rush")
    wf_pass = build_walkforward(game_off_pass, games, cfg, cache_key="pass")
    wf_rush = build_walkforward(game_off_rush, games, cfg, cache_key="rush")
    split_epa = split_net_epa_matchup(wf_pass, wf_rush, games, cfg)
    challenger = challenger.merge(split_epa, on="game_id", how="left")

    # Tests a different, follow-on claim: not whether rush ratings deserve their own
    # linear term (the split test above already rejected that), but whether the EFFECT of
    # facing a worse defense scales with how good the offense already is -- a genuine
    # multiplicative interaction the current additive off_weight*off + def_weight*def
    # formula cannot represent at all. Pooled (general form) and rush-only (the specific
    # "this rushing style beats this front" claim raised in review) versions, both free --
    # `wf`/`wf_rush`/`wf_pass` are already fit above. Pass-specific added 2026-08-18 as a
    # follow-up (GATES.md D13 flagged it, not run at the time): the pooled and rush-only
    # interactions were tested but not the pass one, and passing is where the pass/rush
    # split test found the stronger standalone signal -- worth checking whether that
    # extends to an interaction term too, not just assumed either way.
    interactions = interaction_matchup(wf, games, prefix="pooled_").merge(
        interaction_matchup(wf_rush, games, prefix="rush_"), on="game_id", how="left"
    ).merge(
        interaction_matchup(wf_pass, games, prefix="pass_"), on="game_id", how="left")
    challenger = challenger.merge(interactions, on="game_id", how="left")

    frame = walk_forward(cfg, market, wf, challenger_features=challenger)
    print(f"backtest frame: {len(frame):,} graded games")

    windows = [
        ("2021-2025 restricted (PRIMARY)", cfg.graded_seasons, True),
        ("2023-2025 restricted (pre-committed)", cfg.graded_seasons_secondary, True),
        ("2021-2025 full FBS (contrast)", cfg.graded_seasons, False),
    ]
    tables, fits_by_window = [], {}
    for label, seasons, restricted in windows:
        sub = select_window(frame, seasons, restricted=restricted)
        if len(sub) < 50:
            continue
        tbl, fits = report_window(sub, label)
        tables.append(tbl)
        fits_by_window[label] = (fits, sub)

    print()
    print(RULE)
    print("PER-WINDOW FITS -- graded vs OPENER, free intercept")
    print(RULE)
    print(pd.concat(tables, ignore_index=True).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print()
    print(RULE)
    print("FILTERED RECORDS AND CLV")
    print(RULE)
    for label, (fits, sub) in fits_by_window.items():
        print(f"\n{label}")
        for kind in ("spread", "total"):
            won, picked = filtered_record(sub, cfg, kind)
            if not len(won):
                print(f"  {kind:<7} no positions cleared the edge threshold")
                continue
            rate, lo, hi = bootstrap(won,picked["season"].to_numpy())
            line = (f"  {kind:<7} {len(won):>4} positions  win {rate:.4f}  "
                    f"90% CI [{lo:.4f}, {hi:.4f}]")
            if kind == "spread":
                line += f"  CLV {clv(picked, 'side'):.4f}"
            print(line)
            if lo < 0.50:
                print("          5th pct below 50% -- not distinguishable from noise")

    primary_fits, _ = fits_by_window["2021-2025 restricted (PRIMARY)"]
    completed = market[market["completed"].fillna(False)]
    print()
    print(RULE)
    print("SIMULATOR-BACKED DISTRIBUTION CHECK (GATE_KEY_NUMBERS) -- pooled sample,")
    print("in-sample ratings; tests shape only, not predictive power")
    print(RULE)
    # pooled_margin_pmf's per-game context needs venue_id (for venue-HFA and, via
    # weather, dome/lat/lon), which ingest.load_games alone does not carry -- attach_venues
    # is a separate, one-call CFBD pull, same as project_game.py's live-projection path.
    games_with_venues = attach_venues(games, load_venues(client))
    sim_pmf = pooled_margin_pmf(cfg, games_with_venues, drives, wf, frame)
    key_numbers_gate = gate_key_numbers(sim_pmf, market)
    print(key_numbers_gate)
    if key_numbers_gate.detail:
        print(f"      {key_numbers_gate.detail}")

    print()
    print(RULE)
    print("CALIBRATED COVER PROBABILITIES (GATE_CALIBRATED) -- walk-forward, one season")
    print("of simulation at a time, reweighted onto this build's own model_spread/total")
    print(RULE)
    calibrated = attach_calibration(frame, cfg, games_with_venues, drives, wf)
    print(f"  {len(calibrated):,} graded games carry a calibrated cover probability")
    calib_table = calibration_table(calibrated)
    calibrated_gate = gate_calibrated(calib_table)
    print(calibrated_gate)
    if len(calib_table):
        print(calib_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    gates = [
        gate_opener_coverage(market, cfg, cfg.graded_seasons),
        gate_no_lookahead(build_features(completed, wf, cfg), wf),
        gate_garbage(garbage, cfg),
        gate_unbiased(primary_fits, cfg),
        gate_unbiased_by_week(
            select_window(frame, cfg.graded_seasons, restricted=True), cfg),
        gate_scale(primary_fits, cfg),
        gate_blend_informative(primary_fits, cfg),
        # Absolute accuracy, close-anchored (Phase 0D). Listed after the blend gate
        # deliberately: GATE_BLEND_INFORMATIVE can pass on an errors-in-variables
        # artifact, and GATE_RMSE is the one that cannot.
        rmse_gate(select_window(frame, cfg.graded_seasons, restricted=True), cfg,
                  kind="total"),
        rmse_gate(select_window(frame, cfg.graded_seasons, restricted=True), cfg,
                  kind="spread"),
        gate_api_budget(client.calls_used, cfg),
        key_numbers_gate,
        calibrated_gate,
    ]
    cutoff=pd.to_datetime(frame.get("kickoff"),utc=True,errors="coerce").max()
    # Stamp the version with the evidence that is actually PERSISTED, not the in-memory
    # frame. walk_forward returns its working copy while caching a parquet whose dtypes
    # differ on first write, so hashing the in-memory frame produced a version no reader
    # could ever reproduce: the adapter reads the parquet, recomputes a different digest,
    # and repudiates every artifact this script writes. bets_allowed() then returned
    # False because verification was broken, not because a gate had failed -- and
    # fail-closed made those two indistinguishable.
    persisted = CACHE_DIR / "backtest_frame_default.parquet"
    evidence = pd.read_parquet(persisted) if persisted.exists() else frame
    model_version=build_model_version("ncaa",cfg.path.parent.parent,cfg.path,evidence)
    artifact_path=write_gate_artifact(CACHE_DIR,league="ncaa",model_version=model_version,
        data_cutoff=None if pd.isna(cutoff) else cutoff.isoformat(),gates=gates,
        promotion_names={"GATE_OPENER_COVERAGE","GATE_NO_LOOKAHEAD","GATE_GARBAGE_FILTER","GATE_UNBIASED","GATE_UNBIASED_BY_WEEK","GATE_BLEND_INFORMATIVE","GATE_RMSE_TOTAL","GATE_RMSE_SPREAD","GATE_KEY_NUMBERS","GATE_CALIBRATED"})

    calibration_path = fit_and_write_calibration(
        select_window(frame, cfg.graded_seasons, restricted=True), cfg,
        cache_dir=CACHE_DIR, model_version=model_version,
        data_cutoff=None if pd.isna(cutoff) else cutoff.isoformat(),
    )
    print(f"  calibration evidence ({CALIBRATION_ANCHOR}-anchored): {calibration_path}")
    print()
    print(RULE)
    print("ACCEPTANCE GATES")
    print(RULE)
    for g in gates:
        print(g)
        if g.detail:
            print(f"      {g.detail}")

    if args.sweep:
        print()
        print(RULE)
        print("PRE-REGISTERED 12-CELL GRID (DECISIONS D3) -- all cells, tuned on "
              "2021-2023, held out 2024-2025")
        print(RULE)
        sweep = run_sweep(cfg, market, games, game_off)
        print(sweep.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        sweep.to_csv("output/sweep_results.csv", index=False)

    t_total = primary_fits["total"].t_b
    t_spread = primary_fits["spread"].t_b
    print()
    print(RULE)
    print("KILL CRITERION -- held-out t(b_model) on TOTALS, restricted, vs OPENER")
    print(f"   totals  t(b) = {t_total:+.3f}")
    print(f"   spreads t(b) = {t_spread:+.3f}")
    decision=load_gate_artifact(artifact_path,expected_league="ncaa",expected_model_version=model_version)
    verdict=decision["bets_allowed"] is True; diagnostic=t_total>2.0
    label="PROMOTED -- every required gate passed" if verdict else ("DIAGNOSTIC SIGNAL ONLY -- promotion blocked by "+", ".join(decision["blockers"]) if diagnostic else "NULL / NOT PROMOTED -- stop")
    print(f"   VERDICT: {label}")
    print(f"   authoritative decision: {artifact_path}")
    print(RULE)
    return 0 if args.diagnostic_exit_zero else (0 if verdict else 2)


if __name__ == "__main__":
    sys.exit(main())
