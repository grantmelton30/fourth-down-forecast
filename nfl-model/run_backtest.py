#!/usr/bin/env python
"""Backtest CLI (§8).

    python run_backtest.py
    python run_backtest.py --refresh          # force a network pull
    python run_backtest.py --tune-lambdas     # §5c grid search, writes config/nfl.yaml

Prints the metrics table, the calibration table, the bootstrap CI and all six acceptance
gates. A failed gate blocks bet-sheet generation -- the diagnostic report is the output
instead.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict

import pandas as pd

from src import ingest, nfelo
from src.backtest import (
    backtest_frame_path,
    bootstrap_ats,
    calibration_table,
    clv_proxy,
    compute_metrics,
    evaluate_both_markets,
    filtered_bet_results,
    pooled_margin_pmf,
    run_gates,
    time_weekly_refresh,
    walk_forward,
)
from src.config import CACHE_DIR, load_config, write_tuned_block
from src._shared import SHARED_DIR  # noqa: F401  (adds shared/ to sys.path)
from calibration_evidence import write_calibration_evidence
from gate_artifact import load_gate_artifact, write_gate_artifact
from model_identity import build_model_version
from src.market import fit_blend_weights
from src.ratings import build_game_offense_table, tune_lambdas
from src.stadiums import crosscheck_roofs

RULE = "=" * 88


def main() -> int:
    ap = argparse.ArgumentParser(description="NFL model walk-forward backtest")
    ap.add_argument("--refresh", action="store_true", help="force a network data pull")
    ap.add_argument("--tune-lambdas", action="store_true", help="run the §5c grid search")
    ap.add_argument("--n-sims", type=int, default=None, help="simulations per game")
    ap.add_argument("--no-cache", action="store_true", help="recompute the backtest frame")
    ap.add_argument("--validation-games", type=int, default=400)
    ap.add_argument(
        "--diagnostic-exit-zero", action="store_true",
        help="return success after a valid build even when betting gates remain closed",
    )
    args = ap.parse_args()

    cfg = load_config()
    pd.set_option("display.width", 200)

    print(RULE)
    print("NFL SPREAD & TOTAL SIMULATOR -- WALK-FORWARD BACKTEST")
    print(f"train {cfg.seasons.train_start}+   graded "
          f"{cfg.seasons.backtest_start}-{cfg.seasons.current - 1}   "
          f"weeks 4+   n_sims={args.n_sims or cfg.simulation.n_sims}")
    print(RULE)

    # --- data ------------------------------------------------------------------------
    schedules = ingest.load_schedules(cfg.train_seasons, refresh=args.refresh)
    pbp = ingest.load_pbp(cfg.train_seasons, refresh=args.refresh)
    ingest.ingest_health(schedules, pbp)

    roof_issues = crosscheck_roofs(schedules)
    if len(roof_issues):
        print("\n  WARNING: stadium roof disagreements vs schedules.roof:")
        print(roof_issues.to_string(index=False))

    # --- optional lambda tuning ------------------------------------------------------
    if args.tune_lambdas:
        print("\n--- §5c lambda grid search (walk-forward) ---")
        game_off = build_game_offense_table(pbp, schedules, cache_key="main")
        grid = tune_lambdas(game_off, schedules, cfg)
        print(grid.head(10).to_string(index=False))
        best = grid.iloc[0]
        write_tuned_block(cfg, {
            "lambda_off": float(best["lambda_off"]),
            "lambda_def": float(best["lambda_def"]),
            "rmse": float(best["rmse"]),
        })
        print(f"  wrote tuned lambdas to {cfg.path}: "
              f"off={best['lambda_off']:.0f} def={best['lambda_def']:.0f}")
        cfg = load_config()

    # --- nfelo -----------------------------------------------------------------------
    print("\n--- nfelo integration (§7.5) ---")
    qb_elos = nfelo.load_qb_elos(refresh=args.refresh)
    qb_adj = nfelo.qb_adjustments_by_game(qb_elos, schedules)
    matched = int(qb_adj["qb_adj_home"].notna().sum()) if len(qb_adj) else 0
    print(f"  qb_elos.csv: {'loaded' if qb_elos is not None else 'UNAVAILABLE'}"
          f"  ({matched} games matched to a QB adjustment)")

    # --- the walk --------------------------------------------------------------------
    print("\n--- walk-forward ---")
    frame = walk_forward(
        cfg, schedules, pbp, qb_adj_table=qb_adj,
        n_sims=args.n_sims, cache=not args.no_cache,
    )
    print(f"  {len(frame):,} graded games")

    # --- blend -----------------------------------------------------------------------
    print("\n--- §7c blend (residual form) ---")
    weights = fit_blend_weights(frame, cfg)
    # Persisted further down, once the model version exists to bind them to. Weights
    # without that binding cannot be proven to belong to this build, and the calibration
    # gate refuses them.
    print(f"  spread a = {weights.a_spread:+.4f} "
          f"(se {weights.se_a_spread:.4f}, t = {weights.t_a_spread:.2f})")
    print(f"  spread b = {weights.b_model_raw:+.4f} raw / {weights.b_model:.4f} applied "
          f"(se {weights.se_model:.4f}, t = {weights.t_model:.2f}; "
          f"{weights.covariance_type_spread})")
    print(f"  R^2 = {weights.r_squared:.4f}   resid sd = {weights.resid_sd:.2f}   "
          f"n = {weights.n}")
    if weights.n_total:
        print(f"  total a = {weights.a_total:+.4f} "
              f"(se {weights.se_a_total:.4f}, t = {weights.t_a_total:.2f})")
        print(f"  total b = {weights.b_model_total_raw:+.4f} raw / "
              f"{weights.b_model_total:.4f} applied "
              f"(t = {weights.t_model_total:.2f}, n = {weights.n_total}; "
              f"{weights.covariance_type_total})")
    for c in weights.clamped:
        print(f"  CLAMPED: {c}  <-- investigate, do not bet this")
    print(f"  reading: {_blend_reading(weights)}")

    if weights.b_model < 0.10:
        print("\n" + "*" * 88)
        print("*** NO EDGE DETECTED -- model adds no information beyond the market. "
              "Emitting zero bets. ***")
        print("*" * 88)

    # --- metrics ---------------------------------------------------------------------
    print("\n--- §8a metrics ---")
    metrics = compute_metrics(frame, weights)
    print(metrics.round(4).to_string(index=False))

    print("\n--- calibration (5% bins) ---")
    calib = calibration_table(frame)
    print(calib.round(4).to_string(index=False))

    print("\n--- filtered bets, bootstrap CI ---")
    results, result_seasons = filtered_bet_results(
        frame, cfg, weights, return_seasons=True
    )
    rate, lo, hi = bootstrap_ats(results, result_seasons)
    if len(results):
        print(f"  {len(results)} filtered bets, win rate {rate:.4f}, "
              f"90% CI [{lo:.4f}, {hi:.4f}]")
        if lo < 0.50:
            print("  PLAIN LANGUAGE: the 5th percentile is below 50%, so this edge is "
                  "NOT distinguishable from noise at this sample size.")
        if rate < cfg.betting.breakeven_prob_110:
            print(f"  PLAIN LANGUAGE: the win rate is below the "
                  f"{cfg.betting.breakeven_prob_110:.4f} break-even at -110, so this "
                  "filter loses money as it stands.")
    else:
        print("  no bets cleared the filter -- a model that says 'no bets' is working "
              "correctly.")

    clv = clv_proxy(frame, schedules)
    print("\n  CLV proxy: " + (
        f"{clv:.4f}" if clv is not None else
        "unavailable -- nflverse publishes closing lines only, and grading closing "
        "against closing would be circular"
    ))

    # --- PATCH 01 §5: spreads and totals, judged separately ---------------------------
    print("\n" + RULE)
    print("PATCH 01 §5 -- SPREADS vs TOTALS, EVALUATED SEPARATELY")
    print(RULE)
    both = evaluate_both_markets(frame, cfg)
    for kind in ("spread", "total"):
        e = both[kind]
        if not e.get("n"):
            print(f"\n{kind.upper()}: no usable rows")
            continue
        print(f"\n{kind.upper()}S  (n = {e['n']:,})")
        print(f"  RMSE      model {e['model_rmse']:.4f}   market {e['market_rmse']:.4f}   "
              f"model-rescaled {e['rescaled_rmse']:.4f}")
        print(f"  MAE       model {e['model_mae']:.4f}   market {e['market_mae']:.4f}")
        print(f"  SD        model {e['model_sd']:.3f}   market {e['market_sd']:.3f}   "
              f"ratio {e['sd_ratio']:.3f}   (GATE_SCALE wants |ratio-1| < 0.15)")
        print(f"  residual  raw a {e['a_raw']:+.4f}, b {e['b_raw']:+.4f} "
              f"(t(b) = {e['t_raw']:+.2f})   rescaled a {e['a_rescaled']:+.4f}, "
              f"b {e['b_rescaled']:+.4f} (t(b) = {e['t_rescaled']:+.2f})")
        if e["n_bets"]:
            side = "ATS" if kind == "spread" else "O/U"
            print(f"  filtered  {e['n_bets']} bets, {side} {e['win_rate']:.4f}, "
                  f"90% CI [{e['ci_low']:.4f}, {e['ci_high']:.4f}]")
            if e["ci_low"] < 0.50:
                print("            5th percentile below 50% -- not distinguishable from noise.")
            if e["win_rate"] < cfg.betting.breakeven_prob_110:
                print(f"            below the {cfg.betting.breakeven_prob_110:.4f} "
                      "break-even at -110.")
        else:
            print("  filtered  no bets cleared the threshold")
        calib = e["calibration"]
        big = calib[calib["n"] >= 100] if len(calib) else calib
        if len(big):
            print(f"  calib     worst bin (n>=100) off by {100 * big['abs_error'].max():.2f}pp")
        else:
            print("  calib     no bin with n>=100")
    verdict_spread = abs(both["spread"].get("t_rescaled", 0.0)) > 2.0
    verdict_total = abs(both["total"].get("t_rescaled", 0.0)) > 2.0
    print(f"\n  VERDICT: spreads {'PASS' if verdict_spread else 'no demonstrated edge'}, "
          f"totals {'PASS' if verdict_total else 'no demonstrated edge'}  "
          "(|t| > 2.0 after rescaling)")
    print(RULE)

    # --- simulator validation + gates ------------------------------------------------
    print("\n--- §6d simulator validation (pooled margins) ---")
    sim_pmf = pooled_margin_pmf(cfg, schedules, pbp, n_games=args.validation_games)
    weekly_seconds = time_weekly_refresh(cfg, schedules, pbp)

    gates = run_gates(
        weights, metrics, calib, sim_pmf, schedules, weekly_seconds, frame
    )
    # Hash the persisted frame, not the in-memory one: the adapter verifies against what
    # is on disk, so a version derived from anything else can never be reproduced and
    # every artifact is silently repudiated. See the NCAA counterpart for the full note.
    persisted = backtest_frame_path(cfg)
    evidence = pd.read_parquet(persisted) if persisted.exists() else frame
    model_version = build_model_version("nfl", cfg.path.parent.parent, cfg.path, evidence)
    cutoff_col = next(
        (c for c in ("kickoff", "gameday", "game_date", "date") if c in frame), None
    )
    data_cutoff = None
    if cutoff_col and frame[cutoff_col].notna().any():
        data_cutoff = str(pd.to_datetime(frame[cutoff_col], errors="coerce").max().date())
    artifact_path = write_gate_artifact(
        CACHE_DIR,
        league="nfl",
        model_version=model_version,
        data_cutoff=data_cutoff,
        gates=gates,
        promotion_names={
            "GATE_KEY_NUMBERS",
            "GATE_NO_LOOKAHEAD",
            "GATE_BEATS_ELO",
            "GATE_BLEND_INFORMATIVE",
            "GATE_CALIBRATED",
            "GATE_RMSE_SPREAD",
            "GATE_RMSE_TOTAL",
            "GATE_UNBIASED",
        },
    )
    write_calibration_evidence(
        CACHE_DIR,
        league="nfl",
        model_version=model_version,
        data_cutoff=data_cutoff,
        weights=asdict(weights),
    )

    print("\n" + RULE)
    print("ACCEPTANCE GATES (§8b)")
    print(RULE)
    for g in gates:
        print(g)
        if g.detail:
            print(f"      {g.detail}")
    decision = load_gate_artifact(
        artifact_path, expected_league="nfl", expected_model_version=model_version
    )
    failed = list(decision["blockers"])
    diagnostic_failed = [
        g.name for g in gates if not g.passed and g.name not in set(failed)
    ]
    print(RULE)
    if failed:
        print(f"{len(failed)} PROMOTION GATE(S) FAILED: {', '.join(failed)}")
        print("Bet-sheet generation is blocked while a gate is failing (§8b).")
    else:
        print("ALL PROMOTION GATES PASS.")
    if diagnostic_failed:
        print("Non-promotion diagnostics failed: " + ", ".join(diagnostic_failed))
    print(f"Authoritative betting decision: {artifact_path}")
    print(RULE)
    return 0 if args.diagnostic_exit_zero else (1 if failed else 0)


def _blend_reading(w) -> str:
    """The §7c diagnostic table, in words."""
    if abs(w.t_model) > 2.0:
        return "the model carries signal beyond the market -- understand why before trusting it"
    return "the model does not beat the market on this sample -- emit zero bets"


if __name__ == "__main__":
    sys.exit(main())
