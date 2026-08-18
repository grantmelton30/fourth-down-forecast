"""Isolates whether the 2026-08-17 Appendix A drift (GATES.md "revisited again") was
caused by the FCS-ratings fix (`ingest.py::build_game_offense`, same day).

DECISIONS D9 confirmed the drift is NOT the challenger-promotion mechanism suspected
2026-08-16 (`feature_total_promoted` is False on every graded row). The next candidate is
the FCS-ratings fix itself: `build_game_offense` used to collapse every non-FBS opponent
into one shared bucket; it now keeps FBS *and* FCS teams as real, individually-fitted
teams. The FCS plan predicted this would leave FBS-vs-FBS relative ratings invariant,
"because that's what a connected ridge fit already does" -- a reasoned prediction, never
actually checked against `model_total`'s residual coefficient.

This script checks it directly: rebuild `game_off` with the OLD (pre-fix) bucketing logic,
re-run the walk-forward ratings and backtest frame from that counterfactual `game_off`, and
compare the same Appendix A statistics against both the post-fix (current) and pre-fix
(historical) readings. No new CFBD calls -- reuses the existing cached games/drives/plays.

    .venv/Scripts/python.exe -m analysis.appendix_a_fcs_counterfactual
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.power import analyse_frame
from src import ingest
from src.backtest import build_features, project_walkforward, walk_forward
from src.cfbd_client import BudgetedCFBD
from src.config import CACHE_DIR, build_cache_signature, frame_signature, load_config
from src.features import load_free_preseason, matchup_feature_table, normalize_preseason_sources
from src.ratings import build_walkforward


def _old_build_game_offense(plays, drives, games, cfg):
    """`build_game_offense` as it read before the 2026-08-17 FCS-ratings fix: every
    non-FBS opponent, FCS included, collapses into one shared bucket."""
    scrimmage = plays[
        plays["competitive"]
        & plays["ppa"].notna()
        & ~plays["playType"].isin(ingest.SPECIAL_PLAY_TYPES)
    ]
    eff = scrimmage.groupby(["game_id", "offense"], as_index=False).agg(
        ppa_per_play=("ppa", "mean"), n_plays=("ppa", "size"),
    )
    dr = drives.groupby(["game_id", "offense"], as_index=False).agg(
        drives=("driveNumber", "nunique")
    )
    agg = eff.merge(dr, on=["game_id", "offense"], how="left")

    g = games[[
        "game_id", "season", "week", "kickoff", "neutralSite", "completed",
        "homeTeam", "awayTeam", "homeConference", "awayConference",
        "homeClassification", "awayClassification", "homePoints", "awayPoints",
    ]]
    df = agg.merge(g, on="game_id", how="inner")

    fcs = cfg.teams.fcs_bucket_name
    df["home_norm"] = np.where(df["homeClassification"] == "fbs", df["homeTeam"], fcs)
    df["away_norm"] = np.where(df["awayClassification"] == "fbs", df["awayTeam"], fcs)
    is_home_off = df["offense"] == df["homeTeam"]
    df["offense_norm"] = np.where(is_home_off, df["home_norm"], df["away_norm"])
    df["defense_norm"] = np.where(is_home_off, df["away_norm"], df["home_norm"])
    df["is_home_offense"] = np.where(
        df["neutralSite"].fillna(False), 0.5, is_home_off.astype(float)
    )
    df["game_weight"] = np.where(
        (df["offense_norm"] == fcs) | (df["defense_norm"] == fcs),
        cfg.teams.fcs_game_weight, 1.0,
    )
    df = df[df["completed"].fillna(False)]
    df = df.sort_values("kickoff").reset_index(drop=True)
    return df


def _appendix_a_stats(frame: pd.DataFrame, label: str) -> None:
    graded = frame[frame["has_opener"] & frame["season"].between(2021, 2025)]
    restricted = graded[graded["restricted"]].dropna(
        subset=["model_total", "total_open", "total_close", "actual_total"]
    )
    close = analyse_frame(restricted, market_col="total_close", label="close")
    opener = analyse_frame(restricted, market_col="total_open", label="open")
    full = graded[graded["fbs_only"]].dropna(
        subset=["model_total", "total_close", "actual_total"]
    )
    full_p = analyse_frame(full, market_col="total_close", label="full FBS")
    ci_upper = full_p.b + 1.96 * full_p.se_b

    print(f"\n{label}")
    print(f"  restricted close  n={len(restricted):5d}  b={close.b:+.4f}  t={close.t_b:+.4f}")
    print(f"  restricted open   n={len(restricted):5d}  b={opener.b:+.4f}  t={opener.t_b:+.4f}")
    print(f"  full FBS close    n={len(full):5d}  b={full_p.b:+.4f}  se={full_p.se_b:.4f}  "
          f"CI_upper={ci_upper:+.4f}")


def main() -> int:
    cfg = load_config()
    client = BudgetedCFBD(cfg)

    print("Loading cached games/drives/plays/lines (no new CFBD calls expected)...")
    games = ingest.load_games(client, cfg.all_seasons)
    drives = ingest.load_drives(client, cfg.all_seasons)
    plays = ingest.load_plays(client, games, cfg.all_seasons)
    lines = ingest.load_lines(client, cfg, cfg.all_seasons)
    market = ingest.build_market(lines, games, cfg)
    print(f"  CFBD calls used this run: {client.calls_used}")

    print("\n=== POST-FIX (current committed code) ===")
    game_off_new = ingest.build_game_offense(plays, drives, games, cfg)
    wf_new = build_walkforward(game_off_new, games, cfg, cache_key="default")
    frame_new = walk_forward(cfg, market, wf_new, cache_key="default")
    _appendix_a_stats(frame_new, "POST-FIX (FCS teams individually rated)")

    print("\n=== COUNTERFACTUAL (pre-fix bucketing, reverted in-memory only) ===")
    game_off_old = _old_build_game_offense(plays, drives, games, cfg)
    wf_old = build_walkforward(game_off_old, games, cfg, cache_key="counterfactual_prefcs")
    frame_old = walk_forward(cfg, market, wf_old, cache_key="counterfactual_prefcs")
    _appendix_a_stats(frame_old, "COUNTERFACTUAL (FCS teams bucketed, old behavior)")

    print("\n=== REFERENCE (historical readings, for comparison) ===")
    print("  2026-08-04 Appendix A:      restricted close b=+0.076 t=+0.89 | "
          "restricted open b=+0.195 t=+2.23")
    print("  2026-08-16 revisit:         full FBS close   b=+0.0911 se=0.0631 CI_upper=+0.2148")
    print("  2026-08-17 revisit (post-fix): restricted close b=+0.1549 t=+1.745 | "
          "full FBS close b=+0.1157 se=0.0623 CI_upper=+0.2379")

    print("\nIf the counterfactual's numbers land back near the 2026-08-04/08-16 readings, "
          "the FCS-ratings fix is confirmed as the cause. If the counterfactual still shows "
          "the drift, something else moved and this hypothesis is rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
