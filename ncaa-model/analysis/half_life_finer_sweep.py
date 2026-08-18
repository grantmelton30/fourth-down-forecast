"""Finer `half_life_games` sweep, pre-registered in DECISIONS.md D11 before this ran.

D8 ran the pre-registered 12-cell grid (DECISIONS D3) against the post-FCS-fix rating
universe and found `half_life_games=16` scoring best of the three tested points (6/10/16),
monotonically increasing across them -- too coarse to tell whether 16 is a real interior
optimum or the trend was still climbing when the grid stopped. This tests six points
centered on 16, holding `prior_weight_games=8.0` and `off_weight=1.45` fixed at their
already-validated live values -- `ratings.half_life_games` only, not the separately-tuned
`pace.half_life_games`.

    .venv/Scripts/python.exe -m analysis.half_life_finer_sweep
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import ingest
from src.backtest import select_window, walk_forward
from src.cfbd_client import BudgetedCFBD
from src.config import load_config
from src.market import residual_fit, season_week_groups
from src.ratings import build_walkforward

# DECISIONS D11 -- declared before this ran. No cell outside this table is evaluated.
GRID = [8, 12, 16, 20, 24, 28]


def main() -> int:
    cfg = load_config()
    client = BudgetedCFBD(cfg)

    print("Loading cached games/plays/lines (no new CFBD calls expected)...")
    games = ingest.load_games(client, cfg.all_seasons)
    drives = ingest.load_drives(client, cfg.all_seasons)
    plays = ingest.load_plays(client, games, cfg.all_seasons)
    lines = ingest.load_lines(client, cfg, cfg.all_seasons)
    game_off = ingest.build_game_offense(plays, drives, games, cfg)
    market = ingest.build_market(lines, games, cfg)
    print(f"  CFBD calls used this run: {client.calls_used}")

    tune = [s for s in cfg.graded_seasons if s <= 2023]
    hold = [s for s in cfg.graded_seasons if s >= 2024]

    rows = []
    for i, half_life in enumerate(GRID, 1):
        c = cfg.with_ratings(half_life_games=half_life, prior_weight_games=8.0,
                              off_weight=1.45, def_weight=1.0)
        wf = build_walkforward(game_off, games, c, cache_key=f"hlsweep{i}")
        frame = walk_forward(c, market, wf, cache_key=f"hlsweep{i}")
        r = {"cell": i, "half_life_games": half_life}
        for name, seasons in (("tune", tune), ("hold", hold)):
            sub = select_window(frame, seasons, restricted=True)
            if len(sub) < 100:
                r[f"{name}_t_total"] = np.nan
                r[f"{name}_rmse_total"] = np.nan
                r[f"{name}_sd_total"] = np.nan
                r[f"{name}_t_spread"] = np.nan
                continue
            ft = residual_fit(sub["model_total"], sub["total_open"], sub["actual_total"],
                               "total", groups=season_week_groups(sub))
            fs = residual_fit(sub["model_spread"], sub["spread_open"], sub["actual_margin"],
                               "spread", groups=season_week_groups(sub))
            r[f"{name}_t_total"] = ft.t_b
            r[f"{name}_rmse_total"] = ft.rmse
            r[f"{name}_sd_total"] = ft.sd_ratio
            r[f"{name}_t_spread"] = fs.t_b
        rows.append(r)
        print(f"  cell {i}/{len(GRID)} (half_life_games={half_life}) done", flush=True)

    sweep = pd.DataFrame(rows)
    print()
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print()
    print("GATE_SCALE tolerance: |SD ratio - 1.0| < 0.15 (need hold_sd_total in [0.85, 1.15])")
    within = sweep[(sweep["hold_sd_total"] >= 0.85) & (sweep["hold_sd_total"] <= 1.15)]
    print(f"  cells within GATE_SCALE tolerance: {sorted(within['half_life_games'].tolist())}")

    print()
    best = sweep.loc[sweep["hold_t_total"].idxmax()]
    print(f"Best held-out t(b_total): cell {int(best.cell)} "
          f"(half_life_games={int(best.half_life_games)}), t={best.hold_t_total:+.4f}, "
          f"RMSE={best.hold_rmse_total:.4f}, SD_ratio={best.hold_sd_total:.4f}")

    sweep.to_csv("output/half_life_finer_sweep.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
