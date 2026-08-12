"""GATE_NO_LOOKAHEAD (§8b.2). NON-NEGOTIABLE #1.

For a random sample of backtest rows, assert that no game used in fitting that row's
ratings has a kickoff at or after the predicted game's kickoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_frame_path
from src.config import CACHE_DIR, load_config
from src.ingest import load_pbp, load_schedules
from src.ratings import build_game_offense_table, fit_ratings, week_cutoffs


def check_no_lookahead(sample: int = 200, seed: int = 20260730):
    """Returns (ok, detail). Used both by pytest and by the gate runner."""
    cfg = load_config()
    frame_path = backtest_frame_path(load_config())
    if not frame_path.exists():
        return False, "no backtest_frame.parquet -- run the backtest first"

    schedules = load_schedules(cfg.train_seasons)
    frame = pd.read_parquet(frame_path)
    pbp = load_pbp(cfg.train_seasons)
    game_off = build_game_offense_table(pbp, schedules, cache_key="main")

    cutoffs = week_cutoffs(schedules, cfg.train_seasons).set_index(["season", "week"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frame), size=min(sample, len(frame)), replace=False)
    rows = frame.iloc[idx]

    violations = []
    for r in rows.itertuples(index=False):
        as_of = pd.Timestamp(cutoffs.loc[(r.season, r.week), "as_of"])
        kickoff = pd.Timestamp(r.kickoff)
        # 1. The cutoff must not be after the game we are predicting.
        if as_of > kickoff:
            violations.append(f"{r.game_id}: as_of {as_of} > kickoff {kickoff}")
            continue
        # 2. No game visible to that fit may kick off at or after the predicted game.
        used = game_off[game_off["kickoff"] < as_of]
        if len(used) and used["kickoff"].max() >= kickoff:
            violations.append(
                f"{r.game_id}: fit saw a game at {used['kickoff'].max()} >= {kickoff}"
            )

    if violations:
        return False, f"{len(violations)} violations, e.g. {violations[:3]}"
    return True, f"{len(rows)} sampled rows, no game used from at or after kickoff"


def test_no_lookahead_in_backtest():
    ok, detail = check_no_lookahead()
    if detail.startswith("no backtest_frame"):
        pytest.skip(detail)
    assert ok, detail


def test_fit_ratings_only_sees_the_past():
    cfg = load_config()
    schedules = load_schedules(cfg.train_seasons)
    pbp = load_pbp(cfg.train_seasons)
    game_off = build_game_offense_table(pbp, schedules, cache_key="main")

    as_of = pd.Timestamp("2023-10-01")
    fit = fit_ratings(game_off, as_of, cfg)
    assert fit.n_games > 0

    visible = game_off[game_off["kickoff"] < as_of]
    assert visible["kickoff"].max() < as_of


def test_walkforward_cutoff_is_week_first_kickoff():
    cfg = load_config()
    schedules = load_schedules(cfg.train_seasons)
    cutoffs = week_cutoffs(schedules, [2023])
    reg = schedules[(schedules["season"] == 2023) & schedules["game_type"].eq("REG")]
    for row in cutoffs.itertuples(index=False):
        assert row.as_of == reg[reg["week"] == row.week]["kickoff"].min()
