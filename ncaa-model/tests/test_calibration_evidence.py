"""NCAA calibration evidence must be anchored to the close, not the opener.

GATES.md records that NCAA's only apparent edge -- a totals signal -- was a false
positive that existed solely against the opener and vanished against the close. The
`fit_blend` default (`grade="open"`) exists for `run_paper.py`'s exploratory grading of
that same phenomenon. But the "Calibrated" column published to users is a claim that a
number has been validated, and every other honest NCAA gate (GATE_RMSE_*) grades against
the close -- so calibration evidence must clear that same, harder bar rather than the one
that produced the false positive.

Requires no cache: the frame is synthetic and self-contained.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

import run_backtest
from src.config import load_config
from src.market import fit_blend


def _synthetic_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Opener and close are constructed to disagree, so a fit anchored to one differs
    detectably from a fit anchored to the other."""
    rng = np.random.default_rng(seed)
    model_spread = rng.normal(0, 10, size=n)
    spread_open = model_spread * 0.10 + rng.normal(0, 9, size=n)
    spread_close = model_spread * 0.95 + rng.normal(0, 2, size=n)
    actual_margin = spread_close + rng.normal(0, 10, size=n)

    model_total = rng.normal(55, 8, size=n)
    total_open = model_total * 0.10 + rng.normal(50, 7, size=n)
    total_close = model_total * 0.95 + rng.normal(50, 2, size=n)
    actual_total = total_close + rng.normal(0, 10, size=n)

    return pd.DataFrame({
        "season": rng.integers(2021, 2025, size=n),
        "week": rng.integers(1, 14, size=n),
        "model_spread": model_spread, "spread_open": spread_open,
        "spread_close": spread_close, "actual_margin": actual_margin,
        "model_total": model_total, "total_open": total_open,
        "total_close": total_close, "actual_total": actual_total,
    })


def test_calibration_evidence_is_close_anchored(tmp_path):
    cfg = load_config()
    frame = _synthetic_frame()

    path = run_backtest.fit_and_write_calibration(
        frame, cfg, cache_dir=tmp_path, model_version="ncaa-test",
        data_cutoff="2024-01-01",
    )
    published = json.loads(path.read_text())["weights"]

    expected_close = asdict(fit_blend(frame, cfg, grade="close"))
    expected_open = asdict(fit_blend(frame, cfg, grade="open"))

    assert published["b_model_spread"] == pytest.approx(expected_close["b_model_spread"])
    assert published["b_model_total"] == pytest.approx(expected_close["b_model_total"])
    # And it must NOT match an opener-anchored fit -- the two are constructed to diverge.
    assert published["b_model_spread"] != pytest.approx(
        expected_open["b_model_spread"], rel=1e-6
    )


def test_calibration_evidence_carries_model_version_and_league(tmp_path):
    cfg = load_config()
    frame = _synthetic_frame()

    path = run_backtest.fit_and_write_calibration(
        frame, cfg, cache_dir=tmp_path, model_version="ncaa-abc123",
        data_cutoff="2024-01-01",
    )
    payload = json.loads(path.read_text())

    assert payload["league"] == "ncaa"
    assert payload["model_version"] == "ncaa-abc123"
