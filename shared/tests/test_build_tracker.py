"""The forward-record arithmetic (scripts/build_tracker.py).

The numbers on the Record tab are the only scoreboard this project has, and the combined
view pools two rules, so the maths has to be right in one place rather than two.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "build_tracker", ROOT / "scripts" / "build_tracker.py")
bt = importlib.util.module_from_spec(spec)
sys.modules["build_tracker"] = bt
spec.loader.exec_module(bt)


def test_no_settled_bets_is_not_zero_percent():
    """An empty record must read 'no data', never 0% -- which would look like total failure
    on the day a rule is registered."""
    s = bt._stats(0, 0)
    assert s["win_pct"] is None and s["units"] == 0.0


def test_units_price_a_loss_at_minus_110():
    """A loss costs 1.1 units, a win pays 1.0. Treating them symmetrically would show a
    50% record as break-even when it is actually losing money."""
    assert bt._stats(1, 1)["units"] == pytest.approx(-0.1)
    assert bt._stats(10, 0)["units"] == pytest.approx(10.0)
    assert bt._stats(0, 10)["units"] == pytest.approx(-11.0)


def test_break_even_rate_is_roughly_units_neutral():
    """52.38% is the break-even claim the whole site is measured against; at that rate the
    units should sit near zero, which is what makes the number meaningful."""
    wins, losses = 5238, 4762
    assert bt._stats(wins, losses)["units"] == pytest.approx(0.0, abs=5.0)


def test_interval_needs_more_than_one_bet():
    assert bt._stats(1, 0)["ci_low"] is None
    assert bt._stats(30, 20)["ci_low"] is not None


def test_win_rate_and_interval_are_sane():
    s = bt._stats(60, 40)
    assert s["win_pct"] == 60.0
    assert s["ci_low"] < 60.0 < s["ci_high"]


def test_breakeven_constant_matches_minus_110():
    """-110 both sides: 110/210. If this drifts, every 'vs break-even' figure on the site
    silently moves with it."""
    assert bt.BREAKEVEN == pytest.approx(100 * 110 / 210, abs=0.01)


# --- real prices, not an assumed -110 -------------------------------------------------

def test_loss_cost_and_breakeven_track_the_actual_price():
    """-110 is an assumption, not a fact. The rate you must beat moves with the price, and a
    rule measured at 54% is profitable at -105 and losing at -120 -- so the price, not the
    model, decides the sign of that edge."""
    assert bt._loss_units(-110) == pytest.approx(1.10)
    assert bt._loss_units(-105) == pytest.approx(1.05)
    assert bt._loss_units(-120) == pytest.approx(1.20)
    assert bt._loss_units(+120) == pytest.approx(100 / 120)
    assert 100 * bt._breakeven(-110) == pytest.approx(52.38, abs=0.01)
    assert 100 * bt._breakeven(-105) == pytest.approx(51.22, abs=0.01)
    assert 100 * bt._breakeven(-120) == pytest.approx(54.55, abs=0.01)


def test_missing_price_costs_the_assumed_vig_not_nothing():
    """A bet with no recorded price is unknown, never free."""
    assert bt._loss_units(None) == pytest.approx(1.10)
    assert bt._loss_units(float("nan")) == pytest.approx(1.10)


def test_priced_units_use_the_same_convention_as_the_headline_units():
    """`_stats` reports wins - losses*1.1 and every figure in GATES.md and both
    registrations is on that basis. If the priced number used risk-one-unit instead, every
    total would shift about 10% and the site would contradict its own documentation."""
    wins, losses = 347, 254
    assert _stats_units(wins, losses) == pytest.approx(wins - losses * bt._loss_units(-110))


def _stats_units(w, l):
    return bt._stats(w, l)["units"]


def test_legacy_rows_do_not_advance_active_checkpoint(tmp_path):
    import pandas as pd
    path = tmp_path / "p1.csv"
    pd.DataFrame([
      {"result":"WIN", "line_basis":"opener"},
      {"result":"LOSS", "line_basis":"current"},
      {"result":"WIN", "line_basis":"current", "protocol_version":"P1-current-v2"},
    ]).to_csv(path,index=False)
    spec = {**bt.RULES["P1"], "log":path}
    summary, rows = bt.summarise("P1", spec)
    assert summary["settled"] == 1 and summary["wins"] == 1
    assert len(summary["cohorts"]) == 3 and len(rows) == 3
    assert "199" in summary["status"]
    assert summary["priced_units"] is None and summary["priced_bets"] == 0


def test_priced_units_exclude_unpriced_rows(tmp_path):
    import pandas as pd
    path = tmp_path / "p1.csv"
    pd.DataFrame([
      {"result":"WIN", "price":120, "protocol_version":"P1-current-v2"},
      {"result":"LOSS", "price":-120, "protocol_version":"P1-current-v2"},
      {"result":"LOSS", "protocol_version":"P1-current-v2"},
    ]).to_csv(path,index=False)
    summary,_ = bt.summarise("P1",{**bt.RULES["P1"],"log":path})
    assert summary["priced_bets"] == 2 and summary["priced_units"] == pytest.approx(-.2)


def test_wilson_interval_has_uncertainty_at_extreme_records():
    assert 0 < bt._stats(2,0)["ci_low"] < 100
    assert 0 < bt._stats(0,2)["ci_high"] < 100


def test_generated_payload_reconciles_source_rows_and_scoreboards():
    payload = bt.build_payload()
    bt.validate_payload(payload)


def test_tracker_validation_rejects_duplicate_games_and_bad_combined_counts():
    payload = bt.build_payload()
    if payload["bets"]:
        duplicate = copy.deepcopy(payload)
        duplicate["bets"].append(copy.deepcopy(duplicate["bets"][0]))
        with pytest.raises(ValueError, match="duplicate tracker bet"):
            bt.validate_payload(duplicate)

    broken = copy.deepcopy(payload)
    broken["combined"]["logged"] += 1
    with pytest.raises(ValueError, match="logged count does not reconcile"):
        bt.validate_payload(broken)
