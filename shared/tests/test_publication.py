from __future__ import annotations

import pytest

from prediction_contract import Forecast, MarketEvidence
from publication import (assess_quality, build_public_record, calibrated_forecast,
                         confidence_label)


def _forecast(spread, total, source):
    return Forecast.from_spread_total(spread, total, None, source)


def test_calibration_does_not_overwrite_independent_projection():
    model, market = _forecast(7, 50, "independent"), _forecast(3, 46, "consensus")
    calibrated = calibrated_forecast(
        model, market, spread_intercept=.2, spread_weight=.25,
        total_intercept=-.5, total_weight=.5,
    )
    assert model.spread == 7
    assert calibrated.spread == pytest.approx(4.2)
    assert calibrated.total == pytest.approx(47.5)


def test_record_has_all_four_views_and_honest_consensus_label():
    model, market = _forecast(7, 50, "independent"), _forecast(3, 46, "market")
    evidence = MarketEvidence("3-book median", True, 3, 3, ("A", "B", "C"))
    calibrated = calibrated_forecast(
        model, market, spread_intercept=0, spread_weight=.25,
        total_intercept=0, total_weight=.25,
    )
    record = build_public_record(
        league="nfl", game_id="g", season=2026, week=1,
        kickoff="2026-09-10T23:00:00Z", home_team="H", away_team="A",
        independent=model, market=market, market_evidence=evidence,
        calibrated=calibrated, model_version="v", data_cutoff="2026-09-10T18:00:00Z",
    )
    assert record.independent.spread == 7
    assert record.market.spread == 3
    assert record.calibrated.spread == 4
    assert record.model_market_difference.spread == 4
    assert record.market_evidence.is_consensus


def _assess(**overrides):
    kwargs = dict(
        league="ncaa", week=8, home_games_observed=7, away_games_observed=7,
        unavailable_features=(),
        market_evidence=MarketEvidence("3-book median", True, 3, 3, ("A", "B", "C")),
        spread_difference=1.5, calibration_status={"spread": False, "total": False},
        bets_allowed=False,
    )
    kwargs.update(overrides)
    return assess_quality(**kwargs)


def test_unvalidated_calibration_discloses_why_the_evidence_was_refused():
    """"Not validated" and "we threw the evidence out" are different facts to a reader."""
    silent = _assess()
    assert any("calibration is not validated" in r for r in silent.reasons)

    disclosed = _assess(
        calibration_block_reason="calibration evidence was fitted by a different model build")
    assert "calibration evidence was fitted by a different model build" in disclosed.reasons


def test_calibration_block_reason_is_dropped_once_calibration_is_validated():
    quality = _assess(
        calibration_status={"spread": True, "total": True},
        calibration_block_reason=None,
    )
    assert not any("calibration" in r for r in quality.reasons)


def test_single_line_is_never_called_consensus():
    evidence = MarketEvidence("published line", False, 1, 1, ("source",))
    assert confidence_label(unavailable_features=(), market_evidence=evidence,
                            calibrated=None, bets_allowed=False) == "single-line baseline"
