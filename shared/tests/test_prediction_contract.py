from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from prediction_contract import (
    Forecast,
    PredictionLedger,
    PredictionRecord,
    build_prediction_record,
)


def _forecast(spread: float, total: float, source: str) -> Forecast:
    return Forecast.from_spread_total(
        spread=spread,
        total=total,
        home_win_probability=0.61,
        source=source,
    )


def test_four_outputs_remain_explicit_and_difference_is_independent_minus_market():
    record = build_prediction_record(
        league="nfl",
        game_id="2026_01_DAL_PHI",
        season=2026,
        week=1,
        kickoff="2026-09-10T00:20:00Z",
        home_team="PHI",
        away_team="DAL",
        independent=_forecast(4.0, 48.0, "football-model"),
        market=_forecast(6.0, 46.0, "three-book median"),
        calibrated=_forecast(5.4, 46.4, "prior-season blend"),
        model_version="nfl-test-v1",
        data_cutoff="2026-09-09T18:00:00Z",
        generated_at="2026-09-09T18:01:00Z",
    )

    assert record.independent.spread == 4.0
    assert record.market.spread == 6.0
    assert record.calibrated.spread == 5.4
    assert record.model_market_difference.spread == -2.0
    assert record.model_market_difference.total == 2.0
    assert record.independent.home_score == 26.0
    assert record.independent.away_score == 22.0


def test_market_absence_does_not_contaminate_independent_forecast():
    record = build_prediction_record(
        league="ncaa",
        game_id="game-1",
        season=2026,
        week=1,
        kickoff="2026-08-29T16:00:00Z",
        home_team="A",
        away_team="B",
        independent=_forecast(3.0, 51.0, "football-model"),
        market=None,
        calibrated=None,
        model_version="ncaa-test-v1",
        data_cutoff="2026-08-28T20:00:00Z",
    )

    assert record.independent.spread == 3.0
    assert record.market is None
    assert record.calibrated is None
    assert record.model_market_difference is None


def test_ledger_is_idempotent_but_refuses_overwrite(tmp_path):
    ledger = PredictionLedger(tmp_path / "predictions.jsonl")
    record = build_prediction_record(
        league="nfl", game_id="g", season=2026, week=1,
        kickoff="2026-09-10T00:20:00Z", home_team="H", away_team="A",
        independent=_forecast(2.0, 44.0, "football-model"), market=None,
        calibrated=None, model_version="v1", data_cutoff="2026-09-09T18:00:00Z",
        generated_at="2026-09-09T18:01:00Z",
    )
    assert ledger.append(record) is True
    assert ledger.append(record) is False

    payload = record.to_dict()
    payload["independent"]["spread"] = 9.0
    conflicting = PredictionRecord.from_dict(payload)
    with pytest.raises(ValueError, match="immutable"):
        ledger.append(conflicting)

    lines = ledger.path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["prediction_id"] == record.prediction_id


def test_data_cutoff_must_precede_kickoff():
    with pytest.raises(ValueError, match="data_cutoff"):
        build_prediction_record(
            league="nfl", game_id="g", season=2026, week=1,
            kickoff="2026-09-10T00:20:00Z", home_team="H", away_team="A",
            independent=_forecast(2.0, 44.0, "football-model"), market=None,
            calibrated=None, model_version="v1", data_cutoff="2026-09-10T01:00:00Z",
        )


def test_quality_evidence_round_trips_in_public_record():
    record = build_prediction_record(
        league="ncaa", game_id="q", season=2026, week=1,
        kickoff="2026-08-29T16:00:00Z", home_team="H", away_team="A",
        independent=_forecast(2, 50, "model"), market=_forecast(18, 52, "market"),
        calibrated=None, model_version="v", data_cutoff="2026-08-28T18:00:00Z",
        confidence="incomplete", quality_reasons=("missing returning production",),
        warnings=("Extreme model/market disagreement; pick suppressed",),
        pick_eligible=False, out_of_distribution=True,
        calibration_status={"spread": False, "total": True},
    )
    restored = PredictionRecord.from_dict(record.to_dict())
    assert restored.quality_reasons == ("missing returning production",)
    assert restored.out_of_distribution is True
    assert restored.calibration_status == {"spread": False, "total": True}
