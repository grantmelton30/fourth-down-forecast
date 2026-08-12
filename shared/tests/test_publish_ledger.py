from __future__ import annotations

import importlib.util
from pathlib import Path

from prediction_contract import Forecast, PredictionLedger, build_prediction_record

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("publish_ledger", ROOT / "scripts/publish_ledger.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(generated_at, spread):
    return build_prediction_record(
        league="nfl", game_id="g", season=2026, week=1,
        kickoff="2026-09-10T23:00:00Z", home_team="H", away_team="A",
        independent=Forecast.from_spread_total(spread, 44, .55, "model"),
        market=None, calibrated=None, model_version="v", data_cutoff="2026-09-10T18:00:00Z",
        generated_at=generated_at,
    )


def test_feed_keeps_latest_immutable_record_per_game(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.jsonl")
    ledger.append(_record("2026-09-10T18:01:00Z", 2))
    ledger.append(_record("2026-09-10T19:01:00Z", 3))
    feed = MODULE.build_feed(ledger)
    assert feed["record_count"] == 1
    assert feed["records"][0]["independent"]["spread"] == 3
