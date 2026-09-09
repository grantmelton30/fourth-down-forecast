from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from prediction_contract import Forecast, MarketEvidence, build_prediction_record


PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_slate.py"
SPEC = importlib.util.spec_from_file_location("build_slate_shadow_test", PATH)
build_slate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_slate)


class Adapter:
    def totals_shadow(self, game_id, market_total):
        assert game_id == "g1" and market_total == 52.5
        return {"market_total":52.5,"bias_total":52.1,"tempo_total":51.9,
                "shadow_total":51.9,"spec_version":"T1-shadow-v1",
                "model_version":"shadow1","selected_model":"tempo_efficiency",
                "training_through":2025,"status":"prospective_shadow_not_a_pick"}


def inputs():
    market = Forecast.from_spread_total(3, 52.5, None, "Book")
    evidence = MarketEvidence("Book",False,1,1,("Book",),"2026-09-12T11:55:00Z")
    record = build_prediction_record(
        league="ncaa",game_id="g1",season=2026,week=3,
        kickoff="2026-09-12T17:00:00Z",home_team="Home",away_team="Away",
        independent=Forecast.from_spread_total(2,51,None,"model"),market=market,
        calibrated=None,model_version="main1",data_cutoff="2026-09-12T11:59:00Z",
        generated_at="2026-09-12T12:00:00Z")
    return market,evidence,record


def test_build_path_appends_exact_prediction_and_quote_identity(tmp_path):
    market,evidence,record=inputs()
    path=tmp_path/"shadow.jsonl"
    assert build_slate._append_totals_shadow(
        path,Adapter(),{"game_id":"g1"},market,evidence,record) is True
    text=path.read_text()
    assert record.prediction_id in text
    assert "2026-09-12T11:55:00+00:00" in text


def test_build_path_requires_observed_market_timestamp(tmp_path):
    market,evidence,record=inputs()
    evidence=SimpleNamespace(observed_at=None)
    assert build_slate._append_totals_shadow(
        tmp_path/"shadow.jsonl",Adapter(),{"game_id":"g1"},market,evidence,record) is False
