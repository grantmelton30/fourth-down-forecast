import importlib.util
import json
from pathlib import Path
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("grade_all_games",Path(__file__).resolve().parents[2]/"scripts/grade_all_games.py")
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)


def record(made="2026-09-05T16:00:00Z", model=55):
    return {"league":"ncaa","game_id":"1","kickoff":"2026-09-05T18:00:00Z",
            "generated_at":made,"data_cutoff":made,"prediction_id":made,
            "independent":{"total":model,"spread":5},"market":{"total":50,"spread":3},
            "market_evidence":{"providers":["Book"],"book_count_total":1,"book_count_spread":1}}


def test_horizon_never_uses_later_forecast_or_postkickoff_record():
    first=record();later=record("2026-09-05T17:30:00Z",45)
    post=record("2026-09-05T19:00:00Z",100)
    bundles,_=grade._published([first,later,post])
    assert ("ncaa","1",24) not in bundles
    bundle=bundles[("ncaa","1",1)]
    assert bundle["decision"] == first and bundle["last"] == later
    later["market"]["total"]=52
    row=grade._grade_row("ncaa","1",1,bundle,(30,25),[],pd.Timestamp.now(tz="UTC"))
    assert row["reference_movement_total"] == 2
    assert row["total_result"] == "WIN"
    assert row["same_book_line_clv_total"] is None


def shadow(prediction_id="2026-09-05T16:00:00Z"):
    return {"prediction_id":prediction_id,"game_id":"1","season":2026,"week":1,
            "home_team":"H","away_team":"A","kickoff":"2026-09-05T18:00:00Z",
            "generated_at":"2026-09-05T16:00:00Z","data_cutoff":"2026-09-05T16:00:00Z",
            "market_observed_at":"2026-09-05T15:59:00Z","market_label":"Book",
            "providers":["Book"],"market_total":50,"bias_total":51,"tempo_total":55,
            "shadow_total":55,"spec_version":"T1-shadow-v1","model_version":"s1",
            "selected_model":"tempo_efficiency","training_through":2025,
            "status":"prospective_shadow_not_a_pick"}


def test_shadow_is_paired_only_to_its_exact_prediction():
    published=record()
    mapped=grade._shadow_map([shadow()])
    assert grade._matching_shadow(published,mapped)["tempo_total"] == 55
    changed=record()
    changed["market"]["total"]=51
    assert grade._matching_shadow(changed,mapped) is None


def test_shadow_forecasts_are_graded_without_replacing_independent_model():
    published=record()
    bundle={"decision":published,"last":published,
            "kickoff":pd.Timestamp(published["kickoff"])}
    row=grade._grade_row("ncaa","1",1,bundle,(30,25),[],
                         pd.Timestamp("2026-09-06",tz="UTC"),shadow())
    assert row["model_total"] == 55
    assert row["shadow_bias_total"] == 51
    assert row["shadow_tempo_total"] == 55
    assert row["shadow_bias_result"] == "WIN"
    assert row["shadow_tempo_result"] == "WIN"


def test_stale_horizon_and_future_data_cutoff_are_excluded():
    stale=record("2026-09-04T11:59:00Z")
    leak=record();leak["data_cutoff"]="2026-09-05T17:00:00Z"
    bundles,_=grade._published([stale,leak])
    assert not bundles


def test_horizon_coverage_reports_missing_snapshots_explicitly():
    now=pd.Timestamp("2026-09-06",tz="UTC")
    bundle={"kickoff":pd.Timestamp("2026-09-05T18:00:00Z")}
    future={"kickoff":pd.Timestamp("2026-09-07T18:00:00Z")}
    result=grade._horizon_coverage(
        {("ncaa","1",1):bundle,("ncaa","2",1):future},
        [{"league":"ncaa","game_id":"3","decision_hours":1}],now,
    )
    assert result == [{"league":"ncaa","decision_hours":1,"available":1,
                       "missing":1,"total":2,"coverage_rate":.5}]


def quote(seen,total=50,provider="Book"):
    return {"league":"ncaa","game_id":"1","provider":provider,"status":"quoted",
            "quote_phase":"current","observed_at":seen,"ingested_at":seen,
            "total":total,"spread":-3,"source_id":"source"}


def test_same_book_close_requires_a_fresh_matching_quote():
    r=record();ko=pd.Timestamp(r["kickoff"])
    entry=quote("2026-09-05T15:59:00Z")
    close=quote("2026-09-05T17:30:00Z",52)
    assert grade._same_book_close(r,[entry,close],"total",ko)[0] == 52
    assert grade._same_book_close(r,[entry],"total",ko)[0] is None
    assert grade._same_book_close(r,[entry,quote("2026-09-05T18:01:00Z",52)],"total",ko)[0] is None
    assert grade._same_book_close(r,[entry,quote("2026-09-05T17:30:00Z",52,"Other")],"total",ko)[0] is None


def test_grading_repeat_does_not_rewrite_existing_decision(tmp_path,monkeypatch):
    predictions=tmp_path/"predictions.jsonl"
    predictions.write_text(json.dumps(record())+"\n")
    monkeypatch.setattr(grade,"PREDICTIONS",predictions)
    monkeypatch.setattr(grade,"QUOTES",tmp_path/"quotes.jsonl")
    monkeypatch.setattr(grade,"LEDGER",tmp_path/"graded.csv")
    monkeypatch.setattr(grade,"STATUS",tmp_path/"status.json")
    monkeypatch.setattr(grade,"_espn_final",lambda gid:(30,25))
    assert grade.grade() == 0
    first=grade.LEDGER.read_bytes()
    monkeypatch.setattr(grade,"_espn_final",lambda gid:pytest.fail("already graded game fetched again"))
    assert grade.grade() == 0 and grade.LEDGER.read_bytes() == first


def test_source_failure_is_visible_and_not_recorded_as_a_loss(tmp_path,monkeypatch):
    monkeypatch.setattr(grade,"PREDICTIONS",tmp_path/"p.jsonl")
    grade.PREDICTIONS.write_text(json.dumps(record())+"\n")
    monkeypatch.setattr(grade,"QUOTES",tmp_path/"q.jsonl")
    monkeypatch.setattr(grade,"LEDGER",tmp_path/"g.csv")
    monkeypatch.setattr(grade,"STATUS",tmp_path/"s.json")
    def fail(gid): raise TimeoutError()
    monkeypatch.setattr(grade,"_espn_final",fail)
    assert grade.grade() == 1
    assert not grade.LEDGER.exists()
    assert json.loads(grade.STATUS.read_text())["source_failures"][0]["error"] == "TimeoutError"
