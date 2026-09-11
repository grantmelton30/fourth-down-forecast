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


def two_book(taken="Bovada",label=None,**evidence):
    """A survey of two books whose line was taken from one of them."""
    r=record()
    r["market_evidence"]={"providers":["Bovada","DraftKings"],
                          "book_count_total":2,"book_count_spread":2,
                          "label":label if label is not None else f"{taken} (of 2 books)",
                          "is_consensus":False,**evidence}
    return r


def test_two_book_survey_still_has_a_named_entry_book():
    # Requiring a single surveyed provider discarded 138 of 286 graded rows whose entry
    # book was named all along. The number still has to match that book's own quote.
    r=two_book();ko=pd.Timestamp(r["kickoff"])
    entry=quote("2026-09-05T15:59:00Z",provider="Bovada")
    close=quote("2026-09-05T17:30:00Z",52,provider="Bovada")
    assert grade._same_book_close(r,[entry,close],"total",ko)[:2] == (52,"Bovada")
    # The OTHER surveyed book's quotes are not the entry book's and settle nothing.
    other=[quote("2026-09-05T15:59:00Z",provider="DraftKings"),
           quote("2026-09-05T17:30:00Z",52,provider="DraftKings")]
    assert grade._same_book_close(r,other,"total",ko)[0] is None


def test_blended_and_consensus_references_still_fail_closed():
    ko=pd.Timestamp(record()["kickoff"])
    quotes=[quote("2026-09-05T15:59:00Z",provider="Bovada"),
            quote("2026-09-05T17:30:00Z",52,provider="Bovada")]
    # A label naming no single surveyed book is a blend, not a book.
    assert grade._entry_book(two_book(label="two-book reference")["market_evidence"]) is None
    assert grade._same_book_close(two_book(label="two-book reference"),quotes,"total",ko)[0] is None
    # An explicit consensus flag fails closed even when one provider is listed.
    assert grade._entry_book({"providers":["Bovada"],"is_consensus":True}) is None


def test_published_line_baseline_is_not_treated_as_a_book():
    # "nflverse published line" is not equal to "nflverse", so an exact-match exclusion
    # let the baseline through and would have scored it as same-book CLV.
    r=record()
    r["market_evidence"]={"providers":["nflverse published line"],
                          "book_count_total":1,"book_count_spread":1,
                          "label":"published line baseline","is_consensus":False}
    ko=pd.Timestamp(r["kickoff"])
    quotes=[quote("2026-09-05T15:59:00Z",provider="nflverse published line"),
            quote("2026-09-05T17:30:00Z",52,provider="nflverse published line")]
    assert grade._same_book_close(r,quotes,"total",ko)[0] is None


def test_status_reports_closing_line_coverage_not_just_snapshot_coverage():
    # A ledger can be fully snapshot-covered and still resolve almost no closing lines.
    frame=pd.DataFrame([{"league":"ncaa","close_spread":None,"close_total":1.0},
                        {"league":"ncaa","close_spread":None,"close_total":None},
                        {"league":"ncaa","close_spread":None,"close_total":None},
                        {"league":"ncaa","close_spread":None,"close_total":None}])
    coverage=grade._clv_coverage(frame)
    assert {"league":"ncaa","market":"spread","captured":0,"total":4,
            "coverage_rate":0.0} in coverage
    assert {"league":"ncaa","market":"total","captured":1,"total":4,
            "coverage_rate":.25} in coverage
    assert grade._clv_coverage(pd.DataFrame()) == []


def test_starved_closing_line_coverage_is_reported_without_faking_a_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(grade,"PREDICTIONS",tmp_path/"p.jsonl")
    grade.PREDICTIONS.write_text(json.dumps(record())+"\n")
    monkeypatch.setattr(grade,"QUOTES",tmp_path/"q.jsonl")
    monkeypatch.setattr(grade,"LEDGER",tmp_path/"g.csv")
    monkeypatch.setattr(grade,"STATUS",tmp_path/"s.json")
    monkeypatch.setattr(grade,"_espn_final",lambda gid:(30,25))
    assert grade.grade() == 0
    status=json.loads(grade.STATUS.read_text())
    # No quotes were supplied, so nothing resolved -- and the run still says "ok",
    # because thin evidence is not a source failure. It must not be silent about it.
    assert status["status"] == "ok"
    assert all(item["coverage_rate"] == 0.0 for item in status["clv_coverage"])
    assert status["clv_starved"] and status["clv_coverage_floor"] == grade.CLV_COVERAGE_FLOOR


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
