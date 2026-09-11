#!/usr/bin/env python3
"""Grade archived forecasts at fixed pre-kickoff horizons.

Research diagnostics only. Reference movement is not execution CLV. Old grading
files are preserved; v2 records identify the exact forecast and decision horizon.
No model is rebuilt, and no post-kickoff forecast can enter the evaluation.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "ncaa-model" / "src"))
from totals_shadow_ledger import validate_shadow_record  # noqa: E402
from totals_shadow_evaluation import evaluate_totals_shadow  # noqa: E402

PREDICTIONS = ROOT / "data/predictions.jsonl"
QUOTES = ROOT / "data/market_quotes.jsonl"
SHADOWS = ROOT / "data/internal/ncaa_totals_shadow.jsonl"
LEDGER = ROOT / "data/internal/graded_games_v2.csv"
STATUS = ROOT / "data/internal/grading_status.json"
HORIZONS = (24, 1)
MAX_SNAPSHOT_AGE_HOURS = 6
CLOSE_WINDOW_HOURS = 1
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/{sport}/summary?event={gid}"


def _stamp(value):
    result = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(result) else result


def _num(record, block, field):
    try:
        value = float((record.get(block) or {}).get(field))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _records(path):
    if not path.exists():
        return []
    # A corrupt ledger is a failure, not an invitation to silently omit games.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _published(records=None, horizons=HORIZONS):
    groups = {}
    for r in (_records(PREDICTIONS) if records is None else records):
        ko, made, cutoff = (_stamp(r.get(k)) for k in ("kickoff", "generated_at", "data_cutoff"))
        if None in (ko,made,cutoff) or made >= ko or cutoff > made:
            continue
        key = (r.get("league"), str(r.get("game_id")))
        groups.setdefault(key, []).append(r)
    bundles = {}
    for key, rows in groups.items():
        rows.sort(key=lambda r: _stamp(r["generated_at"]))
        # Use the last known pregame kickoff when schedules changed.
        ko = _stamp(rows[-1]["kickoff"])
        before = [r for r in rows if _stamp(r["generated_at"]) < ko]
        for hours in horizons:
            deadline = ko-pd.Timedelta(hours=hours)
            eligible = [r for r in before if deadline-pd.Timedelta(hours=MAX_SNAPSHOT_AGE_HOURS)
                        <= _stamp(r["generated_at"]) <= deadline]
            if eligible:
                bundles[(*key,hours)] = {"decision":eligible[-1],"last":before[-1],"kickoff":ko}
    return bundles, groups


def _result(model, line, actual):
    if None in (model,line,actual) or model == line:
        return None
    if actual == line:
        return "PUSH"
    return "WIN" if (actual > line) == (model > line) else "LOSS"


def _movement(model, taken, later):
    if None in (model,taken,later) or model == taken:
        return None
    return later-taken if model > taken else taken-later


NON_BOOK_TOKENS = ("nflverse", "consensus", "unknown")


def _entry_book(evidence):
    """The one book whose number became the recorded line, or None if it is a blend.

    ``providers`` lists every book SURVEYED, not the book that supplied the line, so a
    two-book survey that took Bovada's number reads as ``["Bovada", "DraftKings"]``.
    Requiring a single provider therefore discarded games whose entry book is known and
    named -- 138 of the 286 rows graded up to 2026-09-10, every one of them labelled
    "Bovada (of 2 books)". ``label`` names the book actually taken; a label naming no
    single surveyed book ("two-book reference") is a blend and still fails closed, as
    does anything flagged ``is_consensus``.

    Dropping the old ``book_count == 1`` condition is safe because it was never the
    check doing the work: the caller still requires the chosen book's own entry quote to
    equal the recorded line exactly, so picking the wrong book fails closed there.
    """
    providers = evidence.get("providers") or []
    if evidence.get("is_consensus"):
        return None
    if len(providers) == 1:
        return providers[0]
    named = (evidence.get("label") or "").split(" (of ")[0].strip()
    return named if named in providers else None


def _same_book_close(record, quotes, market, kickoff):
    """Same-provider point movement, only with a documented fresh closing sample.

    This is line CLV, never a claim of executable price CLV. Consensus references,
    stale observations, post-kickoff samples and mismatched entry lines fail closed.
    """
    evidence = record.get("market_evidence") or {}
    provider = _entry_book(evidence)
    if provider is None:
        return None, None, None
    # Substring, not equality: the published-line baseline arrives as the provider
    # "nflverse published line", which an exact-match check let through.
    if any(token in provider.lower() for token in NON_BOOK_TOKENS):
        return None, None, None
    made = _stamp(record["generated_at"])
    observations=[]
    for q in quotes:
        if (q.get("league"),str(q.get("game_id")),q.get("provider")) != (record["league"],str(record["game_id"]),provider):
            continue
        if q.get("status") != "quoted" or q.get("quote_phase") != "current":
            continue
        seen, ingested = _stamp(q.get("observed_at")), _stamp(q.get("ingested_at"))
        try:
            number=float(q.get(market))
        except (TypeError,ValueError):
            continue
        if not math.isfinite(number) or seen is None or seen >= kickoff:
            continue
        # The market archive stores negative-home-favorite, projections positive.
        observations.append((seen,-number if market=="spread" else number,ingested,q.get("source_id")))
    entry = [q for q in observations if made-pd.Timedelta(hours=MAX_SNAPSHOT_AGE_HOURS)<=q[0]<=made
             and q[2] is not None and q[2]<=made]
    if not entry:
        return None,None,None
    entry = max(entry,key=lambda q:q[0])
    if entry[1] != _num(record,"market",market):
        return None,None,None
    closing = [q for q in observations if kickoff-pd.Timedelta(hours=CLOSE_WINDOW_HOURS)<=q[0]<kickoff
               and q[0]>=entry[0] and q[3]==entry[3]]
    if not closing:
        return None,None,None
    closing=max(closing,key=lambda q:q[0])
    return closing[1],provider,closing[0].isoformat()


def _shadow_map(records):
    found = {}
    for raw in records:
        row = validate_shadow_record(raw)
        prediction_id = row["prediction_id"]
        if prediction_id in found and found[prediction_id] != row:
            raise ValueError("conflicting totals shadows for one prediction")
        found[prediction_id] = row
    return found


def _matching_shadow(record, shadows):
    shadow = shadows.get(str(record.get("prediction_id")))
    if shadow is None:
        return None
    identity = (str(record.get("game_id")), _stamp(record.get("kickoff")),
                _stamp(record.get("generated_at")), _stamp(record.get("data_cutoff")))
    candidate = (shadow["game_id"], _stamp(shadow["kickoff"]),
                 _stamp(shadow["generated_at"]), _stamp(shadow["data_cutoff"]))
    if candidate != identity or shadow["market_total"] != _num(record,"market","total"):
        return None
    return shadow


def _grade_row(league,gid,hours,bundle,actual,quotes,now,shadow=None):
    r,last,ko = bundle["decision"],bundle["last"],bundle["kickoff"]
    home,away=actual
    row={"league":league,"game_id":gid,"decision_hours":hours,"schema_version":2,
         "season":r.get("season"),"week":r.get("week"),"home_team":r.get("home_team"),
         "away_team":r.get("away_team"),"kickoff":ko.isoformat(),
         "prediction_id":r.get("prediction_id"),"model_version":r.get("model_version"),
         "projection_generated_at":r["generated_at"],"graded_at":now.isoformat(),
         "matchup_group":r.get("matchup_group","unknown"),
         "actual_margin":home-away,"actual_total":home+away,
         "confidence":r.get("confidence"),"out_of_distribution":r.get("out_of_distribution"),
         "actual_horizon_hours":(ko-_stamp(r["generated_at"])).total_seconds()/3600}
    interval_low=_num(r,"independent","interval_80_low")
    interval_high=_num(r,"independent","interval_80_high")
    row["margin_interval_80_covered"]=(interval_low<=home-away<=interval_high
                                        if None not in (interval_low,interval_high) else None)
    for market,observed in (("spread",home-away),("total",home+away)):
        model,taken,later=(_num(rec,block,market) for rec,block in
                           ((r,"independent"),(r,"market"),(last,"market")))
        close,book,seen=_same_book_close(r,quotes,market,ko)
        row.update({"model_"+market:model,"market_"+market:taken,
                    market+"_result":_result(model,taken,observed),
                    "reference_movement_"+market:_movement(model,taken,later),
                    "same_book_line_clv_"+market:_movement(model,taken,close),
                    "close_"+market:close,"close_book_"+market:book,"close_observed_at_"+market:seen})
    if shadow is not None:
        row.update({"shadow_spec_version":shadow["spec_version"],
                    "shadow_model_version":shadow["model_version"],
                    "shadow_market_total":shadow["market_total"]})
        for name in ("bias", "tempo"):
            forecast=float(shadow[name+"_total"])
            row["shadow_"+name+"_total"] = forecast
            row["shadow_"+name+"_result"] = _result(
                forecast, shadow["market_total"], home+away
            )
    return row


def _espn_final(gid):
    url=ESPN.format(sport="college-football",gid=gid)
    with urllib.request.urlopen(url,timeout=25) as response:
        p=json.load(response)
    c=p["header"]["competitions"][0]
    if c["status"]["type"]["state"] != "post":
        return None
    scores={v["homeAway"]:float(v["score"]) for v in c["competitors"]}
    return scores["home"],scores["away"]


def _nfl_finals():
    import nflreadpy as nfl
    frame=nfl.load_schedules().to_pandas().dropna(subset=["home_score","away_score"])
    return {str(r.game_id):(float(r.home_score),float(r.away_score)) for r in frame.itertuples()}


def _atomic_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    tmp.replace(path)


def _horizon_coverage(bundles, missing, now):
    available=Counter(
        (league,hours) for (league,_gid,hours),bundle in bundles.items()
        if bundle["kickoff"] < now
    )
    absent=Counter((row["league"],row["decision_hours"]) for row in missing)
    result=[]
    for league,hours in sorted(set(available)|set(absent)):
        found=int(available[(league,hours)])
        missed=int(absent[(league,hours)])
        total=found+missed
        result.append({"league":league,"decision_hours":int(hours),
                       "available":found,"missing":missed,"total":total,
                       "coverage_rate":found/total if total else 0.0})
    return result


def _clv_coverage(frame):
    """How often a closing line was actually resolved, per league and market.

    ``horizon_coverage`` measures whether a forecast snapshot existed; it says nothing
    about whether the close was captured, so a run could report "ok" on 96% of rows
    having no CLV at all -- which is what happened up to 2026-09-10. CLV is the whole
    reason this grader exists, so its capture rate is reported beside the rest.
    """
    result=[]
    if frame is None or not len(frame):
        return result
    for league,rows in frame.groupby("league"):
        for market in ("spread","total"):
            column="close_"+market
            if column not in rows:
                continue
            captured=int(rows[column].notna().sum())
            total=int(len(rows))
            result.append({"league":str(league),"market":market,"captured":captured,
                           "total":total,
                           "coverage_rate":captured/total if total else 0.0})
    return result


# Below this, the ledger cannot answer the question it was built for and says so.
CLV_COVERAGE_FLOOR = 0.25


def grade(limit=None,verbose=True):
    now=pd.Timestamp.now(tz="UTC")
    ledger=pd.read_csv(LEDGER,dtype={"game_id":str}) if LEDGER.exists() else pd.DataFrame()
    already=set(zip(ledger.league,ledger.game_id,ledger.decision_hours)) if len(ledger) else set()
    bundles,groups=_published()
    quotes=_records(QUOTES)
    shadows=_shadow_map(_records(SHADOWS))
    pending={key:b for key,b in bundles.items() if key not in already and b["kickoff"]<now}
    game_keys=sorted({key[:2] for key in pending})
    missing=[]
    for key,rows in groups.items():
        if _stamp(rows[-1]["kickoff"])>=now:
            continue
        for h in HORIZONS:
            if (*key,h) not in bundles:
                missing.append({"league":key[0],"game_id":key[1],"decision_hours":h})
    failures=[];finals={};nfl_finals=None
    for league,gid in game_keys[:limit] if limit is not None else game_keys:
        try:
            if league=="nfl":
                if nfl_finals is None:
                    nfl_finals=_nfl_finals()
                actual=nfl_finals.get(gid)
            elif league=="ncaa":
                actual=_espn_final(gid)
            else:
                raise ValueError("unknown league")
            if actual is not None:
                finals[(league,gid)]=actual
        except Exception as exc:
            failures.append({"league":league,"game_id":gid,"error":type(exc).__name__})
    quote_groups={}
    for q in quotes:
        quote_groups.setdefault((q.get("league"),str(q.get("game_id"))),[]).append(q)
    fresh=[_grade_row(l,g,h,b,finals[(l,g)],quote_groups.get((l,g),[]),now,
                      _matching_shadow(b["decision"],shadows))
           for (l,g,h),b in pending.items() if (l,g) in finals]
    if fresh:
        LEDGER.parent.mkdir(parents=True,exist_ok=True)
        frame=pd.DataFrame(fresh)
        if len(ledger):
            frame=pd.concat([ledger,frame],ignore_index=True)
        tmp=LEDGER.with_suffix(".csv.tmp")
        frame.to_csv(tmp,index=False)
        tmp.replace(LEDGER)
    graded_frame=frame if len(fresh) else ledger
    shadow_evaluation=evaluate_totals_shadow(graded_frame)
    horizon_coverage=_horizon_coverage(bundles,missing,now)
    clv_coverage=_clv_coverage(graded_frame)
    starved=[item for item in clv_coverage if item["coverage_rate"]<CLV_COVERAGE_FLOOR]
    _atomic_json(STATUS,{"generated_at":now.isoformat(),"schema_version":2,
                        "new_rows":len(fresh),"pending_game_count":len(game_keys),
                        "missing_horizon_snapshots":missing,"source_failures":failures,
                        "horizon_coverage":horizon_coverage,
                        "clv_coverage":clv_coverage,
                        "clv_coverage_floor":CLV_COVERAGE_FLOOR,
                        "clv_starved":starved,
                        "ncaa_totals_shadow":shadow_evaluation,
                        "status":"degraded" if failures else "ok"})
    if failures:
        print("::warning::Full-slate grading has source failures; see grading_status.json")
    print(f"graded {len(fresh)} horizon rows; {len(missing)} missing snapshots; {len(failures)} source failures")
    for item in horizon_coverage:
        print(f"  {item['league']} {item['decision_hours']}h snapshot coverage: "
              f"{item['available']}/{item['total']} ({item['coverage_rate']:.1%})")
    for item in clv_coverage:
        print(f"  {item['league']} {item['market']} closing-line coverage: "
              f"{item['captured']}/{item['total']} ({item['coverage_rate']:.1%})")
    if starved:
        worst=", ".join(f"{i['league']} {i['market']} {i['coverage_rate']:.1%}" for i in starved)
        # Not a source failure: the run is healthy, the evidence is thin. Saying so is
        # the point -- a silent "ok" is how 2.8% CLV coverage went unnoticed for a week.
        print(f"::warning::Closing-line coverage below {CLV_COVERAGE_FLOOR:.0%} ({worst}); "
              "same-book CLV cannot be read from this ledger yet")
    return 1 if failures else 0


def report():
    if not LEDGER.exists():
        print("no v2 graded forecasts yet")
        return 0
    frame=pd.read_csv(LEDGER)
    shadow_evaluation=evaluate_totals_shadow(frame)
    print("NCAA totals shadow:",json.dumps(shadow_evaluation,sort_keys=True))
    for (league,hours),group in frame.groupby(["league","decision_hours"]):
        print(f"{league} at {hours}h: {len(group)} archived forecasts")
        for market,actual in (("spread","actual_margin"),("total","actual_total")):
            paired=group.dropna(subset=["model_"+market,"market_"+market,actual])
            if paired.empty:
                continue
            rmse=lambda col: float(((paired[col]-paired[actual])**2).mean()**.5)
            print(f"  {market}: model RMSE {rmse('model_'+market):.3f}; market {rmse('market_'+market):.3f}; n={len(paired)}")
            print("  outcomes:",paired[market+"_result"].value_counts().to_dict())
            for field in ("reference_movement_","same_book_line_clv_"):
                values=paired[field+market].dropna()
                print(f"  {field}{market}: n={len(values)}, mean={values.mean() if len(values) else 'unavailable'}")
        if league=="ncaa":
            for name in ("bias","tempo"):
                column="shadow_"+name+"_total"
                paired=group.dropna(subset=[column,"actual_total"]) if column in group else pd.DataFrame()
                if len(paired):
                    rmse=float(((paired[column]-paired.actual_total)**2).mean()**.5)
                    outcomes=paired["shadow_"+name+"_result"].value_counts().to_dict()
                    print(f"  shadow {name}: RMSE {rmse:.3f}; n={len(paired)}; outcomes={outcomes}")
    print("Reference diagnostics only. Neither reference movement nor line CLV proves executable returns.")
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",action="store_true")
    p.add_argument("--quiet",action="store_true")
    p.add_argument("--limit",type=int)
    a=p.parse_args()
    if a.limit is not None and a.limit<0:
        p.error("limit must be nonnegative")
    if a.report:
        return report()
    result=grade(a.limit,not a.quiet)
    report()
    return result


if __name__=="__main__":
    raise SystemExit(main())
