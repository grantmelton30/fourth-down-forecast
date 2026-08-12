"""Versioned, fail-closed gate results shared by both football models."""
from __future__ import annotations
import json, os
from datetime import datetime,timezone
from pathlib import Path
from typing import Iterable,Mapping
SCHEMA_VERSION=1; FILENAME="gate_results.json"
REQUIRED_PROMOTION_GATES={
 "nfl":frozenset({"GATE_KEY_NUMBERS","GATE_NO_LOOKAHEAD","GATE_BEATS_ELO","GATE_BLEND_INFORMATIVE","GATE_CALIBRATED","GATE_RMSE_SPREAD","GATE_RMSE_TOTAL","GATE_UNBIASED"}),
 "ncaa":frozenset({"GATE_OPENER_COVERAGE","GATE_NO_LOOKAHEAD","GATE_GARBAGE_FILTER","GATE_UNBIASED","GATE_UNBIASED_BY_WEEK","GATE_BLEND_INFORMATIVE","GATE_RMSE_TOTAL","GATE_RMSE_SPREAD","GATE_KEY_NUMBERS","GATE_CALIBRATED"})}
class GateArtifactError(ValueError): pass
def _row(gate):
    raw=dict(gate) if isinstance(gate,Mapping) else {k:getattr(gate,k,"" if k in ("observed","detail") else None) for k in ("name","passed","observed","detail")}
    if not raw.get("name"): raise GateArtifactError("gate record has no name")
    if raw.get("passed") not in (True,False,None): raise GateArtifactError("passed must be true, false, or null")
    out={"name":str(raw["name"]),"passed":raw.get("passed"),"observed":str(raw.get("observed","")),"detail":str(raw.get("detail",""))}
    for k in ("market","evaluation_line","n","estimate","ci_low","ci_high","threshold","operator"): out[k]=raw.get(k)
    out["promotion"]=bool(raw.get("promotion",False)); return out
def write_gate_artifact(cache_dir:Path,*,league:str,model_version:str,data_cutoff:str|None,gates:Iterable[object],promotion_names:Iterable[str]):
    league=league.lower(); promote=set(REQUIRED_PROMOTION_GATES.get(league,()))|set(promotion_names); rows=[]; seen=set()
    for gate in gates:
        r=_row(gate); r["promotion"]=r["name"] in promote; rows.append(r); seen.add(r["name"])
    for name in sorted(promote-seen): rows.append({"name":name,"passed":None,"observed":"required promotion gate was not produced","detail":"","market":None,"evaluation_line":None,"n":None,"estimate":None,"ci_low":None,"ci_high":None,"threshold":None,"operator":None,"promotion":True})
    blockers=[r["name"] for r in rows if r["promotion"] and r["passed"] is not True]
    payload={"schema_version":SCHEMA_VERSION,"league":league,"model_version":str(model_version),"data_cutoff":data_cutoff,"generated_at":datetime.now(timezone.utc).isoformat(),"producer":os.environ.get("CODEX_MODEL_BUILD","run_backtest.py"),"bets_allowed":not blockers,"blockers":blockers,"gates":rows}
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True); path=cache_dir/FILENAME; tmp=path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); tmp.replace(path); return path
def load_gate_artifact(path:Path,*,expected_league:str|None=None,expected_model_version:str|None=None):
    path=Path(path)
    if not path.exists(): raise GateArtifactError(f"gate artifact missing: {path}")
    try: payload=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError) as exc: raise GateArtifactError("gate artifact unreadable") from exc
    if payload.get("schema_version")!=SCHEMA_VERSION: raise GateArtifactError("unknown gate schema")
    if expected_league and payload.get("league")!=expected_league.lower(): raise GateArtifactError("gate artifact league mismatch")
    if expected_model_version is not None and payload.get("model_version")!=expected_model_version: raise GateArtifactError("gate artifact was produced by a different model/configuration version")
    gates=payload.get("gates")
    if not isinstance(gates,list) or not gates: raise GateArtifactError("gate artifact has no gate records")
    for r in gates: _row(r)
    by_name={r.get("name"):r for r in gates}
    for required in REQUIRED_PROMOTION_GATES.get(payload.get("league"),()):
        r=by_name.get(required)
        if r is None or r.get("promotion") is not True: raise GateArtifactError(f"required promotion gate {required} is absent or not marked promotion")
    blockers=[r.get("name") for r in gates if r.get("promotion") and r.get("passed") is not True]
    if bool(payload.get("bets_allowed"))!=(not blockers): raise GateArtifactError("gate artifact decision disagrees with its gate records")
    return payload
