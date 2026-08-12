"""Fail-closed manifests and atomic writes for derived model artifacts."""
from __future__ import annotations
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
MANIFEST_SCHEMA=1

def _jsonable(v):
    if isinstance(v,Path): return str(v)
    if isinstance(v,dict): return {str(k):_jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,set)): return [_jsonable(x) for x in v]
    if hasattr(v,"item"):
        try: return v.item()
        except (TypeError,ValueError): pass
    return v

def digest_payload(payload): return hashlib.sha256(json.dumps(_jsonable(payload),sort_keys=True,default=str).encode()).hexdigest()
def file_digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def frame_signature(frame, columns: Iterable[str]|None=None):
    import pandas as pd
    cols=[c for c in list(columns or frame.columns) if c in frame.columns]
    data=pd.util.hash_pandas_object(frame[cols],index=True).to_numpy().tobytes()
    return {"rows":int(len(frame)),"columns":list(map(str,frame.columns)),"selected_columns":cols,"content_sha256":hashlib.sha256(data).hexdigest()}

def _source_digest(builder: Path):
    model_repo=builder.parents[1] if builder.parent.name=="src" else builder.parent; workspace=model_repo.parent
    files=sorted((model_repo/"src").rglob("*.py"))+sorted((workspace/"shared").glob("*.py")); h=hashlib.sha256()
    for p in files: h.update(str(p.relative_to(workspace)).encode()); h.update(b"\0"); h.update(p.read_bytes()); h.update(b"\0")
    return h.hexdigest()

def build_signature(*,builder:Path,config:Any,inputs:Any,artifact_version:int=1):
    builder=Path(builder)
    return {"artifact_version":int(artifact_version),"builder":str(builder),"builder_sha256":file_digest(builder),"source_tree_sha256":_source_digest(builder),"config_sha256":digest_payload(config),"inputs":_jsonable(inputs)}

def manifest_path(path): return Path(str(path)+".manifest.json")
def read_parquet(path,expected_cols,signature):
    import pandas as pd
    path=Path(path); sidecar=manifest_path(path)
    if not path.exists() or not sidecar.exists(): return None
    try: manifest=json.loads(sidecar.read_text())
    except (OSError,json.JSONDecodeError): return None
    if manifest.get("manifest_schema")!=MANIFEST_SCHEMA or manifest.get("signature_sha256")!=digest_payload(signature): return None
    try: frame=pd.read_parquet(path)
    except Exception: return None
    if set(expected_cols)-set(frame.columns) or manifest.get("rows")!=len(frame): return None
    return frame

def write_parquet(frame,path,signature):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); sidecar=manifest_path(path)
    tmp_data=path.with_name(path.name+f".{os.getpid()}.tmp"); tmp_manifest=sidecar.with_name(sidecar.name+f".{os.getpid()}.tmp")
    frame.to_parquet(tmp_data,index=False)
    manifest={"manifest_schema":MANIFEST_SCHEMA,"signature_sha256":digest_payload(signature),"signature":signature,"rows":int(len(frame)),"columns":list(map(str,frame.columns)),"built_at":datetime.now(timezone.utc).isoformat()}
    tmp_manifest.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); tmp_data.replace(path); tmp_manifest.replace(sidecar); return path
