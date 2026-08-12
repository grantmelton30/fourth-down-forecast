"""Deterministic identity for model code, configuration, and backtest evidence."""
from __future__ import annotations
import hashlib
from pathlib import Path

def evidence_digest(frame) -> str:
    import pandas as pd
    preferred = ["game_id","season","week","kickoff","gameday","game_date","date",
        "model_spread","model_total","market_spread","market_total","spread_open",
        "spread_close","total_open","total_close","actual_margin","actual_total"]
    columns = [c for c in preferred if c in frame.columns]
    if not columns:
        raise ValueError("backtest evidence has no identity or prediction columns")
    data = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().tobytes()
    return hashlib.sha256(data).hexdigest()

def build_model_version(league: str, repo: Path, config_path: Path, frame) -> str:
    repo = Path(repo); shared = repo.parent / "shared"
    paths = [Path(config_path), repo / "run_backtest.py"]
    paths += sorted((repo / "src").rglob("*.py")) + sorted(shared.glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repo.parent)).encode()); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    digest.update(evidence_digest(frame).encode())
    return f"{league.lower()}-{digest.hexdigest()[:20]}"

__all__ = ["build_model_version", "evidence_digest"]
