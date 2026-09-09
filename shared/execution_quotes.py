"""Intake for timestamped, book-specific totals quotes supplied by an operator/feed.

The reference providers do not prove execution. Only attach a price when league,
game, kickoff, book and line all match, and the quote was observed recently.
"""
from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

MAX_AGE_MINUTES = 15


def validate_quote(row):
    out = dict(row)
    for key in ("league", "game_id", "book", "source"):
        if not str(out.get(key) or "").strip():
            raise ValueError(f"quote requires {key}")
    if out["league"] not in ("nfl", "ncaa"):
        raise ValueError("unknown league")
    for key in ("kickoff", "observed_at", "ingested_at"):
        value = pd.Timestamp(out[key])
        if pd.isna(value) or value.tzinfo is None:
            raise ValueError("quote timestamps must be known and timezone-aware")
        out[key] = value.tz_convert("UTC").isoformat()
    if not pd.Timestamp(out["observed_at"]) <= pd.Timestamp(out["ingested_at"]) < pd.Timestamp(out["kickoff"]):
        raise ValueError("quote must be observed and ingested before kickoff")
    for key in ("line", "over_price", "under_price"):
        out[key] = float(out[key])
        if not math.isfinite(out[key]):
            raise ValueError("quote numbers must be finite")
    if out["line"] <= 0 or min(abs(out["over_price"]), abs(out["under_price"])) < 100:
        raise ValueError("positive total and valid American prices required")
    out["game_id"] = str(out["game_id"])
    payload = {k:v for k,v in out.items() if k != "quote_id"}
    out["quote_id"] = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return out


def append_quote(path, row):
    row = validate_quote({**row, "ingested_at": datetime.now(timezone.utc).isoformat()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,sort_keys=True)+"\n")
    return row


def matching_quote(path, *, league, game_id, kickoff, book, line, now):
    path=Path(path)
    if not path.exists():
        return None
    now=pd.Timestamp(now)
    found=[]
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row=validate_quote(json.loads(raw))
        if (row["league"],row["game_id"],row["book"],row["line"]) != (league,str(game_id),book,float(line)):
            continue
        if pd.Timestamp(row["kickoff"]) != pd.Timestamp(kickoff):
            continue
        if (now-pd.Timedelta(minutes=MAX_AGE_MINUTES) <= pd.Timestamp(row["observed_at"]) <= now
                and pd.Timestamp(row["ingested_at"]) <= now):
            found.append(row)
    return max(found,key=lambda r:r["observed_at"]) if found else None
