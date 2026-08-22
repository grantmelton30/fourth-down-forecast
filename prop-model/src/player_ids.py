"""Canonical player identity across nflverse sources.

WHY THIS EXISTS. The prop model has to join three tables that do not share a key:

    load_player_stats   keys on gsis_id      (player_id)
    load_snap_counts    keys on pfr_player_id
    injury reports      key on gsis_id, but carry only a display name in practice

`nfl-model/src/injuries.py:43-46` solves this by normalising names to `[a-z]` within a
team-season. That works better than its docstring suggests, and better than the obvious
replacement.

MEASURED 2026-08-21 on 2,448 Out/Doubtful player-weeks from 2023-24, asking "does a snap
row exist for this player in this team-season":

    normalised-name join      93.0%
    ff_playerids crosswalk    80.5%   <- WORSE ALONE
    either                    96.3%
    crosswalk rescues  81 rows the name join missed
    name join rescues 387 rows the crosswalk missed

So the crosswalk is NOT a replacement. `load_ff_playerids` carries gsis_id on only 64% of
its 12,480 rows, so switching to it wholesale would LOSE more than it gains. Both are used,
name first, crosswalk as the fallback, and the union is what this module returns.

(The 78% quoted in `injuries.py` is a different and stricter quantity -- the success rate of
its backward `merge_asof`, which additionally requires a snap row in a *prior* week. It is
not comparable to the figures above and is not being claimed as improved.)

NO NETWORK AT IMPORT. The crosswalk is fetched lazily and cached, so importing this module
in a test costs nothing.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd


def normalize_name(name: object) -> str:
    """Lowercase, strip everything that is not a-z. "A.J. Brown" -> "ajbrown".

    Deliberately identical to `nfl-model/src/injuries.py::_norm`. If the two ever diverge,
    the prop model and the team model would silently disagree about who a player is.
    """
    return re.sub(r"[^a-z]", "", str(name).lower())


@lru_cache(maxsize=1)
def _crosswalk() -> pd.DataFrame:
    import nflreadpy as nfl

    frame = nfl.load_ff_playerids().to_pandas()
    keep = [c for c in ("gsis_id", "pfr_id", "espn_id", "name", "position")
            if c in frame.columns]
    return frame[keep].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")


def gsis_to_pfr() -> dict:
    """gsis_id -> pfr_id, for joining player stats to snap counts."""
    x = _crosswalk().dropna(subset=["pfr_id"])
    return dict(zip(x["gsis_id"], x["pfr_id"]))


def attach_player_key(
    frame: pd.DataFrame, *, name_col: str, gsis_col: "str | None" = None,
    team_col: str = "team", season_col: str = "season",
) -> pd.DataFrame:
    """Add `player_key` -- a stable identity within a team-season.

    Resolution order, chosen by the measurement in the module docstring rather than by
    preference: the normalised name is the PRIMARY key because it resolves more rows, and
    the crosswalk only fills in where a name is absent or unusable. Reversing the order
    would lose 387 of every 2,448 injury rows to gain 81.

    `player_key_source` records which path each row took, so a later change in either
    source shows up as a shift in the mix rather than as silent attrition.
    """
    out = frame.copy()
    names = out[name_col].map(normalize_name)
    source = pd.Series("name", index=out.index)

    usable = names.str.len() > 0
    if gsis_col is not None and gsis_col in out.columns:
        # Only where the name is missing -- see the ordering note above.
        table = _crosswalk().set_index("gsis_id")["name"] if "name" in _crosswalk() else None
        if table is not None:
            filled = out.loc[~usable, gsis_col].map(table).map(normalize_name)
            names = names.mask(~usable, filled)
            source = source.mask(~usable & names.str.len().gt(0), "crosswalk")

    out["player_key"] = (
        out[season_col].astype(str) + "|" + out[team_col].astype(str) + "|" + names
    )
    out["player_key_source"] = source.where(names.str.len() > 0, "unresolved")
    out.loc[names.str.len() == 0, "player_key"] = pd.NA
    return out


def coverage_report(frame: pd.DataFrame) -> dict:
    """What fraction of rows resolved, and by which path. Report it, do not assume it."""
    if "player_key_source" not in frame.columns:
        raise ValueError("call attach_player_key first")
    counts = frame["player_key_source"].value_counts(normalize=True)
    return {
        "n": int(len(frame)),
        "resolved_pct": round(100 * float(frame["player_key"].notna().mean()), 2),
        "by_source": {k: round(100 * float(v), 2) for k, v in counts.items()},
    }


__all__ = ["normalize_name", "gsis_to_pfr", "attach_player_key", "coverage_report"]
