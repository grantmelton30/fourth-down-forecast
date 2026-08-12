"""Prospective NFL availability features from free feeds and explicit overrides.

Absence of a report is unknown, never healthy.  The nflverse injury archive is useful
historically but is not a dependable live 2025+ feed, so a dated manual CSV is a first-
class source.  Every row must carry ``observed_at`` so it can be joined as-of kickoff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


UNAVAILABLE = {"out", "doubtful", "inactive", "injured reserve", "ir"}
QUESTIONABLE = {"questionable", "limited"}
POSITION_GROUP = {
    "C": "offensive_line", "G": "offensive_line", "OG": "offensive_line",
    "T": "offensive_line", "OT": "offensive_line", "OL": "offensive_line",
    "WR": "skill", "RB": "skill", "FB": "skill", "TE": "skill",
    "QB": "quarterback",
}
OUTPUT_COLUMNS = [
    "season", "week", "team", "offensive_line_burden", "skill_burden",
    "quarterback_burden", "defense_burden", "availability_confirmed",
    "availability_source", "availability_observed_at",
]


def read_manual_availability(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"season", "week", "team", "full_name", "position", "report_status",
                "observed_at", "source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manual availability missing columns: {sorted(missing)}")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    if frame["observed_at"].isna().any():
        raise ValueError("manual availability contains an invalid observed_at")
    return frame


def build_availability_features(
    reports: pd.DataFrame, snap_counts: pd.DataFrame, *, questionable_weight: float = 0.25,
) -> pd.DataFrame:
    """Return position-weighted team-week burdens using only earlier snap shares.

    One unit is one full-time player. Questionable/limited players contribute a small,
    configurable expected burden and are never treated as confirmed absences.
    """
    if reports is None or reports.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    required = {"season", "week", "team", "full_name", "position", "report_status"}
    missing = required - set(reports.columns)
    if missing:
        raise ValueError(f"availability reports missing columns: {sorted(missing)}")
    if snap_counts is None or snap_counts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    from .injuries import _norm

    rep, snaps = reports.copy(), snap_counts.copy()
    for frame in (rep, snaps):
        frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
        frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    rep["k"] = rep["full_name"].map(_norm)
    snaps["k"] = snaps["player"].map(_norm)
    snaps["play_pct"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    snaps = snaps.sort_values(["season", "team", "k", "week"])
    snaps["prior_pct"] = snaps.groupby(["season", "team", "k"])["play_pct"].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    prior = snaps[["season", "team", "k", "week", "prior_pct"]].dropna().sort_values(
        ["week"]
    )
    rep = rep.sort_values("week")
    joined = pd.merge_asof(rep, prior, on="week", by=["season", "team", "k"],
                           direction="backward")
    status = joined["report_status"].astype(str).str.strip().str.lower()
    joined["status_weight"] = np.select(
        [status.isin(UNAVAILABLE), status.isin(QUESTIONABLE)],
        [1.0, float(questionable_weight)], default=0.0,
    )
    joined["burden"] = joined["prior_pct"].fillna(0.0) * joined["status_weight"]
    joined["group"] = joined["position"].astype(str).str.upper().map(POSITION_GROUP).fillna(
        "defense"
    )
    wide = joined.pivot_table(index=["season", "week", "team"], columns="group",
                              values="burden", aggfunc="sum", fill_value=0.0)
    wide = wide.rename(columns={
        "offensive_line": "offensive_line_burden", "skill": "skill_burden",
        "quarterback": "quarterback_burden", "defense": "defense_burden",
    }).reset_index()
    for column in OUTPUT_COLUMNS[3:7]:
        if column not in wide:
            wide[column] = 0.0

    meta = rep.groupby(["season", "week", "team"], as_index=False).agg(
        availability_confirmed=("report_status", lambda values: bool(
            values.astype(str).str.lower().isin(UNAVAILABLE).any()
        )),
        availability_source=("source", lambda values: ", ".join(sorted(set(map(str, values)))))
        if "source" in rep else ("report_status", lambda _: "unknown"),
        availability_observed_at=("observed_at", "max")
        if "observed_at" in rep else ("week", lambda _: pd.NaT),
    )
    return wide.merge(meta, on=["season", "week", "team"], how="left")[OUTPUT_COLUMNS]


def matchup_availability(features: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away position burdens for held-out challenger fitting."""
    base = schedules[["game_id", "season", "week", "home_team", "away_team"]].copy()
    if features is None or features.empty:
        return base[["game_id"]]
    burdens = [c for c in OUTPUT_COLUMNS if c.endswith("_burden")]
    h = features.rename(columns={"team": "home_team",
        **{c: f"home_{c}" for c in burdens}})
    a = features.rename(columns={"team": "away_team",
        **{c: f"away_{c}" for c in burdens}})
    out = base.merge(h[["season", "week", "home_team", *[f"home_{c}" for c in burdens]]],
                     on=["season", "week", "home_team"], how="left")
    out = out.merge(a[["season", "week", "away_team", *[f"away_{c}" for c in burdens]]],
                    on=["season", "week", "away_team"], how="left")
    for burden in burdens:
        out[f"{burden}_diff"] = out[f"home_{burden}"] - out[f"away_{burden}"]
    return out[["game_id", *[f"{c}_diff" for c in burdens]]]


__all__ = ["OUTPUT_COLUMNS", "build_availability_features", "matchup_availability",
           "read_manual_availability"]
