"""QB adjustment (§12).

The largest single source of NFL model error is a quarterback change the ratings have not
absorbed, so it is handled explicitly rather than left to the ridge to notice three weeks
late.

The primary signal is nfeloqb's `qb1_adj` / `qb2_adj` (see src/nfelo.py) -- already on a
points scale, maintained back through the 538 era, refreshed twice weekly in season. The
in-repo EPA+CPOE composite below is kept as a cross-check rather than the primary, and any
game where the two disagree by more than 3 points is logged: those disagreements are
usually a depth-chart read gone wrong on one side or the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MANUAL_DIR, Config

DISAGREEMENT_THRESHOLD_POINTS = 3.0


@dataclass(frozen=True)
class QBAdjustment:
    points: float          # home perspective, already capped
    source: str            # nfelo | composite | none
    note: str = ""


# --------------------------------------------------------------------------------------
# The in-repo composite (cross-check)
# --------------------------------------------------------------------------------------

def qb_composites(pbp: pd.DataFrame, as_of: pd.Timestamp, cfg: Config) -> pd.DataFrame:
    """EPA-per-dropback and CPOE composite per passer, regressed toward the positional
    mean by `regression_dropbacks`.

    A rookie with 40 great dropbacks is not an elite quarterback, and the regression is
    what stops the model believing he is.
    """
    df = pbp[
        (pbp["qb_dropback"] == 1)
        & pbp["passer_player_id"].notna()
        & pbp["epa"].notna()
    ]
    if "kickoff" in df.columns:
        df = df[df["kickoff"] < as_of]

    grp = df.groupby(["passer_player_id", "passer_player_name"], as_index=False).agg(
        dropbacks=("epa", "size"),
        epa_per_dropback=("epa", "mean"),
        cpoe=("cpoe", "mean"),
    )
    league_epa = float(grp["epa_per_dropback"].mean())
    league_cpoe = float(grp["cpoe"].mean())

    # The composite weights EPA more heavily than CPOE; CPOE is a stabilizer, not the
    # signal.
    raw = 0.85 * (grp["epa_per_dropback"] - league_epa) + 0.15 * (
        (grp["cpoe"] - league_cpoe) / 100.0
    )
    shrink = grp["dropbacks"] / (grp["dropbacks"] + cfg.qb.regression_dropbacks)
    grp["composite"] = raw * shrink
    grp["usable"] = grp["dropbacks"] >= cfg.qb.min_dropbacks
    return grp


def embedded_qb(pbp: pd.DataFrame, team: str, as_of: pd.Timestamp) -> "str | None":
    """The QB with the most dropbacks inside the rating window -- the one the team's
    current offensive rating is really describing."""
    df = pbp[
        (pbp["posteam"] == team)
        & (pbp["qb_dropback"] == 1)
        & pbp["passer_player_id"].notna()
    ]
    if "kickoff" in df.columns:
        df = df[df["kickoff"] < as_of]
    if df.empty:
        return None
    return df["passer_player_id"].value_counts().idxmax()


# --------------------------------------------------------------------------------------
# The primary signal
# --------------------------------------------------------------------------------------

def qb_points_for_game(
    game_id: str, qb_adj_table: "pd.DataFrame | None", cfg: Config
) -> QBAdjustment:
    """Points of spread adjustment from nfeloqb, home perspective, capped.

    The cap exists so a bad depth-chart read cannot wreck a line.
    """
    if not cfg.qb.enabled or qb_adj_table is None or qb_adj_table.empty:
        return QBAdjustment(0.0, "none")
    row = qb_adj_table[qb_adj_table["game_id"] == game_id]
    if row.empty:
        return QBAdjustment(0.0, "none")

    home_adj, away_adj = row["qb_adj_home"].iloc[0], row["qb_adj_away"].iloc[0]
    if pd.isna(home_adj) or pd.isna(away_adj):
        return QBAdjustment(0.0, "none")

    points = float(np.clip(
        float(home_adj) - float(away_adj),
        -cfg.qb.max_adjustment_points,
        cfg.qb.max_adjustment_points,
    ))
    return QBAdjustment(
        points, "nfelo", f"nfeloqb {row['qb_home'].iloc[0]} vs {row['qb_away'].iloc[0]}"
    )


def log_signal_disagreements(
    games: pd.DataFrame,
    nfelo_points: pd.Series,
    composite_points: pd.Series,
    threshold: float = DISAGREEMENT_THRESHOLD_POINTS,
) -> pd.DataFrame:
    """Games where nfeloqb and the in-repo composite disagree by more than `threshold`.

    Worth looking at rather than silently averaging -- one of the two has the wrong
    starter.
    """
    diff = (nfelo_points - composite_points).abs()
    mask = diff > threshold
    return pd.DataFrame({
        "game_id": games.loc[mask, "game_id"] if "game_id" in games else None,
        "nfelo_points": nfelo_points[mask],
        "composite_points": composite_points[mask],
        "abs_diff": diff[mask],
    })


# --------------------------------------------------------------------------------------
# Starter detection (§12.3)
# --------------------------------------------------------------------------------------

def detect_starters(
    depth_charts: pd.DataFrame,
    injuries: "pd.DataFrame | None",
    season: int,
    week: int,
) -> pd.DataFrame:
    """Auto-detect the projected starter, then let the user override.

    Depth-chart QB1, demoted to QB2 when the injury report says Out or Doubtful. Written
    to data/manual/qb_starters.csv as a pre-filled default. Rows whose `source` is
    `manual` are never overwritten -- injury reports are strategically vague,
    "Questionable" resolves both ways, and beat reporters know before the report does.
    """
    dc = depth_charts[
        (depth_charts.get("season") == season) & (depth_charts.get("week") == week)
    ].copy()
    pos_col = next(
        (c for c in ("position", "depth_position", "pos") if c in dc.columns), None
    )
    if pos_col is not None:
        dc = dc[dc[pos_col].astype(str).str.upper().eq("QB")]

    order_col = next(
        (c for c in ("depth_team", "depth_chart_order", "rank") if c in dc.columns), None
    )
    if order_col is not None:
        dc = dc.sort_values(order_col)

    id_col = next((c for c in ("gsis_id", "player_id") if c in dc.columns), None)
    name_col = next(
        (c for c in ("player_name", "full_name", "football_name") if c in dc.columns),
        None,
    )

    out_ids: set = set()
    if injuries is not None and len(injuries):
        inj = injuries[
            (injuries.get("season") == season) & (injuries.get("week") == week)
        ]
        status_col = next(
            (c for c in ("report_status", "game_status") if c in inj.columns), None
        )
        inj_id_col = next((c for c in ("gsis_id", "player_id") if c in inj.columns), None)
        if status_col and inj_id_col:
            out_ids = set(
                inj.loc[inj[status_col].isin(["Out", "Doubtful"]), inj_id_col].dropna()
            )

    rows = []
    for team, grp in dc.groupby("team"):
        available, source = grp, "depth_chart"
        if id_col and out_ids:
            healthy = grp[~grp[id_col].isin(out_ids)]
            if len(healthy) and len(healthy) < len(grp):
                available, source = healthy, "injury_report"
        if available.empty:
            continue
        top = available.iloc[0]
        rows.append({
            "team": team,
            "week": week,
            "starter_id": top[id_col] if id_col else None,
            "starter_name": top[name_col] if name_col else None,
            "source": source,
        })
    return pd.DataFrame(rows)


def write_starters(detected: pd.DataFrame):
    """Merge detected starters into data/manual/qb_starters.csv, preserving manual rows."""
    path = MANUAL_DIR / "qb_starters.csv"
    if path.exists() and len(detected):
        existing = pd.read_csv(path)
        manual = existing[existing["source"] == "manual"]
        keys = set(zip(manual["team"], manual["week"]))
        detected = detected[
            ~detected.apply(lambda r: (r["team"], r["week"]) in keys, axis=1)
        ]
        detected = pd.concat([manual, detected], ignore_index=True)
    detected.to_csv(path, index=False)
    return path
