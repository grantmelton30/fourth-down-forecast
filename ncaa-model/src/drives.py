"""CFBD drives payload -> the drive table the shared simulator expects.

This is the NCAA half of the port. The simulator itself lives in `../shared/sim_core.py`
and is the same one nfl-model uses; all that differs is how each sport produces a drive
table. Emit these columns and the Monte Carlo works unchanged.

CFBD already gives drive-level rows, so unlike the NFL side there is no play-by-play to
aggregate and no kickoff-row trap: `startYardsToGoal` is exactly the NFL's
`start_yardline_100` (yards to the opponent's end zone), already numeric and already
measured at the drive's first scrimmage snap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from ._shared import sim_core
from .config import build_cache_signature,frame_signature,read_cached_frame,write_cached_frame

TD, FG, NO_SCORE, TURNOVER = "TD", "FG", "NO_SCORE", "TURNOVER"
DRIVE_CLASSES = sim_core.DRIVE_CLASSES
DRIVE_CLASS_INDEX = sim_core.DRIVE_CLASS_INDEX
StartFieldPosition = sim_core.StartFieldPosition
fit_start_field_position = sim_core.fit_start_field_position

# Schema, in one place so the builder and the cache guard cannot drift apart. Adding a
# column here is what invalidates an old cache on disk.
DRIVE_COLUMNS = [
    "game_id", "season", "week", "offense", "defense", "drive_number",
    "start_yardline_100", "n_plays", "result", "points_scored_on_drive",
    "is_home_offense", "prev_drive_turnover", "start_score_diff", "start_qtr",
    "start_gsr",
]

# CFBD `driveResult` -> our four classes, from the OFFENSE's perspective.
#
# The two families worth stating explicitly, because both are easy to get backwards:
#   * A defensive return touchdown ("INT TD", "FUMBLE RETURN TD", "DOWNS TD") is a
#     TURNOVER for this offense. The seven points belong to the opponent and are generated
#     by `defensive_td_lambda` in the simulator, exactly as the NFL side handles
#     "Opp touchdown".
#   * A kicking-play return touchdown ("PUNT TD", "PUNT RETURN TD", "MISSED FG TD",
#     "FG TD") is NO_SCORE for this offense -- the drive simply failed to score. Those
#     points come from `special_teams_td_lambda`.
# A safety is NO_SCORE here and is produced by `safety_lambda`, again matching the NFL map.
#
# The map was originally validated against 2023 alone. Extending it to 2019-2025 turned up
# five values it did not cover and two it covered wrongly; both were settled by reading the
# score actually recorded on the drive (`endOffenseScore - startOffenseScore` against the
# same for the defense) rather than by reading the label. See
# `analysis/measure_simulation_constants.py`.
_RESULT_MAP = {
    "TD": TD,
    # 2021 only, 17 drives between them: that season's payload names a few touchdown
    # drives by how they were scored. Offense +6/+7/+8, defense +0 -- ordinary TD drives.
    "RUSHING TD": TD,
    "PASSING TD": TD,
    # NOT touchdowns for this offense. On all 16 of these the offense scored nothing and
    # the DEFENSE scored 6 or 7 -- they are a defensive return TD on the game's or half's
    # last snap. Mapping them to TD credited the wrong team; they belong with the other
    # turnover-return results, whose points come from `defensive_td_lambda`.
    "END OF GAME TD": TURNOVER,
    "END OF HALF TD": TURNOVER,
    "FG": FG,
    # 2021-only spellings of FG / MISSED FG. Offense +3 and +0 respectively.
    "FG GOOD": FG,
    "FG MISSED": NO_SCORE,
    "PUNT": NO_SCORE,
    "DOWNS": NO_SCORE,
    "MISSED FG": NO_SCORE,
    "BLOCKED FG": NO_SCORE,
    "BLOCKED PUNT": NO_SCORE,
    "END OF HALF": NO_SCORE,
    "END OF GAME": NO_SCORE,
    "END OF 4TH QUARTER": NO_SCORE,
    "SF": NO_SCORE,
    "PUNT TD": NO_SCORE,
    "PUNT RETURN TD": NO_SCORE,
    "MISSED FG TD": NO_SCORE,
    "FG TD": NO_SCORE,
    "INT": TURNOVER,
    "FUMBLE": TURNOVER,
    "INT TD": TURNOVER,
    # 2021, one drive. "TOUCH" is a truncated TOUCHDOWN, not a touchback: the offense
    # scored 0 and the defense scored 7. Same treatment as "INT TD".
    "INT RETURN TOUCH": TURNOVER,
    "FUMBLE TD": TURNOVER,
    "FUMBLE RETURN TD": TURNOVER,
    "DOWNS TD": TURNOVER,
}

# Rows that are not a scrimmage drive at all. CFBD emits a few hundred of these a season;
# they carry no usable field position and the simulator models none of them.
_DROP_RESULTS = {"Uncategorized", "KICKOFF", "POSSESSION (FOR OT DRIVES)"}

_PERIOD_SECONDS = 900.0


def _game_seconds_remaining(df: pd.DataFrame) -> np.ndarray:
    """Seconds left in regulation at the drive's first snap.

    CFBD gives a clock within the period as {"minutes": M, "seconds": S}. That arrives in
    one of two shapes depending on how the payload was loaded: a column of dicts from raw
    JSON, or flattened `startTime.minutes` / `startTime.seconds` columns once it has been
    through `pd.json_normalize` -- which is what `ingest.load_drives` does. Handle both;
    reading only the dict shape silently yields an all-null column and disables the
    endgame layer, which is where the key-number spikes come from.

    The simulator wants seconds remaining in the whole game, matching nflverse's
    `game_seconds_remaining`. Overtime periods (>4) clamp to zero.
    """
    period = df["startPeriod"].astype(float)
    if "startTime.minutes" in df.columns:
        within = (
            df["startTime.minutes"].astype(float).fillna(0) * 60.0
            + df["startTime.seconds"].astype(float).fillna(0)
        ).to_numpy(dtype=float)
    elif "startTime" in df.columns:
        def _secs(v):
            if isinstance(v, dict):
                return float(v.get("minutes") or 0) * 60.0 + float(v.get("seconds") or 0)
            return np.nan
        within = df["startTime"].map(_secs).to_numpy(dtype=float)
    else:
        return np.full(len(df), np.nan)
    periods_left = np.clip(4 - period.to_numpy(dtype=float), 0, None)
    return periods_left * _PERIOD_SECONDS + within


def build_drive_table(
    drives: pd.DataFrame, games: pd.DataFrame, cache_path=None, refresh: bool = False
) -> pd.DataFrame:
    """One row per drive, in the shared simulator's schema.

    `games` supplies the week number, which the CFBD drives endpoint does not return.
    """
    signature=build_cache_signature(builder=Path(__file__),config={"drive_columns":DRIVE_COLUMNS},inputs={"drives":frame_signature(drives,["game_id","driveNumber","driveResult","startYardsToGoal","startOffenseScore","startDefenseScore","endOffenseScore","startPeriod","startTime","startTime.minutes","startTime.seconds","isHomeOffense","plays","season","offense","defense"]),"games":frame_signature(games,["game_id","season","week"])},artifact_version=2)
    if cache_path is not None and not refresh:
        cached=read_cached_frame(cache_path,DRIVE_COLUMNS,signature)
        if cached is not None: return cached

    df = drives.copy()
    df = df[~df["driveResult"].isin(_DROP_RESULTS)]

    unmapped = sorted(set(df["driveResult"].dropna()) - set(_RESULT_MAP))
    if unmapped:
        raise ValueError(
            f"drives.build_drive_table: unmapped CFBD driveResult values {unmapped}. "
            "Add them to _RESULT_MAP rather than letting them fall through to NO_SCORE."
        )
    df["result"] = df["driveResult"].map(_RESULT_MAP)

    df["start_yardline_100"] = df["startYardsToGoal"].astype(float).clip(1, 99)
    df["start_qtr"] = df["startPeriod"].astype(float)
    df["start_gsr"] = _game_seconds_remaining(df)
    df["start_score_diff"] = (
        df["startOffenseScore"].astype(float) - df["startDefenseScore"].astype(float)
    )
    df["is_home_offense"] = df["isHomeOffense"].astype(bool).astype(int)
    df["n_plays"] = (
        df["plays"].astype(float) if "plays" in df
        else pd.Series(np.nan, index=df.index)
    )
    df["points_scored_on_drive"] = (
        df["endOffenseScore"].astype(float) - df["startOffenseScore"].astype(float)
    ).clip(lower=0)
    df = df.rename(columns={"driveNumber": "drive_number"})

    wk = games[["game_id", "week"]].drop_duplicates()
    df = df.merge(wk, on="game_id", how="left")

    df = df.sort_values(["game_id", "drive_number"])
    prev = df.groupby("game_id")["result"].shift(1)
    df["prev_drive_turnover"] = (prev == TURNOVER).fillna(False)

    df = df[df["start_yardline_100"].notna()].copy()
    out = df[DRIVE_COLUMNS].reset_index(drop=True)
    if cache_path is not None:
        write_cached_frame(out,cache_path,signature)
    return out
