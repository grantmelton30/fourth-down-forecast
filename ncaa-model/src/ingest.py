"""CFBD -> tidy pandas tables.

Everything here goes through the budgeted client, so the API cost of a cold build is
bounded and visible. Historical seasons are immutable and cached permanently.

Two conventions are established here and never re-litigated downstream:
  * spreads are POSITIVE when the home team is favored (CFBD quotes the opposite);
  * the graded market line is the OPENER, with the close retained only for CLV.
"""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .cfbd_client import BudgetedCFBD
from .config import CACHE_DIR,Config,build_cache_signature,frame_signature,read_cached_frame,write_cached_frame

PLAY_COLUMNS = [
    "gameId", "driveId", "driveNumber", "playNumber", "offense", "defense",
    "offenseConference", "defenseConference", "offenseScore", "defenseScore",
    "home", "away", "period", "down", "distance", "yardsGained", "yardsToGoal",
    "playType", "ppa",
]

# Non-scrimmage play types: excluded before any efficiency number is computed.
SPECIAL_PLAY_TYPES = {
    "Kickoff", "Kickoff Return (Offense)", "Kickoff Return Touchdown", "Punt",
    "Punt Return Touchdown", "Blocked Punt", "Blocked Punt Touchdown",
    "Field Goal Good", "Field Goal Missed", "Blocked Field Goal",
    "Blocked Field Goal Touchdown", "Missed Field Goal Return",
    "Missed Field Goal Return Touchdown", "Extra Point Good", "Extra Point Missed",
    "Two Point Pass", "Two Point Rush", "Defensive 2pt Conversion", "Timeout",
    "End Period", "End of Half", "End of Game", "End of Regulation",
}


def _parquet(name: str):
    return CACHE_DIR / f"{name}.parquet"


# --------------------------------------------------------------------------------------
# Raw loaders
# --------------------------------------------------------------------------------------

def _split_live(client: BudgetedCFBD, seasons: list) -> tuple:
    """Split requested seasons into (immutable history, the one live season or None).

    THE WHOLE POINT. Every parquet below is keyed on a season RANGE, and every raw JSON
    payload is cached forever. That is exactly right for finished seasons and exactly wrong
    for the one being played: a `plays_2019_2026.parquet` written in week 0 would be handed
    back unchanged in week 8, and nothing about the filename would look stale. This repo has
    already lost five days to a cache whose key did not encode what produced it; here the
    input that changes is time.

    So history keeps the existing range-named artifacts -- which means the 2019-2025 caches
    already on disk stay valid and nothing has to be rebuilt -- and the live season is
    fetched separately under a TTL and never written into them.

    The live season is read off `client.cfg` rather than passed in, so that no call site can
    forget it and silently re-freeze the season.
    """
    live = client.cfg.live_season
    hist = [s for s in seasons if s != live]
    return hist, (live if live in seasons else None)


def _live_ttl(client: BudgetedCFBD) -> float:
    return float(client.cfg.api.live_refresh_hours)


GAME_COLUMNS = [
    "id", "season", "week", "startDate", "completed", "neutralSite",
    "conferenceGame", "homeTeam", "homeConference", "homeClassification",
    "homePoints", "awayTeam", "awayConference", "awayClassification", "awayPoints",
    # Venue, for weather. CFBD gives a stable venueId that joins to /venues, which
    # carries latitude, longitude, timezone and a dome flag.
    "venueId", "venue",
]


def _games_frame(client: BudgetedCFBD, seasons: list,
                 ttl: "float | None" = None) -> pd.DataFrame:
    frames = [
        client.frame("games", f"games_{year}", ttl_hours=ttl,
                     year=year, seasonType="regular")
        for year in seasons
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df[[c for c in GAME_COLUMNS if c in df.columns]].rename(
        columns={"id": "game_id"}
    )
    df["kickoff"] = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    return df


def load_games(client: BudgetedCFBD, seasons: list) -> pd.DataFrame:
    """Schedule and results. One call per season."""
    hist, live = _split_live(client, seasons)
    parts = []
    if hist:
        path = _parquet(f"games_{min(hist)}_{max(hist)}")
        cached = None
        if path.exists():
            c = pd.read_parquet(path)
            # Schema guard, same reason as load_drives: `venueId` was added to `keep` after
            # this cache was first written, and without this the old narrow parquet keeps
            # being served and every game silently has no venue -- hence no coordinates,
            # hence no weather. Rebuilding costs nothing; the raw JSON is cached.
            if not {"venueId"} - set(c.columns):
                cached = c
        if cached is None:
            cached = _games_frame(client, hist)
            cached.to_parquet(path, index=False)
        parts.append(cached)
    if live is not None:
        # Never written to the range parquet: it would be stale the moment a game finishes.
        parts.append(_games_frame(client, [live], ttl=_live_ttl(client)))
    return pd.concat(parts, ignore_index=True)


# The score/clock fields feed the shared simulator's endgame layer; without them the drive
# table cannot be built. See src/drives.py.
#
# "startTime" (bug, fixed 2026-08-17): CFBD's raw drive payload has no column literally
# named "startTime" -- client.frame() runs it through pd.json_normalize, which flattens the
# nested {"minutes": M, "seconds": S} clock into "startTime.minutes"/"startTime.seconds".
# Neither ever matched the old "startTime" entry, so `[c for c in DRIVE_KEEP if c in
# df.columns]` silently dropped the clock from EVERY drive ever loaded through this
# function -- drives.py's `_game_seconds_remaining` then saw an all-null column, so
# `start_gsr` was null for 100% of rows, `fit_endgame_table`'s "late game" filter matched
# zero drives, and the entire endgame layer (where the 3/7-point key-number spikes come
# from, per its own docstring) never actually ran. `project_game.py` was unaffected: it
# reads raw drives_<year>.json directly, bypassing this keep-list entirely.
DRIVE_KEEP = [
    "gameId", "season", "offense", "defense", "driveNumber", "driveResult",
    "startPeriod", "startYardsToGoal", "isHomeOffense", "plays", "scoring",
    "startOffenseScore", "startDefenseScore", "endOffenseScore",
    "startTime.minutes", "startTime.seconds",
]


def _drives_frame(client: BudgetedCFBD, seasons: list,
                  ttl: "float | None" = None) -> pd.DataFrame:
    frames = []
    for year in seasons:
        d = client.frame("drives", f"drives_{year}", ttl_hours=ttl,
                         year=year, seasonType="regular")
        d["season"] = year
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df[[c for c in DRIVE_KEEP if c in df.columns]].rename(
        columns={"gameId": "game_id"}
    )


def load_drives(client: BudgetedCFBD, seasons: list) -> pd.DataFrame:
    """Drive-level table. One call per season -- cleaner than deriving it from plays."""
    hist, live = _split_live(client, seasons)
    parts = []
    if hist:
        path = _parquet(f"drives_{min(hist)}_{max(hist)}")
        cached = None
        if path.exists():
            c = pd.read_parquet(path)
            # Schema guard. Without it, extending DRIVE_KEEP silently keeps serving the old
            # narrower cache -- the exact failure that poisoned nfl-model's drive table for
            # the lifetime of that repo. This guard itself checked for "startTime" (which
            # never existed as a column, see DRIVE_KEEP above) rather than the flattened
            # "startTime.minutes"/"startTime.seconds" names, so it always evaluated as
            # missing and silently forced a full rebuild on every call -- which is exactly
            # why the fix above never got a chance to catch a stale on-disk parquet.
            if not {"startOffenseScore", "startTime.minutes", "startTime.seconds"} - set(c.columns):
                cached = c
        if cached is None:
            cached = _drives_frame(client, hist)
            cached.to_parquet(path, index=False)
        parts.append(cached)
    if live is not None:
        parts.append(_drives_frame(client, [live], ttl=_live_ttl(client)))
    return pd.concat(parts, ignore_index=True)


def load_plays(client: BudgetedCFBD, games: pd.DataFrame, seasons: list) -> pd.DataFrame:
    """Play-by-play, paginated by week.

    Weeks come from the schedule rather than being guessed, so no call is spent on a week
    that does not exist. Columns are subset immediately -- the raw payload is ~27 fields
    across roughly two million plays.
    """
    hist, live = _split_live(client, seasons)
    parts = []
    if hist:
        path = _parquet(f"plays_{min(hist)}_{max(hist)}")
        cached = pd.read_parquet(path) if path.exists() else None
        # Down/distance were added to the challenger contract. Old narrow parquets are
        # rebuilt from the permanent per-week JSON cache, not silently served forever.
        if cached is None or {"down", "distance"} - set(cached.columns):
            built = _plays_frame(client, games, hist)
            built.to_parquet(path, index=False)
            parts.append(built)
        else:
            parts.append(cached)
    if live is not None:
        # Only the live season's weeks carry a TTL. Finished weeks inside it still refetch
        # on expiry, which costs a call each; the alternative is deciding when a week is
        # "final", and a wrong guess there silently freezes results.
        parts.append(_plays_frame(client, games, [live], ttl=_live_ttl(client)))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=[*PLAY_COLUMNS, "season", "week"])
    return pd.concat(parts, ignore_index=True)


def _plays_frame(client: BudgetedCFBD, games: pd.DataFrame, seasons: list,
                 ttl: "float | None" = None) -> pd.DataFrame:
    frames = []
    for year in seasons:
        g = games[games["season"] == year]
        # ONLY WEEKS THAT HAVE ACTUALLY BEEN PLAYED. The schedule for the live season lists
        # every future week, and asking for their plays returns an empty list -- while still
        # spending a call. With a 12h TTL that is ~16 empty requests twice a day, which on
        # its own would exhaust the 900/month budget. Historical seasons are unaffected:
        # every one of their weeks has completed games, so this filter removes nothing.
        if "completed" in g.columns:
            g = g[g["completed"].fillna(False).astype(bool)]
        weeks = sorted(g["week"].dropna().unique())
        for wk in weeks:
            wk = int(wk)
            raw = client.call(
                "plays", f"plays_{year}_{wk}", ttl_hours=ttl,
                year=year, week=wk, seasonType="regular",
            )
            if not raw:
                continue
            d = pd.json_normalize(raw)
            d = d[[c for c in PLAY_COLUMNS if c in d.columns]].copy()
            d["season"], d["week"] = year, wk
            frames.append(d)
    if not frames:
        # A live season before its first game has a schedule but no plays. Empty is the
        # correct answer, not a crash.
        return pd.DataFrame(columns=[*PLAY_COLUMNS, "season", "week"])
    df = pd.concat(frames, ignore_index=True).rename(columns={"gameId": "game_id"})
    return add_garbage_flag(df)


def load_lines(client: BudgetedCFBD, cfg: Config, seasons: list) -> pd.DataFrame:
    """Betting lines, one call per season, resolved to a single provider per game.

    Provider selection prefers a book that actually carries BOTH openers (DECISIONS D1);
    `consensus` is excluded entirely because it structurally carries none.
    """
    hist, live = _split_live(client, seasons)
    parts = []
    if hist:
        path = _parquet(f"lines_{min(hist)}_{max(hist)}")
        if path.exists():
            parts.append(pd.read_parquet(path))
        else:
            built = _lines_frame(client, cfg, hist)
            built.to_parquet(path, index=False)
            parts.append(built)
    if live is not None:
        # Lines move all week, so this is the payload that most needs the TTL.
        parts.append(_lines_frame(client, cfg, [live], ttl=_live_ttl(client)))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _lines_frame(client: BudgetedCFBD, cfg: Config, seasons: list,
                 ttl: "float | None" = None) -> pd.DataFrame:
    priority = list(cfg.market.provider_priority)
    rows = []
    for year in seasons:
        for g in client.call("lines", f"lines_{year}", ttl_hours=ttl,
                             year=year, seasonType="regular"):
            chosen = _pick_provider(g.get("lines") or [], priority)
            rows.append({
                "game_id": g["id"], "season": g["season"], "week": g["week"],
                "home_team": g["homeTeam"], "away_team": g["awayTeam"],
                "home_conference": g.get("homeConference"),
                "away_conference": g.get("awayConference"),
                "home_classification": g.get("homeClassification"),
                "away_classification": g.get("awayClassification"),
                "home_points": g.get("homeScore"), "away_points": g.get("awayScore"),
                "provider": chosen.get("provider"),
                "spread_close_raw": chosen.get("spread"),
                "spread_open_raw": chosen.get("spreadOpen"),
                "total_close": chosen.get("overUnder"),
                "total_open": chosen.get("overUnderOpen"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # SIGN CONVENTION. CFBD quotes negative = home favored; we use positive = home
    # favored, matching the NFL build. Normalized once, here, and asserted in tests.
    for src, dst in (
        ("spread_close_raw", "spread_close"), ("spread_open_raw", "spread_open"),
    ):
        df[dst] = -pd.to_numeric(df[src], errors="coerce")
    for c in ("total_close", "total_open"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop(columns=["spread_close_raw", "spread_open_raw"])


def _pick_provider(lines: list, priority: list) -> dict:
    """First provider by priority that carries both openers, then spread-only, then any."""
    by_provider = {str(l.get("provider")): l for l in lines}
    for p in priority:
        l = by_provider.get(p)
        if l and l.get("spreadOpen") is not None and l.get("overUnderOpen") is not None:
            return l
    for p in priority:
        l = by_provider.get(p)
        if l and l.get("spreadOpen") is not None:
            return l
    for p in priority:
        if p in by_provider:
            return by_provider[p]
    return lines[0] if lines else {}


# --------------------------------------------------------------------------------------
# Garbage time (§6a) -- runs before any efficiency number is computed
# --------------------------------------------------------------------------------------

def add_garbage_flag(plays: pd.DataFrame) -> pd.DataFrame:
    """Rules-based filter. CFBD ships no win-probability column, and an explicit rule is
    more transparent than a reconstructed one.

    A 49-3 game contains maybe 40 competitive plays and 100 against backups; unfiltered
    efficiency is dominated by garbage time in a way it simply is not in the NFL.
    """
    margin = (plays["offenseScore"] - plays["defenseScore"]).abs()
    period = plays["period"]
    garbage = (
        ((period == 2) & (margin > 38))
        | ((period == 3) & (margin > 28))
        | ((period >= 4) & (margin > 22))
    )
    plays["garbage"] = garbage.fillna(False)
    plays["competitive"] = ~plays["garbage"]
    return plays


def garbage_share(plays: pd.DataFrame) -> float:
    return float(plays["garbage"].mean())


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

def build_game_offense(
    plays: pd.DataFrame, drives: pd.DataFrame, games: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """One row per (game, offensive unit): efficiency, drive count, matchup context.

    Efficiency is mean PPA per scrimmage play on competitive plays only. Non-FBS opponents
    collapse into a single synthetic team so their games still inform the FBS side.
    """
    path = _parquet("game_offense")
    signature=build_cache_signature(builder=Path(__file__),config=asdict(cfg),inputs={"plays":frame_signature(plays,["game_id","offense","defense","period","ppa","playType","competitive"]),"drives":frame_signature(drives,["game_id","offense","driveNumber"]),"games":frame_signature(games,["game_id","season","week","kickoff","completed","homePoints","awayPoints"])},artifact_version=2)
    cached=read_cached_frame(path,["game_id","offense_norm","defense_norm","ppa_per_play","drives"],signature)
    if cached is not None:return cached

    scrimmage = plays[
        plays["competitive"]
        & plays["ppa"].notna()
        & ~plays["playType"].isin(SPECIAL_PLAY_TYPES)
    ]
    eff = scrimmage.groupby(["game_id", "offense"], as_index=False).agg(
        ppa_per_play=("ppa", "mean"), n_plays=("ppa", "size"),
    )
    dr = drives.groupby(["game_id", "offense"], as_index=False).agg(
        drives=("driveNumber", "nunique")
    )
    agg = eff.merge(dr, on=["game_id", "offense"], how="left")

    g = games[[
        "game_id", "season", "week", "kickoff", "neutralSite", "completed",
        "homeTeam", "awayTeam", "homeConference", "awayConference",
        "homeClassification", "awayClassification", "homePoints", "awayPoints",
    ]]
    df = agg.merge(g, on="game_id", how="inner")

    fcs = cfg.teams.fcs_bucket_name
    df["home_norm"] = np.where(df["homeClassification"] == "fbs", df["homeTeam"], fcs)
    df["away_norm"] = np.where(df["awayClassification"] == "fbs", df["awayTeam"], fcs)
    is_home_off = df["offense"] == df["homeTeam"]
    df["offense_norm"] = np.where(is_home_off, df["home_norm"], df["away_norm"])
    df["defense_norm"] = np.where(is_home_off, df["away_norm"], df["home_norm"])
    df["is_home_offense"] = np.where(
        df["neutralSite"].fillna(False), 0.5, is_home_off.astype(float)
    )

    # FCS games carry real information about the FBS side, but less of it.
    df["game_weight"] = np.where(
        (df["offense_norm"] == fcs) | (df["defense_norm"] == fcs),
        cfg.teams.fcs_game_weight, 1.0,
    )
    df = df[df["completed"].fillna(False)]
    df = df.sort_values("kickoff").reset_index(drop=True)
    write_cached_frame(df,path,signature)
    return df


def build_market(lines: pd.DataFrame, games: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per game: the opener we grade against, the close we measure CLV against,
    and the restricted-universe flag."""
    g = games[["game_id", "kickoff", "neutralSite", "completed"]]
    df = lines.merge(g, on="game_id", how="left")

    df["home_p5"] = [
        cfg.is_p5(c, t) for c, t in zip(df["home_conference"], df["home_team"])
    ]
    df["away_p5"] = [
        cfg.is_p5(c, t) for c, t in zip(df["away_conference"], df["away_team"])
    ]
    df["fbs_only"] = (
        (df["home_classification"] == "fbs") & (df["away_classification"] == "fbs")
    )
    # Restricted universe: FBS vs FBS, excluding P5-vs-P5. G5 games plus cross-tier.
    df["restricted"] = df["fbs_only"] & ~(df["home_p5"] & df["away_p5"])
    df["cross_tier"] = df["home_p5"] != df["away_p5"]

    df["actual_margin"] = df["home_points"] - df["away_points"]
    df["actual_total"] = df["home_points"] + df["away_points"]
    df["has_opener"] = df["spread_open"].notna() & df["total_open"].notna()
    return df
