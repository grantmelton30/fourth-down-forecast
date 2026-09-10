"""nflverse -> tidy pandas tables, with a parquet disk cache.

Everything crosses the polars/pandas boundary exactly once, here. Downstream modules see
pandas only. Every ingest function fails loudly with a specific cause rather than
returning an empty frame (DATA_SOURCES.md §9) -- an empty frame propagates into the ridge
as "this team has no data", which shrinks it to league average and produces a
plausible-looking line built on nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR

# --------------------------------------------------------------------------------------
# Team abbreviation normalization (§4)
# --------------------------------------------------------------------------------------
# nflverse changed abbreviations over time and different tables disagree with each other.
# Applied to every team column in every table at ingest. Deferring this silently corrupts
# joins.
TEAM_ALIASES = {
    "OAK": "LV",     # Raiders -> Las Vegas, 2020
    "SD": "LAC",     # Chargers -> Los Angeles, 2017
    "STL": "LA",     # Rams -> Los Angeles, 2016
    "SL": "LA",
    "LAR": "LA",     # rosters/depth charts spell the Rams LAR; schedules use LA
    "JAC": "JAX",
    "WSH": "WAS",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
}

CANONICAL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

_TEAM_COLUMNS = (
    "team", "posteam", "defteam", "home_team", "away_team", "td_team", "club_code",
    "recent_team", "team_abbr", "penalty_team", "timeout_team",
)

PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "home_team",
    "away_team", "drive", "fixed_drive", "fixed_drive_result", "drive_start_yard_line",
    "play_type", "epa", "wpa", "success", "yards_gained", "down", "ydstogo",
    "yardline_100", "qtr", "game_seconds_remaining", "score_differential",
    "passer_player_id", "passer_player_name", "rusher_player_id", "cpoe", "qb_dropback",
    "penalty", "special", "touchdown", "field_goal_result", "interception",
    "fumble_lost", "sack", "safety", "td_team", "return_touchdown", "extra_point_result",
    "two_point_conv_result", "wp",
]


def normalize_team(s: "pd.Series | str") -> "pd.Series | str":
    """Map any historical/alternate abbreviation onto the canonical current one."""
    if isinstance(s, str):
        return TEAM_ALIASES.get(s, s)
    return s.map(lambda v: TEAM_ALIASES.get(v, v) if isinstance(v, str) else v)


def normalize_all_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in _TEAM_COLUMNS:
        if col in df.columns:
            df[col] = normalize_team(df[col])
    return df


# --------------------------------------------------------------------------------------
# Cache plumbing
# --------------------------------------------------------------------------------------

def _cache_path(name: str, seasons: "list[int] | None" = None) -> Path:
    if seasons:
        tag = f"{min(seasons)}_{max(seasons)}"
        return CACHE_DIR / f"{name}_{tag}.parquet"
    return CACHE_DIR / f"{name}.parquet"


def _cached(name: str, seasons: "list[int] | None", refresh: bool, fetch,
            expected_cols: "list[str] | None" = None):
    """Read parquet if present and not refreshing, else fetch, normalize, and write.

    `expected_cols` guards the CACHED read specifically. `load_pbp`'s own `fetch()`
    already validates a fresh pull against `PBP_COLUMNS`, but that check only runs on a
    cache miss -- an old parquet built under a narrower expected-columns list is served
    from `path.exists()` forever without ever reaching it. This is the exact failure
    config.py's own docstring already names for this repo: "nfl-model's `drives_main.parquet`
    was built before the builder learned to skip kickoff rows." Passing `expected_cols`
    closes that gap for the cached path the same way `read_cached_frame`
    (`ncaa-model/src/config.py`) already does for its own caches.
    """
    path = _cache_path(name, seasons)
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if expected_cols is None or not set(expected_cols) - set(cached.columns):
            return cached
    try:
        df = fetch()
    except Exception as exc:  # noqa: BLE001 - re-raised with the actionable cause
        raise RuntimeError(
            f"ingest.{name}: nflverse fetch failed ({type(exc).__name__}: {exc}). "
            "Check https://github.com/nflverse/nflverse-data for release status, or read "
            "the parquet URLs directly (DATA_SOURCES.md §1 fallback)."
        ) from exc
    if df is None or len(df) == 0:
        raise RuntimeError(
            f"ingest.{name}: source returned zero rows for seasons={seasons}. Refusing to "
            "cache an empty frame -- an empty frame silently becomes 'league average'."
        )
    df.to_parquet(path, index=False)
    return df


def _unpublished_current_release(exc: Exception, season: int) -> bool:
    """True only for nflverse's exact current-season parquet-not-found response."""
    import nflreadpy as nfl

    try:
        if season != int(nfl.get_current_season()):
            return False
    except Exception:  # source metadata failure is not evidence that a dataset is absent
        return False
    messages = []
    current: BaseException | None = exc
    while current is not None:
        messages.append(str(current))
        current = current.__cause__
    joined = " ".join(messages)
    return "404 Client Error" in joined and f"_{season}.parquet" in joined


def _cached_with_current_fallback(name: str, seasons: list[int], refresh: bool,
                                  fetch_for_seasons, expected_cols=None):
    """Retry without an as-yet unpublished current-season dataset.

    nflverse can publish the schedule and advance ``get_current_season()`` before every
    play-level parquet exists. Only that exact current-season 404 is recoverable. Older
    missing seasons, schema errors, empty frames, and all other source failures still stop
    the build rather than silently shrinking the training sample.
    """
    try:
        return _cached(name, seasons, refresh, lambda: fetch_for_seasons(seasons),
                       expected_cols=expected_cols)
    except RuntimeError as exc:
        latest = max(seasons)
        fallback = [season for season in seasons if season != latest]
        if not fallback or not _unpublished_current_release(exc, latest):
            raise
        print(f"ingest.{name}: nflverse has not published {latest}; "
              f"using completed seasons through {max(fallback)}")
        return _cached(name, fallback, refresh, lambda: fetch_for_seasons(fallback),
                       expected_cols=expected_cols)


def _to_pandas(obj) -> pd.DataFrame:
    """nflreadpy returns polars; cross to pandas here and only here."""
    return obj.to_pandas() if hasattr(obj, "to_pandas") else obj


def published_seasons(seasons: list[int]) -> list[int]:
    """Clamp a requested season range to what nflverse has actually published.

    `config.seasons.current` is the season being *played*, and the play-by-play loaders
    reject a season before its first game. During the offseason the config's current
    season legitimately has schedules (the slate is announced) but no plays yet, so the
    play-level loaders ask only for seasons up to nflverse's current one. Schedules are
    deliberately exempt -- next season's slate is exactly what run_week.py needs.
    """
    import nflreadpy as nfl

    latest = nfl.get_current_season()
    keep = [s for s in seasons if s <= latest]
    if not keep:
        raise RuntimeError(
            f"ingest: no requested season {seasons} has published data "
            f"(nflverse current season is {latest})."
        )
    return keep


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------

def load_schedules(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Games, results, and closing betting lines. The backbone table (§1).

    `spread_line` is positive when the home team is favored and is aligned with
    `result = home_score - away_score`. Adds a `kickoff` timestamp used as the `as_of`
    cutoff everywhere downstream.
    """

    def fetch() -> pd.DataFrame:
        import nflreadpy as nfl

        df = _to_pandas(nfl.load_schedules())
        df = df[df["season"].isin(seasons)].copy()
        df = normalize_all_team_columns(df)
        df["kickoff"] = _kickoff_timestamp(df)
        return df.sort_values("kickoff").reset_index(drop=True)

    return _cached("schedules", seasons, refresh, fetch,
                   expected_cols=["season", "home_team", "away_team",
                                  "spread_line", "result", "kickoff"])


def _kickoff_timestamp(df: pd.DataFrame) -> pd.Series:
    """Combine gameday + gametime into a single naive timestamp.

    nflverse publishes `gametime` in US/Eastern. The whole build stays in naive ET rather
    than localizing: every comparison is between two nflverse kickoffs, so a consistent
    frame is all that is required, and tz-aware timestamps in parquet caches invite dtype
    friction for no benefit. Missing times default to 13:00 ET, the modal kickoff.
    """
    gametime = df["gametime"].fillna("13:00")
    return pd.to_datetime(
        df["gameday"].astype(str) + " " + gametime.astype(str),
        format="mixed",
        errors="coerce",
    )


def load_pbp(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Play-by-play, subset to the columns this build uses and tagged `competitive`.

    Full PBP is ~370 columns; subsetting at ingest keeps the cache small and every
    downstream groupby fast.
    """

    seasons = published_seasons(seasons)

    def fetch(requested: list[int]) -> pd.DataFrame:
        import nflreadpy as nfl

        raw = nfl.load_pbp(requested)
        missing = [c for c in PBP_COLUMNS if c not in raw.columns]
        if missing:
            raise RuntimeError(
                f"ingest.load_pbp: nflverse schema changed, missing columns {missing}"
            )
        df = _to_pandas(raw.select(PBP_COLUMNS))
        df = normalize_all_team_columns(df)
        return add_competitive_flag(df)

    return _cached_with_current_fallback(
        "pbp", seasons, refresh, fetch, expected_cols=PBP_COLUMNS)


def add_competitive_flag(pbp: pd.DataFrame) -> pd.DataFrame:
    """Garbage-time filter (§4).

    Unfiltered EPA rewards teams for running up the score on backups and punishes teams
    for prevent defense; both are noise. A null `wp` (a handful of plays per season,
    mostly end-of-half administrivia) passes the win-probability leg rather than being
    dropped outright -- the quarter/score legs still apply.
    """
    wp = pbp["wp"]
    wp_ok = ((wp >= 0.05) & (wp <= 0.95)) | wp.isna()
    sd = pbp["score_differential"].abs()
    blowout_q4 = ((pbp["qtr"] == 4) & (sd > 21)).fillna(False)
    kneel_down = (
        (pbp["qtr"] == 4) & (pbp["game_seconds_remaining"] < 120) & (sd > 8)
    ).fillna(False)
    pbp["competitive"] = wp_ok & ~blowout_q4 & ~kneel_down
    return pbp


def load_rosters(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    seasons = published_seasons(seasons)

    def fetch() -> pd.DataFrame:
        import nflreadpy as nfl

        return normalize_all_team_columns(_to_pandas(nfl.load_rosters_weekly(seasons)))

    return _cached("rosters", seasons, refresh, fetch)


def load_depth_charts(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Weekly depth charts -- the QB module's starter auto-detection (§12)."""
    seasons = published_seasons(seasons)

    def fetch() -> pd.DataFrame:
        import nflreadpy as nfl

        return normalize_all_team_columns(_to_pandas(nfl.load_depth_charts(seasons)))

    # season/week are the two columns qb.py::detect_starters indexes unconditionally
    # (`.get("season")`/`.get("week")`); the rest (position, depth order, player id/name)
    # are looked up through a first-match-wins list of aliases specifically because their
    # exact name has moved before, so guarding on any one of them would be another
    # instance of the same over-specific-guard mistake this audit exists to fix.
    return _cached("depth_charts", seasons, refresh, fetch, expected_cols=["season", "week"])


def load_snap_counts(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Per-player snap shares, 2012+. Used to weight the injury report by how much of a
    team's playing time is actually missing -- see src/injuries.py.

    Snap counts start in 2012 and nflreadpy rejects a season beyond the last completed
    one, so the range is trimmed at both ends; earlier weeks are what the weighting needs
    anyway.
    """
    seasons = [s for s in published_seasons(seasons) if s >= 2012]
    if not seasons:
        return pd.DataFrame()

    def fetch(requested: list[int]) -> pd.DataFrame:
        import nflreadpy as nfl

        return normalize_all_team_columns(_to_pandas(nfl.load_snap_counts(requested)))

    # injuries.py::build_injury_burden indexes all five of these directly (no alias
    # fallback), so a schema drift here silently breaks the injury-burden feature -- the
    # one real, measured NFL injury signal this repo has (NEXT_SESSION.md).
    return _cached_with_current_fallback(
        "snap_counts", seasons, refresh, fetch, expected_cols=[
            "season", "week", "player", "offense_pct", "defense_pct",
        ])


def load_injuries(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Historical report plus a dated manual live file.

    nflverse's injury source is not published beyond 2024. Requesting later seasons makes
    a weekly build fail before it can disclose that live availability is unknown, so the
    automated archive is clamped and any current report comes from the explicit free/
    manual evidence file. Missing manual evidence stays missing; it is never "healthy".
    """
    seasons = [s for s in published_seasons(seasons) if s <= 2024]

    def fetch() -> pd.DataFrame:
        import nflreadpy as nfl

        return normalize_all_team_columns(_to_pandas(nfl.load_injuries(seasons)))
    historical = _cached("injuries", seasons, refresh, fetch) if seasons else pd.DataFrame()
    manual_path = Path(__file__).resolve().parents[1] / "data" / "manual" / "availability.csv"
    if not manual_path.exists():
        return historical
    manual = pd.read_csv(manual_path)
    required = {"season", "week", "team", "full_name", "position", "report_status",
                "observed_at", "source"}
    missing = required - set(manual)
    if missing:
        raise ValueError(f"availability.csv missing columns {sorted(missing)}")
    manual["observed_at"] = pd.to_datetime(manual["observed_at"], utc=True, errors="coerce")
    if manual["observed_at"].isna().any():
        raise ValueError("availability.csv contains invalid observed_at")
    return normalize_all_team_columns(pd.concat([historical, manual], ignore_index=True))


# --------------------------------------------------------------------------------------
# Health check (DATA_SOURCES.md §9)
# --------------------------------------------------------------------------------------

def ingest_health(
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    stream=sys.stdout,
) -> pd.DataFrame:
    """Print rows per table, max date per table, teams with fewer than 3 games, and the
    share of plays dropped as garbage time. Run before every weekly refresh; it catches
    most of the §9 failure table before it reaches the bet sheet."""
    played = schedules[schedules["result"].notna()]
    counts = pd.concat([played["home_team"], played["away_team"]]).value_counts()
    thin = counts[counts < 3]
    dropped = 1.0 - float(pbp["competitive"].mean())
    last_season = pbp["season"].max()
    last_week = pbp.loc[pbp["season"] == last_season, "week"].max()

    rows = [
        ("schedules rows", f"{len(schedules):,}"),
        ("schedules max kickoff", str(schedules["kickoff"].max())),
        ("schedules games with result", f"{len(played):,}"),
        ("pbp rows", f"{len(pbp):,}"),
        ("pbp max season/week", f"{last_season}/{last_week}"),
        ("teams with <3 games", f"{len(thin)}" + (f" {list(thin.index)}" if len(thin) else "")),
        ("plays dropped as garbage time", f"{dropped:.1%}"),
    ]
    print("\n--- ingest_health ---", file=stream)
    for k, v in rows:
        print(f"  {k:<32} {v}", file=stream)
    if dropped > 0.35 or dropped < 0.05:
        print(
            f"  WARNING: garbage-time drop rate of {dropped:.1%} is outside the expected "
            "10-30% band; check the competitive filter and normalize_team() coverage.",
            file=stream,
        )
    return pd.DataFrame(rows, columns=["metric", "value"])


def game_kickoffs(schedules: pd.DataFrame) -> pd.Series:
    """game_id -> kickoff, the lookup every no-lookahead filter is built on."""
    return schedules.set_index("game_id")["kickoff"]
