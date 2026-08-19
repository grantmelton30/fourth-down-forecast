"""Config loading. Every tunable lives in config/ncaa.yaml; nothing hardcoded in modules."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path: sys.path.insert(0,str(SHARED_DIR))
from cache_manifest import build_signature as build_cache_signature, frame_signature
from cache_manifest import read_parquet as read_manifested_frame, write_parquet as write_manifested_frame

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "ncaa.yaml"
CACHE_DIR = Path(
    os.environ.get("NCAA_MODEL_CACHE_DIR", REPO_ROOT / "data" / "cache")
).expanduser()
MANUAL_DIR = REPO_ROOT / "data" / "manual"
OUTPUT_DIR = REPO_ROOT / "output"
BUDGET_PATH = REPO_ROOT / "data" / "api_budget.json"


def read_cached_frame(path: Path, expected_cols: list, signature=None) -> "object | None":
    """Load a cached parquet, but only if it still matches the schema the caller expects.

    WHY THIS EXISTS. A bare `if path.exists(): return pd.read_parquet(path)` is silent
    poison the moment its builder changes, and it has already cost this codebase months:
    nfl-model's `drives_main.parquet` was built before the builder learned to skip kickoff
    rows, so every drive model and backtest after that was fit on a table whose average
    drive started at midfield instead of a team's own 29. The same guard is why
    `ingest.load_drives` re-fetches when `startOffenseScore` is absent.

    A cache whose schema has drifted is treated as absent and rebuilt. This cannot catch a
    builder that changes values without changing columns, so such a change must still bump
    a column name or a cache key -- but it makes the common, already-observed failure
    impossible.
    """
    import pandas as pd

    if signature is not None: return read_manifested_frame(path,expected_cols,signature)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if set(expected_cols) - set(df.columns):
        return None
    return df

def write_cached_frame(frame,path:Path,signature:dict)->Path:
    return write_manifested_frame(frame,path,signature)


class _Section:
    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name for f in fields(cls)}
        missing = known - set(d)
        if missing:
            raise KeyError(f"{cls.__name__}: missing config keys {sorted(missing)}")
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class SeasonsCfg(_Section):
    train_start: int
    backtest_start: int
    backtest_secondary: int
    current: int
    exclude_from_scoring: list


@dataclass(frozen=True)
class ApiCfg(_Section):
    monthly_call_budget: int
    cache_permanent: bool
    # How long a LIVE-season payload may be served from cache before it is refetched.
    # Historical seasons ignore this entirely -- they are immutable and cached forever.
    live_refresh_hours: float


@dataclass(frozen=True)
class TeamsCfg(_Section):
    division: str
    fcs_bucket_name: str
    fcs_off_rating: float
    fcs_def_rating: float
    fcs_game_weight: float
    p5_conferences: list
    p5_teams: list


@dataclass(frozen=True)
class RatingsCfg(_Section):
    half_life_games: float
    lambda_off: float
    lambda_def: float
    lambda_grid: list
    prior_weight_games: float
    league_mean_plays_per_game: float
    offseason_regression: float
    off_weight: float
    def_weight: float
    conference_shrink_weight: float


@dataclass(frozen=True)
class PaceCfg(_Section):
    league_mean_drives_per_team: float
    drives_sd: float
    half_life_games: float
    lambda_: float


@dataclass(frozen=True)
class ContextCfg(_Section):
    hfa_league_mean: float
    hfa_venue_shrink_games: float
    hfa_neutral_site: float
    wind_total_center_mph: float
    wind_total_points_per_mph: float
    dome_total_bump: float


@dataclass(frozen=True)
class QbCfg(_Section):
    """Quarterback adjustment (DECISIONS.md D17). NCAA-fitted; nothing is shared with
    `nfl-model`'s `qb:` block -- the two leagues differ sharply wherever both have been
    measured (wind 0.1867 vs 0.3339 points per mph, dome 1.31 vs 3.16), so a borrowed
    constant would be a bug rather than a shortcut."""

    enabled: bool
    max_adjustment_points: float
    min_attempts: int
    regression_attempts: int
    points_per_starter_out: float


@dataclass(frozen=True)
class SimulationCfg(_Section):
    """Constants the shared drive simulator reads off `cfg.simulation`.

    All measured from CFBD by `analysis/measure_simulation_constants.py`; the yaml carries
    the provenance for each. Deliberately does NOT mirror nfl-model's `drive_count_dist`,
    which that repo declares and never reads.
    """

    n_sims: int
    seed: int
    xp_make_prob: float
    two_point_attempt_rate: float
    two_point_success_prob: float
    defensive_td_lambda: float
    special_teams_td_lambda: float
    safety_lambda: float
    endgame_second_drive_prob: float
    two_point_chart: list
    chart_follow_rate: float
    overtime_max_exchanges: int


@dataclass(frozen=True)
class MarketCfg(_Section):
    devig_method: str
    model_weight_cap: float
    provider_priority: list
    grade_against: str


@dataclass(frozen=True)
class BettingCfg(_Section):
    breakeven_prob_110: float
    min_cover_prob: float
    min_edge_points_spread: float
    min_edge_points_total: float
    max_abs_market_spread: float
    min_total: float
    max_total: float
    exclude_fcs_games: bool
    exclude_weeks_total: list
    max_team_pick_share: float
    max_conference_pick_share: float
    kelly_fraction: float
    max_stake_pct_bankroll: float
    bankroll: float
    default_price: int


@dataclass(frozen=True)
class GatesCfg(_Section):
    opener_coverage_min: float
    opener_separation_min: float
    unbiased_max_abs_a: float
    scale_ratio_tolerance: float
    blend_t_threshold: float
    garbage_drop_min: float
    garbage_drop_max: float
    api_budget_max_cold: int
    # Absolute-accuracy precondition (Phase 0D). The model's RMSE against the actual
    # outcome, divided by the market line's, must not exceed this. 1.0 means "at least as
    # accurate as the number you are betting against".
    rmse_max_ratio: float


@dataclass(frozen=True)
class OutputCfg(_Section):
    excel_path: str


@dataclass(frozen=True)
class Config:
    seasons: SeasonsCfg
    api: ApiCfg
    teams: TeamsCfg
    ratings: RatingsCfg
    pace: PaceCfg
    context: ContextCfg
    qb: QbCfg
    simulation: SimulationCfg
    market: MarketCfg
    betting: BettingCfg
    gates: GatesCfg
    output: OutputCfg
    path: Path

    @property
    def all_seasons(self) -> list:
        """Every season ingested, including pre-opener years used only for history.

        INCLUSIVE of `current`, unlike `graded_seasons` below. Ingesting and grading are
        different questions, and this property used to answer both with one exclusive
        bound -- which made the repo a 2019-2025 historical tool that could not see the
        season being played. The in-progress season must be ingested to project this week's
        games, and must never be graded, because it has no outcomes yet.

        nfl-model has always drawn this distinction (`train_seasons` is `current + 1`,
        `backtest_seasons` is `current`); this brings the two repos into line.
        """
        return list(range(self.seasons.train_start, self.seasons.current + 1))

    @property
    def live_season(self) -> int:
        """The in-progress season -- the one whose caches must NOT be permanent.

        Everything before it is immutable and cached forever, which is what keeps a cold
        build inside the free tier. This season's results change week to week, so a cache
        keyed only on a season range would serve whatever snapshot happened to be taken
        first and never update again. See `ingest._split_live` and `BudgetedCFBD.call`.
        """
        return int(self.seasons.current)

    @property
    def graded_seasons(self) -> list:
        """Primary window: seasons with opener coverage (DECISIONS D2)."""
        return [
            s for s in range(self.seasons.backtest_start, self.seasons.current)
            if s not in self.seasons.exclude_from_scoring
        ]

    @property
    def graded_seasons_secondary(self) -> list:
        """The literal pre-committed fallback window, reported alongside."""
        return [
            s for s in range(self.seasons.backtest_secondary, self.seasons.current)
            if s not in self.seasons.exclude_from_scoring
        ]

    def with_ratings(self, **kw) -> "Config":
        """Copy with overridden ratings hyperparameters, for the pre-registered sweep."""
        return replace(self, ratings=replace(self.ratings, **kw))

    def is_p5(self, conference, team) -> bool:
        return (conference in self.teams.p5_conferences) or (team in self.teams.p5_teams)


def load_config(path: "str | Path" = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    pace_raw = dict(raw["pace"])
    pace_raw["lambda_"] = pace_raw.pop("lambda")

    cfg = Config(
        seasons=SeasonsCfg.from_dict(raw["seasons"]),
        api=ApiCfg.from_dict(raw["api"]),
        teams=TeamsCfg.from_dict(raw["teams"]),
        ratings=RatingsCfg.from_dict(raw["ratings"]),
        pace=PaceCfg.from_dict(pace_raw),
        context=ContextCfg.from_dict(raw["context"]),
        qb=QbCfg.from_dict(raw["qb"]),
        simulation=SimulationCfg.from_dict(raw["simulation"]),
        market=MarketCfg.from_dict(raw["market"]),
        betting=BettingCfg.from_dict(raw["betting"]),
        gates=GatesCfg.from_dict(raw["gates"]),
        output=OutputCfg.from_dict(raw["output"]),
        path=path,
    )
    for d in (CACHE_DIR, MANUAL_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def load_api_key() -> str:
    """CFBD_API_KEY from the environment, else from .env. Never logged."""
    key = os.environ.get("CFBD_API_KEY")
    if key:
        return key.strip()
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("CFBD_API_KEY"):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "CFBD_API_KEY not found. Put it in ncaa-model/.env or the environment. "
        "Free key: https://collegefootballdata.com/key"
    )
