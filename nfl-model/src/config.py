"""Configuration loading. Every tunable lives in config/nfl.yaml; nothing is hardcoded
in the modules. Loaded into frozen dataclasses so a typo raises instead of silently
returning None."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path: sys.path.insert(0,str(SHARED_DIR))
from cache_manifest import build_signature as build_cache_signature, frame_signature
from cache_manifest import read_parquet as read_manifested_frame, write_parquet as write_manifested_frame

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "nfl.yaml"

# Defaults to the spec's data/cache/. The override exists because this checkout lives on
# an iCloud-synced Desktop, and a hot parquet cache inside a sync root is actively
# harmful: reads stall for minutes behind the sync daemon, which blows GATE_SPEED. Point
# NFL_MODEL_CACHE_DIR at any local path to opt out. Cache contents are disposable and
# gitignored either way.
CACHE_DIR = Path(
    os.environ.get("NFL_MODEL_CACHE_DIR", REPO_ROOT / "data" / "cache")
).expanduser()
MANUAL_DIR = REPO_ROOT / "data" / "manual"
OUTPUT_DIR = REPO_ROOT / "output"


def read_cached_frame(path: Path, expected_cols: "list[str]", signature=None) -> "Any | None":
    """Load a cached parquet, but only if it still matches the schema the caller expects.

    WHY THIS EXISTS. Every derived cache in this repo was previously read back with a bare
    `if path.exists(): return pd.read_parquet(path)`. That is silent poison the moment a
    builder changes, and it already happened: `drives_main.parquet` was built before
    build_drive_table learned to skip kickoff rows, so it recorded the average drive as
    starting at midfield (yardline_100 53) instead of a team's own 29 (yardline_100 71).
    Every drive model, every field-position distribution, and every backtest since was fit
    on that table. It also predated the endgame layer, so it lacked start_qtr / start_gsr /
    start_score_diff and crashed nine simulator tests outright.

    A cache whose schema has drifted is treated as absent and rebuilt. This cannot catch a
    builder change that alters values without altering columns, so a builder that changes
    semantics must still bump a column name or clear its cache -- but it makes the common,
    already-observed failure impossible.
    """
    import pandas as pd

    if signature is not None: return read_manifested_frame(path,expected_cols,signature)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if set(expected_cols) - set(df.columns):
        return None
    return df

def write_cached_frame(frame:Any,path:Path,signature:dict)->Path:
    return write_manifested_frame(frame,path,signature)


class _Section:
    """Mixin: build a frozen dataclass from a yaml mapping, ignoring unknown keys but
    failing loudly on missing ones."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        known = {f.name for f in fields(cls)}
        missing = known - set(d)
        if missing:
            raise KeyError(f"{cls.__name__}: missing config keys {sorted(missing)}")
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class SeasonsCfg(_Section):
    train_start: int
    backtest_start: int
    current: int


@dataclass(frozen=True)
class RatingsCfg(_Section):
    half_life_games: float
    lambda_off: float
    lambda_def: float
    prior_weight_games: float
    league_mean_plays_per_game: float
    offseason_regression: float
    off_weight: float
    def_weight: float
    lambda_grid: list[float]


@dataclass(frozen=True)
class PaceCfg(_Section):
    league_mean_drives_per_team: float
    drives_sd: float
    half_life_games: float
    # `lambda` is a Python keyword; the yaml key is remapped in load_config.
    lambda_: float


@dataclass(frozen=True)
class QBCfg(_Section):
    enabled: bool
    max_adjustment_points: float
    min_dropbacks: int
    regression_dropbacks: int


@dataclass(frozen=True)
class ContextCfg(_Section):
    hfa_league_mean: float
    hfa_venue_shrink_games: float
    hfa_neutral_site: float
    rest_point_per_day: float
    rest_max_points: float
    short_week_penalty: float
    travel_points_per_1000mi: float
    timezone_cross_penalty: float
    wind_total_center_mph: float
    wind_total_points_per_mph: float
    wind_max_plausible_mph: float
    dome_total_bump: float
    week18_rest_starters_points: float
    injury_points_per_burden: float
    injury_max_points: float


@dataclass(frozen=True)
class SimulationCfg(_Section):
    n_sims: int
    seed: int
    drive_count_dist: str
    xp_make_prob: float
    two_point_attempt_rate: float
    two_point_success_prob: float
    defensive_td_lambda: float
    special_teams_td_lambda: float
    safety_lambda: float
    endgame_second_drive_prob: float
    two_point_chart: list[int]
    chart_follow_rate: float
    overtime_max_exchanges: int


@dataclass(frozen=True)
class MarketCfg(_Section):
    devig_method: str
    model_weight_cap: float


@dataclass(frozen=True)
class BettingCfg(_Section):
    breakeven_prob_110: float
    min_cover_prob: float
    min_edge_points_spread: float
    min_edge_points_total: float
    kelly_fraction: float
    max_stake_pct_bankroll: float
    bankroll: float
    default_price: int


@dataclass(frozen=True)
class OutputCfg(_Section):
    excel_path: str


@dataclass(frozen=True)
class ProjectionCfg(_Section):
    """Which estimator produces `E[margin]`. See config/nfl.yaml for the measurement.

    `simulator` is what this repo shipped from the start. `linear` is the L1/L2 split that
    PATCH_03 §5.1 prescribed, NCAA adopted, and NFL never implemented -- the simulator still
    produces the distribution, but a walk-forward least-squares fit on (net_diff,
    context_points) produces the mean and the simulated margins are recentred onto it.

    Deliberately NOT applied to totals: measured on this repo's own data the simulator BEATS
    the linear projection on totals (13.445 vs 13.630) while losing on spreads (13.668 vs
    13.196), both significant. One knob, because only one market moved.
    """

    spread_source: str
    min_train_games: int

    def __post_init__(self):
        if self.spread_source not in ("linear", "simulator"):
            raise ValueError(
                f"projection.spread_source must be 'linear' or 'simulator', "
                f"got {self.spread_source!r}"
            )


@dataclass(frozen=True)
class Config:
    seasons: SeasonsCfg
    ratings: RatingsCfg
    pace: PaceCfg
    qb: QBCfg
    context: ContextCfg
    simulation: SimulationCfg
    market: MarketCfg
    betting: BettingCfg
    output: OutputCfg
    projection: ProjectionCfg
    tuned: dict[str, Any]
    path: Path

    @property
    def backtest_seasons(self) -> list[int]:
        """Seasons graded in the walk-forward backtest: backtest_start..current-1."""
        return list(range(self.seasons.backtest_start, self.seasons.current))

    @property
    def train_seasons(self) -> list[int]:
        """Every season that must be ingested: train_start..current."""
        return list(range(self.seasons.train_start, self.seasons.current + 1))

    def with_sims(self, n_sims: int) -> "Config":
        """Return a copy with a different simulation count (the backtest runs fewer sims
        per game than the weekly slate does)."""
        return replace(self, simulation=replace(self.simulation, n_sims=n_sims))

    def with_ratings(self, **overrides) -> "Config":
        """A copy with named `ratings:` fields replaced, for sweeping memory parameters.

        `prior_weight_games`, `offseason_regression` and `half_life_games` have never been
        swept on this build -- they carry no provenance comment, unlike the lambdas. The
        NCAA build swept its equivalents and moved 0.42/5.0 -> 1.00/8.0 for 0.155 of RMSE,
        with the largest gain in the early-season weeks where priors dominate. This model
        shows the same early-season weakness (weeks 4-6 run a 1.079 ratio against the
        market versus ~1.055 later), which is what makes the knobs worth testing.
        """
        return replace(self, ratings=replace(self.ratings, **overrides))

    def with_lambdas(self, lambda_off: float, lambda_def: float) -> "Config":
        """A copy pinned to specific ridge penalties, for a grid search that scores the
        SHIPPED projection (`ratings.tune_lambdas_shipped`).

        Overrides the `tuned:` block rather than `ratings:`, because `effective_lambdas`
        prefers `tuned:` -- setting only the seed values would be silently ignored whenever
        a tuned block exists, which it does.
        """
        merged = dict(self.tuned or {})
        merged["lambda_off"] = float(lambda_off)
        merged["lambda_def"] = float(lambda_def)
        return replace(self, tuned=merged)

    @property
    def effective_lambdas(self) -> tuple[float, float]:
        """Tuned lambdas from §5c if tune_lambdas() has been run, else the seed values."""
        t = self.tuned or {}
        return (
            float(t.get("lambda_off", self.ratings.lambda_off)),
            float(t.get("lambda_def", self.ratings.lambda_def)),
        )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    pace_raw = dict(raw["pace"])
    pace_raw["lambda_"] = pace_raw.pop("lambda")

    cfg = Config(
        seasons=SeasonsCfg.from_dict(raw["seasons"]),
        ratings=RatingsCfg.from_dict(raw["ratings"]),
        pace=PaceCfg.from_dict(pace_raw),
        qb=QBCfg.from_dict(raw["qb"]),
        context=ContextCfg.from_dict(raw["context"]),
        simulation=SimulationCfg.from_dict(raw["simulation"]),
        market=MarketCfg.from_dict(raw["market"]),
        betting=BettingCfg.from_dict(raw["betting"]),
        output=OutputCfg.from_dict(raw["output"]),
        projection=ProjectionCfg.from_dict(raw["projection"]),
        tuned=raw.get("tuned") or {},
        path=path,
    )
    for d in (CACHE_DIR, MANUAL_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def write_tuned_block(cfg: Config, tuned: dict[str, Any]) -> None:
    """Persist tuned lambdas back into config/nfl.yaml under a `tuned:` block (§5c)."""
    with open(cfg.path) as fh:
        raw = yaml.safe_load(fh)
    raw["tuned"] = tuned
    with open(cfg.path, "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False, default_flow_style=False)
