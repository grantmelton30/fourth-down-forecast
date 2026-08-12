"""Walk-forward backtest and the six acceptance gates (§8).

This is the heart of the acceptance process. Weeks 1-3 are excluded from scoring because
the prior dominates and the sample is not representative; the model still *produces*
numbers for them in live use, it just is not graded on them.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import ingest
from ._shared import robust_inference, validated_model
from .config import (
    CACHE_DIR,
    Config,
    build_cache_signature,
    frame_signature,
    read_cached_frame,
    write_cached_frame,
)
from .context import build_context, estimate_venue_hfa, load_resting_starters
from .drive_model import fit_drive_model, fit_endgame_table
from .drives import build_drive_table, fit_start_field_position
from .elo import run_elo
from .features import FEATURE_COLUMNS, matchup_feature_table
from .market import (
    BlendWeights,
    _residual_fit,
    apply_blend,
    apply_blend_total,
    fit_blend_weights,
)
from .qb import qb_points_for_game
from .ratings import (
    build_game_offense_table,
    build_walkforward_ratings,
    net_epa,
    ratings_at,
)
from .simulate import simulate_game

FIRST_GRADED_WEEK = 4


@dataclass
class GateResult:
    name: str
    passed: bool
    observed: str
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"  [{status}] {self.name:<24} {self.observed}"


# --------------------------------------------------------------------------------------
# The walk
# --------------------------------------------------------------------------------------

def backtest_frame_path(cfg: Config) -> Path:
    """Where the walk-forward backtest frame for THIS configuration lives.

    THE CONFIG IS IN THE FILENAME ON PURPOSE. This used to be a bare
    `backtest_frame.parquet`, returned whenever it existed. Every constant that shapes the
    frame -- ratings penalties, pace, the drive-simulation constants, every context
    adjustment -- could change underneath it and the cached frame would keep being served,
    so the gates would grade a model that no longer existed.

    That is not hypothetical. ncaa-model had the identical bug in `build_walkforward`: its
    pace lambda was retuned 900 -> 15 and the stale artifact kept serving the old fit for
    five days, until the simulator projected the two slowest offenses in college football
    at league-average pace. Fixed there by tagging; fixed here the same way.

    A short hash rather than a readable tag because the NFL context block alone carries a
    dozen fields. Over-hashing costs a rebuild; under-hashing costs a wrong answer.
    """
    payload = json.dumps(
        {
            "seasons": asdict(cfg.seasons),
            "ratings": asdict(cfg.ratings),
            "pace": asdict(cfg.pace),
            "qb": asdict(cfg.qb),
            "context": asdict(cfg.context),
            "simulation": asdict(cfg.simulation),
            "projection": asdict(cfg.projection),
            "tuned": cfg.tuned,
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    return CACHE_DIR / f"backtest_frame_{digest}.parquet"


def _fit_l1_spread(train: list[dict], cfg: Config) -> "np.ndarray | None":
    """Least-squares `E[margin]` on (net_diff, context_points), or None to use the sim.

    Returns None when the config asks for the simulator, or when there is not yet enough
    history to fit on -- least squares on two seasons of one league is not automatically
    better than the generative model it replaces, and a silent fit on 40 games would be.

    `train` carries only games from strictly earlier seasons; the caller refits once per
    season before that season's games are graded, which is the same walk-forward discipline
    the ratings themselves use. Fitting on all seasons at once would leak the future into
    every projection and quietly make this look far better than it is.
    """
    if cfg.projection.spread_source != "linear":
        return None
    if len(train) < cfg.projection.min_train_games:
        return None
    t = pd.DataFrame(train)
    X = np.column_stack([np.ones(len(t)), t["net_diff"], t["context_points"]])
    return np.linalg.lstsq(X, t["actual_margin"].to_numpy(float), rcond=None)[0]


def _fit_feature_challenger(train: list[dict], cfg: Config):
    """Promote richer features only when the latest earlier season improves."""
    if len(train) < max(450, cfg.projection.min_train_games):
        return None
    frame = pd.DataFrame(train)
    candidates = [c for c in frame if c.endswith("_matchup_diff")]
    candidates += [c for c in frame if c.endswith("_burden_diff")]
    candidates += [c for c in ("rest_days_diff", "coach_continuity_diff") if c in frame]
    if not candidates or frame["season"].nunique() < 3:
        return None
    try:
        return validated_model.fit_validated_ridge(
            frame, base_features=["net_diff", "context_points"],
            candidate_features=candidates, target="actual_margin", alpha=35.0,
            minimum_fit_rows=300, minimum_improvement=0.05,
        )
    except ValueError:
        return None


def walk_forward(
    cfg: Config,
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    qb_adj_table: "pd.DataFrame | None" = None,
    n_sims: "int | None" = None,
    verbose: bool = True,
    cache: bool = True,
) -> pd.DataFrame:
    """For each season and each week from 4 onward, fit on the past and predict the week.

    The ratings walk is precomputed once (`build_walkforward_ratings`), which is what makes
    this affordable: every fit it contains used only games strictly before that week's
    first kickoff, so reading from it cannot leak the future.
    """
    path = backtest_frame_path(cfg)
    from .injuries import build_injury_burden, burden_lookup
    from .availability import build_availability_features, matchup_availability

    injuries = ingest.load_injuries(cfg.train_seasons)
    snap_counts = ingest.load_snap_counts(cfg.train_seasons)
    injury_burden_frame = build_injury_burden(injuries, snap_counts)
    availability = build_availability_features(injuries, snap_counts)
    injury_burden = burden_lookup(injury_burden_frame)
    resting = load_resting_starters()
    signature = build_cache_signature(
        builder=Path(__file__),
        config={"model": asdict(cfg), "n_sims": n_sims or cfg.simulation.n_sims},
        inputs={
            "schedules": frame_signature(
                schedules,
                ["game_id", "season", "week", "kickoff", "result", "total",
                 "spread_line", "total_line", "home_team", "away_team", "game_type",
                 "location", "home_rest", "away_rest", "roof", "stadium", "temp",
                 "wind"],
            ),
            "pbp": frame_signature(
                pbp,
                ["game_id", "season", "week", "posteam", "defteam", "epa",
                 "competitive", "fixed_drive", "down", "play_type", "success",
                 "yards_gained", "qb_dropback", "yardline_100", "sack",
                 "interception", "fumble_lost", "touchdown", "special",
                 "game_seconds_remaining"],
            ),
            "qb_adjustments": None if qb_adj_table is None else frame_signature(qb_adj_table),
            "injuries": frame_signature(
                injuries, ["season", "week", "team", "full_name", "position",
                           "report_status", "observed_at", "source"]
            ),
            "snap_counts": frame_signature(
                snap_counts,
                ["season", "week", "team", "player", "offense_pct", "defense_pct"],
            ),
            "resting_starters": sorted(resting),
        },
        artifact_version=3,
    )
    if cache:
        cached = read_cached_frame(
            path,
            ["game_id", "model_spread", "model_total", "market_spread",
             "market_total", "actual_margin", "actual_total"],
            signature,
        )
        if cached is not None:
            return cached

    sim_cfg = cfg.with_sims(n_sims) if n_sims else cfg
    game_off = build_game_offense_table(pbp, schedules, cache_key="main")
    feature_table = matchup_feature_table(pbp, schedules)
    feature_table = feature_table.merge(
        matchup_availability(availability, schedules), on="game_id", how="left")
    feature_lookup = feature_table.set_index("game_id").to_dict("index")
    drives = build_drive_table(pbp, cache_key="main")
    walkforward = build_walkforward_ratings(game_off, schedules, cfg)
    start_fp = fit_start_field_position(drives)

    graded = schedules[
        schedules["season"].isin(cfg.backtest_seasons)
        & schedules["game_type"].eq("REG")
        & (schedules["week"] >= FIRST_GRADED_WEEK)
        & schedules["result"].notna()
        & schedules["spread_line"].notna()
    ].sort_values("kickoff")

    # Injury burden, built once: snap-weighted share of each team's playing time that is
    # listed Out or Doubtful. See src/injuries.py.
    rows = []
    # L1 training history. Appended to as games are graded, and read only by the refit at
    # the top of each season, so the fit for season S never sees season S or later.
    l1_train: list[dict] = []
    for season in cfg.backtest_seasons:
        season_games = graded[graded["season"] == season]
        if season_games.empty:
            continue
        l1_beta = _fit_l1_spread(l1_train, cfg)
        feature_challenger = _fit_feature_challenger(l1_train, cfg)
        # Refit the drive model and the endgame table once per season -- both are slowly
        # varying calibration layers, and refitting weekly is wasted compute (§8 step 2).
        drive_model = fit_drive_model(drives, walkforward, cfg, as_of_season=season)
        endgame = fit_endgame_table(drives, cfg, as_of_season=season)
        venue_hfa = estimate_venue_hfa(schedules, walkforward, cfg, before_season=season)
        rng = np.random.default_rng(cfg.simulation.seed + season)

        for week, week_games in season_games.groupby("week"):
            try:
                ratings = ratings_at(walkforward, season, week)
            except KeyError:
                continue
            for _, g in week_games.iterrows():
                if (g["home_team"] not in ratings.index
                        or g["away_team"] not in ratings.index):
                    continue
                ctx = build_context(
                    g, cfg, venue_hfa=venue_hfa, allow_network=False,
                    resting_starters=resting, injury_burden=injury_burden,
                )
                qb = qb_points_for_game(g["game_id"], qb_adj_table, cfg)
                ctx = ctx.with_qb(qb.points, qb.note)

                sim = simulate_game(
                    g["home_team"], g["away_team"], ratings, drive_model, ctx,
                    sim_cfg, start_fp, endgame=endgame, rng=rng,
                    neutral_site=g.get("location") == "Neutral",
                )
                # L1/L2 (§9.2). The simulator always produces the DISTRIBUTION; which
                # estimator produces its CENTRE is config, and measured per market -- see
                # config/nfl.yaml `projection`. Totals are not touched: the simulator wins
                # there.
                net_diff = (
                    net_epa(ratings.at[g["home_team"], "off_rating"],
                            ratings.at[g["away_team"], "def_rating"], cfg)
                    - net_epa(ratings.at[g["away_team"], "off_rating"],
                              ratings.at[g["home_team"], "def_rating"], cfg)
                )
                model_spread = sim.mean_margin
                spread_source = "simulator"
                probs = sim
                if l1_beta is not None:
                    model_spread = float(
                        l1_beta[0]
                        + l1_beta[1] * net_diff
                        + l1_beta[2] * ctx.spread_points
                    )
                    spread_source = "linear"
                    # Reweight rather than translate or resimulate, so the integer lattice
                    # and its key-number atoms survive. Cover probabilities must use the
                    # calibrated weights or they describe a different projected centre.
                    probs = sim.recentered(model_spread)
                candidate = feature_lookup.get(g["game_id"], {})
                challenger_promoted = bool(
                    feature_challenger is not None
                    and feature_challenger.challenger_promoted
                )
                if challenger_promoted:
                    prediction_row = {"net_diff": net_diff,
                                      "context_points": ctx.spread_points, **candidate}
                    model_spread = float(feature_challenger.predict(
                        pd.DataFrame([prediction_row]))[0])
                    spread_source = "validated-feature-ridge"
                    probs = sim.recentered(model_spread)
                training_row = {
                    "season": season,
                    "net_diff": net_diff,
                    "context_points": ctx.spread_points,
                    "actual_margin": float(g["result"]),
                }
                training_row.update(candidate)
                l1_train.append(training_row)

                rows.append({
                    "game_id": g["game_id"],
                    "season": season,
                    "week": week,
                    "kickoff": g["kickoff"],
                    "home": g["home_team"],
                    "away": g["away_team"],
                    "model_spread": model_spread,
                    "model_spread_source": spread_source,
                    "feature_challenger_promoted": challenger_promoted,
                    "model_total": sim.mean_total,
                    "market_spread": float(g["spread_line"]),
                    "market_total": (
                        float(g["total_line"]) if pd.notna(g["total_line"]) else np.nan
                    ),
                    "actual_margin": float(g["result"]),
                    "actual_total": (
                        float(g["total"]) if pd.notna(g["total"]) else np.nan
                    ),
                    "cover_prob_home": probs.cover_prob(float(g["spread_line"]), "home"),
                    "over_prob": (
                        sim.total_prob(float(g["total_line"]), "over")
                        if pd.notna(g["total_line"]) else np.nan
                    ),
                    "hfa_points": ctx.components.get("hfa_venue_deviation", np.nan),
                    "qb_points": qb.points,
                    "context_points": ctx.spread_points,
                    "data_incomplete": ctx.data_incomplete,
                    "roof": g.get("roof"),
                    "div_game": g.get("div_game"),
                })
        if verbose:
            print(f"  backtest {season}: {len(rows)} games cumulative", flush=True)

    frame = pd.DataFrame(rows)
    elo_preds, _ = run_elo(schedules, cfg)
    frame = frame.merge(elo_preds[["game_id", "elo_spread"]], on="game_id", how="left")
    if cache:
        write_cached_frame(frame, path, signature)
    return frame


# --------------------------------------------------------------------------------------
# 8a. Metrics
# --------------------------------------------------------------------------------------

def _err(pred: pd.Series, actual: pd.Series) -> tuple:
    mask = pred.notna() & actual.notna()
    if not mask.any():
        return np.nan, np.nan
    e = pred[mask] - actual[mask]
    return float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))


def compute_metrics(frame: pd.DataFrame, weights: BlendWeights) -> pd.DataFrame:
    """The §8a comparison table: model, blended, market, Elo baseline."""
    blended_spread = frame.apply(
        lambda r: apply_blend(r["market_spread"], r["model_spread"], weights),
        axis=1,
    )
    blended_total = frame.apply(
        lambda r: apply_blend_total(r["market_total"], r["model_total"], weights), axis=1
    )

    rows = []
    for label, spread, total in (
        ("Model", frame["model_spread"], frame["model_total"]),
        ("Blended", blended_spread, blended_total),
        ("Market", frame["market_spread"], frame["market_total"]),
        ("Elo baseline", frame.get("elo_spread"), None),
    ):
        if spread is None:
            continue
        mae, rmse = _err(spread, frame["actual_margin"])
        t_mae, t_rmse = (
            _err(total, frame["actual_total"]) if total is not None else (np.nan, np.nan)
        )
        rows.append({
            "model": label,
            "spread_mae": mae,
            "spread_rmse": rmse,
            "total_mae": t_mae,
            "total_rmse": t_rmse,
            "n": int((spread.notna() & frame["actual_margin"].notna()).sum()),
        })
    out = pd.DataFrame(rows)
    out["ats_all"] = np.nan
    out.loc[out["model"] == "Model", "ats_all"] = _ats_record(frame)
    out["brier"] = np.nan
    out.loc[out["model"] == "Model", "brier"] = _brier(frame)
    return out


def _ats_record(frame: pd.DataFrame) -> float:
    """Win rate of always taking the side the model prefers, pushes excluded."""
    pick_home = (frame["model_spread"] > frame["market_spread"]).to_numpy()
    margin = frame["actual_margin"].to_numpy()
    line = frame["market_spread"].to_numpy()
    won = np.where(pick_home, margin > line, margin < line)
    resolved = margin != line
    return float(won[resolved].mean()) if resolved.any() else np.nan


def _brier(frame: pd.DataFrame) -> float:
    p = frame["cover_prob_home"]
    outcome = (frame["actual_margin"] > frame["market_spread"]).astype(float)
    resolved = frame["actual_margin"] != frame["market_spread"]
    if not resolved.any():
        return np.nan
    return float(((p[resolved] - outcome[resolved]) ** 2).mean())


def calibration_table(frame: pd.DataFrame, bin_width: float = 0.05) -> pd.DataFrame:
    """Predicted cover probability vs realized cover rate, in 5% bins.

    Systematic overconfidence here is the most common cause of a backtest that looks good
    and a live account that bleeds.
    """
    resolved = frame[frame["actual_margin"] != frame["market_spread"]].copy()
    resolved["outcome"] = (
        resolved["actual_margin"] > resolved["market_spread"]
    ).astype(float)
    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    resolved["bin"] = pd.cut(resolved["cover_prob_home"], bins, include_lowest=True)
    grp = resolved.groupby("bin", observed=True).agg(
        n=("outcome", "size"),
        predicted=("cover_prob_home", "mean"),
        realized=("outcome", "mean"),
    ).reset_index()
    grp["abs_error"] = (grp["realized"] - grp["predicted"]).abs()
    return grp


def bootstrap_ats(
    bet_results: np.ndarray,
    seasons: "np.ndarray | None" = None,
    n_boot: int = 10000,
    seed: int = 20260730,
) -> tuple:
    """5th/95th percentile of filtered win rate, resampling whole seasons when known.

    If the 5th percentile is below 50%, the edge is not distinguishable from noise at this
    sample size, and the report has to say so in plain language.
    """
    if len(bet_results) == 0:
        return np.nan, np.nan, np.nan
    if seasons is not None:
        result = robust_inference.season_block_bootstrap(
            np.asarray(bet_results, dtype=float),
            seasons,
            n_boot=n_boot,
            seed=seed,
            confidence=0.90,
        )
        return result.point, result.low, result.high
    rng = np.random.default_rng(seed)
    draws = rng.choice(bet_results, size=(n_boot, len(bet_results)), replace=True)
    rates = draws.mean(axis=1)
    return (
        float(bet_results.mean()),
        float(np.percentile(rates, 5)),
        float(np.percentile(rates, 95)),
    )


def clv_proxy(frame: pd.DataFrame, schedules: pd.DataFrame):
    """Fraction of the time the market moved toward the model's side.

    The most informative single number in the report -- but it needs opening lines, which
    nflverse does not publish (`spread_line` is the closing number). Returns None when the
    data is absent rather than grading closing against closing, which would be circular.
    """
    if "spread_line_open" not in schedules.columns:
        return None
    opens = schedules.set_index("game_id")["spread_line_open"]
    merged = frame.join(opens, on="game_id")
    mask = merged["spread_line_open"].notna()
    if not mask.any():
        return None
    moved = merged.loc[mask, "market_spread"] - merged.loc[mask, "spread_line_open"]
    model_side = np.sign(
        merged.loc[mask, "model_spread"] - merged.loc[mask, "spread_line_open"]
    )
    return float((np.sign(moved) == model_side).mean())


# --------------------------------------------------------------------------------------
# 8b. Gates
# --------------------------------------------------------------------------------------

def gate_key_numbers(sim_pmf: dict, schedules: pd.DataFrame, tol: float = 2.0):
    """Simulated margin PMF vs historical at 3, 7, 6, 10, 14, 4, within 2 points each."""
    done = schedules[schedules["result"].notna() & schedules["season"].ge(2019)]
    real = done["result"].abs().value_counts(normalize=True)

    rows, worst, worst_k = [], 0.0, None
    for k in (3, 7, 6, 10, 14, 4):
        sim_pct = 100 * (sim_pmf.get(k, 0.0) + sim_pmf.get(-k, 0.0))
        real_pct = 100 * float(real.get(k, 0.0))
        diff = abs(sim_pct - real_pct)
        rows.append(
            f"|{k}|: sim {sim_pct:5.2f}%  real {real_pct:5.2f}%  ({sim_pct - real_pct:+.2f}pp)"
        )
        if diff > worst:
            worst, worst_k = diff, k
    return GateResult(
        "GATE_KEY_NUMBERS",
        worst <= tol,
        f"worst gap {worst:.2f}pp at |margin|={worst_k} (tolerance {tol:.0f}pp)",
        "\n      ".join(rows),
    )


def gate_no_lookahead(sample: int = 200) -> GateResult:
    from tests.test_no_lookahead import check_no_lookahead

    ok, detail = check_no_lookahead(sample=sample)
    return GateResult("GATE_NO_LOOKAHEAD", ok, detail)


def gate_beats_elo(metrics: pd.DataFrame) -> GateResult:
    model = metrics.loc[metrics["model"] == "Model", "spread_rmse"].iloc[0]
    elo_rows = metrics.loc[metrics["model"] == "Elo baseline", "spread_rmse"]
    if elo_rows.empty or pd.isna(elo_rows.iloc[0]):
        return GateResult("GATE_BEATS_ELO", False, "no Elo baseline available")
    elo = float(elo_rows.iloc[0])
    return GateResult(
        "GATE_BEATS_ELO", model < elo, f"model RMSE {model:.3f} vs Elo {elo:.3f}"
    )


def gate_blend_informative(weights: BlendWeights) -> GateResult:
    return GateResult(
        "GATE_BLEND_INFORMATIVE",
        weights.informative,
        f"spread t={weights.t_model:+.2f}, total t={weights.t_model_total:+.2f} "
        "(need positive applied b and t > 2.0 in at least one market)",
        f"spread b={weights.b_model_raw:+.3f} raw/{weights.b_model:.3f} applied; "
        f"total b={weights.b_model_total_raw:+.3f} raw/"
        f"{weights.b_model_total:.3f} applied",
    )


def gate_calibrated(calib: pd.DataFrame, tol: float = 0.06) -> GateResult:
    big = calib[calib["n"] >= 100]
    if big.empty:
        return GateResult("GATE_CALIBRATED", False, "no bin with 100+ observations")
    worst = float(big["abs_error"].max())
    row = big.loc[big["abs_error"].idxmax()]
    return GateResult(
        "GATE_CALIBRATED", worst <= tol,
        f"worst bin off by {100 * worst:.2f}pp (bin {row['bin']}, n={int(row['n'])}, "
        f"tolerance {100 * tol:.0f}pp)",
    )


def gate_rmse(frame: pd.DataFrame, kind: str, max_ratio: float = 1.0) -> GateResult:
    """Absolute accuracy versus the closing market, as a promotion precondition."""
    if kind == "spread":
        model, market, actual = "model_spread", "market_spread", "actual_margin"
    elif kind == "total":
        model, market, actual = "model_total", "market_total", "actual_total"
    else:
        raise ValueError(f"unknown market {kind!r}")
    sub = frame[[model, market, actual]].dropna()
    name = f"GATE_RMSE_{kind.upper()}"
    if len(sub) < 200:
        return GateResult(name, False, f"only {len(sub)} gradeable games")
    model_rmse = float(np.sqrt(((sub[model] - sub[actual]) ** 2).mean()))
    market_rmse = float(np.sqrt(((sub[market] - sub[actual]) ** 2).mean()))
    ratio = model_rmse / market_rmse if market_rmse > 0 else float("inf")
    return GateResult(
        name,
        ratio <= max_ratio,
        f"model RMSE {model_rmse:.3f} vs close {market_rmse:.3f} "
        f"(ratio {ratio:.3f}, need <= {max_ratio:.3f}), n={len(sub)}",
    )


def gate_unbiased(frame: pd.DataFrame, tol: float = 0.5) -> GateResult:
    """Mean model projection matches mean outcome, on both markets (§8b).

    Specified by PATCH 02 and never implemented. It is the cheapest gate in this file and
    it would have caught a live bug: `league_mean_drives_per_team` was set to 11.4 against
    a measured 11.099, which put +1.22 points of bias into every simulated total for the
    lifetime of the repo. Implemented now mainly as a regression guard on that class of
    unchecked constant.
    """
    rows, worst, worst_m = [], 0.0, None
    for label, model, actual in (
        ("spread", "model_spread", "actual_margin"),
        ("total", "model_total", "actual_total"),
    ):
        sub = frame[[model, actual]].dropna()
        bias = float(sub[model].mean() - sub[actual].mean()) if len(sub) else float("nan")
        rows.append(f"{label}: bias {bias:+.3f} pts (n={len(sub)})")
        if abs(bias) > worst:
            worst, worst_m = abs(bias), label
    return GateResult(
        "GATE_UNBIASED", worst <= tol,
        f"worst bias {worst:.3f} pts on {worst_m} (tolerance {tol:.1f})",
        "\n      ".join(rows),
    )


def gate_scale(frame: pd.DataFrame, tol: float = 0.15) -> GateResult:
    """Model projections vary as much as the market's do, on both markets (§8b).

    Also specified by PATCH 02 and never implemented.

    DO NOT "FIX" A FAILURE HERE BY RESCALING THE MODEL OUTPUT. This gate assumes the model
    carries roughly as much information as the market, and this one does not, so the
    target it enforces is wrong for it. Regressing outcomes on the model's own projections
    gives slopes of 0.605 (spread) and 0.753 (total): the variance the model's signal
    actually supports is sd 4.87 on spreads and 2.07 on totals, against market sds of 6.42
    and 4.48. Inflating the model to hit a ratio of 1.0 would make it more than twice as
    confident on totals as its information justifies -- worse predictions that score better
    on this one line. Read a failure here as a measurement of the signal gap, not a defect
    to tune away; it will clear on its own if and only if the model gets genuinely better.
    """
    rows, worst, worst_m = [], 0.0, None
    for label, model, market in (
        ("spread", "model_spread", "market_spread"),
        ("total", "model_total", "market_total"),
    ):
        sub = frame[[model, market]].dropna()
        msd = float(sub[market].std())
        ratio = float(sub[model].std()) / msd if msd > 0 else float("nan")
        rows.append(
            f"{label}: sd ratio {ratio:.3f} (model {sub[model].std():.3f} "
            f"vs market {msd:.3f})"
        )
        if abs(ratio - 1.0) > worst:
            worst, worst_m = abs(ratio - 1.0), label
    return GateResult(
        "GATE_SCALE", worst <= tol,
        f"worst |ratio-1| {worst:.3f} on {worst_m} (tolerance {tol:.2f})",
        "\n      ".join(rows),
    )


def gate_speed(seconds: float, budget: float = 180.0) -> GateResult:
    return GateResult(
        "GATE_SPEED", seconds < budget,
        f"weekly refresh on cached data took {seconds:.1f}s (budget {budget:.0f}s)",
    )


def run_gates(
    weights: BlendWeights,
    metrics: pd.DataFrame,
    calib: pd.DataFrame,
    sim_pmf: dict,
    schedules: pd.DataFrame,
    weekly_seconds: float,
    frame: pd.DataFrame,
) -> list:
    """All nine gates from §8b. GATE_UNBIASED and GATE_SCALE were specified by PATCH 02
    and left unimplemented until now, so 'all gates pass' was never a statement about
    nine gates."""
    return [
        gate_key_numbers(sim_pmf, schedules),
        gate_no_lookahead(),
        gate_beats_elo(metrics),
        gate_blend_informative(weights),
        gate_calibrated(calib),
        gate_rmse(frame, "spread"),
        gate_rmse(frame, "total"),
        gate_unbiased(frame),
        gate_scale(frame),
        gate_speed(weekly_seconds),
    ]


# --------------------------------------------------------------------------------------
# Simulator validation input (§6d) and speed measurement
# --------------------------------------------------------------------------------------

def _shared_artifacts(cfg: Config, schedules: pd.DataFrame, pbp: pd.DataFrame, season: int):
    game_off = build_game_offense_table(pbp, schedules, cache_key="main")
    drives = build_drive_table(pbp, cache_key="main")
    walkforward = build_walkforward_ratings(game_off, schedules, cfg)
    start_fp = fit_start_field_position(drives)
    drive_model = fit_drive_model(drives, walkforward, cfg, as_of_season=season)
    endgame = fit_endgame_table(drives, cfg, as_of_season=season)
    return walkforward, start_fp, drive_model, endgame


def pooled_margin_pmf(
    cfg: Config,
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    n_games: int = 400,
    n_sims: int = 4000,
) -> dict:
    """Pool simulated margins across many historical games and compare the shape to
    history.

    In-sample ratings are fine here -- this tests the *distribution shape*, not predictive
    power.
    """
    season = max(cfg.backtest_seasons)
    walkforward, start_fp, drive_model, endgame = _shared_artifacts(
        cfg, schedules, pbp, season
    )
    venue_hfa = estimate_venue_hfa(schedules, walkforward, cfg)

    games = schedules[
        schedules["season"].isin(cfg.backtest_seasons)
        & schedules["game_type"].eq("REG")
        & (schedules["week"] >= FIRST_GRADED_WEEK)
        & schedules["result"].notna()
    ]
    games = games.sample(min(n_games, len(games)), random_state=cfg.simulation.seed)

    sim_cfg = cfg.with_sims(n_sims)
    rng = np.random.default_rng(cfg.simulation.seed)
    pooled = []
    for _, g in games.iterrows():
        try:
            ratings = ratings_at(walkforward, g["season"], g["week"])
        except KeyError:
            continue
        if g["home_team"] not in ratings.index or g["away_team"] not in ratings.index:
            continue
        ctx = build_context(g, cfg, venue_hfa=venue_hfa, allow_network=False)
        sim = simulate_game(
            g["home_team"], g["away_team"], ratings, drive_model, ctx, sim_cfg,
            start_fp, endgame=endgame, rng=rng,
            neutral_site=g.get("location") == "Neutral",
        )
        pooled.append(sim.margins)

    allm = np.concatenate(pooled)
    vals, counts = np.unique(allm, return_counts=True)
    return {int(v): float(c / counts.sum()) for v, c in zip(vals, counts)}


def time_weekly_refresh(cfg: Config, schedules: pd.DataFrame, pbp: pd.DataFrame) -> float:
    """GATE_SPEED: time a full weekly refresh on cached data."""
    t0 = time.time()
    season = max(cfg.backtest_seasons)
    walkforward, start_fp, drive_model, endgame = _shared_artifacts(
        cfg, schedules, pbp, season
    )
    venue_hfa = estimate_venue_hfa(schedules, walkforward, cfg, before_season=season)

    week_games = schedules[
        (schedules["season"] == season) & (schedules["week"] == 10)
        & schedules["game_type"].eq("REG")
    ]
    ratings = ratings_at(walkforward, season, 10)
    rng = np.random.default_rng(cfg.simulation.seed)
    for _, g in week_games.iterrows():
        if g["home_team"] not in ratings.index or g["away_team"] not in ratings.index:
            continue
        ctx = build_context(g, cfg, venue_hfa=venue_hfa, allow_network=False)
        simulate_game(
            g["home_team"], g["away_team"], ratings, drive_model, ctx, cfg, start_fp,
            endgame=endgame, rng=rng,
        )
    return time.time() - t0


# --------------------------------------------------------------------------------------
# PATCH 01 §5 -- spreads and totals evaluated separately
# --------------------------------------------------------------------------------------

def rescale_to_market(component: pd.Series, market: pd.Series) -> pd.Series:
    """Rescale a component to the market's dispersion about the market line.

    An overdispersed component's disagreement with the market is dominated by scale noise,
    which suppresses the t-statistic on its blend weight. Rescaling is upstream of any
    verdict on whether the component carries signal (PATCH 01 §4).

    Note this rescales the *component*, not its disagreement. Shrinking the disagreement
    by a scalar cannot change the t-statistic on its own coefficient -- it rescales b and
    its standard error identically. Compressing the component toward its own mean changes
    which part of the disagreement survives, and that is what lifts the t-statistic when
    overdispersion is the thing suppressing it.
    """
    c_sd, m_sd = float(component.std()), float(market.std())
    if c_sd <= 0 or m_sd <= 0:
        return component.copy()
    return component.mean() + (component - component.mean()) * (m_sd / c_sd)


def residual_fit(
    component: pd.Series,
    market: pd.Series,
    actual: pd.Series,
    debias: bool = False,
    sd_match: bool = False,
) -> dict:
    """Residual-form regression WITH a free intercept (PATCH 02 §1).

        actual - market = a + b * (component - market)

    The intercept is not optional. With the market coefficient pinned at 1.0 and no
    intercept, any constant level bias in a component loads onto its disagreement slope --
    which is exactly how +1.43 points of systematic over-projection on totals presented
    itself as b = +0.060 when the real game-level coefficient was negative.

    `a` is reported alongside every `b`, and GATE_UNBIASED requires |a| < 0.5.
    """
    df = pd.concat([component, market, actual], axis=1).dropna()
    comp, mkt, act = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    if sd_match:
        comp = rescale_to_market(comp, mkt)
    if debias:
        comp = comp - comp.mean() + mkt.mean()

    y = (act - mkt).to_numpy(float)
    fit = robust_inference.fit_ols(
        y, (comp - mkt).to_numpy(float), covariance="hc3"
    )
    beta, se = fit.beta, fit.se
    return {
        "a": float(beta[0]),
        "se_a": float(se[0]),
        "t_a": float(beta[0] / se[0]) if se[0] > 0 else 0.0,
        "b": float(beta[1]),
        "se_b": float(se[1]),
        "t_b": float(beta[1] / se[1]) if se[1] > 0 else 0.0,
        "r2": fit.r_squared,
        "resid_sd": fit.residual_sd,
        "n": len(y),
        "sd_ratio": float(comp.std() / mkt.std()) if mkt.std() else np.nan,
        "rmse": float(np.sqrt(((comp - act) ** 2).mean())),
        "noise_points": float(
            np.sqrt(max(comp.std() ** 2 - mkt.std() ** 2, 0.0))
        ),
    }


def evaluate_one_market(
    frame: pd.DataFrame,
    cfg: Config,
    kind: str,
) -> dict:
    """RMSE, blend weight (raw and rescaled), filtered record + bootstrap CI, calibration.

    `kind` is "spread" or "total". Elo cannot produce a total, so it appears only for
    spreads.
    """
    if kind == "spread":
        model, market, actual = "model_spread", "market_spread", "actual_margin"
        prob_col, edge_min = "cover_prob_home", cfg.betting.min_edge_points_spread
    else:
        model, market, actual = "model_total", "market_total", "actual_total"
        prob_col, edge_min = "over_prob", cfg.betting.min_edge_points_total

    df = frame.dropna(subset=[model, market, actual]).copy()
    if df.empty:
        return {"kind": kind, "n": 0}

    _, raw_fit = _residual_fit(df, model, market, actual)
    a_raw, b_raw = map(float, raw_fit.beta)
    t_raw = float(raw_fit.t_values[1])

    rescaled = rescale_to_market(df[model], df[market])
    rescaled_frame = df.assign(_rescaled_component=rescaled)
    _, rescaled_fit = _residual_fit(
        rescaled_frame, "_rescaled_component", market, actual
    )
    a_rescaled, b_rescaled = map(float, rescaled_fit.beta)
    t_resc = float(rescaled_fit.t_values[1])

    mae_m, rmse_m = _err(df[model], df[actual])
    mae_k, rmse_k = _err(df[market], df[actual])
    _, rmse_resc = _err(rescaled, df[actual])

    # Filtered record: take the model's side when it disagrees by at least the threshold.
    edge = (df[model] - df[market]).to_numpy(float)
    take_high = edge > 0
    qualifies = np.abs(edge) >= edge_min
    a = df[actual].to_numpy(float)
    k = df[market].to_numpy(float)
    sel = qualifies & (a != k)
    if sel.any():
        won = np.where(take_high[sel], a[sel] > k[sel], a[sel] < k[sel]).astype(float)
        rate, lo, hi = bootstrap_ats(won, df.loc[sel, "season"].to_numpy())
    else:
        won, rate, lo, hi = np.array([]), np.nan, np.nan, np.nan

    return {
        "kind": kind,
        "n": len(df),
        "model_rmse": rmse_m,
        "market_rmse": rmse_k,
        "rescaled_rmse": rmse_resc,
        "model_mae": mae_m,
        "market_mae": mae_k,
        "model_sd": float(df[model].std()),
        "market_sd": float(df[market].std()),
        "sd_ratio": float(df[model].std() / df[market].std()) if df[market].std() else np.nan,
        "a_raw": a_raw,
        "b_raw": b_raw,
        "t_raw": t_raw,
        "a_rescaled": a_rescaled,
        "b_rescaled": b_rescaled,
        "t_rescaled": t_resc,
        "n_bets": int(sel.sum()),
        "win_rate": rate,
        "ci_low": lo,
        "ci_high": hi,
        "calibration": calibration_for(df, prob_col, model, market, actual, kind),
    }


def calibration_for(
    df: pd.DataFrame, prob_col: str, model: str, market: str, actual: str, kind: str,
    bin_width: float = 0.05,
) -> pd.DataFrame:
    """Predicted probability vs realized rate, for either market."""
    if prob_col not in df.columns or df[prob_col].isna().all():
        return pd.DataFrame(columns=["bin", "n", "predicted", "realized", "abs_error"])
    d = df[df[actual] != df[market]].copy()
    d["outcome"] = (d[actual] > d[market]).astype(float)
    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    d["bin"] = pd.cut(d[prob_col], bins, include_lowest=True)
    grp = d.groupby("bin", observed=True).agg(
        n=("outcome", "size"),
        predicted=(prob_col, "mean"),
        realized=("outcome", "mean"),
    ).reset_index()
    grp["abs_error"] = (grp["realized"] - grp["predicted"]).abs()
    return grp


def evaluate_both_markets(frame: pd.DataFrame, cfg: Config) -> dict:
    return {k: evaluate_one_market(frame, cfg, k) for k in ("spread", "total")}


def filtered_bet_results(
    frame: pd.DataFrame,
    cfg: Config,
    weights: BlendWeights,
    *,
    return_seasons: bool = False,
):
    """Win/loss array for bets clearing the §9 spread thresholds, pushes dropped."""
    blended = frame.apply(
        lambda r: apply_blend(r["market_spread"], r["model_spread"], weights),
        axis=1,
    )
    edge = blended - frame["market_spread"]
    pick_home = (edge > 0).to_numpy()
    qualifies = (edge.abs() >= cfg.betting.min_edge_points_spread).to_numpy()
    margin = frame["actual_margin"].to_numpy()
    line = frame["market_spread"].to_numpy()
    sel = qualifies & (margin != line)
    if not sel.any():
        empty = np.array([])
        return (empty, empty) if return_seasons else empty
    won = np.where(pick_home[sel], margin[sel] > line[sel], margin[sel] < line[sel])
    results = won.astype(float)
    if return_seasons:
        return results, frame.loc[sel, "season"].to_numpy()
    return results
