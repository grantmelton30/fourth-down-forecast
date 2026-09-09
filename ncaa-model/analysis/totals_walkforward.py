#!/usr/bin/env python3
"""Compare fixed NCAA totals candidates using earlier-season-only fits.

This is a historical research instrument, not a promotion gate. The underlying seasons
have been inspected before. Its useful output is a reproducible prospective shadow-model
candidate, if one survives the fixed comparisons in TOTALS_RESEARCH.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME = ROOT / "data" / "cache" / "backtest_frame_default.parquet"
DEFAULT_JSON = ROOT / "data" / "internal" / "ncaa_totals_walkforward.json"
DEFAULT_MARKDOWN = ROOT / "data" / "internal" / "ncaa_totals_walkforward.md"
TRAIN_START = 2019
EVAL_SEASONS = (2022, 2023, 2024, 2025)
MIN_TRAIN_ROWS = 300
RIDGE_ALPHA = 45.0

FAMILIES = {
    "pure_residual": ("pure_disagreement",),
    "tempo_efficiency": (
        "pace_sum", "eff_sum", "pace_efficiency_interaction", "abs_net_diff",
    ),
    "play_quality": (
        "success_rate_matchup_sum", "explosive_rate_matchup_sum",
        "havoc_avoidance_matchup_sum", "red_zone_td_rate_matchup_sum",
    ),
}
ADAPTIVE_CANDIDATES = ("market", "bias", *FAMILIES)


def _frame_label(frame_path: Path) -> str:
    try:
        return frame_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return frame_path.name


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id", "season", "week", "fbs_only", "restricted", "home_p5", "away_p5",
        "actual_total", "total_close", "spread_close", "model_total_pure",
        "model_total_market_adjusted", "pace_sum", "eff_sum", "net_diff",
        *FAMILIES["play_quality"],
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"totals research frame missing columns: {sorted(missing)}")
    if frame.duplicated("game_id").any():
        raise ValueError("totals research requires one row per game_id")
    out = frame.copy()
    for column in required - {"game_id", "fbs_only", "restricted", "home_p5", "away_p5"}:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[
        out["fbs_only"].fillna(False)
        & out["season"].between(TRAIN_START, max(EVAL_SEASONS))
        & out["actual_total"].notna()
        & out["total_close"].notna()
    ].copy()
    out["market_error"] = out["actual_total"] - out["total_close"]
    out["pure_disagreement"] = out["model_total_pure"] - out["total_close"]
    out["pace_efficiency_interaction"] = out["pace_sum"] * out["eff_sum"]
    out["abs_net_diff"] = out["net_diff"].abs()
    return out.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _fit_ridge(train: pd.DataFrame, features: tuple[str, ...]):
    usable = train.dropna(subset=["market_error"])
    if features:
        usable = usable[usable.loc[:, features].notna().any(axis=1)]
    if len(usable) < MIN_TRAIN_ROWS:
        raise ValueError("insufficient earlier-season rows")
    if not features:
        return np.array([]), np.array([]), np.array([]), np.array([usable.market_error.mean()])
    values = usable.loc[:, features].to_numpy(float)
    medians = np.nanmedian(values, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("candidate contains an entirely missing feature")
    values = np.where(np.isnan(values), medians, values)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    design = np.column_stack([np.ones(len(values)), (values - center) / scale])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ usable.market_error.to_numpy(float))
    return medians, center, scale, beta


def _predict_ridge(test: pd.DataFrame, features: tuple[str, ...], fitted) -> np.ndarray:
    medians, center, scale, beta = fitted
    if not features:
        return np.repeat(beta[0], len(test))
    values = test.loc[:, features].to_numpy(float)
    available = np.isfinite(values).any(axis=1)
    values = np.where(np.isnan(values), medians, values)
    prediction = np.column_stack([np.ones(len(values)), (values - center) / scale]) @ beta
    prediction[~available] = np.nan
    return prediction


def walkforward_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare(frame)
    results = []
    for season in EVAL_SEASONS:
        train = data[data.season < season]
        test = data[data.season == season].copy()
        if test.empty:
            continue
        test["market"] = test["total_close"]
        test["bias"] = test["total_close"] + _predict_ridge(test, (), _fit_ridge(train, ()))
        for name, features in FAMILIES.items():
            try:
                residual = _predict_ridge(test, features, _fit_ridge(train, features))
            except ValueError:
                residual = np.full(len(test), np.nan)
            test[name] = test["total_close"] + residual
        test["archived_pure"] = test["model_total_pure"]
        test["archived_adjusted"] = test["model_total_market_adjusted"]
        results.append(test)
    if not results:
        raise ValueError("no populated evaluation seasons")
    return apply_adaptive(pd.concat(results, ignore_index=True))


def apply_adaptive(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["adaptive"] = pd.to_numeric(out["market"], errors="coerce").astype(float)
    out["adaptive_choice"] = "market"
    for season in sorted(out.season.unique()):
        previous = out[(out.season < season) & (out.season >= min(EVAL_SEASONS))]
        if previous.season.nunique() < 2:
            continue
        scores = {}
        for name in ADAPTIVE_CANDIDATES:
            paired = previous.dropna(subset=[name, "actual_total"])
            if len(paired) >= MIN_TRAIN_ROWS:
                scores[name] = float(np.sqrt(np.mean((paired[name] - paired.actual_total) ** 2)))
        if not scores:
            continue
        choice = min(scores, key=lambda name: (scores[name], ADAPTIVE_CANDIDATES.index(name)))
        mask = out.season == season
        available = out.loc[mask, choice].notna()
        index = out.index[mask]
        out.loc[index[available], "adaptive"] = out.loc[index[available], choice]
        out.loc[mask, "adaptive_choice"] = choice
    return out


def _bootstrap_gain(base_error, candidate_error, blocks, *, n_boot, seed=20260909):
    base = np.asarray(base_error, dtype=float)
    candidate = np.asarray(candidate_error, dtype=float)
    groups: dict[str, list[int]] = {}
    for i, block in enumerate(blocks):
        groups.setdefault(str(block), []).append(i)
    positions = [np.asarray(value) for value in groups.values()]
    point = float(np.sqrt(np.mean(base**2)) - np.sqrt(np.mean(candidate**2)))
    if not n_boot:
        return point, None, None
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.integers(0, len(positions), len(positions))
        index = np.concatenate([positions[j] for j in chosen])
        values[i] = np.sqrt(np.mean(base[index] ** 2)) - np.sqrt(np.mean(candidate[index] ** 2))
    return point, float(np.quantile(values, .05)), float(np.quantile(values, .95))


def metrics(predictions: pd.DataFrame, *, n_boot=5000) -> list[dict]:
    models = ["market", "bias", *FAMILIES, "adaptive", "archived_pure", "archived_adjusted"]
    rows = []
    for offset, model in enumerate(models):
        paired = predictions.dropna(subset=[model, "actual_total", "market", "bias"]).copy()
        error = paired[model].to_numpy(float) - paired.actual_total.to_numpy(float)
        market_error = paired.market.to_numpy(float) - paired.actual_total.to_numpy(float)
        bias_error = paired.bias.to_numpy(float) - paired.actual_total.to_numpy(float)
        blocks = paired.season.astype(str) + "-W" + paired.week.astype(int).astype(str)
        market_gain = _bootstrap_gain(market_error, error, blocks, n_boot=n_boot, seed=20260909 + offset)
        bias_gain = _bootstrap_gain(bias_error, error, blocks, n_boot=n_boot, seed=20261009 + offset)
        rows.append({
            "model": model, "n": len(paired), "coverage": len(paired) / len(predictions),
            "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "rmse_gain_vs_market": market_gain[0], "gain_vs_market_90ci": list(market_gain[1:]),
            "rmse_gain_vs_bias": bias_gain[0], "gain_vs_bias_90ci": list(bias_gain[1:]),
        })
    return rows


def season_metrics(predictions: pd.DataFrame) -> list[dict]:
    rows = []
    for season, group in predictions.groupby("season"):
        for model in ("market", "bias", *FAMILIES, "adaptive"):
            paired = group.dropna(subset=[model, "actual_total"])
            error = paired[model] - paired.actual_total
            rows.append({"season": int(season), "model": model, "n": len(paired),
                         "rmse": float(np.sqrt(np.mean(error**2))),
                         "bias": float(error.mean()),
                         "adaptive_choice": (str(group.adaptive_choice.iloc[0])
                                             if model == "adaptive" else None)})
    return rows


def diagnostics(predictions: pd.DataFrame) -> list[dict]:
    groups = {
        "restricted": predictions.restricted.fillna(False),
        "p5_vs_p5": predictions.home_p5.fillna(False) & predictions.away_p5.fillna(False),
        "weeks_1_3": predictions.week <= 3,
        "weeks_4_plus": predictions.week >= 4,
        "spread_abs_over_28": predictions.spread_close.abs() > 28,
        "spread_abs_at_most_28": predictions.spread_close.abs() <= 28,
        "market_total_at_most_50": predictions.total_close <= 50,
        "market_total_over_50": predictions.total_close > 50,
    }
    rows = []
    for group_name, mask in groups.items():
        group = predictions[mask]
        for model in ("market", "bias", *FAMILIES, "adaptive"):
            paired = group.dropna(subset=[model, "actual_total"])
            if paired.empty:
                rows.append({"group": group_name, "model": model, "n": 0,
                             "rmse": None, "bias": None, "available": False})
                continue
            error = paired[model] - paired.actual_total
            rows.append({"group": group_name, "model": model, "n": len(paired),
                         "rmse": float(np.sqrt(np.mean(error**2))), "bias": float(error.mean()),
                         "available": True})
    return rows


def p1_summary(predictions: pd.DataFrame) -> dict:
    predictions = predictions[predictions.season.between(2021, 2025)].copy()
    edge = predictions.model_total_pure - predictions.total_close
    selected = predictions[
        predictions.restricted.fillna(False) & edge.abs().ge(.5) & edge.abs().lt(6)
        & predictions.model_total_pure.notna()
    ].copy()
    wins = ((selected.actual_total > selected.total_close) ==
            (selected.model_total_pure > selected.total_close))
    pushes = selected.actual_total == selected.total_close
    decided = wins[~pushes]
    win_count = int(decided.sum())
    losses = int(len(decided) - win_count)
    return {"n": len(selected), "wins": win_count, "losses": losses,
            "pushes": int(pushes.sum()),
            "win_rate": (win_count / len(decided) if len(decided) else None),
            "assumed_units_at_minus_110": win_count - 1.1 * losses}


def build_report(frame_path: Path, *, n_boot=5000) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_parquet(frame_path)
    population = prepare(frame)
    predictions = walkforward_predictions(frame)
    report = {
        "schema_version": 1,
        "status": "exploratory_historical_research_not_authorized_for_promotion",
        "frame": _frame_label(frame_path),
        "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        "source_rows": len(frame), "population_rows": len(population),
        "evaluation_rows": len(predictions),
        "available_evaluation_weeks": sorted(int(x) for x in predictions.week.unique()),
        "evaluation_seasons": list(EVAL_SEASONS), "minimum_train_rows": MIN_TRAIN_ROWS,
        "ridge_alpha": RIDGE_ALPHA, "families": {k: list(v) for k, v in FAMILIES.items()},
        "metrics": metrics(predictions, n_boot=n_boot),
        "season_metrics": season_metrics(predictions),
        "diagnostics": diagnostics(predictions), "p1_historical_close": p1_summary(population),
        "adaptive_choices": {str(int(s)): str(g.adaptive_choice.iloc[0])
                             for s, g in predictions.groupby("season")},
    }
    return report, predictions


def markdown(report: dict) -> str:
    by_model = {row["model"]: row for row in report["metrics"]}
    lines = [
        "# NCAA totals walk-forward research", "",
        "Historical exploratory analysis. No automatic promotion or betting authorization.", "",
        f"Evaluated {report['evaluation_rows']:,} FBS-vs-FBS games across "
        f"{', '.join(map(str, report['evaluation_seasons']))}.", "",
        "| model | n | coverage | RMSE | gain vs market | 90% CI | gain vs bias | 90% CI | bias |", "",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["metrics"]:
        market_interval = row["gain_vs_market_90ci"]
        bias_interval = row["gain_vs_bias_90ci"]
        market_shown = ("unavailable" if market_interval[0] is None else
                        f"[{market_interval[0]:+.3f}, {market_interval[1]:+.3f}]")
        bias_shown = ("unavailable" if bias_interval[0] is None else
                      f"[{bias_interval[0]:+.3f}, {bias_interval[1]:+.3f}]")
        lines.append(f"| {row['model']} | {row['n']} | {row['coverage']:.2%} | {row['rmse']:.3f} | "
                     f"{row['rmse_gain_vs_market']:+.3f} | {market_shown} | "
                     f"{row['rmse_gain_vs_bias']:+.3f} | {bias_shown} | {row['bias']:+.3f} |")
    lines.extend(["", "Adaptive choices by season: " + ", ".join(
        f"{season}={choice}" for season, choice in report["adaptive_choices"].items()), ""])
    if not any(row["available"] for row in report["diagnostics"]
               if row["group"] == "weeks_1_3"):
        lines.extend(["Weeks 1–3 diagnostic: unavailable; the cached backtest begins at week 4.", ""])
    p1 = report["p1_historical_close"]
    rate = "unavailable" if p1["win_rate"] is None else f"{p1['win_rate']:.2%}"
    lines.extend([
        f"Frozen P1 historical-close diagnostic: {p1['wins']}-{p1['losses']}-{p1['pushes']} "
        f"({rate}), {p1['assumed_units_at_minus_110']:+.1f} assumed units.", "",
        "The closing market is a historical information benchmark, not a recorded executable quote. "
        "Intervals are descriptive and do not correct for prior inspection of these seasons.", "",
    ])
    best = min(report["metrics"], key=lambda row: (row["rmse"], row["model"]))
    lines.append(f"Lowest aggregate RMSE: {best['model']} ({best['rmse']:.3f}).")
    if by_model["adaptive"]["gain_vs_market_90ci"][0] is not None:
        low = by_model["adaptive"]["gain_vs_market_90ci"][0]
        conclusion = ("The adaptive candidate cleared zero across its descriptive interval."
                      if low > 0 else
                      "The adaptive candidate did not clearly improve on the market.")
        lines.extend(["", conclusion])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("bootstrap must be nonnegative")
    report, _ = build_report(args.frame, n_boot=args.bootstrap)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
