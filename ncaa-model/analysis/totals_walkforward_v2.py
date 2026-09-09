#!/usr/bin/env python3
"""Evaluate the preregistered NCAA totals V2 challenger without changing live models."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import totals_walkforward as v1  # noqa: E402


DEFAULT_FRAME = ROOT / "data" / "cache" / "backtest_frame_default.parquet"
DEFAULT_JSON = ROOT / "data" / "internal" / "ncaa_totals_walkforward_v2.json"
DEFAULT_MARKDOWN = ROOT / "data" / "internal" / "ncaa_totals_walkforward_v2.md"
TEAM_SHRINKAGE = 20.0

MOVEMENT_CONTEXT = (
    "line_move", "line_move_missing", "market_level", "market_level_squared",
    "abs_spread_close", "indoor", "wind_mph",
)
FOOTBALL_CONTEXT = (
    *v1.FAMILIES["tempo_efficiency"],
    "early_down_efficiency_matchup_sum", "explosive_rate_matchup_sum",
    "red_zone_td_rate_matchup_sum", "starting_field_position_matchup_sum",
    "pass_block_success_matchup_sum", "run_block_success_matchup_sum",
    "abs_qb_delta_gap",
)
FAMILIES = {
    "movement_context": MOVEMENT_CONTEXT,
    "football_context": FOOTBALL_CONTEXT,
    "team_memory": ("team_memory_feature",),
    "combined_v2": tuple(dict.fromkeys(
        (*MOVEMENT_CONTEXT, *FOOTBALL_CONTEXT, "team_memory_feature")
    )),
}
MODELS = ("market", "bias", "tempo_efficiency", *FAMILIES)


def _team_memory(data: pd.DataFrame) -> pd.Series:
    """Pregame team residuals; games at the same kickoff are updated as one batch."""
    ordered = data.sort_values(["kickoff", "game_id"]).copy()
    history: dict[str, list[float]] = {}
    values = pd.Series(index=ordered.index, dtype=float)
    for _kickoff, group in ordered.groupby("kickoff", sort=True):
        for index, row in group.iterrows():
            priors = []
            for team in (str(row.home_team), str(row.away_team)):
                total, count = history.get(team, [0.0, 0.0])
                priors.append(total / (count + TEAM_SHRINKAGE))
            values.at[index] = float(np.mean(priors))
        for row in group.itertuples():
            residual = float(row.market_error)
            for team in (str(row.home_team), str(row.away_team)):
                total, count = history.get(team, [0.0, 0.0])
                history[team] = [total + residual, count + 1.0]
    return values.reindex(data.index)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "home_team", "away_team", "kickoff", "total_open", "indoor", "wind_mph",
        "qb_delta_gap", "early_down_efficiency_matchup_sum",
        "explosive_rate_matchup_sum", "red_zone_td_rate_matchup_sum",
        "starting_field_position_matchup_sum", "pass_block_success_matchup_sum",
        "run_block_success_matchup_sum",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"V2 frame missing columns: {sorted(missing)}")
    out = v1.prepare(frame)
    numeric = required - {"home_team", "away_team", "kickoff"}
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["kickoff"] = pd.to_datetime(out["kickoff"], utc=True, errors="coerce")
    if out["kickoff"].isna().any():
        raise ValueError("V2 requires a known kickoff for causal team memory")
    out["line_move_missing"] = out["total_open"].isna().astype(float)
    out["line_move"] = (out["total_close"] - out["total_open"]).fillna(0.0)
    out["market_level"] = out["total_close"] - 50.0
    out["market_level_squared"] = out["market_level"] ** 2
    out["abs_spread_close"] = out["spread_close"].abs()
    out["abs_qb_delta_gap"] = out["qb_delta_gap"].abs()
    out["team_memory_feature"] = _team_memory(out)
    return out.sort_values(["season", "week", "kickoff", "game_id"]).reset_index(drop=True)


def walkforward_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare(frame)
    results = []
    for season in v1.EVAL_SEASONS:
        train = data[data.season < season]
        test = data[data.season == season].copy()
        if test.empty:
            continue
        test["market"] = test["total_close"]
        test["bias"] = test["total_close"] + v1._predict_ridge(test, (), v1._fit_ridge(train, ()))
        tempo = v1.FAMILIES["tempo_efficiency"]
        test["tempo_efficiency"] = test["total_close"] + v1._predict_ridge(
            test, tempo, v1._fit_ridge(train, tempo)
        )
        for name, features in FAMILIES.items():
            try:
                correction = v1._predict_ridge(test, features, v1._fit_ridge(train, features))
            except ValueError:
                correction = np.full(len(test), np.nan)
            test[name] = test["total_close"] + correction
        results.append(test)
    if not results:
        raise ValueError("no populated V2 evaluation seasons")
    return pd.concat(results, ignore_index=True)


def evaluate(predictions: pd.DataFrame, *, n_boot: int = 5_000) -> list[dict]:
    rows = []
    for offset, model in enumerate(MODELS):
        paired = predictions.dropna(subset=[model, "market", "bias", "actual_total"]).copy()
        error = paired[model].to_numpy(float) - paired.actual_total.to_numpy(float)
        market_error = paired.market.to_numpy(float) - paired.actual_total.to_numpy(float)
        bias_error = paired.bias.to_numpy(float) - paired.actual_total.to_numpy(float)
        blocks = paired.season.astype(str) + "-W" + paired.week.astype(int).astype(str)
        market_gain = v1._bootstrap_gain(market_error, error, blocks, n_boot=n_boot,
                                         seed=20261109 + offset)
        bias_gain = v1._bootstrap_gain(bias_error, error, blocks, n_boot=n_boot,
                                       seed=20261209 + offset)
        season_gains = {}
        for season, group in paired.groupby("season"):
            candidate_rmse = float(np.sqrt(np.mean((group[model] - group.actual_total) ** 2)))
            market_rmse = float(np.sqrt(np.mean((group.market - group.actual_total) ** 2)))
            bias_rmse = float(np.sqrt(np.mean((group.bias - group.actual_total) ** 2)))
            season_gains[str(int(season))] = {
                "vs_market": market_rmse - candidate_rmse,
                "vs_bias": bias_rmse - candidate_rmse,
            }
        coverage = len(paired) / len(predictions)
        positive_seasons = sum(
            value["vs_market"] > 0 and value["vs_bias"] > 0
            for value in season_gains.values()
        )
        eligible = bool(
            model in FAMILIES and coverage >= .95 and market_gain[1] is not None
            and market_gain[1] > 0 and bias_gain[1] > 0 and positive_seasons >= 3
        )
        rows.append({
            "model": model, "n": int(len(paired)), "coverage": coverage,
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias_value": float(np.mean(error)),
            "gain_vs_market": market_gain[0], "gain_vs_market_ci90": list(market_gain[1:]),
            "gain_vs_bias": bias_gain[0], "gain_vs_bias_ci90": list(bias_gain[1:]),
            "season_gains": season_gains, "positive_seasons": int(positive_seasons),
            "eligible_for_new_shadow": eligible,
        })
    return rows


def build_report(frame_path: Path, *, n_boot: int = 5_000) -> dict:
    predictions = walkforward_predictions(pd.read_parquet(frame_path))
    return {
        "schema_version": 1,
        "status": "preregistered_historical_challenger_not_authorized_for_promotion",
        "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        "evaluation_rows": int(len(predictions)),
        "team_shrinkage": TEAM_SHRINKAGE,
        "ridge_alpha": v1.RIDGE_ALPHA,
        "families": {key: list(value) for key, value in FAMILIES.items()},
        "metrics": evaluate(predictions, n_boot=n_boot),
    }


def markdown(report: dict) -> str:
    lines = [
        "# NCAA totals V2 preregistered challenger", "",
        "Historical model-selection evidence only. No live model or pick was changed.", "",
        "| model | n | coverage | RMSE | gain vs market | 90% CI | gain vs bias | 90% CI | positive seasons | shadow eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["metrics"]:
        mci = row["gain_vs_market_ci90"]
        bci = row["gain_vs_bias_ci90"]
        lines.append(
            f"| {row['model']} | {row['n']} | {row['coverage']:.2%} | {row['rmse']:.3f} | "
            f"{row['gain_vs_market']:+.3f} | [{mci[0]:+.3f}, {mci[1]:+.3f}] | "
            f"{row['gain_vs_bias']:+.3f} | [{bci[0]:+.3f}, {bci[1]:+.3f}] | "
            f"{row['positive_seasons']}/4 | {'yes' if row['eligible_for_new_shadow'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("bootstrap must be nonnegative")
    report = build_report(args.frame, n_boot=args.bootstrap)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
