"""Frozen prospective decision rule for the NCAA totals shadow experiment."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


SPEC_VERSION = "T1-shadow-v1"
PRIMARY_HORIZON_HOURS = 1
MIN_GAMES = 300
MIN_WEEKS = 8
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260909


def _rmse(frame: pd.DataFrame, column: str) -> float:
    return float(np.sqrt(np.mean((frame[column].to_numpy(float)
                                  - frame["actual_total"].to_numpy(float)) ** 2)))


def evaluate_totals_shadow(
    graded: pd.DataFrame, *, replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Evaluate one fixed horizon without treating repeated snapshots as new games."""
    base = {
        "spec_version": SPEC_VERSION,
        "primary_horizon_hours": PRIMARY_HORIZON_HOURS,
        "minimum_games": MIN_GAMES,
        "minimum_weeks": MIN_WEEKS,
        "bootstrap_replicates": int(replicates),
        "status": "collecting",
        "eligible_for_review": False,
        "reason": "no eligible prospective grades",
        "n_games": 0,
        "n_weeks": 0,
    }
    required = {
        "league", "game_id", "season", "week", "decision_hours", "actual_total",
        "shadow_spec_version", "shadow_model_version", "shadow_market_total",
        "shadow_bias_total", "shadow_tempo_total",
    }
    if graded is None or graded.empty or not required <= set(graded.columns):
        return base
    frame = graded[
        graded["league"].astype(str).str.lower().eq("ncaa")
        & pd.to_numeric(graded["decision_hours"], errors="coerce").eq(PRIMARY_HORIZON_HOURS)
        & graded["shadow_spec_version"].astype(str).eq(SPEC_VERSION)
    ].copy()
    numeric = ["season", "week", "actual_total", "shadow_market_total",
               "shadow_bias_total", "shadow_tempo_total"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[*numeric, "game_id", "shadow_model_version"])
    if frame.empty:
        return base
    if frame.duplicated(["game_id"]).any():
        return {**base, "status": "invalid", "reason": "duplicate games at the primary horizon"}
    versions = sorted(frame["shadow_model_version"].astype(str).unique())
    if len(versions) != 1:
        return {**base, "status": "invalid", "reason": "multiple shadow model versions",
                "model_versions": versions}
    finite = np.isfinite(frame[numeric].to_numpy(float)).all(axis=1)
    frame = frame.loc[finite].copy()
    if frame.empty:
        return base
    frame["block"] = frame["season"].astype(int).astype(str) + "-" + frame["week"].astype(int).astype(str)
    blocks = sorted(frame["block"].unique())
    result = {
        **base,
        "model_version": versions[0],
        "n_games": int(len(frame)),
        "n_weeks": int(len(blocks)),
        "market_rmse": _rmse(frame, "shadow_market_total"),
        "bias_rmse": _rmse(frame, "shadow_bias_total"),
        "tempo_rmse": _rmse(frame, "shadow_tempo_total"),
    }
    result["tempo_rmse_gain_vs_market"] = result["market_rmse"] - result["tempo_rmse"]
    result["tempo_rmse_gain_vs_bias"] = result["bias_rmse"] - result["tempo_rmse"]
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    members = {block: frame.index[frame["block"].eq(block)].to_numpy() for block in blocks}
    gains_market = np.empty(int(replicates))
    gains_bias = np.empty(int(replicates))
    for i in range(int(replicates)):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        take = np.concatenate([members[block] for block in sampled])
        boot = frame.loc[take]
        tempo = _rmse(boot, "shadow_tempo_total")
        gains_market[i] = _rmse(boot, "shadow_market_total") - tempo
        gains_bias[i] = _rmse(boot, "shadow_bias_total") - tempo
    result["tempo_gain_vs_market_ci90"] = [float(x) for x in np.quantile(gains_market, [.05, .95])]
    result["tempo_gain_vs_bias_ci90"] = [float(x) for x in np.quantile(gains_bias, [.05, .95])]
    enough = len(frame) >= MIN_GAMES and len(blocks) >= MIN_WEEKS
    clears = (result["tempo_gain_vs_market_ci90"][0] > 0
              and result["tempo_gain_vs_bias_ci90"][0] > 0)
    result["eligible_for_review"] = bool(enough and clears)
    if not enough:
        result["reason"] = "minimum game/week evidence has not accumulated"
    elif not clears:
        result["status"] = "not_promoted"
        result["reason"] = "tempo has not cleared both paired 90% intervals"
    else:
        result["status"] = "review_candidate"
        result["reason"] = "tempo cleared the frozen evidence gate; human review is still required"
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite totals shadow result: {key}")
    return result


__all__ = ["evaluate_totals_shadow", "SPEC_VERSION", "PRIMARY_HORIZON_HOURS",
           "MIN_GAMES", "MIN_WEEKS"]
