"""L3 -- rank-preserving quantile calibration (PATCH 01 §2).

The blended line says where the distribution sits. This layer says what shape it has, by
mapping onto the empirical conditional distribution of actual outcomes given a line.

Why a quantile map rather than a better simulator: an independent-possession simulator
cannot reproduce the spikes in a football scoring distribution at any level of refinement,
because they come from teams playing to the scoreboard rather than from the scoring
process. That was established on the NFL build. The map achieves the same shape by
construction, in roughly forty lines, and is verifiable.

Rank-preserving matters. The projection's game-specific ordering is the part that carries
information; only the shape is corrected.

NO LOOKAHEAD: the empirical distribution is built strictly from seasons before
`as_of_season`, and `build_conditional` refuses to do otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConditionalPMF:
    """Empirical distribution of an outcome, bucketed by the line."""

    centers: np.ndarray          # bucket centers (the line)
    supports: list               # per-bucket sorted outcome values
    cdfs: list                   # per-bucket cumulative probabilities
    kind: str
    n_by_bucket: np.ndarray
    as_of_season: int

    def _bucket(self, line: float) -> int:
        return int(np.argmin(np.abs(self.centers - float(line))))

    def support_and_cdf(self, line: float):
        b = self._bucket(line)
        return self.supports[b], self.cdfs[b]

    def quantiles(self, line: float, q: np.ndarray) -> np.ndarray:
        v, cdf = self.support_and_cdf(line)
        return v[np.searchsorted(cdf, q).clip(0, len(v) - 1)]

    def mean(self, line: float) -> float:
        v, cdf = self.support_and_cdf(line)
        pmf = np.diff(np.concatenate([[0.0], cdf]))
        return float((v * pmf).sum())

    def prob_over(self, line: float, threshold: float) -> float:
        """P(outcome > threshold | line), pushes excluded so the two sides sum to 1."""
        v, cdf = self.support_and_cdf(line)
        pmf = np.diff(np.concatenate([[0.0], cdf]))
        over = float(pmf[v > threshold].sum())
        under = float(pmf[v < threshold].sum())
        denom = over + under
        return over / denom if denom > 0 else 0.5


def build_conditional(
    history: pd.DataFrame,
    line_col: str,
    outcome_col: str,
    as_of_season: int,
    kind: str = "total",
    bucket_width: float = 0.5,
    pool: float = 1.5,
    min_bucket: int = 150,
) -> ConditionalPMF:
    """Bucket by line in half-point bins, pooling +/- `pool` points to keep counts healthy.

    Only seasons strictly before `as_of_season` are used -- the same no-lookahead rule the
    ratings obey, applied to the calibration layer as PATCH 01 §2 requires.
    """
    past = history[
        (history["season"] < as_of_season)
        & history[line_col].notna()
        & history[outcome_col].notna()
    ]
    if len(past) < 500:
        raise ValueError(
            f"build_conditional: only {len(past)} games before {as_of_season}; "
            "not enough to estimate a conditional distribution."
        )

    lines = past[line_col].to_numpy(float)
    outcomes = past[outcome_col].to_numpy(float)
    lo, hi = np.percentile(lines, [1, 99])
    centers = np.arange(
        np.floor(lo / bucket_width) * bucket_width,
        np.ceil(hi / bucket_width) * bucket_width + bucket_width,
        bucket_width,
    )

    supports, cdfs, counts = [], [], []
    for c in centers:
        sel = np.abs(lines - c) <= pool
        vals = outcomes[sel]
        if len(vals) < min_bucket:
            # Widen to the nearest games rather than emit a PMF built on a handful.
            order = np.argsort(np.abs(lines - c))
            vals = outcomes[order[:min_bucket]]
        v, cnt = np.unique(vals, return_counts=True)
        supports.append(v)
        cdfs.append(np.cumsum(cnt / cnt.sum()))
        counts.append(len(vals))

    return ConditionalPMF(
        centers=centers, supports=supports, cdfs=cdfs, kind=kind,
        n_by_bucket=np.array(counts), as_of_season=as_of_season,
    )


def calibrate_draws(draws: np.ndarray, line: float, pmf: ConditionalPMF) -> np.ndarray:
    """Map draws onto the empirical conditional distribution, preserving rank order."""
    draws = np.asarray(draws, dtype=float)
    q = (np.argsort(np.argsort(draws)) + 0.5) / len(draws)
    return pmf.quantiles(line, q)


def calibrated_prob(line: float, threshold: float, pmf: ConditionalPMF) -> float:
    """P(outcome > threshold) read straight off the calibrated distribution.

    The bet sheet needs a probability, not a sample, so this reads the PMF directly rather
    than drawing and counting -- same answer, no Monte Carlo error.
    """
    return pmf.prob_over(line, threshold)


def validate(
    pmf: ConditionalPMF, history: pd.DataFrame, line_col: str,
    outcome_col: str, season: int,
) -> pd.DataFrame:
    """Compare calibrated means against realized outcomes, by line decile.

    A calibration layer that shifts the level is broken; this checks that it only reshapes.
    """
    cur = history[
        (history["season"] == season)
        & history[line_col].notna()
        & history[outcome_col].notna()
    ].copy()
    if cur.empty:
        return pd.DataFrame()
    cur["calibrated_mean"] = [pmf.mean(x) for x in cur[line_col]]
    cur["decile"] = pd.qcut(cur[line_col], 10, duplicates="drop")
    return cur.groupby("decile", observed=True).agg(
        n=(outcome_col, "size"),
        line=(line_col, "mean"),
        calibrated=("calibrated_mean", "mean"),
        actual=(outcome_col, "mean"),
    ).reset_index()
