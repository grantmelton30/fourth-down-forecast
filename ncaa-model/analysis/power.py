"""Power arithmetic for the residual blend regression.

    actual - market = a + b * (component - market)

`b` is the only quantity this build bets on, so the only question that matters about any
sample is: **if the effect were real, could this sample have seen it?** Nothing here fits a
model or changes one. It converts a standard error into the three numbers that make a null
readable.

Why this module exists
----------------------
The build has twice stated a claim of the form "the effect is absent in subgroup X" off a
subsample that could not have detected the effect had it been there at full strength.

- `DECISIONS.md` D6d: "the signal is absent from week 13 on", read off `b_late = -0.146`
  against `b_early = +0.242`. The difference test gives t = 1.53, and at n = 254 the
  subsample required `b >= 0.471` to reach t = 2 — roughly twice the effect ever claimed.
  The partition was underpowered by construction. The claim was retracted.
- `README.md`, the 2023-2025 window read as a veto on 2021-2025. Same shape: `b` moved
  0.262 -> 0.243 while the standard error moved 0.092 -> 0.127. One finding at two sample
  sizes, reported as a disagreement.

Both were caught late, by hand, after the claim had been written down. The standing rule is
now that the power calculation happens *before* the claim, and this module is what makes
that cheap enough that there is no excuse for skipping it.

The distinction the verdicts enforce
------------------------------------
A measured `b` near zero means one of two very different things, and they are not
distinguishable without `mde`:

- **NULL** — the sample could have detected a real effect of the reference size and did
  not. This is evidence of absence, and it is a result worth recording.
- **UNDERPOWERED** — the sample could not have detected that effect either way. This is
  the absence of evidence, and it supports no claim in either direction.

`Power.verdict` never returns "no effect" for the second case. That is the entire point.

Standard errors are asymptotic and assume independent observations. Games within a season
share ratings, so the true se is somewhat larger than reported and every MDE here is
therefore mildly optimistic — the errors run in the direction of claiming more power than
the sample has, which is the safe direction to be wrong for a module whose job is to block
claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.market import residual_fit

# The effect size a null is judged against. `b` around 0.20-0.26 is what this build claimed
# on NCAA totals before the close-anchored test collapsed it (Appendix A: +0.195 opener,
# +0.076 close). A null is only meaningful against a stated alternative, so that claimed
# value is the default alternative. Override it deliberately; do not tune it.
REFERENCE_B = 0.20

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


@dataclass(frozen=True)
class Power:
    """Everything derivable about what a sample of this size could have seen."""

    label: str
    n: int
    b: float
    se_b: float
    t_b: float
    disagreement_sd: float
    resid_sd: float
    reference_b: float = REFERENCE_B
    alpha: float = DEFAULT_ALPHA
    target_power: float = DEFAULT_POWER

    @property
    def _z_alpha(self) -> float:
        return float(stats.norm.ppf(1.0 - self.alpha / 2.0))

    @property
    def _z_power(self) -> float:
        return float(stats.norm.ppf(self.target_power))

    @property
    def required_b(self) -> float:
        """The |b| this sample would have to *observe* to reach significance.

        Not the same as `mde`. This is the reporting threshold — what the point estimate
        must exceed for the t-statistic to clear `alpha`. A true effect exactly this size
        is detected only half the time, which is why `mde` is larger.
        """
        return self._z_alpha * self.se_b

    @property
    def mde(self) -> float:
        """Minimum detectable effect: the smallest TRUE b found with `target_power`."""
        return (self._z_alpha + self._z_power) * self.se_b

    def power_at(self, b_true: "float | None" = None) -> float:
        """Probability this sample rejects the null when the true effect is `b_true`."""
        b_true = self.reference_b if b_true is None else b_true
        if self.se_b <= 0:
            return float("nan")
        lam = abs(b_true) / self.se_b
        return float(
            stats.norm.cdf(lam - self._z_alpha) + stats.norm.cdf(-lam - self._z_alpha)
        )

    def required_n(self, b_true: "float | None" = None) -> int:
        """Sample size at which `b_true` becomes detectable with `target_power`.

        `se_b` scales as 1/sqrt(n), so this is `n * (mde / b_true)^2`.
        """
        b_true = self.reference_b if b_true is None else b_true
        if b_true == 0:
            return -1
        return int(np.ceil(self.n * (self.mde / abs(b_true)) ** 2))

    @property
    def adequately_powered(self) -> bool:
        return self.power_at() >= self.target_power

    @property
    def verdict(self) -> str:
        if abs(self.t_b) >= self._z_alpha:
            return "DETECTED"
        if self.adequately_powered:
            return "NULL (adequately powered)"
        return "UNDERPOWERED — no claim permitted"

    def report(self) -> str:
        return "\n".join([
            f"{self.label}",
            f"  n                    {self.n}",
            f"  b observed           {self.b:+.4f}   (se {self.se_b:.4f}, "
            f"t {self.t_b:+.2f})",
            f"  required |b| at t={self._z_alpha:.2f}  {self.required_b:.4f}",
            f"  MDE at {self.target_power:.0%} power     {self.mde:.4f}",
            f"  power at b={self.reference_b:.2f}      {self.power_at():.1%}",
            f"  n needed for b={self.reference_b:.2f}   {self.required_n():,}"
            f"   ({self.required_n() / max(self.n, 1):.1f}x this sample)",
            f"  VERDICT              {self.verdict}",
        ])


@dataclass(frozen=True)
class Difference:
    """A between-subgroup comparison of `b`, with its own power calculation.

    Required before any claim that an effect is present in one subgroup and absent in
    another. Reading two point estimates side by side is not a test: the quantity under
    test is the difference, and its standard error is larger than either component's.

    Assumes the two subsamples are disjoint, which is what a partition gives.
    """

    label_a: str
    label_b: str
    a: Power
    b: Power
    alpha: float = DEFAULT_ALPHA
    target_power: float = DEFAULT_POWER

    @property
    def diff(self) -> float:
        return self.a.b - self.b.b

    @property
    def se_diff(self) -> float:
        return float(np.hypot(self.a.se_b, self.b.se_b))

    @property
    def t_diff(self) -> float:
        return self.diff / self.se_diff if self.se_diff > 0 else 0.0

    @property
    def mde_diff(self) -> float:
        z_a = float(stats.norm.ppf(1.0 - self.alpha / 2.0))
        z_p = float(stats.norm.ppf(self.target_power))
        return (z_a + z_p) * self.se_diff

    @property
    def ci(self) -> tuple:
        z = float(stats.norm.ppf(1.0 - self.alpha / 2.0))
        return (self.diff - z * self.se_diff, self.diff + z * self.se_diff)

    @property
    def verdict(self) -> str:
        """A difference is only "absent" if a reference-sized difference was detectable.

        The reference difference is the larger subgroup's own reference effect: claiming
        the effect vanishes in one subgroup is claiming a difference of at least that
        size, so that is what the sample must have been able to see.
        """
        z = float(stats.norm.ppf(1.0 - self.alpha / 2.0))
        if abs(self.t_diff) >= z:
            return "SUBGROUPS DIFFER"
        if self.mde_diff <= self.a.reference_b:
            return "NO DIFFERENCE DETECTED (adequately powered)"
        return "UNDERPOWERED — the subgroups cannot be distinguished at this n"

    def report(self) -> str:
        lo, hi = self.ci
        return "\n".join([
            f"DIFFERENCE  {self.label_a}  vs  {self.label_b}",
            f"  b            {self.a.b:+.4f} (n={self.a.n})   "
            f"{self.b.b:+.4f} (n={self.b.n})",
            f"  difference   {self.diff:+.4f}   se {self.se_diff:.4f}   "
            f"t {self.t_diff:+.2f}",
            f"  {100 * (1 - self.alpha):.0f}% CI      [{lo:+.4f}, {hi:+.4f}]",
            f"  MDE on diff  {self.mde_diff:.4f}   "
            f"(need <= {self.a.reference_b:.2f} to call absence)",
            f"  VERDICT      {self.verdict}",
        ])


def analyse(
    component: pd.Series,
    market: pd.Series,
    actual: pd.Series,
    label: str = "",
    reference_b: float = REFERENCE_B,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_POWER,
) -> Power:
    """Fit the residual regression and wrap it in its power calculation.

    The fit is `src.market.residual_fit`, not a reimplementation, so the power statement
    always describes exactly the regression the gates read.
    """
    fit = residual_fit(component, market, actual, label=label)
    df = pd.concat([component, market], axis=1).dropna()
    disagreement_sd = float((df.iloc[:, 0] - df.iloc[:, 1]).std())
    return Power(
        label=label or fit.label,
        n=fit.n,
        b=fit.b,
        se_b=fit.se_b,
        t_b=fit.t_b,
        disagreement_sd=disagreement_sd,
        resid_sd=fit.resid_sd,
        reference_b=reference_b,
        alpha=alpha,
        target_power=target_power,
    )


def analyse_frame(
    frame: pd.DataFrame,
    model_col: str = "model_total",
    market_col: str = "total_close",
    actual_col: str = "actual_total",
    label: str = "",
    **kw,
) -> Power:
    """`analyse` for the column names this repo uses.

    `market_col` defaults to the CLOSE. Appendix A established that anchoring the
    measurement on the opener correlates the regressor and the response through the
    opener's own error, which manufactured the entire apparent edge (t = +2.23 opener,
    +0.89 close, on an identical model). The opener remains the correct *execution*
    number and the wrong *measurement* anchor; passing `total_open` here is a deliberate
    act that must be justified at the call site.
    """
    return analyse(
        frame[model_col], frame[market_col], frame[actual_col],
        label=label or f"{model_col} vs {market_col}", **kw,
    )


def compare(
    frame: pd.DataFrame,
    mask_a: pd.Series,
    mask_b: pd.Series,
    label_a: str,
    label_b: str,
    **kw,
) -> Difference:
    """Power-checked subgroup comparison. Use this instead of reading two `b`s."""
    return Difference(
        label_a=label_a,
        label_b=label_b,
        a=analyse_frame(frame[mask_a], label=label_a, **kw),
        b=analyse_frame(frame[mask_b], label=label_b, **kw),
    )


def main() -> int:
    """Report power for the standing windows, and re-test the two retracted claims."""
    from src.config import CACHE_DIR

    path = CACHE_DIR / "backtest_frame_default.parquet"
    if not path.exists():
        print(f"no cached backtest frame at {path}; run run_backtest.py first")
        return 1

    frame = pd.read_parquet(path)
    cols = ["model_total", "total_open", "total_close", "actual_total"]
    base = frame[frame["has_opener"] & frame["season"].between(2021, 2025)]
    graded = base[base["restricted"]].dropna(subset=cols)

    rule = "=" * 78
    print(rule)
    print("POWER — NCAA totals, restricted universe, 2021-2025")
    print(rule)

    for anchor in ("total_close", "total_open"):
        print()
        print(analyse_frame(graded, market_col=anchor,
                            label=f"totals, anchored on {anchor}").report())

    # THE THREE DECLARED WINDOWS, CLOSE-ANCHORED.
    #
    # This is not a new slice. DECISIONS.md D2a names these three windows and reports all
    # three, but reports them anchored on the OPENER — the anchor Appendix A then proved
    # invalid as a measurement. Recomputing an already-declared window on the corrected
    # anchor is finishing that correction, not searching for a friendlier subsample. All
    # three are reported whatever they say, which is the condition that makes it legitimate.
    #
    # It matters because the restricted window is underpowered against its own reference
    # effect (MDE 0.239 vs a claimed 0.20), so "no edge" there is not yet a settled null.
    print()
    print(rule)
    print("THE THREE DECLARED WINDOWS (D2a), CLOSE-ANCHORED")
    print(rule)
    windows = [
        ("2021-2025 restricted", graded),
        ("2023-2025 restricted", graded[graded["season"] >= 2023]),
        ("2021-2025 full FBS", base[base["fbs_only"]].dropna(subset=cols)),
    ]
    for label, sub in windows:
        print()
        print(analyse_frame(sub, market_col="total_close", label=label).report())

    print()
    print(rule)
    print("THE TWO RETRACTED CLAIMS, re-tested")
    print(rule)
    print()
    print(compare(
        graded, graded["week"] < 13, graded["week"] >= 13,
        "weeks 4-12", "weeks 13+",
    ).report())
    print()
    print(compare(
        graded, graded["season"] <= 2022, graded["season"] >= 2023,
        "2021-2022", "2023-2025",
    ).report())
    print()
    print(rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
