"""Market handling: the residual-form blend, with a free intercept.

THE INTERCEPT IS NOT OPTIONAL (PATCH 02 §1). With the market coefficient pinned at 1.0 and
no intercept, any constant level bias in a component loads onto its disagreement slope. On
the NFL build a totals projection running +1.43 points high reported b = +0.060, and
de-biasing flipped it to -0.050 -- the apparent weight was entirely the level error.

Rescaling, where used, rescales the COMPONENT about its own mean. Rescaling the
disagreement is a scalar multiple: b and se both scale by 1/c and t is exactly invariant.
The strategic consequence is that shrinkage can never change significance -- only changes
that re-rank which games the model disagrees with the market about can move t.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ._shared import robust_inference
from .config import CACHE_DIR, Config


def american_to_prob(price: float) -> float:
    price = float(price)
    return -price / (-price + 100.0) if price < 0 else 100.0 / (price + 100.0)


def devig_two_way(price_a: int, price_b: int, method: str = "multiplicative") -> tuple:
    pa, pb = american_to_prob(price_a), american_to_prob(price_b)
    total = pa + pb
    if total <= 0:
        raise ValueError("devig_two_way: non-positive implied probability total")
    if method == "multiplicative":
        return pa / total, pb / total
    if method == "power":
        from scipy import optimize

        try:
            k = optimize.brentq(lambda k: pa ** k + pb ** k - 1.0, 0.2, 5.0)
        except ValueError:
            return pa / total, pb / total
        return pa ** k, pb ** k
    raise ValueError(f"unknown devig method {method!r}")


def ols_with_se(X: np.ndarray, y: np.ndarray):
    fit=robust_inference.fit_ols(y,X,covariance="hc3",add_intercept=False)
    return fit.beta,fit.se,fit.r_squared,fit.residual_sd


def rescale_to_market(component: pd.Series, market: pd.Series) -> pd.Series:
    """Rescale the COMPONENT about its own mean -- see the module docstring."""
    c_sd, m_sd = float(component.std()), float(market.std())
    if c_sd <= 0 or m_sd <= 0:
        return component.copy()
    return component.mean() + (component - component.mean()) * (m_sd / c_sd)


@dataclass
class ResidualFit:
    a: float
    se_a: float
    t_a: float
    b: float
    se_b: float
    t_b: float
    r2: float
    resid_sd: float
    n: int
    sd_ratio: float
    rmse: float
    label: str = ""
    covariance_type: str = "hc3"
    n_clusters: int | None = None

    @property
    def unbiased(self) -> bool:
        return abs(self.a) < 0.5


def residual_fit(
    component: pd.Series,
    market: pd.Series,
    actual: pd.Series,
    label: str = "",
    sd_match: bool = False,
    groups: "pd.Series | None" = None,
) -> ResidualFit:
    """actual - market = a + b * (component - market), with a free intercept."""
    columns={"component":component,"market":market,"actual":actual}
    if groups is not None: columns["cluster"]=groups
    df=pd.concat(columns,axis=1).dropna(); comp,mkt,act=df["component"],df["market"],df["actual"]
    if sd_match:
        comp = rescale_to_market(comp, mkt)
    y = (act - mkt).to_numpy(float)
    fit=robust_inference.fit_ols(y,(comp-mkt).to_numpy(float),covariance="cluster" if groups is not None else "hc3",groups=df["cluster"].tolist() if groups is not None else None)
    beta,se=fit.beta,fit.se
    return ResidualFit(
        a=float(beta[0]), se_a=float(se[0]),
        t_a=float(beta[0] / se[0]) if se[0] > 0 else 0.0,
        b=float(beta[1]), se_b=float(se[1]),
        t_b=float(beta[1] / se[1]) if se[1] > 0 else 0.0,
        r2=fit.r_squared, resid_sd=fit.residual_sd, n=len(y),
        sd_ratio=float(comp.std() / mkt.std()) if mkt.std() else float("nan"),
        rmse=float(np.sqrt(((comp - act) ** 2).mean())),
        label=label,
        covariance_type=fit.covariance_type,n_clusters=fit.n_clusters,
    )

def season_week_groups(frame):
    if not {"season","week"}<=set(frame.columns): return None
    return pd.Series(list(zip(frame["season"],frame["week"])),index=frame.index,dtype=object,name="season_week")


@dataclass
class BlendWeights:
    b_model_spread: float = 0.0
    t_model_spread: float = 0.0
    a_spread: float = 0.0
    b_model_total: float = 0.0
    t_model_total: float = 0.0
    a_total: float = 0.0
    n_spread: int = 0
    n_total: int = 0
    window: str = ""
    clamped: list = field(default_factory=list)

    @property
    def informative(self) -> bool:
        return max(abs(self.t_model_spread), abs(self.t_model_total)) > 2.0

    def save(self, path=None) -> None:
        (path or CACHE_DIR / "blend_weights.json").write_text(
            json.dumps(asdict(self), indent=2)
        )

    @staticmethod
    def load(path=None):
        p = path or CACHE_DIR / "blend_weights.json"
        return BlendWeights(**json.loads(p.read_text())) if p.exists() else None


def fit_blend(
    frame: pd.DataFrame, cfg: Config, window: str = "", grade: str = "open"
) -> BlendWeights:
    """Fit spreads and totals separately, against the OPENER by default."""
    cap = cfg.market.model_weight_cap
    clamped = []
    groups=season_week_groups(frame)
    fs = residual_fit(
        frame["model_spread"], frame[f"spread_{grade}"], frame["actual_margin"], "spread", groups=groups
    )
    ft = residual_fit(
        frame["model_total"], frame[f"total_{grade}"], frame["actual_total"], "total", groups=groups
    )
    for f in (fs, ft):
        if not 0.0 <= f.b <= cap:
            clamped.append(f"{f.label} b={f.b:.3f} -> {float(np.clip(f.b, 0, cap)):.3f}")

    return BlendWeights(
        b_model_spread=float(np.clip(fs.b, 0.0, cap)), t_model_spread=fs.t_b,
        a_spread=fs.a, n_spread=fs.n,
        b_model_total=float(np.clip(ft.b, 0.0, cap)), t_model_total=ft.t_b,
        a_total=ft.a, n_total=ft.n,
        window=window, clamped=clamped,
    )


def clv(frame: pd.DataFrame, side_col: str, grade: str = "open") -> float:
    """Share of positions where the close moved toward the model's side.

    The single most informative number available: it says whether the market agreed with
    the model after the model committed, which is what an edge against the opener means.
    """
    move = frame["spread_close"] - frame[f"spread_{grade}"]
    toward = np.sign(move) == np.sign(frame[side_col])
    valid = frame["spread_close"].notna() & frame[f"spread_{grade}"].notna() & (move != 0)
    return float(toward[valid].mean()) if valid.any() else float("nan")
