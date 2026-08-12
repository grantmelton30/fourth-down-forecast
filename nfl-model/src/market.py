"""Market handling (§7): devigging, the blend, key numbers, and line acquisition.

`fit_blend_weights` is the single most important function in the build. It is what stops
the model output from being used raw (NON-NEGOTIABLE #2).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import pandas as pd
import requests
from scipy import optimize

from ._shared import robust_inference
from .config import CACHE_DIR, MANUAL_DIR, Config

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
MANUAL_LINES_COLUMNS = [
    "away_team", "home_team", "spread_home", "total", "spread_price_home",
    "spread_price_away", "total_price_over", "total_price_under",
]


# --------------------------------------------------------------------------------------
# 7a. Devig
# --------------------------------------------------------------------------------------

def american_to_prob(price: float) -> float:
    """American odds -> implied probability, vig included."""
    price = float(price)
    return -price / (-price + 100.0) if price < 0 else 100.0 / (price + 100.0)


def american_to_decimal(price: float) -> float:
    price = float(price)
    return 1.0 + (100.0 / -price if price < 0 else price / 100.0)


def devig_two_way(price_a: int, price_b: int, method: str = "multiplicative") -> tuple:
    """Strip the vig from a two-way market.

    `multiplicative` is the default for spreads and totals: they are priced near
    -110/-110, the book sum is only about 1.048, and all three methods agree to within a
    fraction of a percent. The other two exist for lopsided markets where
    favorite-longshot bias is real -- do not use multiplicative on a -2000 moneyline.
    """
    pa, pb = american_to_prob(price_a), american_to_prob(price_b)
    total = pa + pb
    if total <= 0:
        raise ValueError("devig_two_way: non-positive implied probability total")

    if method == "multiplicative":
        return pa / total, pb / total

    if method == "power":
        # Solve for k such that pa^k + pb^k = 1.
        def f(k: float) -> float:
            return pa ** k + pb ** k - 1.0

        try:
            k = optimize.brentq(f, 0.2, 5.0)
        except ValueError:
            return pa / total, pb / total
        return pa ** k, pb ** k

    if method == "shin":
        # Solve for the informed-money proportion z.
        def adjust(p: float, z: float) -> float:
            return (np.sqrt(z ** 2 + 4 * (1 - z) * p ** 2 / total) - z) / (2 * (1 - z))

        def g(z: float) -> float:
            return adjust(pa, z) + adjust(pb, z) - 1.0

        try:
            z = optimize.brentq(g, 1e-6, 0.35)
        except ValueError:
            return pa / total, pb / total
        out = [adjust(pa, z), adjust(pb, z)]
        s = sum(out)
        return out[0] / s, out[1] / s

    raise ValueError(f"devig_two_way: unknown method {method!r}")


# --------------------------------------------------------------------------------------
# 7c. The blend
# --------------------------------------------------------------------------------------

@dataclass
class BlendWeights:
    """Fitted disagreement weights, plus everything needed to judge them."""

    b_model: float
    se_model: float
    t_model: float
    r_squared: float
    resid_sd: float
    n: int
    schema_version: int = 2
    a_spread: float = 0.0
    se_a_spread: float = 0.0
    t_a_spread: float = 0.0
    b_model_raw: float = 0.0
    b_model_total: float = 0.0
    se_model_total: float = 0.0
    t_model_total: float = 0.0
    a_total: float = 0.0
    se_a_total: float = 0.0
    t_a_total: float = 0.0
    b_model_total_raw: float = 0.0
    resid_sd_total: float = 0.0
    n_total: int = 0
    covariance_type_spread: str = "hc3"
    n_clusters_spread: "int | None" = None
    covariance_type_total: str = "hc3"
    n_clusters_total: "int | None" = None
    clamped: list = field(default_factory=list)

    @property
    def informative(self) -> bool:
        """GATE_BLEND_INFORMATIVE: the disagreement term carries real signal."""
        return ((self.b_model_raw>0 and self.b_model>0 and self.t_model>2) or
                (self.b_model_total_raw>0 and self.b_model_total>0 and self.t_model_total>2))

    def save(self, path=None) -> None:
        path = path or CACHE_DIR / "blend_weights.json"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path=None) -> "BlendWeights | None":
        path = path or CACHE_DIR / "blend_weights.json"
        if not path.exists():
            return None
        try: raw=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError): return None
        if raw.get("schema_version")!=2: return None
        required={"a_spread","se_a_spread","t_a_spread","b_model_raw","b_model","t_model","a_total","se_a_total","t_a_total","b_model_total_raw","b_model_total","t_model_total"}
        if required-set(raw): return None
        # Ignore keys this class no longer has. An artifact written before the nfelo
        # blend term was removed still carries b_nfelo / se_nfelo / t_nfelo, and a bare
        # BlendWeights(**raw) would raise on it.
        known = {f.name for f in fields(BlendWeights)}
        try: return BlendWeights(**{k:v for k,v in raw.items() if k in known})
        except TypeError: return None


def _ols_with_se(X: np.ndarray, y: np.ndarray):
    """Plain OLS returning coefficients, standard errors, R^2 and residual sd."""
    fit=robust_inference.fit_ols(y,X,covariance="hc3",add_intercept=False)
    return fit.beta,fit.se,fit.r_squared,fit.residual_sd

def _residual_fit(frame,model,market,actual):
    cols=[model,market,actual]; grouped={"season","week"}<=set(frame.columns)
    if grouped: cols += ["season","week"]
    df=frame[cols].dropna(); groups=list(zip(df["season"],df["week"])) if grouped else None
    fit=robust_inference.fit_ols((df[actual]-df[market]).to_numpy(float),(df[model]-df[market]).to_numpy(float),covariance="cluster" if grouped else "hc3",groups=groups)
    return df,fit


def fit_blend_weights(backtest_frame: pd.DataFrame, cfg: Config) -> BlendWeights:
    """Fit the blend in residual form (§7c).

    The response is `actual_margin - market_spread` and the regressors are the two
    disagreement terms, which is how the market coefficient is constrained to exactly 1.0.
    That constraint is the right prior -- the market is close to unbiased, so what is
    worth estimating is deviation from it, not the market itself.

    Residual form is not cosmetic. `model_spread` and `market_spread` correlate above
    0.85; differencing against the market drops the correlation between regressors to
    roughly 0.2, which is what makes the t-statistics in GATE_BLEND_INFORMATIVE
    trustworthy. It matters more with nfelo in the model, because nfelo regresses to
    market by design and is therefore collinear with it.
    """
    df,fit=_residual_fit(backtest_frame,"model_spread","market_spread","actual_margin")
    a_spread,b_model=map(float,fit.beta); se_a_spread,se_model=map(float,fit.se)
    t_a_spread=a_spread/se_a_spread if se_a_spread>0 else 0.0
    t_model = b_model / se_model if se_model > 0 else 0.0

    # Clamp to [0, cap]. A negative fitted weight means the model is anti-predictive on
    # this sample, which is almost always a sign error or a lookahead leak rather than a
    # real finding -- investigate it, do not bet it.
    cap = cfg.market.model_weight_cap
    clamped = []
    if not 0.0 <= b_model <= cap:
        clamped.append(f"b_model {b_model:.3f} -> {float(np.clip(b_model, 0.0, cap)):.3f}")

    totals = _fit_total_blend(backtest_frame, cap)
    total_raw=totals.get("b_model_total_raw")
    if total_raw is not None and not 0<=total_raw<=cap: clamped.append(f"b_model_total {total_raw:.3f} -> {float(np.clip(total_raw,0,cap)):.3f}")
    return BlendWeights(
        b_model=float(np.clip(b_model, 0.0, cap)),
        b_model_raw=b_model,a_spread=a_spread,se_a_spread=se_a_spread,t_a_spread=t_a_spread,
        se_model=se_model,
        t_model=t_model,
        r_squared=fit.r_squared,resid_sd=fit.residual_sd,n=len(df),
        covariance_type_spread=fit.covariance_type,n_clusters_spread=fit.n_clusters,
        clamped=clamped,
        **totals,
    )


def _fit_total_blend(df: pd.DataFrame, cap: float) -> dict:
    """The same regression for totals. nfelo publishes no totals model, so this has one
    term."""
    need = {"model_total", "market_total", "actual_total"}
    if not need <= set(df.columns):
        return {}
    sub = df.dropna(subset=list(need))
    if len(sub) < 50:
        return {}
    sub,fit=_residual_fit(df,"model_total","market_total","actual_total")
    a,b=map(float,fit.beta); se_a,s=map(float,fit.se)
    return {
        "a_total":a,"se_a_total":se_a,"t_a_total":a/se_a if se_a>0 else 0.0,
        "b_model_total_raw":b,
        "b_model_total": float(np.clip(b, 0.0, cap)),
        "se_model_total": s,
        "t_model_total": b / s if s > 0 else 0.0,
        "resid_sd_total":fit.residual_sd,"n_total":len(sub),"covariance_type_total":fit.covariance_type,"n_clusters_total":fit.n_clusters,
    }


def apply_blend(
    market_spread: float, model_spread: float, weights: BlendWeights
) -> float:
    """blended = market + b_model * (model - market)."""
    return market_spread+weights.a_spread+weights.b_model*(model_spread-market_spread)


def apply_blend_total(
    market_total: float, model_total: float, weights: BlendWeights
) -> float:
    return market_total+weights.a_total+weights.b_model_total*(model_total-market_total)


# --------------------------------------------------------------------------------------
# 7d. Key numbers
# --------------------------------------------------------------------------------------

KEY_NUMBERS = (3, 7, 6, 10, 14, 4)


def key_number_value(model_line: float, market_line: float, pmf: dict) -> float:
    """How much of an edge comes from crossing a key number rather than a smooth shift.

    Returns the simulated probability mass sitting strictly between the two lines. Moving
    from -3.5 to -2 crosses the 3 spike and is worth far more than the same 1.5 points
    from -8.5 to -7. Surfaced as `key_cross` and used as a ranking tiebreaker, never as a
    multiplier on stake.
    """
    lo, hi = sorted((float(model_line), float(market_line)))
    return float(sum(p for m, p in pmf.items() if lo < m < hi))


# --------------------------------------------------------------------------------------
# 7b. Line acquisition
# --------------------------------------------------------------------------------------

def acquire_lines(schedules_week: pd.DataFrame, cfg: Config, refresh: bool = False):
    """Live first, manual CSV second. Returns (frame, source).

    The manual path is not a degraded one: the numbers typed off your own app are the
    numbers you can actually bet, which beats a consensus feed containing books you have
    no account at.
    """
    live = _fetch_odds_api(refresh)
    parsed = _parse_odds_api(live, schedules_week) if live else None
    if parsed is not None and len(parsed):
        return parsed, "the-odds-api"
    manual = _read_manual_lines()
    if manual is not None and len(manual):
        return manual, "manual-csv"
    path = write_manual_template(schedules_week)
    raise RuntimeError(
        f"market.acquire_lines: no live odds and no usable manual file. A pre-filled "
        f"template has been written to {path} -- fill in the numbers and re-run."
    )


def _fetch_odds_api(refresh: bool):
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return None
    try:
        resp = requests.get(
            ODDS_API_URL,
            params={
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "apiKey": key,
            },
            timeout=25,
        )
        if resp.status_code == 403:
            print("  odds api: 403 -- your tier does not include NFL. Falling back.")
            return None
        resp.raise_for_status()
        payload = resp.json()
        (CACHE_DIR / "odds_api_latest.json").write_text(
            json.dumps({"fetched": str(pd.Timestamp.now()), "data": payload})
        )
        return payload
    except Exception as exc:  # noqa: BLE001 - fall through to the manual path
        print(f"  odds api: {exc}. Falling back to manual lines.")
        return None


def _parse_odds_api(payload, schedules_week: pd.DataFrame):
    rows = []
    for event in payload or []:
        book = next(iter(event.get("bookmakers", [])), None)
        if not book:
            continue
        rec = {"home_raw": event.get("home_team"), "away_raw": event.get("away_team")}
        for market in book.get("markets", []):
            for oc in market.get("outcomes", []):
                if market["key"] == "spreads" and oc["name"] == event.get("home_team"):
                    rec["spread_home"] = -float(oc["point"])
                    rec["spread_price_home"] = oc.get("price")
                elif market["key"] == "spreads":
                    rec["spread_price_away"] = oc.get("price")
                elif market["key"] == "totals" and str(oc["name"]).lower() == "over":
                    rec["total"] = float(oc["point"])
                    rec["total_price_over"] = oc.get("price")
                elif market["key"] == "totals":
                    rec["total_price_under"] = oc.get("price")
        rows.append(rec)
    if not rows:
        return None

    df = pd.DataFrame(rows)

    # The Odds API uses full team names; match them onto abbreviations via the slate.
    def match(name):
        if not isinstance(name, str):
            return None
        token = name.split()[-1].upper()
        for _, g in schedules_week.iterrows():
            for abbr in (g["home_team"], g["away_team"]):
                if token.startswith(abbr[:3]) or abbr[:3] in token:
                    return abbr
        return None

    df["home_team"] = df["home_raw"].map(match)
    df["away_team"] = df["away_raw"].map(match)
    df = df.dropna(subset=["home_team", "away_team"])
    return df.drop(columns=["home_raw", "away_raw"], errors="ignore")


def _read_manual_lines():
    path = MANUAL_DIR / "this_week_lines.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    missing = [c for c in MANUAL_LINES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"this_week_lines.csv missing columns {missing}")
    return df.dropna(subset=["spread_home", "total"])


def write_manual_template(schedules_week: pd.DataFrame):
    """Pre-fill matchups with blank numbers so the fallback is one paste away."""
    path = MANUAL_DIR / "this_week_lines.csv"
    pd.DataFrame({
        "away_team": schedules_week["away_team"],
        "home_team": schedules_week["home_team"],
        "spread_home": np.nan,
        "total": np.nan,
        "spread_price_home": -110,
        "spread_price_away": -110,
        "total_price_over": -110,
        "total_price_under": -110,
    }).to_csv(path, index=False)
    return path
