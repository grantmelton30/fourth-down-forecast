"""Measure the wind effect on college totals, for `context.wind_total_*`.

    NCAA_MODEL_CACHE_DIR=~/.cache/ncaa-model python -m analysis.measure_wind

Reads only the cached Open-Meteo readings in `weather.parquet`; it makes no API calls of
its own. Warm that cache first by running weather over a slate with `allow_network=True`.

WHAT IS BEING ESTIMATED, AND WHAT IS NOT. The question is physical -- does wind suppress
scoring -- not whether the market misprices it. Those need different regressions and give
different answers, and conflating them is how a projection model turns into a fake edge.
So the total is regressed on the model's own totals predictors (pace and efficiency, the
same two `backtest.project_walkforward` uses) PLUS a wind term, and the wind coefficient is
read off that. Regressing the market residual on wind instead would answer the other
question and must not be used to set this constant.

Wind enters as a hinge, `max(0, wind - threshold)`, because the effect is not linear from
zero: a 5 mph breeze does nothing. The threshold is swept rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import ingest, venues
from src.cfbd_client import BudgetedCFBD
from src.config import CACHE_DIR, load_config
from src.ratings import net_epa_vec, walkforward_path
from src.weather import WEATHER_COLUMNS

EXCLUDE_SEASONS = (2020,)


def _attach_cached_weather(games: pd.DataFrame) -> pd.DataFrame:
    """Join the nearest-to-kickoff cached reading onto each game. Cache-only."""
    path = CACHE_DIR / "weather.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- warm the weather cache first.")
    wx = pd.read_parquet(path)[WEATHER_COLUMNS].drop_duplicates()

    g = games.copy()
    utc = g["kickoff"].dt.tz_convert("UTC")
    g["_lat"] = g["lat"].round(4)
    g["_lon"] = g["lon"].round(4)
    # The cache is keyed on venue-local date/hour; recover both the same way weather.py did.
    tz_local = [
        utc.iloc[i].tz_convert(g["tz"].iloc[i]) if pd.notna(g["tz"].iloc[i]) else None
        for i in range(len(g))
    ]
    g["_date"] = [t.strftime("%Y-%m-%d") if t is not None else None for t in tz_local]
    g["_hour"] = [t.hour if t is not None else np.nan for t in tz_local]

    wx = wx.rename(columns={"lat": "_lat", "lon": "_lon", "date": "_date"})
    merged = g.merge(wx, on=["_lat", "_lon", "_date"], how="inner")
    merged["_gap"] = (merged["hour"] - merged["_hour"]).abs()
    idx = merged.groupby("game_id")["_gap"].idxmin()
    return merged.loc[idx].reset_index(drop=True)


def fit_wind(df: pd.DataFrame, threshold: float) -> dict:
    """OLS of actual total on [pace_sum, eff_sum, hinge(wind)]."""
    hinge = np.maximum(0.0, df["wind_mph"].to_numpy(float) - threshold)
    X = np.column_stack([
        np.ones(len(df)), df["pace_sum"].to_numpy(float),
        df["eff_sum"].to_numpy(float), hinge,
    ])
    y = df["actual_total"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(df) - X.shape[1])
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    return {
        "threshold": threshold,
        "coef": float(beta[3]),
        "se": float(se[3]),
        "t": float(beta[3] / se[3]) if se[3] > 0 else 0.0,
        "n_over": int((hinge > 0).sum()),
        "rmse": float(np.sqrt(sigma2)),
    }


def main() -> None:
    cfg = load_config()
    client = BudgetedCFBD(cfg)
    games = ingest.load_games(client, list(range(2019, 2026)))
    games = venues.attach_venues(games, venues.load_venues(client))
    wf = pd.read_parquet(walkforward_path(cfg))

    g = games[
        (games["homeClassification"] == "fbs")
        & (games["awayClassification"] == "fbs")
        & games["homePoints"].notna() & games["awayPoints"].notna()
        & games["kickoff"].notna() & games["lat"].notna()
        & ~games["dome"].fillna(True)
        & ~games["season"].isin(EXCLUDE_SEASONS)
    ].copy()
    g["actual_total"] = g["homePoints"].astype(float) + g["awayPoints"].astype(float)

    h = wf.rename(columns={"team": "homeTeam", "off_rating": "h_off",
                           "def_rating": "h_def", "pace_rating": "h_pace"})
    a = wf.rename(columns={"team": "awayTeam", "off_rating": "a_off",
                           "def_rating": "a_def", "pace_rating": "a_pace"})
    cols = ["season", "week"]
    g = g.merge(h[cols + ["homeTeam", "h_off", "h_def", "h_pace"]],
                on=cols + ["homeTeam"], how="inner")
    g = g.merge(a[cols + ["awayTeam", "a_off", "a_def", "a_pace"]],
                on=cols + ["awayTeam"], how="inner")
    g["eff_sum"] = (net_epa_vec(g["h_off"], g["a_def"], cfg)
                    + net_epa_vec(g["a_off"], g["h_def"], cfg))
    g["pace_sum"] = g["h_pace"] + g["a_pace"]

    df = _attach_cached_weather(g)
    print(f"{len(df):,} outdoor FBS-vs-FBS games with a cached reading "
          f"(of {len(g):,} eligible)")
    if len(df) < 300:
        print("too few to fit; warm more of the weather cache first.")
        return
    print(f"wind mph: mean {df.wind_mph.mean():.1f}  p90 {df.wind_mph.quantile(.9):.1f}  "
          f"max {df.wind_mph.max():.1f}")
    print(f"\n{'thresh':>7s} {'pts/mph over':>13s} {'se':>7s} {'t':>7s} {'n over':>8s}")
    best = None
    for thr in (0, 5, 8, 10, 12, 15, 18, 20):
        r = fit_wind(df, thr)
        print(f"{thr:7.0f} {r['coef']:13.4f} {r['se']:7.4f} {r['t']:7.2f} "
              f"{r['n_over']:8,}")
        if r["n_over"] >= 200 and (best is None or r["rmse"] < best["rmse"]):
            best = r
    if best:
        print(f"\nbest by RMSE: threshold {best['threshold']:.0f} mph, "
              f"{-best['coef']:.4f} points lost per mph over  (t = {best['t']:.2f})")
    print(f"config currently: threshold {cfg.context.wind_total_threshold_mph}, "
          f"{cfg.context.wind_total_points_per_mph_over} points per mph over")


if __name__ == "__main__":
    main()
