"""nfelo integration (§7.5): the QB valuation half, and only that half.

`qb_elos.csv` is a genuine model INPUT. nfeloqb maintains 538's QB Elo model in 538's
original schema, and `qb1_adj` / `qb2_adj` are already a points-scale adjustment for the
starting QB relative to the team's baseline. That is exactly what §12 needs, from a
maintained source with a decade of history. It is a legitimate input rather than
wrapper-ism because it is a QB valuation model, not a game prediction model: it does not
consume betting market data, so folding it in does not double-count the market.

WHAT WAS REMOVED, AND WHY. §7.5 also described using nfelo's game SPREAD as a third term
in the blend. That was never populated -- building the series requires running the full
nfelo model locally, which the playbook flags as the most involved piece of the build and
explicitly permits skipping. The cost of leaving the scaffolding in place was not zero:
`fit_blend_weights` reported `t_nfelo = +0.00` next to two genuinely measured
t-statistics, so every gate readout claimed nfelo had been tested and found uninformative
when it had never been tested at all. A measurement that never happened must not be
reported as one.

It is also the half that was never worth much. nfelo regresses to market spreads by
design, so its independent contribution is mostly the publish-to-market lag -- which for
a public model is approximately zero.

Credit: nfelo (nfeloapp.com), open source at github.com/greerreNFL.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from .config import CACHE_DIR
from .ingest import normalize_team

QB_ELOS_URL = "https://raw.githubusercontent.com/greerreNFL/nfeloqb/main/qb_elos.csv"
_QB_COLUMNS = [
    "season", "date", "team1", "team2", "qb1", "qb2", "qb1_value_pre", "qb2_value_pre",
    "qb1_adj", "qb2_adj", "qbelo1_pre", "qbelo2_pre",
]


def load_qb_elos(refresh: bool = False) -> "pd.DataFrame | None":
    """Plain HTTP GET, cached to parquet. No key, no package, refresh weekly.

    Returns None rather than raising: the QB layer is valuable, but the build must not
    block on a third-party file being reachable.
    """
    path = CACHE_DIR / "qb_elos.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    try:
        resp = requests.get(QB_ELOS_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
    except Exception as exc:  # noqa: BLE001
        print(
            f"  nfelo: could not fetch qb_elos.csv ({exc}). QB layer degrades to the "
            "in-repo EPA+CPOE composite."
        )
        return None

    df = df[[c for c in _QB_COLUMNS if c in df.columns]].copy()
    for col in ("team1", "team2"):
        if col in df.columns:
            df[col] = normalize_team(df[col])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.to_parquet(path, index=False)
    return df


def qb_adjustments_by_game(
    qb_elos: "pd.DataFrame | None", schedules: pd.DataFrame
) -> pd.DataFrame:
    """Join nfeloqb's per-start QB adjustments onto our schedule.

    `team1` is the home team in 538's schema. Returns one row per game with the home and
    away points-scale QB adjustments and the starter names behind them.
    """
    empty = pd.DataFrame(
        columns=["game_id", "qb_adj_home", "qb_adj_away", "qb_home", "qb_away"]
    )
    if qb_elos is None or qb_elos.empty:
        return empty

    sched = schedules[["game_id", "season", "gameday", "home_team", "away_team"]].copy()
    sched["date"] = pd.to_datetime(sched["gameday"], errors="coerce")

    merged = sched.merge(
        qb_elos,
        left_on=["date", "home_team", "away_team"],
        right_on=["date", "team1", "team2"],
        how="left",
    )
    return pd.DataFrame({
        "game_id": merged["game_id"],
        "qb_adj_home": merged.get("qb1_adj"),
        "qb_adj_away": merged.get("qb2_adj"),
        "qb_home": merged.get("qb1"),
        "qb_away": merged.get("qb2"),
    })
