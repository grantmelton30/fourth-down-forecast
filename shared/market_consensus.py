"""Provider-neutral aggregation for free/manual sportsbook line snapshots."""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_market_consensus(
    lines: pd.DataFrame, *, minimum_books: int = 3
) -> pd.DataFrame:
    required = {"game_id", "provider", "spread", "total", "observed_at"}
    missing = required - set(lines.columns)
    if missing:
        raise ValueError(f"market lines missing columns: {sorted(missing)}")
    frame = lines.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    # A provider contributes only its latest known snapshot to a given game.
    frame = (frame.dropna(subset=["observed_at"])
                  .sort_values("observed_at")
                  .drop_duplicates(["game_id", "provider"], keep="last"))
    rows = []
    for game_id, game in frame.groupby("game_id", sort=False):
        spread = game["spread"].dropna().astype(float)
        total = game["total"].dropna().astype(float)
        n_spread, n_total = int(len(spread)), int(len(total))
        rows.append({
            "game_id": game_id,
            "spread": float(spread.median()) if n_spread else np.nan,
            "total": float(total.median()) if n_total else np.nan,
            "spread_dispersion": float(spread.max() - spread.min()) if n_spread else np.nan,
            "total_dispersion": float(total.max() - total.min()) if n_total else np.nan,
            "book_count_spread": n_spread,
            "book_count_total": n_total,
            "is_consensus": bool(n_spread >= minimum_books and n_total >= minimum_books),
            "observed_at": game["observed_at"].max(),
            "providers": tuple(sorted(game["provider"].astype(str).unique())),
        })
    return pd.DataFrame(rows)


__all__ = ["aggregate_market_consensus"]
