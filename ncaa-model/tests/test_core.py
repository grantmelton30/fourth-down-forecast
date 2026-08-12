"""The three NCAA assertions that catch silent killers.

Not a broad suite. Each guards a failure mode that produces entirely plausible-looking
numbers while being wrong:

1. Spread sign — CFBD quotes negative = home favored, nflverse the opposite. A flip
   produces a model that looks fine and is exactly backwards.
2. No lookahead — a leak inflates every metric and is invisible in the output.
3. Universe size — the restricted universe is quoted as 1,543 games throughout the
   documentation and Appendix A. If a filter changes silently, every reported figure
   refers to a different population than the one documented.

Requires the parquet cache:
    NCAA_MODEL_CACHE_DIR=~/.cache/ncaa-model pytest tests/
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import CACHE_DIR, load_config
from src.ratings import walkforward_path

DOCUMENTED_RESTRICTED_UNIVERSE = 1543


def _need(path):
    if not path.exists():
        pytest.skip(f"cache artifact missing: {path.name}; set NCAA_MODEL_CACHE_DIR")
    return path


# --------------------------------------------------------------------------------------
# 1. Spread sign convention
# --------------------------------------------------------------------------------------

def test_spread_sign_convention():
    """Positive `spread_close` must mean the HOME team is favored.

    Asserted three ways, because this is the bug the playbook names as the silent killer:
    against the raw CFBD payload (opposite convention), against outcome correlation, and
    against heavy home favorites actually winning.
    """
    market = pd.read_parquet(_need(CACHE_DIR / "market.parquet"))
    played = market.dropna(subset=["spread_close", "actual_margin"])
    assert len(played) > 1000, "not enough completed games to test the sign"

    # (a) Correlation with the realised margin must be positive and substantial.
    r = float(played["spread_close"].corr(played["actual_margin"]))
    assert r > 0.3, f"spread_close correlates {r:+.3f} with actual margin — sign flipped?"

    # (b) Heavy home favorites must usually win.
    heavy = played[played["spread_close"] >= 14]
    if len(heavy) >= 50:
        win_rate = float((heavy["actual_margin"] > 0).mean())
        assert win_rate > 0.75, (
            f"home teams favored by 14+ won only {win_rate:.1%} — sign flipped?"
        )

    # (c) Our normalisation must be the OPPOSITE of the raw CFBD payload.
    raw_path = CACHE_DIR / "lines_2023.json"
    if raw_path.exists():
        cfg = load_config()
        priority = list(cfg.market.provider_priority)
        checked = 0
        for g in json.loads(raw_path.read_text()):
            by_prov = {str(l.get("provider")): l for l in (g.get("lines") or [])}
            chosen = next(
                (by_prov[p] for p in priority
                 if p in by_prov and by_prov[p].get("spread") is not None),
                None,
            )
            if chosen is None:
                continue
            ours = market.loc[market["game_id"] == g["id"], "spread_close"]
            if ours.empty or pd.isna(ours.iloc[0]) or chosen["spread"] == 0:
                continue
            assert np.sign(ours.iloc[0]) == -np.sign(chosen["spread"]), (
                f"game {g['id']}: ours {ours.iloc[0]} vs raw CFBD {chosen['spread']} — "
                "normalisation is not flipping the CFBD convention"
            )
            checked += 1
            if checked >= 40:
                break
        assert checked > 0, "could not verify against any raw CFBD line"


# --------------------------------------------------------------------------------------
# 2. No lookahead on the walk-forward split
# --------------------------------------------------------------------------------------

def test_no_lookahead_in_walkforward_split():
    """Every rating used for a game must be dated at or before that game's kickoff.

    The ratings `as_of` for a (season, week) is that week's first kickoff, so a game may
    legitimately share its own week's cutoff — but never a later one.
    """
    ratings = pd.read_parquet(_need(walkforward_path(load_config())))
    frame = pd.read_parquet(_need(CACHE_DIR / "backtest_frame_default.parquet"))

    cutoffs = ratings[["season", "week", "as_of"]].drop_duplicates()
    merged = frame.merge(cutoffs, on=["season", "week"], how="left")
    merged = merged.dropna(subset=["as_of", "kickoff"])
    assert len(merged) > 500, "not enough joined rows to test"

    as_of = pd.to_datetime(merged["as_of"], utc=True)
    kickoff = pd.to_datetime(merged["kickoff"], utc=True)
    violations = merged[as_of > kickoff]
    assert violations.empty, (
        f"{len(violations)} games used ratings dated AFTER kickoff, e.g. "
        f"{violations['game_id'].head(3).tolist()}"
    )

    # The cutoff must also be non-decreasing through a season.
    for season, grp in cutoffs.groupby("season"):
        ordered = pd.to_datetime(
            grp.sort_values("week")["as_of"].reset_index(drop=True), utc=True
        )
        assert (ordered.diff().dropna() >= pd.Timedelta(0)).all(), (
            f"season {season}: week cutoffs are not monotonically increasing"
        )


# --------------------------------------------------------------------------------------
# 3. Universe row count
# --------------------------------------------------------------------------------------

def test_restricted_universe_row_count():
    """Pins the restricted universe at its documented size.

    1,543 is quoted in README.md, NCAA_PLAYBOOK.md Appendix A, REPO_INVENTORY.md and every
    reported coefficient. If a filter changes silently, those figures describe a different
    population than the one they claim to.
    """
    from src.backtest import select_window

    cfg = load_config()
    frame = pd.read_parquet(_need(CACHE_DIR / "backtest_frame_default.parquet"))
    universe = select_window(frame, cfg.graded_seasons, restricted=True)

    assert len(universe) == DOCUMENTED_RESTRICTED_UNIVERSE, (
        f"restricted universe is {len(universe)}, documented as "
        f"{DOCUMENTED_RESTRICTED_UNIVERSE}. Either a filter changed or the documentation "
        "is stale — reconcile before trusting any reported coefficient."
    )
    assert universe["restricted"].all(), "non-restricted game in the restricted universe"
    assert universe["has_opener"].all(), "game without an opener in the graded universe"
    assert universe["week"].min() >= 4, "game before week 4 in the graded universe"
    assert set(universe["season"]) <= set(cfg.graded_seasons), "season outside the window"
