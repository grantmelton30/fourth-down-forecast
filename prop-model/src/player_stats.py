"""Player-week production and walk-forward usage shares.

WHAT THIS IS FOR. A prop is a bet on one player's counting stat. That decomposes into two
questions the market prices separately and often prices badly:

    how much of his offence does he get?      <- usage share, modelled here
    what does each opportunity produce?       <- efficiency, modelled downstream

`nflreadpy.load_player_stats` supplies both, free, at one row per player-week. Nothing in
this repository calls it today.

NO LOOKAHEAD, ON KICKOFF TIMESTAMPS, NOT WEEK LABELS. `nfl-model/CLAUDE.md` §1 is explicit
that `as_of` is a kickoff timestamp and the filter is `<` and never `<=`. Grouping by week
number is not the same thing and is not safe: a postponed game can be played after a later
week's games, and a "week" can straddle a Thursday and the following Monday. Usage shares
here are expanding means over games that KICKED OFF STRICTLY EARLIER, joined to the schedule
for the timestamp. Getting this wrong would inflate every backtest silently, which is the
failure mode this codebase has already documented three times.

SHRUNK TOWARD THE TEAM MEAN. A receiver with two games of history is not a 40%-target-share
player because he happened to catch four passes twice. Shares are shrunk by
`n/(n + prior_games)` toward the positional team mean, the same shape `qb.py` uses for
quarterback ratings and `ratings.py` for team ratings.
"""

from __future__ import annotations

import pandas as pd

# The counting stats a prop can be written on, and their opportunity denominator. Yardage is
# deliberately absent from the first pass: PREREG-style discipline says start with discrete
# markets that have real pushes and a clean link to volume.
USAGE_MARKETS = {
    "receptions": ("targets", "receptions"),
    "rushing_attempts": ("carries", "carries"),
    "pass_attempts": ("attempts", "attempts"),
}

STAT_COLUMNS = [
    "player_id", "player_display_name", "position", "team", "season", "week",
    "attempts", "completions", "passing_yards", "passing_tds",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
]


def load_player_weeks(seasons: list[int]) -> pd.DataFrame:
    """One row per player-week, restricted to the columns a prop model actually uses.

    Subset at ingest for the same reason `nfl-model/src/ingest.py` does it for play-by-play:
    the full frame is far wider than anything downstream reads, and a narrow cache is a
    cache someone will actually inspect.
    """
    import nflreadpy as nfl

    raw = nfl.load_player_stats(seasons=seasons).to_pandas()
    missing = [c for c in STAT_COLUMNS if c not in raw.columns]
    if missing:
        raise RuntimeError(
            f"nflverse player-stats schema changed: missing {missing}. Refusing to build "
            "usage shares on a frame whose columns are not what this module was written "
            "against."
        )
    out = raw[STAT_COLUMNS].copy()
    out = out.rename(columns={"team": "team_raw"})
    out["team"] = out["team_raw"]
    return out.drop(columns=["team_raw"])


def attach_kickoffs(stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Attach each player-week's KICKOFF TIMESTAMP, which is what ordering must use.

    A player-week is joined to the game his team played that week. Rows that do not match a
    scheduled game are dropped rather than defaulted -- a stat line with no kickoff cannot be
    placed in time, and guessing its position is exactly the lookahead this module exists to
    prevent.
    """
    games = schedules[["season", "week", "home_team", "away_team", "kickoff"]].copy()
    home = games.rename(columns={"home_team": "team"})[["season", "week", "team", "kickoff"]]
    away = games.rename(columns={"away_team": "team"})[["season", "week", "team", "kickoff"]]
    long = pd.concat([home, away], ignore_index=True)
    out = stats.merge(long, on=["season", "week", "team"], how="inner")
    return out.sort_values("kickoff").reset_index(drop=True)


def usage_shares(stats: pd.DataFrame, *, prior_games: float = 4.0) -> pd.DataFrame:
    """Walk-forward share of team opportunity, per player, per market.

    For every player-week this returns what was knowable BEFORE that game kicked off. The
    row's own game never contributes to its own share.

    `prior_games` is the shrinkage strength: a player with `prior_games` of history is
    weighted half his own rate and half his team's positional mean.
    """
    if "kickoff" not in stats.columns:
        raise ValueError("call attach_kickoffs first -- usage must be ordered by kickoff")

    df = stats.sort_values("kickoff").copy()
    out = df[["player_id", "player_display_name", "position", "team", "season", "week",
              "kickoff"]].copy()

    for market, (denom, _stat) in USAGE_MARKETS.items():
        if denom not in df.columns:
            continue
        # Team opportunity in each game, then each player's share of it. A team with zero
        # opportunity in a market (no pass attempts at all) yields NaN, not zero: it is an
        # absence of information, and a zero would be read as "this player got no share"
        # and dragged into his prior mean as a real observation.
        team_total = df.groupby(["season", "week", "team"])[denom].transform("sum")
        share = (df[denom].astype(float)
                 / team_total.astype(float).replace(0.0, float("nan")))

        by_player = share.groupby(df["player_id"])
        # STRICTLY EARLIER: shift(1) before the expanding mean, so a game never sees itself.
        prior_mean = by_player.transform(lambda s: s.shift(1).expanding().mean())
        prior_n = by_player.transform(lambda s: s.shift(1).expanding().count())

        # Shrink toward the team's positional mean over the same prior window.
        from .prop_model import prior_group_totals
        history = df.assign(_share=share)
        pos_sum, pos_count = prior_group_totals(
            history, ["season", "team", "position"], "_share")
        pos_mean = pos_sum / pos_count.replace(0, float("nan"))
        weight = prior_n / (prior_n + prior_games)
        blended = weight * prior_mean.fillna(0.0) + (1 - weight) * pos_mean.fillna(0.0)

        out[f"{market}_share"] = blended.astype(float)
        out[f"{market}_prior_games"] = prior_n.fillna(0.0).astype(float)
        out[f"{market}_actual"] = df[denom].astype(float)

    return out


__all__ = ["USAGE_MARKETS", "STAT_COLUMNS", "load_player_weeks", "attach_kickoffs",
           "usage_shares"]
