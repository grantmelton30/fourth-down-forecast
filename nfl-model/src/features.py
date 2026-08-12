"""Free, play-derived NFL challenger features with strict prior-game rolling state."""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "early_down_pass_epa", "early_down_rush_epa", "success_rate",
    "explosive_rate", "sack_avoidance", "red_zone_td_rate",
    "turnover_opportunity_rate", "starting_field_position", "special_teams_epa",
)


def _numeric(plays: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric play column without turning a missing feed into an exception."""
    if column not in plays:
        return pd.Series(default, index=plays.index, dtype=float)
    return pd.to_numeric(plays[column], errors="coerce").fillna(default)


def build_team_game_features(pbp: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "season", "week", "posteam", "defteam", "down",
                "play_type", "epa", "success", "yards_gained", "qb_dropback",
                "yardline_100"}
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"NFL features missing PBP columns: {sorted(missing)}")
    plays = pbp.copy()
    if "competitive" in plays:
        plays = plays[plays["competitive"].fillna(False)]
    special = pd.Series(dtype=float, name="special_teams_epa")
    if "special" in plays:
        special = (plays[plays["special"].fillna(0).eq(1) & plays["posteam"].notna()]
                   .groupby(["game_id", "posteam"])["epa"].mean()
                   .rename("special_teams_epa"))
        plays = plays[plays["special"].fillna(0).eq(0)]
    plays = plays[plays["posteam"].notna() & plays["defteam"].notna()]
    early = plays["down"].isin([1, 2])
    passes = plays["qb_dropback"].fillna(0).eq(1)
    rushes = plays["play_type"].eq("run")
    plays["_ed_pass_epa"] = plays["epa"].where(early & passes)
    plays["_ed_rush_epa"] = plays["epa"].where(early & rushes)
    plays["_explosive"] = (
        (passes & plays["yards_gained"].ge(20))
        | (rushes & plays["yards_gained"].ge(12))
    ).astype(float)
    plays["_sack"] = _numeric(plays, "sack")
    plays["_turnover_opportunity"] = (
        _numeric(plays, "interception") + _numeric(plays, "fumble_lost")
    ).clip(upper=1)
    plays["_red_zone"] = plays["yardline_100"].le(20)
    plays["_rz_td"] = _numeric(plays, "touchdown").where(plays["_red_zone"])

    # Field position is the first scrimmage play of a drive, not the mean location of
    # every play.  The latter rewards offenses for moving the ball and double-counts EPA.
    drive_key = "fixed_drive" if "fixed_drive" in plays else "drive"
    if drive_key in plays:
        first = (
            plays.dropna(subset=[drive_key])
            .sort_index()
            .drop_duplicates(["game_id", "posteam", drive_key], keep="first")
        )
        starts = (
            first.groupby(["game_id", "posteam"])["yardline_100"]
            .mean().rename("starting_field_position")
        )
    else:
        starts = pd.Series(dtype=float, name="starting_field_position")
    keys = ["game_id", "season", "week", "posteam", "defteam"]
    out = plays.groupby(keys, as_index=False).agg(
        early_down_pass_epa=("_ed_pass_epa", "mean"),
        early_down_rush_epa=("_ed_rush_epa", "mean"),
        success_rate=("success", "mean"), explosive_rate=("_explosive", "mean"),
        sacks=("_sack", "sum"), dropbacks=("qb_dropback", "sum"),
        red_zone_td_rate=("_rz_td", "mean"),
        turnover_opportunity_rate=("_turnover_opportunity", "mean"),
    ).rename(columns={"posteam": "team", "defteam": "opponent"})
    out = out.merge(
        starts.reset_index().rename(columns={"posteam": "team"}),
        on=["game_id", "team"], how="left",
    )
    out = out.merge(special.reset_index().rename(columns={"posteam": "team"}),
                    on=["game_id", "team"], how="left")
    out["sack_avoidance"] = 1.0 - out["sacks"] / out["dropbacks"].clip(lower=1)
    return out[["game_id", "season", "week", "team", "opponent", *FEATURE_COLUMNS]]


def coaching_continuity(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per-game staff continuity from the free schedule coach fields."""
    required = {"game_id", "season", "week", "home_team", "away_team",
                "home_coach", "away_coach"}
    if not required <= set(schedules):
        return schedules[["game_id"]].assign(coach_continuity_diff=np.nan)
    home = schedules[["game_id", "season", "week", "home_team", "home_coach"]].rename(
        columns={"home_team": "team", "home_coach": "coach"})
    away = schedules[["game_id", "season", "week", "away_team", "away_coach"]].rename(
        columns={"away_team": "team", "away_coach": "coach"})
    team = pd.concat([home, away]).sort_values(["team", "season", "week", "game_id"])
    prior = team.groupby("team")["coach"].shift(1)
    team["coach_continuity"] = team["coach"].eq(prior).where(prior.notna()).astype(float)
    h = team[["game_id", "team", "coach_continuity"]].rename(
        columns={"team": "home_team", "coach_continuity": "home_coach_continuity"})
    a = team[["game_id", "team", "coach_continuity"]].rename(
        columns={"team": "away_team", "coach_continuity": "away_coach_continuity"})
    out = schedules[["game_id", "home_team", "away_team"]].merge(
        h, on=["game_id", "home_team"], how="left").merge(
        a, on=["game_id", "away_team"], how="left")
    out["coach_continuity_diff"] = (out["home_coach_continuity"]
                                     - out["away_coach_continuity"])
    return out[["game_id", "coach_continuity_diff"]]


def rolling_team_features(
    games: pd.DataFrame, half_life_games: float = 6.0
) -> pd.DataFrame:
    """Exponentially weighted team form, shifted so the current game never enters."""
    frame = games.sort_values(["season", "team", "week", "game_id"]).copy()
    alpha = 1.0 - 0.5 ** (1.0 / float(half_life_games))
    for feature in FEATURE_COLUMNS:
        frame[feature] = frame.groupby(["season", "team"], sort=False)[feature].transform(
            lambda values: values.shift(1).ewm(alpha=alpha, adjust=False).mean()
        )
    frame["games_observed"] = frame.groupby(["season", "team"], sort=False).cumcount()
    return frame


__all__ = ["FEATURE_COLUMNS", "build_team_game_features", "rolling_team_features"]


def matchup_feature_table(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per game with offense-vs-opposing-defense, all strictly pre-game.

    ``*_matchup_diff`` is the home offense against the away defense minus the mirror
    matchup.  Defense form is built from what prior opponents produced, so this avoids
    treating two identical offensive rates as equal when one team has faced much harder
    defenses.  The later ridge remains responsible for deciding whether the feature
    improves held-out seasons.
    """
    game = build_team_game_features(pbp)
    rolling = rolling_team_features(game)
    allowed = game.rename(columns={"team": "opponent", "opponent": "team"})
    defense = rolling_team_features(allowed)
    home = rolling.rename(columns={
        "team": "home_team", **{c: f"home_{c}" for c in FEATURE_COLUMNS},
        "games_observed": "home_games_observed",
    })
    away = rolling.rename(columns={
        "team": "away_team", **{c: f"away_{c}" for c in FEATURE_COLUMNS},
        "games_observed": "away_games_observed",
    })
    home_def = defense.rename(columns={
        "team": "home_team", **{c: f"home_allowed_{c}" for c in FEATURE_COLUMNS},
        "games_observed": "home_def_games_observed",
    })
    away_def = defense.rename(columns={
        "team": "away_team", **{c: f"away_allowed_{c}" for c in FEATURE_COLUMNS},
        "games_observed": "away_def_games_observed",
    })
    base = schedules[["game_id", "season", "week", "home_team", "away_team"]].copy()
    base = base.merge(coaching_continuity(schedules), on="game_id", how="left")
    if {"home_rest", "away_rest"} <= set(schedules):
        context = schedules[["game_id", "home_rest", "away_rest"]].copy()
        context["rest_days_diff"] = (pd.to_numeric(context["home_rest"], errors="coerce")
                                     - pd.to_numeric(context["away_rest"], errors="coerce"))
        base = base.merge(context[["game_id", "rest_days_diff"]], on="game_id", how="left")
    hcols = ["game_id", "home_team", *[f"home_{c}" for c in FEATURE_COLUMNS],
             "home_games_observed"]
    acols = ["game_id", "away_team", *[f"away_{c}" for c in FEATURE_COLUMNS],
             "away_games_observed"]
    out = base.merge(home[hcols], on=["game_id", "home_team"], how="left")
    out = out.merge(away[acols], on=["game_id", "away_team"], how="left")
    hdcols = ["game_id", "home_team", *[f"home_allowed_{c}" for c in FEATURE_COLUMNS],
              "home_def_games_observed"]
    adcols = ["game_id", "away_team", *[f"away_allowed_{c}" for c in FEATURE_COLUMNS],
              "away_def_games_observed"]
    out = out.merge(home_def[hdcols], on=["game_id", "home_team"], how="left")
    out = out.merge(away_def[adcols], on=["game_id", "away_team"], how="left")
    for feature in FEATURE_COLUMNS:
        out[f"{feature}_diff"] = out[f"home_{feature}"] - out[f"away_{feature}"]
        out[f"{feature}_sum"] = out[f"home_{feature}"] + out[f"away_{feature}"]
        home_matchup = out[f"home_{feature}"] - out[f"away_allowed_{feature}"]
        away_matchup = out[f"away_{feature}"] - out[f"home_allowed_{feature}"]
        out[f"{feature}_matchup_diff"] = home_matchup - away_matchup
        out[f"{feature}_matchup_sum"] = home_matchup + away_matchup
    return out


__all__.extend(["coaching_continuity", "matchup_feature_table"])
