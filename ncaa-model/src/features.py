"""Leakage-safe NCAA challenger features from CFBD's documented free endpoints."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PLAY_FEATURES = (
    "early_down_efficiency", "success_rate", "explosive_rate", "havoc_avoidance",
    "red_zone_td_rate", "starting_field_position",
)
PRESEASON_FEATURES = (
    "prior_power_rating", "returning_production", "portal_net_rating", "recruiting_rating",
    "talent_composite", "qb_continuity", "head_coach_continuity",
    "offensive_coordinator_continuity", "defensive_coordinator_continuity",
)


def maturity_phase(week: int) -> str:
    """Stable evidence phases used for validation and public quality labels."""
    week = int(week)
    if week <= 1:
        return "preseason"
    if week <= 3:
        return "early"
    if week <= 5:
        return "developing"
    return "established"


def previous_season_power(
    walkforward: pd.DataFrame, seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Leakage-safe final prior-year rating exposed as a preseason candidate."""
    if walkforward is None or walkforward.empty:
        return pd.DataFrame(columns=["season", "team", "prior_power_rating"])
    final = (walkforward.sort_values(["season", "week"])
             .drop_duplicates(["season", "team"], keep="last").copy())
    final["season"] = pd.to_numeric(final["season"], errors="coerce") + 1
    final["prior_power_rating"] = (
        pd.to_numeric(final["off_rating"], errors="coerce")
        - pd.to_numeric(final["def_rating"], errors="coerce")
    )
    if seasons is not None:
        final = final[final["season"].isin(seasons)]
    return final[["season", "team", "prior_power_rating"]].reset_index(drop=True)


def add_preseason_matchup_features(
    games: pd.DataFrame, preseason: "pd.DataFrame | None",
) -> pd.DataFrame:
    """Attach team-season priors and phase interactions to a matchup frame."""
    out = games.copy()
    home_col = "home_team" if "home_team" in out else "homeTeam"
    away_col = "away_team" if "away_team" in out else "awayTeam"
    if preseason is None or preseason.empty:
        for feature in PRESEASON_FEATURES:
            out[f"{feature}_diff"] = np.nan
            out[f"{feature}_sum"] = np.nan
    else:
        priors = preseason.drop_duplicates(["season", "team"], keep="last")
        for feature in PRESEASON_FEATURES:
            if feature not in priors:
                priors[feature] = np.nan
        for side, team_col in (("home", home_col), ("away", away_col)):
            team = priors.rename(columns={
                "team": team_col,
                **{feature: f"{side}_{feature}" for feature in PRESEASON_FEATURES},
            })
            keep = ["season", team_col, *[
                f"{side}_{feature}" for feature in PRESEASON_FEATURES
            ]]
            out = out.merge(team[keep], on=["season", team_col], how="left")
        for feature in PRESEASON_FEATURES:
            out[f"{feature}_diff"] = (
                out[f"home_{feature}"] - out[f"away_{feature}"]
            )
            out[f"{feature}_sum"] = (
                out[f"home_{feature}"] + out[f"away_{feature}"]
            )
    phases = out["week"].map(maturity_phase)
    for feature in PRESEASON_FEATURES:
        for kind in ("diff", "sum"):
            base = f"{feature}_{kind}"
            for phase in ("preseason", "early", "developing", "established"):
                out[f"{base}_{phase}"] = out[base].where(phases.eq(phase))
    return out


def build_team_game_features(plays: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "season", "week", "offense", "defense", "playType",
                "yardsGained", "yardsToGoal", "ppa"}
    missing = required - set(plays.columns)
    if missing:
        raise ValueError(f"NCAA features missing play columns: {sorted(missing)}")
    frame = plays.copy()
    if "garbage" in frame:
        frame = frame[~frame["garbage"].fillna(False)]
    kind = frame["playType"].astype(str)
    is_pass = kind.str.contains("Pass|Sack", case=False, regex=True)
    is_rush = kind.str.contains("Rush", case=False, regex=True)
    # CFBD's free play payload does not consistently expose down/distance for every
    # historical year. Positive PPA is therefore the stable success definition.
    frame["_success"] = pd.to_numeric(frame["ppa"], errors="coerce").gt(0).astype(float)
    yards = pd.to_numeric(frame["yardsGained"], errors="coerce")
    frame["_explosive"] = ((is_pass & yards.ge(20)) | (is_rush & yards.ge(12))).astype(float)
    havoc = kind.str.contains("Sack|Interception|Fumble|Tackle for Loss", case=False,
                              regex=True)
    frame["_havoc_avoidance"] = (~havoc).astype(float)
    red_zone = pd.to_numeric(frame["yardsToGoal"], errors="coerce").le(20)
    touchdown = kind.str.contains("Touchdown", case=False, regex=False)
    frame["_rz_td"] = touchdown.astype(float).where(red_zone)
    if "down" in frame:
        early = pd.to_numeric(frame["down"], errors="coerce").le(2)
        frame["_early_eff"] = pd.to_numeric(frame["ppa"], errors="coerce").where(early)
    else:
        # Do not relabel all-down efficiency as early-down efficiency when an older
        # payload lacks down. Missing stays missing and is imputed only inside training.
        frame["_early_eff"] = np.nan

    drive_key = "driveId" if "driveId" in frame else None
    if drive_key:
        starts = (frame.dropna(subset=[drive_key]).sort_index()
                  .drop_duplicates(["game_id", "offense", drive_key])
                  .groupby(["game_id", "offense"])["yardsToGoal"].mean()
                  .rename("starting_field_position"))
    else:
        starts = pd.Series(dtype=float, name="starting_field_position")
    keys = ["game_id", "season", "week", "offense", "defense"]
    out = frame.groupby(keys, as_index=False).agg(
        early_down_efficiency=("_early_eff", "mean"),
        success_rate=("_success", "mean"), explosive_rate=("_explosive", "mean"),
        havoc_avoidance=("_havoc_avoidance", "mean"),
        red_zone_td_rate=("_rz_td", "mean"),
    ).rename(columns={"offense": "team", "defense": "opponent"})
    out = out.merge(starts.reset_index().rename(columns={"offense": "team"}),
                    on=["game_id", "team"], how="left")
    return out[["game_id", "season", "week", "team", "opponent", *PLAY_FEATURES]]


def rolling_team_features(games: pd.DataFrame, half_life_games: float = 5.0) -> pd.DataFrame:
    frame = games.sort_values(["season", "team", "week", "game_id"]).copy()
    alpha = 1.0 - 0.5 ** (1.0 / float(half_life_games))
    for feature in PLAY_FEATURES:
        frame[feature] = frame.groupby(["season", "team"], sort=False)[feature].transform(
            lambda values: values.shift(1).ewm(alpha=alpha, adjust=False).mean()
        )
    frame["games_observed"] = frame.groupby(["season", "team"], sort=False).cumcount()
    return frame


def matchup_feature_table(
    plays: pd.DataFrame, games: pd.DataFrame, preseason: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per game with strictly prior form and dated preseason attributes."""
    game = build_team_game_features(plays)
    offense = rolling_team_features(game)
    allowed = rolling_team_features(game.rename(columns={"team": "opponent",
                                                          "opponent": "team"}))
    base_cols = ["game_id", "season", "week", "homeTeam", "awayTeam"]
    optional = [c for c in ("homeClassification", "awayClassification") if c in games]
    out = games[base_cols + optional].copy()
    for side, team_col in (("home", "homeTeam"), ("away", "awayTeam")):
        off = offense.rename(columns={"team": team_col,
            **{c: f"{side}_{c}" for c in PLAY_FEATURES}})
        dfn = allowed.rename(columns={"team": team_col,
            **{c: f"{side}_allowed_{c}" for c in PLAY_FEATURES}})
        out = out.merge(off[["game_id", team_col, *[f"{side}_{c}" for c in PLAY_FEATURES]]],
                        on=["game_id", team_col], how="left")
        out = out.merge(dfn[["game_id", team_col,
                            *[f"{side}_allowed_{c}" for c in PLAY_FEATURES]]],
                        on=["game_id", team_col], how="left")
    for feature in PLAY_FEATURES:
        home_match = out[f"home_{feature}"] - out[f"away_allowed_{feature}"]
        away_match = out[f"away_{feature}"] - out[f"home_allowed_{feature}"]
        out[f"{feature}_matchup_diff"] = home_match - away_match
        out[f"{feature}_matchup_sum"] = home_match + away_match

    if preseason is not None and len(preseason):
        for side, team_col in (("home", "homeTeam"), ("away", "awayTeam")):
            team = preseason.rename(columns={"team": team_col,
                **{c: f"{side}_{c}" for c in PRESEASON_FEATURES}})
            out = out.merge(team[["season", team_col,
                                  *[f"{side}_{c}" for c in PRESEASON_FEATURES]]],
                            on=["season", team_col], how="left")
        for feature in PRESEASON_FEATURES:
            out[f"{feature}_diff"] = out[f"home_{feature}"] - out[f"away_{feature}"]
            out[f"{feature}_sum"] = out[f"home_{feature}"] + out[f"away_{feature}"]
    if {"homeClassification", "awayClassification"} <= set(out):
        out["cross_tier"] = out["homeClassification"].ne(out["awayClassification"])
    else:
        out["cross_tier"] = False
    return out


def _column(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(np.nan, index=frame.index)


def _coach_seasons(coaching: pd.DataFrame) -> pd.DataFrame:
    """Flatten CFBD's coaches payload to one row per coach-team-season.

    The endpoint returns a coach, not a team-season: identity is at the top level and
    the tenure sits in a nested `seasons` array holding `school` and `year`. Reading it
    with the flat accessors this module uses elsewhere finds no team column at all and
    silently produces NaN for every row, which is why head-coach continuity had never
    been computed even though 138 coaches for 2026 were already cached on disk.

    The flat shape is still accepted, so a differently-shaped free source keeps working.
    """
    d = coaching.copy()
    if "seasons" in d.columns:
        name = (_column(d, "firstName", "first_name").fillna("").astype(str) + " "
                + _column(d, "lastName", "last_name").fillna("").astype(str)).str.strip()
        exploded = d.assign(_coach=name).explode("seasons").reset_index(drop=True)
        entries = exploded["seasons"].apply(
            lambda value: value if isinstance(value, dict) else {})
        out = pd.DataFrame({
            "season": pd.to_numeric(entries.apply(lambda e: e.get("year")),
                                    errors="coerce"),
            "team": entries.apply(lambda e: e.get("school")),
            "coach": exploded["_coach"],
        })
    else:
        out = pd.DataFrame({
            "season": pd.to_numeric(_column(d, "season", "year"), errors="coerce"),
            "team": _column(d, "team", "school"),
            "coach": _column(d, "headCoach", "head_coach", "coach").astype(str),
        })
    out = out.dropna(subset=["season", "team"])
    return out[out["coach"].astype(str).str.strip().ne("")]


def normalize_preseason_sources(
    *, returning: pd.DataFrame | None = None, portal: pd.DataFrame | None = None,
    recruiting: pd.DataFrame | None = None, talent: pd.DataFrame | None = None,
    coaching: pd.DataFrame | None = None, manual: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize the stable parts of several CFBD schemas into team-season rows.

    Schemas have changed casing over time, so this deliberately accepts the documented
    aliases while refusing to invent zeroes for unavailable data.
    """
    parts: list[pd.DataFrame] = []
    if returning is not None and len(returning):
        d = returning.copy()
        parts.append(pd.DataFrame({
            "season": pd.to_numeric(_column(d, "season", "year"), errors="coerce"),
            "team": _column(d, "team", "school"),
            "returning_production": pd.to_numeric(
                _column(d, "percentPPA", "percent_ppa", "returningProduction"),
                errors="coerce"),
            "qb_continuity": pd.to_numeric(
                _column(d, "percentPassingPPA", "percent_passing_ppa"), errors="coerce"),
        }))
    if portal is not None and len(portal):
        d = portal.copy()
        season = pd.to_numeric(_column(d, "season", "year"), errors="coerce")
        rating = pd.to_numeric(_column(d, "rating", "stars"), errors="coerce").fillna(0)
        incoming = pd.DataFrame({"season": season, "team": _column(d, "destination"),
                                 "portal": rating})
        outgoing = pd.DataFrame({"season": season, "team": _column(d, "origin"),
                                 "portal": -rating})
        p = pd.concat([incoming, outgoing]).dropna(subset=["season", "team"])
        parts.append(p.groupby(["season", "team"], as_index=False)["portal"].sum().rename(
            columns={"portal": "portal_net_rating"}))
    for source, output, value_aliases in (
        (recruiting, "recruiting_rating", ("points", "rating")),
        (talent, "talent_composite", ("talent", "talentComposite")),
    ):
        if source is not None and len(source):
            d = source.copy()
            parts.append(pd.DataFrame({
                "season": pd.to_numeric(_column(d, "season", "year"), errors="coerce"),
                "team": _column(d, "team", "school"),
                output: pd.to_numeric(_column(d, *value_aliases), errors="coerce"),
            }))
    if coaching is not None and len(coaching):
        c = _coach_seasons(coaching)
        if len(c):
            c = c.sort_values(["team", "season"]).drop_duplicates(
                ["team", "season"], keep="last")
            prior = c.groupby("team")["coach"].shift(1)
            # No prior season on record means we do not know whether the coach changed.
            # Scoring that as 0.0 would assert a coaching change that may not have
            # happened, which is exactly the fabrication the missing-data policy forbids.
            c["head_coach_continuity"] = np.where(
                prior.isna(), np.nan, c["coach"].eq(prior).astype(float))
            parts.append(c.drop(columns="coach"))
    if manual is not None and len(manual):
        required = {"season", "team", "observed_at", "source"}
        missing = required - set(manual.columns)
        if missing:
            raise ValueError(f"manual NCAA features missing columns: {sorted(missing)}")
        d = manual.copy()
        d["observed_at"] = pd.to_datetime(d["observed_at"], utc=True, errors="coerce")
        if d["observed_at"].isna().any():
            raise ValueError("manual NCAA features contains invalid observed_at")
        keep = ["season", "team", *[c for c in PRESEASON_FEATURES if c in d]]
        parts.append(d[keep])
    if not parts:
        return pd.DataFrame(columns=["season", "team", *PRESEASON_FEATURES])
    normalized = []
    for part in parts:
        part = part.copy()
        part["season"] = pd.to_numeric(part["season"], errors="coerce")
        part["team"] = part["team"].astype("string").str.strip()
        part = part.dropna(subset=["season", "team"])
        part = part[part["team"].ne("")]
        part["season"] = part["season"].astype(int)
        part["team"] = part["team"].astype(str)
        if len(part):
            normalized.append(part)
    parts = normalized
    if not parts:
        return pd.DataFrame(columns=["season", "team", *PRESEASON_FEATURES])
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["season", "team"], how="outer", suffixes=("", "_new"))
        for col in list(out):
            if col.endswith("_new"):
                base = col[:-4]
                out[base] = out[col].combine_first(out.get(base))
                out = out.drop(columns=col)
    for feature in PRESEASON_FEATURES:
        if feature not in out:
            out[feature] = np.nan
    return out[["season", "team", *PRESEASON_FEATURES]]


def game_uncertainty_multiplier(
    week: int, *, preseason_available: bool, cross_tier: bool,
    returning_production: float | None = None,
    qb_continuity: float | None = None,
) -> float:
    """Conservative widening; never changes the point estimate.

    Returning-production and quarterback-continuity inputs are used only to acknowledge
    that an early-season distribution is less certain when last year's roster is mostly
    gone.  They are deliberately not converted into point adjustments here: the live mean
    model may use those features only after its prior-season validation promotes them.
    """
    def share(value: float | None) -> float | None:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if 1.0 < value <= 100.0:
            value /= 100.0
        return float(np.clip(value, 0.0, 1.0))

    early = max(0, 5 - int(week)) * 0.08
    returning = share(returning_production)
    quarterback = share(qb_continuity)
    # Roster uncertainty is a preseason prior, not a permanent label.  Fade it linearly
    # through Week 5 as current-team games replace offseason inference.
    preseason_share = max(0.0, min(1.0, (6 - int(week)) / 5.0))
    roster = 0.0
    if returning is not None:
        roster += preseason_share * 0.20 * max(0.0, 0.55 - returning) / 0.55
    if quarterback is not None:
        roster += preseason_share * 0.12 * max(0.0, 0.50 - quarterback) / 0.50
    missing_preseason = 0.12 * preseason_share if not preseason_available else 0.0
    return float(1.0 + early + missing_preseason
                 + (0.18 if cross_tier else 0.0) + roster)


def load_free_preseason(client, seasons: list[int]) -> dict[str, pd.DataFrame]:
    """Budget-visible CFBD pulls: five calls per season, then permanent cache."""
    from .cfbd_client import APIBudgetExceeded

    result = {name: [] for name in ("returning", "portal", "recruiting", "talent")}
    endpoints = {
        "returning": "player/returning", "portal": "player/portal",
        "recruiting": "recruiting/teams", "talent": "talent",
    }
    for season in seasons:
        for name, endpoint in endpoints.items():
            result[name].append(client.frame(endpoint, f"{name}_{season}", year=season))
    # Coaches is not naturally a one-row-per-year endpoint; requesting by year keeps the
    # call bounded and the raw response cached even when it is empty.
    result["coaching"] = []
    for season in seasons:
        try:
            result["coaching"].append(
                client.frame("coaches", f"coaches_{season}", year=season)
            )
        except APIBudgetExceeded:
            # Coaching continuity is valuable but not safe to fabricate and not required
            # to produce an independent score. Preserve it as missing so publication
            # quality fails closed for that feature while cached football/roster inputs
            # remain usable on CFBD's free tier.
            break
    return {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            for name, frames in result.items()}


__all__ = ["PLAY_FEATURES", "PRESEASON_FEATURES", "add_preseason_matchup_features",
           "build_team_game_features", "game_uncertainty_multiplier", "load_free_preseason",
           "matchup_feature_table", "maturity_phase", "normalize_preseason_sources",
           "previous_season_power", "rolling_team_features"]
