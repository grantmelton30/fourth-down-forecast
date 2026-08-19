"""Per-game context adjustments: venue home-field deviation and wind.

WHAT THIS DELIBERATELY DOES NOT DO. It does not add a league home-field advantage. The
drive multinomial estimates `is_home_offense` from the drive data itself, so the league
baseline is already inside the simulator; adding `hfa_league_mean` here would double-count
it. That was Bug A on the NFL build and the same trap exists here. What this supplies is
only the per-venue DEVIATION from that baseline, centred on zero. Neutral sites are handled
inside the simulator by setting the home flag to 0.5 on both sides, so there is nothing to
subtract here either.

`cfg.context.hfa_league_mean` (2.6) is therefore NOT read as a spread adjustment. It is the
centre that venue deviations are measured against, and nothing else.

The college build carries fewer adjustments than the NFL one on purpose. The NFL has rest,
travel, timezone and a week-18 resting-starters term; college has no bye-week structure
worth modelling at this level, and its travel effects are confounded with conference
strength in a way ~800 games a season cannot separate. Wind is included because its effect
on totals is large, well established, and measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .weather import GameWeather, weather_for_game


@dataclass(frozen=True)
class ContextAdjustment:
    """Points, home perspective. `spread_points` is added to the home team's expected
    margin; `total_points` is added to the expected total.

    Both are consumed by `sim_core.simulate_game`, which converts them into `net_epa` units
    through the model's own points-per-net_epa slope rather than adding them to a simulated
    score.
    """

    spread_points: float
    total_points: float
    components: dict = field(default_factory=dict)
    data_incomplete: bool = False
    incomplete_reasons: tuple = ()

    def with_qb(self, qb_points: float, note: str = "") -> "ContextAdjustment":
        """Fold the quarterback adjustment in, keeping the component breakdown intact.

        Ported from `nfl-model/src/context.py`, and this is the ONLY route by which the
        quarterback reaches a number anyone actually reads. `project_game.py` and the shared
        viewer both report the raw simulator mean, and the sole injection point into that
        mean is this object -- an adjustment applied in `project_walkforward` moves
        `model_spread`, which is a backtest quantity, and changes nothing a user sees
        (DECISIONS.md D17, 2026-08-19).

        Spread only: a missing quarterback lowers his team's expected margin. Whether the
        total should move is a separate empirical question and `nfl-model/src/injuries.py`
        measured it as nothing (-0.011, t = -0.05), so the total is left alone.
        """
        if not qb_points:
            return self
        comps = dict(self.components)
        comps["qb_points"] = float(qb_points)
        if note:
            comps["qb_note"] = note
        return ContextAdjustment(
            spread_points=self.spread_points + float(qb_points),
            total_points=self.total_points,
            components=comps,
            data_incomplete=self.data_incomplete,
            incomplete_reasons=self.incomplete_reasons,
        )


NULL_CONTEXT = ContextAdjustment(0.0, 0.0)


@dataclass(frozen=True)
class VenueHFA:
    """Per-venue home-field deviation from the league mean, shrunk toward zero.

    A venue needs `hfa_venue_shrink_games` home games to earn half its raw deviation, so a
    first-year venue with three blowouts does not get credited with a five-point edge.
    """

    deviation: dict
    league_mean: float

    def get(self, venue_id, neutral: bool = False) -> float:
        if neutral:
            return 0.0
        return float(self.deviation.get(venue_id, 0.0))


def estimate_venue_hfa(
    games: pd.DataFrame,
    cfg: Config,
    walkforward: "pd.DataFrame | None" = None,
    as_of=None,
) -> VenueHFA:
    """Fit venue HFA deviations on games strictly before `as_of`.

    MEASURED ON MARGIN RESIDUALS, NOT RAW MARGINS. This is the whole difficulty. Ohio State
    wins at home by 25 on average, but almost all of that is Ohio State, not Ohio Stadium --
    they win on the road too. Grouping raw home margins by venue therefore credits the
    stadium with the team's quality, and it is not a subtle error: doing exactly that gave
    Ohio Stadium a +14.5 point "venue advantage" and pushed a projected spread 50 points
    past the market. What is actually wanted is the part of the home margin that team
    strength does NOT explain, so the margin is first regressed on the ratings-implied
    strength difference and the venue effect is read off the residuals.

    The result is then shrunk toward the league residual by n / (n + shrink_games) and
    RE-CENTRED on zero, because the league-average home-field level belongs to the drive
    multinomial's `is_home_offense` term, not here.

    NO LOOKAHEAD. `as_of` is a kickoff timestamp and the filter is strict `<`. Passing None
    fits on everything, which is correct only for a descriptive report -- never a projection.
    """
    from .ratings import net_epa_vec

    df = games[
        games["homePoints"].notna()
        & games["awayPoints"].notna()
        & ~games["neutralSite"].fillna(False).astype(bool)
        & games["venue_id"].notna()
    ].copy()
    if as_of is not None:
        df = df[df["kickoff"] < as_of]
    if df.empty:
        return VenueHFA({}, 0.0)

    df["margin"] = df["homePoints"].astype(float) - df["awayPoints"].astype(float)

    if walkforward is not None:
        wf = walkforward[["season", "week", "team", "off_rating", "def_rating"]]
        h = wf.rename(columns={"team": "homeTeam", "off_rating": "h_off",
                               "def_rating": "h_def"})
        a = wf.rename(columns={"team": "awayTeam", "off_rating": "a_off",
                               "def_rating": "a_def"})
        df = df.merge(h, on=["season", "week", "homeTeam"], how="inner")
        df = df.merge(a, on=["season", "week", "awayTeam"], how="inner")
        if df.empty:
            return VenueHFA({}, 0.0)
        net_diff = (
            net_epa_vec(df["h_off"], df["a_def"], cfg)
            - net_epa_vec(df["a_off"], df["h_def"], cfg)
        )
        # Margin explained by strength alone. The intercept absorbs the league home-field
        # level, so the residual is already centred near zero by construction.
        X = np.column_stack([np.ones(len(df)), net_diff])
        beta = np.linalg.lstsq(X, df["margin"].to_numpy(float), rcond=None)[0]
        df["residual"] = df["margin"].to_numpy(float) - X @ beta
    else:
        # No ratings supplied: fall back to each home team's own overall margin, which nets
        # out most of team strength without needing a rating. Cruder, but never 14 points
        # wrong.
        df["residual"] = df["margin"] - df.groupby("homeTeam")["margin"].transform("mean")

    league_resid = float(df["residual"].mean())
    k = float(cfg.context.hfa_venue_shrink_games)
    grouped = df.groupby("venue_id")["residual"]
    n, mean = grouped.size(), grouped.mean()
    shrunk = (n * mean + k * league_resid) / (n + k)
    deviations = shrunk - shrunk.mean()
    return VenueHFA(
        deviation={vid: float(v) for vid, v in deviations.items()},
        league_mean=league_resid,
    )


def wind_total_adjustment(wind_mph: float, cfg: Config) -> float:
    """Points to add to the expected total for a given wind speed.

    Linear in wind and CENTRED on the league-average wind, so the sign flips: a calm game
    gains points and a windy one loses them. The centring is not cosmetic. The totals
    projection is fit on real games, which already carry average wind, so only the
    deviation from average is new information; applying the raw coefficient would drag
    every projected total down by about 1.3 points.

    College is fit linear from zero rather than with the NFL's 15 mph hinge -- see the
    config for the threshold sweep. Temperature and precipitation are deliberately absent:
    their effect is not distinguishable from noise, and adding them would spend degrees of
    freedom on nothing.

    ONE-SIDED since 2026-08-19, and the asymmetry is physical rather than fitted. Wind is a
    SUPPRESSOR: it degrades throwing and kicking. The absence of wind is not a scoring
    bonus, it is merely the absence of that suppression, so the positive half of the
    centred line -- "a calm game scores 1.31 points above baseline" -- asserts something
    the mechanism does not support. Measured directly on 2,985 graded games with real
    Open-Meteo readings (`DECISIONS.md` D16): applying the positive half made calm games
    WORSE by 0.42% and dome games worse by 1.81%, while the negative half improved 10-15mph
    games by 0.99% and 15mph+ games by 3.26%. Keeping only the half that is both physically
    motivated and empirically helpful improved overall totals RMSE by 0.342% against 0.057%
    for the two-sided form, and cannot by construction harm a game at or below average wind.

    No new constant was introduced to do this -- `wind_total_center_mph` and
    `wind_total_points_per_mph` are unchanged and still carry the measurement that produced
    them. Only the sign gate is new.
    """
    deviation = float(wind_mph) - cfg.context.wind_total_center_mph
    return min(-cfg.context.wind_total_points_per_mph * deviation, 0.0)


def build_context(
    game: pd.Series,
    cfg: Config,
    venue_hfa: "VenueHFA | None" = None,
    allow_network: bool = False,
    weather: "GameWeather | None" = None,
) -> ContextAdjustment:
    """Assemble the context adjustment for one game.

    `allow_network` defaults to False so a backtest cannot silently start making thousands
    of HTTP calls. A caller that wants weather must ask for it -- and must then check
    `data_incomplete`, because "no reading" is not the same as "calm".
    """
    comps: dict = {}
    incomplete: list[str] = []

    neutral = bool(game.get("neutralSite", False))
    hfa = venue_hfa.get(game.get("venue_id"), neutral) if venue_hfa is not None else 0.0
    comps["hfa_venue_deviation"] = hfa

    if weather is None:
        weather = weather_for_game(game, allow_network=allow_network)

    total_adj = 0.0
    comps["weather_source"] = weather.source
    if weather.indoor:
        # A dome is the calm end of the same one-sided relationship, so it gets the same
        # answer a calm outdoor game gets: no adjustment. `dome_total_bump` was derived as
        # `0.1867 * 7.0` -- the positive half of the centred line -- and measuring it
        # against 98 real indoor games made them 1.81% WORSE (DECISIONS.md D16). The
        # constant is left in config as the record of how it was derived; it is no longer
        # applied, for the same reason the positive half of the wind line is not.
        comps["dome_bump"] = 0.0
    elif weather.wind_mph is not None:
        wind_pts = wind_total_adjustment(weather.wind_mph, cfg)
        total_adj += wind_pts
        comps["wind_mph"] = weather.wind_mph
        comps["wind_total_pts"] = wind_pts
    else:
        # An outdoor game with no reading. Say so rather than treating it as calm: a silent
        # zero here is indistinguishable from a measured zero, and that is exactly how the
        # NFL build ran for its whole life without anyone noticing weather never loaded.
        incomplete.append("no weather for an outdoor game")

    return ContextAdjustment(
        spread_points=float(hfa),
        total_points=float(total_adj),
        components=comps,
        data_incomplete=bool(incomplete),
        incomplete_reasons=tuple(incomplete),
    )
