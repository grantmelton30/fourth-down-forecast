"""Context adjustments (§13), returned in points from the home team's perspective.

Everything here is a points-scale number with a named source, so every value that reaches
the workbook is traceable to the line of code that produced it (NON-NEGOTIABLE #4). The
simulator converts these to EPA-per-drive before the multinomial rather than adding them
to a simulated score -- adding points post hoc smears the spikes at 3 and 7 and defeats
the purpose of simulating drives at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MANUAL_DIR, Config
from .injuries import injury_spread_points
from .stadiums import STADIUMS, timezone_gap, travel_distance_miles
from .weather import GameWeather, weather_for_game


@dataclass(frozen=True)
class ContextAdjustment:
    """Points, home perspective. `spread_points` is added to the home team's expected
    margin; `total_points` is added to the expected total."""

    spread_points: float
    total_points: float
    components: dict = field(default_factory=dict)
    data_incomplete: bool = False
    incomplete_reasons: tuple = ()

    def with_qb(self, qb_points: float, note: str = "") -> "ContextAdjustment":
        """Fold the QB adjustment (§12) in, keeping the component breakdown intact."""
        comps = dict(self.components)
        comps["qb"] = qb_points
        if note:
            comps["qb_note"] = note
        return ContextAdjustment(
            spread_points=self.spread_points + qb_points,
            total_points=self.total_points,
            components=comps,
            data_incomplete=self.data_incomplete,
            incomplete_reasons=self.incomplete_reasons,
        )


# --------------------------------------------------------------------------------------
# Home-field advantage
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class VenueHFA:
    """Per-venue home-field *deviation* in points, centered on zero.

    PATCH 01 §0 (Bug A): the league-average home-field advantage now lives entirely in the
    drive multinomial's `is_home_offense` term, which is estimated at the drive level. This
    class carries only what is left over -- how much a particular stadium differs from that
    league baseline. Applying a league HFA here *and* the multinomial's term is the
    double-count that produced a ~4.4-point home edge.

    Estimated from the *residuals* of the ratings model rather than raw home win rate: a
    venue whose home teams happen to have been good would otherwise be credited with an
    advantage it never had.
    """

    by_team: pd.Series          # deviations, mean ~0 across venues
    league_mean: float          # the raw league-average residual, reported not applied
    n_by_team: pd.Series

    def get(self, home_team: str, neutral: bool) -> float:
        if neutral:
            return 0.0
        return float(self.by_team.get(home_team, 0.0))

    @property
    def mean_deviation(self) -> float:
        """Asserted to sit within 0.1 of zero by tests/test_ratings.py."""
        return float(self.by_team.mean()) if len(self.by_team) else 0.0


def estimate_venue_hfa(
    schedules: pd.DataFrame,
    walkforward: pd.DataFrame,
    cfg: Config,
    before_season: "int | None" = None,
) -> VenueHFA:
    """Per-venue deviation from the league-average home-field residual, centered on zero.

    Each venue's mean margin residual is shrunk toward the *league* mean residual by
    `n / (n + hfa_venue_shrink_games)` -- so a venue needs 40 home games to earn half its
    own estimate -- and then the league mean is subtracted, leaving only the stadium's
    idiosyncratic edge. The league level itself belongs to the drive multinomial
    (PATCH 01 §0, Bug A).

    `before_season` keeps this walk-forward: when projecting 2023, only games from 2022 and
    earlier inform the venue estimates.
    """
    games = schedules[
        schedules["result"].notna()
        & schedules["game_type"].eq("REG")
        & schedules["location"].eq("Home")
    ]
    if before_season is not None:
        games = games[games["season"] < before_season]

    league_mean = cfg.context.hfa_league_mean
    if games.empty:
        return VenueHFA(pd.Series(dtype=float), league_mean, pd.Series(dtype=float))

    resid = _margin_residuals(games, walkforward, cfg)
    if resid.empty:
        return VenueHFA(pd.Series(dtype=float), league_mean, pd.Series(dtype=float))

    grp = resid.groupby("home_team")["residual"]
    n, mean = grp.size(), grp.mean()
    observed_league = float(resid["residual"].mean())
    k = cfg.context.hfa_venue_shrink_games
    shrunk = (n * mean + k * observed_league) / (n + k)
    # Center: what survives is each stadium's deviation from the league baseline, and the
    # baseline is the multinomial's job.
    deviations = shrunk - shrunk.mean()
    return VenueHFA(by_team=deviations, league_mean=observed_league, n_by_team=n)


def _margin_residuals(
    games: pd.DataFrame, walkforward: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Actual margin minus the margin implied by team strength alone (no HFA term).

    The strength-to-points scale is estimated on the same historical sample; it is a
    nuisance parameter here, not a forecast, so an in-sample fit is appropriate.
    """
    from .ratings import net_epa_vec

    wf = walkforward.set_index(["season", "week", "team"])[["off_rating", "def_rating"]]
    lookup = wf.to_dict("index")
    rows = []
    for g in games.itertuples(index=False):
        h = lookup.get((g.season, g.week, g.home_team))
        a = lookup.get((g.season, g.week, g.away_team))
        if h is None or a is None:
            continue
        rows.append({
            "home_team": g.home_team,
            "net_diff": float(
                net_epa_vec(h["off_rating"], a["def_rating"], cfg)
                - net_epa_vec(a["off_rating"], h["def_rating"], cfg)
            ),
            "result": float(g.result),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    scale = float(
        np.linalg.lstsq(
            df[["net_diff"]].to_numpy(), df["result"].to_numpy(), rcond=None
        )[0][0]
    )
    df["residual"] = df["result"] - scale * df["net_diff"]
    return df


# --------------------------------------------------------------------------------------
# The main entry point
# --------------------------------------------------------------------------------------

def wind_total_adjustment(wind_mph: float, cfg: Config) -> float:
    """Points to add to the expected total for a given wind speed.

    Linear in wind and CENTRED on the league-average wind, so the sign flips: a calm game
    gains points and a windy one loses them. The centring is not cosmetic -- the totals
    projection is fit on real games that already carry average wind, so only the deviation
    from average is new information. Applying the raw slope would drag every projected
    total down by about 2.7 points.

    Measured at -0.3339 pts/mph (t = -5.03) on 1,792 outdoor games; see config/nfl.yaml
    for the threshold sweep that rejected the old 15 mph hinge.
    """
    deviation = float(wind_mph) - cfg.context.wind_total_center_mph
    return -cfg.context.wind_total_points_per_mph * deviation


def build_context(
    game: pd.Series,
    cfg: Config,
    venue_hfa: "VenueHFA | None" = None,
    allow_network: bool = False,
    resting_starters: "set | None" = None,
    injury_burden: "dict | None" = None,
) -> ContextAdjustment:
    """Assemble every context adjustment for one game."""
    home, away = game["home_team"], game["away_team"]
    neutral = game.get("location") == "Neutral"
    comps: dict = {}
    incomplete: list[str] = []

    # --- HFA: per-venue deviation only (PATCH 01 §0, Bug A) --------------------------
    # The league baseline is the drive multinomial's `is_home_offense` term. Neutral sites
    # are handled inside the simulator by setting that flag to 0.5 on both sides, so there
    # is nothing to subtract here.
    hfa = venue_hfa.get(home, neutral) if venue_hfa is not None else 0.0
    comps["hfa_venue_deviation"] = hfa

    # --- Rest ------------------------------------------------------------------------
    home_rest, away_rest = game.get("home_rest"), game.get("away_rest")
    rest = 0.0
    if pd.notna(home_rest) and pd.notna(away_rest):
        rest = cfg.context.rest_point_per_day * (float(home_rest) - float(away_rest))
        rest = float(
            np.clip(rest, -cfg.context.rest_max_points, cfg.context.rest_max_points)
        )
        # A short week penalizes whoever is on four days, on top of the differential.
        if float(home_rest) <= 4:
            rest -= cfg.context.short_week_penalty
        if float(away_rest) <= 4:
            rest += cfg.context.short_week_penalty
    comps["rest"] = rest

    # --- Travel and time zones -------------------------------------------------------
    travel = tz_pen = 0.0
    if not neutral and home in STADIUMS and away in STADIUMS:
        miles = travel_distance_miles(away, home)
        travel = cfg.context.travel_points_per_1000mi * miles / 1000.0
        kickoff = game.get("kickoff")
        hour = pd.Timestamp(kickoff).hour if pd.notna(kickoff) else 13
        # A West Coast body clock in a 1pm Eastern kickoff: the away team travelling east
        # into an early start.
        if timezone_gap(away, home) >= 2 and hour <= 13:
            tz_pen = cfg.context.timezone_cross_penalty
    comps["travel"] = travel
    comps["timezone"] = tz_pen

    # --- Weather (total only; wind has essentially no effect on the spread) -----------
    wx = weather_for_game(game, allow_network=allow_network)
    total_adj = 0.0
    if wx.indoor:
        total_adj += cfg.context.dome_total_bump
        comps["dome_bump"] = cfg.context.dome_total_bump
    elif wx.wind_mph is not None and wx.wind_mph <= cfg.context.wind_max_plausible_mph:
        total_adj += wind_total_adjustment(wx.wind_mph, cfg)
        comps["wind_mph"] = wx.wind_mph
        comps["wind_total_pts"] = wind_total_adjustment(wx.wind_mph, cfg)
    elif wx.wind_mph is not None:
        # Implausible reading -- treat as missing, never as calm. See
        # wind_max_plausible_mph in the config.
        incomplete.append(f"implausible wind reading {wx.wind_mph:.0f} mph, ignored")
        comps["wind_mph_rejected"] = wx.wind_mph
    else:
        incomplete.append("no weather for an outdoor game")
    comps["weather_source"] = wx.source

    # --- Week 18 resting starters (manual flag file) ---------------------------------
    rest_pts = 0.0
    if resting_starters:
        season, week = game.get("season"), game.get("week")
        if (season, week, home) in resting_starters:
            rest_pts -= cfg.context.week18_rest_starters_points
        if (season, week, away) in resting_starters:
            rest_pts += cfg.context.week18_rest_starters_points
    comps["resting_starters"] = rest_pts

    # --- Injuries (spread only; see src/injuries.py for why not the total) -----------
    inj_pts = 0.0
    if injury_burden:
        season, week = int(game["season"]), int(game["week"])
        h = injury_burden.get((season, week, game["home_team"]), 0.0)
        a = injury_burden.get((season, week, game["away_team"]), 0.0)
        if h or a:
            inj_pts = injury_spread_points(h, a, cfg)
            comps["injury_burden_home"] = h
            comps["injury_burden_away"] = a
            comps["injury_points"] = inj_pts

    spread_points = hfa + rest + travel - tz_pen + rest_pts + inj_pts
    return ContextAdjustment(
        spread_points=spread_points,
        total_points=total_adj,
        components=comps,
        data_incomplete=bool(incomplete),
        incomplete_reasons=tuple(incomplete),
    )


def load_resting_starters() -> set:
    """`data/manual/resting_starters.csv` -> {(season, week, team)}.

    Deliberately manual (§13): the information is qualitative and arrives late.
    """
    path = MANUAL_DIR / "resting_starters.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    needed = {"season", "week", "team"}
    if not needed <= set(df.columns):
        raise ValueError(f"resting_starters.csv must have columns {sorted(needed)}")
    return {
        (int(r.season), int(r.week), str(r.team)) for r in df.itertuples(index=False)
    }


def weather_table(schedules: pd.DataFrame, allow_network: bool = False) -> pd.DataFrame:
    """Resolve weather for a whole slate at once, for reporting and diagnostics."""
    recs = []
    for _, g in schedules.iterrows():
        wx: GameWeather = weather_for_game(g, allow_network)
        recs.append({
            "game_id": g["game_id"], "temp_f": wx.temp_f, "wind_mph": wx.wind_mph,
            "indoor": wx.indoor, "weather_source": wx.source,
        })
    return pd.DataFrame(recs)
