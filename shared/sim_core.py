"""Sport-agnostic drive-level Monte Carlo game simulator.

ONE simulator, used by both nfl-model and ncaa-model. It lives here rather than in either
repo because the two sports must not drift apart: the whole reason this codebase spent
weeks fixing errors and re-creating them was duplicated state (a stale drive cache) getting
out of step with the code that produced it. Two hand-maintained copies of a 700-line
simulator would guarantee the same failure at a larger scale.

WHAT IS GENERIC AND WHAT IS NOT. Everything here works in terms of `net_epa` -- a single
number per offense describing expected efficiency against the current defense -- plus an
empirical field-position distribution and an empirical drive-outcome multinomial. None of
that is NFL-specific. What each sport supplies for itself:

  * how to build a drive table from its own play-by-play or API payload
  * how to fit ratings and turn them into `net_epa` (different ridge, different priors,
    and college needs conference shrinkage and an FCS bucket)
  * its own config, context adjustments, and calibration constants

So `simulate_game` takes `net_home` and `net_away` as plain floats. The caller resolves
ratings; this module never sees a ratings table.

CONVENTION: margins are `home_score - away_score`, and the model spread is `mean_margin`
on the same scale (positive = home favored).

The `cfg` argument is duck-typed. It must expose `cfg.simulation` (n_sims, seed,
xp_make_prob, two_point_attempt_rate, two_point_success_prob, defensive_td_lambda,
special_teams_td_lambda, safety_lambda, endgame_second_drive_prob, two_point_chart,
chart_follow_rate) and `cfg.pace` (league_mean_drives_per_team, drives_sd). Both repos'
Config satisfy this.

Every one of those simulation constants is a MEASURED property of its own sport. None may
be carried across from the other -- the two leagues differ on all eleven, and on several
the college value moves in the opposite direction to the intuition (see the endgame and
two-point block further down, and ncaa-model's config/ncaa.yaml).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

TD = "TD"
FG = "FG"
NO_SCORE = "NO_SCORE"
TURNOVER = "TURNOVER"

# Order is fixed and is the column order of every probability matrix in the simulator.
DRIVE_CLASSES = [TD, FG, NO_SCORE, TURNOVER]
DRIVE_CLASS_INDEX = {c: i for i, c in enumerate(DRIVE_CLASSES)}

# The drive table's schema, in one place so the builder and the cache guard cannot drift
# apart. Adding a column here is what invalidates an old cache on disk.
DRIVE_COLUMNS = [
    "game_id", "season", "week", "season_type", "offense", "defense", "drive_number",
    "start_yardline_100", "n_plays", "result", "points_scored_on_drive",
    "is_home_offense", "prev_drive_turnover", "start_score_diff", "start_qtr",
    "start_gsr",
]



@dataclass(frozen=True)
class StartFieldPosition:
    """Empirical starting-field-position distributions, conditioned on nothing but
    whether the previous drive ended in a turnover (§6b)."""

    values: np.ndarray          # yardline_100 support, shared by both distributions
    normal_cdf: np.ndarray
    post_turnover_cdf: np.ndarray

    def sample(self, rng: np.random.Generator, post_turnover: np.ndarray) -> np.ndarray:
        """Vectorized inverse-CDF draw. `post_turnover` is a boolean array; the result
        has the same shape."""
        u = rng.random(post_turnover.shape)
        normal_idx = np.searchsorted(self.normal_cdf, u)
        turnover_idx = np.searchsorted(self.post_turnover_cdf, u)
        idx = np.where(post_turnover, turnover_idx, normal_idx)
        idx = np.clip(idx, 0, len(self.values) - 1)
        return self.values[idx]

    def marginal_pmf(self, turnover_rate: float = 0.115) -> np.ndarray:
        """Unconditional field-position distribution, mixing the two cuts by how often a
        drive actually follows a turnover.

        Used to average model quantities over field position instead of evaluating them
        at a single "mean" field position, which is biased whenever the quantity is
        non-linear in position -- as expected points is.
        """
        normal = np.diff(np.concatenate([[0.0], self.normal_cdf]))
        post = np.diff(np.concatenate([[0.0], self.post_turnover_cdf]))
        mix = (1.0 - turnover_rate) * normal + turnover_rate * post
        return mix / mix.sum()

    @property
    def mean_normal(self) -> float:
        pmf = np.diff(np.concatenate([[0.0], self.normal_cdf]))
        return float((self.values * pmf).sum())

    @property
    def mean_post_turnover(self) -> float:
        pmf = np.diff(np.concatenate([[0.0], self.post_turnover_cdf]))
        return float((self.values * pmf).sum())


def fit_start_field_position(drives: pd.DataFrame) -> StartFieldPosition:
    """Fit the two empirical distributions over `start_yardline_100`."""
    support = np.arange(1, 100)

    def cdf_for(mask: pd.Series) -> np.ndarray:
        vals = drives.loc[mask, "start_yardline_100"].dropna().astype(int)
        counts = np.bincount(vals.to_numpy(), minlength=100)[1:100].astype(float)
        if counts.sum() == 0:
            raise ValueError("drives.fit_start_field_position: no drives to fit")
        return np.cumsum(counts / counts.sum())

    return StartFieldPosition(
        values=support,
        normal_cdf=cdf_for(~drives["prev_drive_turnover"]),
        post_turnover_cdf=cdf_for(drives["prev_drive_turnover"]),
    )


FEATURES = ["net_epa", "start_fp", "start_fp_sq", "is_home_offense", "net_epa_x_fp"]


@dataclass(frozen=True)
class MultinomialModel:
    """Fitted coefficients, kept as plain arrays so the simulator can evaluate them on
    (n_sims, n_drives) matrices without sklearn in the hot loop."""

    coef: np.ndarray            # (4, 5)
    intercept: np.ndarray       # (4,)
    classes: list[str]
    as_of_season: int
    n_drives_trained: int
    fp_mean: float
    fp_scale: float

    def probabilities(
        self,
        net_epa: np.ndarray,
        start_fp: np.ndarray,
        is_home_offense: "float | np.ndarray",
    ) -> np.ndarray:
        """Softmax probabilities for arbitrarily shaped inputs.

        Returns shape `net_epa.shape + (4,)`, class axis last, in DRIVE_CLASSES order.
        """
        net_epa = np.asarray(net_epa, dtype=float)
        fp = (np.asarray(start_fp, dtype=float) - self.fp_mean) / self.fp_scale
        net_epa, fp = np.broadcast_arrays(net_epa, fp)
        home = np.broadcast_to(np.asarray(is_home_offense, dtype=float), net_epa.shape)
        design = np.stack([net_epa, fp, fp ** 2, home, net_epa * fp], axis=-1)
        logits = design @ self.coef.T + self.intercept
        logits -= logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=-1, keepdims=True)

    def cumulative(self, *args, **kwargs) -> np.ndarray:
        """Cumulative probabilities, for inverse-CDF sampling."""
        return np.cumsum(self.probabilities(*args, **kwargs), axis=-1)

    def to_json(self) -> str:
        return json.dumps({
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "classes": self.classes,
            "as_of_season": self.as_of_season,
            "n_drives_trained": self.n_drives_trained,
            "fp_mean": self.fp_mean,
            "fp_scale": self.fp_scale,
        })

    @staticmethod
    def from_json(s: str) -> "MultinomialModel":
        d = json.loads(s)
        return MultinomialModel(
            coef=np.array(d["coef"]),
            intercept=np.array(d["intercept"]),
            classes=d["classes"],
            as_of_season=d["as_of_season"],
            n_drives_trained=d["n_drives_trained"],
            fp_mean=d["fp_mean"],
            fp_scale=d["fp_scale"],
        )




# Score-state buckets, from the offense's perspective, as (low, high) inclusive.
ENDGAME_BUCKETS = [
    (-999, -9), (-8, -4), (-3, -1), (0, 0), (1, 3), (4, 8), (9, 999),
]


@dataclass(frozen=True)
class EndgameTable:
    """Empirical drive-outcome probabilities late in a game, by score state.

    WHY THIS EXISTS. §6c prescribes independent drives and §6d requires the simulated
    margin distribution to spike at 3 and 7 the way real NFL margins do. Those two
    requirements are in conflict, and it is provable from the data rather than a matter of
    opinion: drawing two independent scores from the *real* historical score distribution
    reproduces the real margin spread (sd 14.1 vs 14.2) but yields P(|margin| = 3) = 6.4%
    against an actual 14.7%. No simulator with independent possessions can clear the §6d
    gate, because the spikes are created by teams playing to the scoreboard, not by lumpy
    scoring increments.

    The fix keeps §6's architecture and adds the missing mechanism, measured from the data
    rather than assumed: on late drives, outcome probabilities are conditioned on the
    current margin. The effect is large and unambiguous -- tied late, a drive ends in a
    field goal 28.5% of the time against a 15.9% baseline (this is literally where the
    3-point spike comes from); trailing by 4-8, teams kick on 0.3% of drives because a
    field goal is worthless to them; leading, 87% of drives end with no score because the
    offense is killing clock. Logged in DECISIONS.md.
    """

    cum: np.ndarray              # (n_buckets, 4) cumulative probabilities
    counts: np.ndarray           # (n_buckets,) sample sizes
    classes: list

    def bucket_index(self, score_diff: np.ndarray) -> np.ndarray:
        """Vectorized bucket lookup for an array of score differentials."""
        idx = np.zeros(np.shape(score_diff), dtype=int)
        for i, (lo, hi) in enumerate(ENDGAME_BUCKETS):
            idx = np.where((score_diff >= lo) & (score_diff <= hi), i, idx)
        return idx

    def to_json(self) -> str:
        return json.dumps({
            "cum": self.cum.tolist(),
            "counts": self.counts.tolist(),
            "classes": self.classes,
        })

    @staticmethod
    def from_json(s: str) -> "EndgameTable":
        d = json.loads(s)
        return EndgameTable(
            cum=np.array(d["cum"]),
            counts=np.array(d["counts"]),
            classes=d["classes"],
        )


def fit_endgame_table(
    drives: pd.DataFrame,
    train_start: int,
    as_of_season: int,
    cache_path: "object | None" = None,
    late_seconds: float = 300.0,
) -> EndgameTable:
    """Empirical P(outcome | score state) for drives inside the last `late_seconds` of
    regulation. Fit on seasons strictly before `as_of_season`.

    `cache_path` is supplied by the caller because each sport owns its own cache directory;
    pass None to skip caching entirely.

    Both a fresh fit AND a load from `cache_path` are validated to have at least one
    counted drive before being trusted. Found 2026-08-17: a run that reached this call
    with an empty `drives` (or one missing `start_qtr`/`start_gsr`/`start_score_diff`)
    silently produced `counts.sum() == 0` and an all-NaN `cum` -- and `path.exists()` was
    the only cache-freshness check, so that broken table was cached once and served to
    every simulated game since. It is not merely inert: `_play_drive`'s
    `(u[:, None] > cum).sum(axis=1)` evaluates to 0 for an all-NaN row (every comparison
    against NaN is False), and index 0 is `TD` -- so every endgame-masked drive (a team's
    final 1-2 drives of every simulated game) was resolving to a GUARANTEED touchdown
    instead of the scoreboard-conditioned outcome this table exists to produce. Same bug
    class this file's own callers have hit three times before: a cache that does not
    validate its inputs. Raising here, rather than adding a runtime NaN guard in
    `_play_drive`, is deliberate -- silently tolerating a degenerate table there would just
    make the next instance of this bug quiet again.
    """
    path = cache_path
    if path is not None and path.exists():
        cached = EndgameTable.from_json(path.read_text())
        if cached.counts.sum() > 0:
            return cached

    late = drives[
        (drives["season"] >= train_start)
        & (drives["season"] < as_of_season)
        & (drives["start_qtr"] == 4)
        & (drives["start_gsr"] <= late_seconds)
        & drives["start_score_diff"].notna()
    ]
    if late.empty:
        raise ValueError(
            f"fit_endgame_table: 0 late-game drives for seasons [{train_start}, "
            f"{as_of_season}) -- refusing to cache a degenerate table. Check that `drives` "
            "carries start_qtr/start_gsr/start_score_diff and covers this season range."
        )

    rows, counts = [], []
    overall = late["result"].value_counts(normalize=True)
    for lo, hi in ENDGAME_BUCKETS:
        grp = late[(late["start_score_diff"] >= lo) & (late["start_score_diff"] <= hi)]
        # Too thin to trust: fall back to the pooled late-game mix rather than to a noisy
        # handful of drives.
        probs = overall if len(grp) < 100 else grp["result"].value_counts(normalize=True)
        rows.append([float(probs.get(c, 0.0)) for c in DRIVE_CLASSES])
        counts.append(len(grp))

    p = np.array(rows)
    p = p / p.sum(axis=1, keepdims=True)
    table = EndgameTable(
        cum=np.cumsum(p, axis=1), counts=np.array(counts), classes=list(DRIVE_CLASSES)
    )
    if path is not None:
        path.write_text(table.to_json())
    return table




_TD = DRIVE_CLASS_INDEX["TD"]
_FG = DRIVE_CLASS_INDEX["FG"]
_TURNOVER = DRIVE_CLASS_INDEX["TURNOVER"]

_MIN_DRIVES, _MAX_DRIVES = 8, 16

# WHICH OF A TEAM'S FINAL DRIVES ARE PLAYED FROM THE SCOREBOARD, and which post-touchdown
# margins pull a two-point try, are both SPORT-SPECIFIC and live on `cfg.simulation`. All
# three were hardcoded at the NFL's values until they were measured for college and found
# to differ on every one. See `analysis/measure_simulation_constants.py` in ncaa-model,
# which reproduces the NFL numbers quoted here from the NFL drive table.
#
# endgame_second_drive_prob -- the endgame table is fit on drives inside the last 5:00 of
#   Q4. Among team-games that get at least one such drive (the relevant conditioning,
#   because the simulator gives every team a final drive by construction) the NFL averages
#   1.3408 of them -- 68% one, 30% two, 2% three -- and college 1.4243. The last drive
#   always qualifies; the one before it qualifies with this probability: 0.34 vs 0.4243.
#
#   Applying it to a flat two drives instead over-uses the mechanism: the endgame table has
#   an 87% no-score rate for a leading offense, so forcing two clock-killing drives per team
#   collapsed simulated margin sd to 12.46 against a real 14.2.
#   tests/test_simulate.py::test_simulated_margin_spread_is_realistic catches exactly that.
#
# two_point_chart / chart_follow_rate -- the league-average attempt rate is right on
#   average, but drawing the attempt at *random* is wrong in a way that costs the model its
#   key numbers. Real two-point decisions are caused by the margin, so they pull final
#   scores onto 3 and 7; an independent coin flip smears them off. Applying the chart only
#   on endgame drives -- the ones already played from the scoreboard -- lifted NFL P(|3|)
#   from 10.4% to within a rounding error of the historical 14.0%.
#
#   The college chart is NOT the NFL's. Fitting attempt rate per post-touchdown margin over
#   45,385 college touchdown drives puts the spikes at -18, -12, -5, -2, +5, +12 and +19
#   against the NFL's -10, -5, -2, +1, +5: only three coincide, and college adds a two-score
#   ladder the NFL chart has no entries for. College coaches also follow their chart far
#   less often -- 0.472 of the time against 0.80.


@dataclass
class SimResult:
    """Joint integer score lattice; calibration changes probability, never outcomes."""

    margins: np.ndarray          # home - away
    totals: np.ndarray
    home_scores: np.ndarray
    away_scores: np.ndarray
    home: str = ""
    away: str = ""
    context: "ContextAdjustment | None" = None
    weights: "np.ndarray | None" = None

    def __post_init__(self):
        arrays=[]
        for name in ("margins","totals","home_scores","away_scores"):
            values=np.asarray(getattr(self,name),dtype=float)
            if values.ndim!=1 or not np.isfinite(values).all(): raise ValueError(f"{name} must be one-dimensional and finite")
            setattr(self,name,values); arrays.append(values)
        n=len(arrays[0])
        if n==0 or any(len(x)!=n for x in arrays[1:]): raise ValueError("simulation arrays must be non-empty and have equal length")
        weights=np.full(n,1/n) if self.weights is None else np.asarray(self.weights,dtype=float)
        if weights.ndim!=1 or len(weights)!=n or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0: raise ValueError("invalid simulation weights")
        self.weights=weights/weights.sum()

    @property
    def mean_margin(self) -> float:
        """The model spread, on nflverse's convention (positive = home favored)."""
        return float(self.weights@self.margins)

    @property
    def median_margin(self) -> float:
        order=np.argsort(self.margins,kind="stable"); cumulative=np.cumsum(self.weights[order])
        if np.all(self.weights==self.weights[0]): return float(np.median(self.margins))
        return float(self.margins[order[min(np.searchsorted(cumulative,.5),len(order)-1)]])

    @property
    def mean_total(self) -> float:
        return float(self.weights@self.totals)

    @property
    def n_sims(self) -> int:
        return len(self.margins)

    # -- probabilities ----------------------------------------------------------------

    def cover_prob(self, line: float, side: str) -> float:
        """P(this side covers | the bet resolves).

        A push is neither a win nor a loss, so the probability returned is conditional on
        the bet resolving: P(win) / (P(win) + P(loss)). Ignoring this systematically
        overstates edge on whole-number spreads -- which are exactly the lines sitting on
        3 and 7.
        """
        home_wins = self._probability(self.margins > line)
        away_wins = self._probability(self.margins < line)
        return _conditional(home_wins, away_wins, side, ("home", "away"))

    def total_prob(self, line: float, side: str) -> float:
        over = self._probability(self.totals > line)
        under = self._probability(self.totals < line)
        return _conditional(over, under, side, ("over", "under"))

    def margin_pmf(self) -> dict:
        return self._pmf(self.margins)

    def total_pmf(self) -> dict: return self._pmf(self.totals)
    def push_prob(self,line:float,market:str="spread"):
        values=self.margins if market=="spread" else self.totals if market=="total" else None
        if values is None: raise ValueError("market must be 'spread' or 'total'")
        return self._probability(values==line)
    def _probability(self,mask): return float(self.weights[np.asarray(mask,dtype=bool)].sum())
    def _pmf(self,values):
        unique,inverse=np.unique(values,return_inverse=True); mass=np.bincount(inverse,weights=self.weights,minlength=len(unique))
        return {(int(round(float(v))) if np.isclose(v,round(float(v)),atol=1e-12,rtol=0) else float(v)):float(p) for v,p in zip(unique,mass)}

    def recentered(self, new_mean_margin: float) -> "SimResult":
        return self._calibrated(float(new_mean_margin),self.mean_total)

    def retotaled(self, new_mean_total: float) -> "SimResult":
        return self._calibrated(self.mean_margin,float(new_mean_total))

    def _calibrated(self,margin,total):
        weights=_calibration_weights(self.margins,self.totals,self.weights,np.array([margin,total]))
        return SimResult(self.margins,self.totals,self.home_scores,self.away_scores,self.home,self.away,self.context,weights)


def _calibration_weights(margins,totals,base_weights,targets):
    if not np.isfinite(targets).all(): raise ValueError("calibration targets must be finite")
    raw=np.column_stack([margins,totals]).astype(float,copy=False); current=base_weights@raw
    if np.allclose(current,targets,atol=1e-12,rtol=0): return base_weights.copy()
    varying=np.ptp(raw,axis=0)>1e-12
    for i,name in enumerate(("margin","total")):
        lo,hi=float(raw[:,i].min()),float(raw[:,i].max())
        if targets[i]<lo-1e-10 or targets[i]>hi+1e-10: raise ValueError(f"requested mean {name} outside simulated support")
        if not varying[i] and not np.isclose(targets[i],lo,atol=1e-10,rtol=0): raise ValueError(f"requested mean {name} is infeasible")
    features=raw[:,varying]; wanted=targets[varying]; center=(features.min(0)+features.max(0))/2; scale=np.ptp(features,axis=0)
    features=(features-center)/scale; wanted=(wanted-center)/scale
    log_base=np.full(len(base_weights),-np.inf); positive=base_weights>0; log_base[positive]=np.log(base_weights[positive])
    def objective(theta):
        logits=log_base+features@theta; peak=float(np.max(logits)); un=np.exp(logits-peak); prob=un/un.sum()
        return peak+np.log(un.sum())-float(theta@wanted), prob@features-wanted
    fit=minimize(objective,np.zeros(features.shape[1]),method="BFGS",jac=True,options={"gtol":1e-11,"maxiter":300})
    logits=log_base+features@fit.x; logits-=float(np.max(logits)); calibrated=np.exp(logits)
    calibrated=np.where(positive,np.maximum(calibrated,np.finfo(float).tiny),0); calibrated/=calibrated.sum()
    if not np.allclose(calibrated@raw,targets,atol=1e-6,rtol=0): raise ValueError("requested means are not jointly feasible on this score lattice")
    return calibrated


def _conditional(a: float, b: float, side: str, names: tuple) -> float:
    denom = a + b
    if denom <= 0:
        return 0.5
    if side == names[0]:
        return a / denom
    if side == names[1]:
        return b / denom
    raise ValueError(f"side must be one of {names}, got {side!r}")


# --------------------------------------------------------------------------------------
# Points <-> net_epa conversion
# --------------------------------------------------------------------------------------

def expected_points_per_drive(
    model: MultinomialModel,
    net: float,
    cfg: Config,
    mean_fp: float = 70.0,
    fp_support: "np.ndarray | None" = None,
    fp_pmf: "np.ndarray | None" = None,
) -> float:
    """Analytic expected points for one drive at a given net_epa.

    Averaged over the empirical field-position distribution when one is supplied.
    Evaluating at a single "mean" field position instead biases the result, because
    expected points is convex in position: the derivative at fp=70 understates the true
    average slope, which made every context adjustment over-apply by about 20%.
    """
    if fp_support is not None and fp_pmf is not None:
        probs = model.probabilities(
            np.full_like(fp_support, net, dtype=float), fp_support, 0.5
        )
        p = (probs * fp_pmf[:, None]).sum(axis=0)
    else:
        p = model.probabilities(np.array(net), np.array(mean_fp), 0.5)
    conv = (
        cfg.simulation.two_point_attempt_rate * 2.0 * cfg.simulation.two_point_success_prob
        + (1 - cfg.simulation.two_point_attempt_rate) * cfg.simulation.xp_make_prob
    )
    return float(p[..., _TD] * (6.0 + conv) + p[..., _FG] * 3.0)


def points_per_net_epa(
    model: MultinomialModel,
    cfg: Config,
    expected_drives: float,
    mean_fp: float = 70.0,
    start_fp: "object | None" = None,
) -> float:
    """How many points of final score one unit of `net_epa` is worth.

    A two-sided numerical derivative of expected points per drive, scaled by the number of
    drives a team gets. Used to convert context adjustments into the model's own feature
    units.
    """
    h = 0.01
    support = pmf = None
    if start_fp is not None:
        support = np.asarray(start_fp.values, dtype=float)
        pmf = start_fp.marginal_pmf()
    hi = expected_points_per_drive(model, h, cfg, mean_fp, support, pmf)
    lo = expected_points_per_drive(model, -h, cfg, mean_fp, support, pmf)
    return (hi - lo) / (2 * h) * expected_drives


def net_epa_shift_for_points(
    points: float,
    model: MultinomialModel,
    cfg: Config,
    expected_drives: float,
    start_fp: "object | None" = None,
) -> float:
    """Convert a points-scale context adjustment into a `net_epa` shift.

    NOTE ON THE PLAYBOOK'S FORMULA. §6c prescribes `epa_per_drive = points /
    expected_drives` and adding that to `net_epa`. Those are different units: `net_epa` is
    EPA *per play* (league spread roughly +/-0.15), while points/drive for a 1.7-point HFA
    is about 0.15 -- so the literal formula would make home-field worth roughly ten times
    its actual value, and every other context adjustment with it. The intent of §6c is
    preserved exactly: the adjustment is applied to `net_epa` *before* the multinomial,
    never added to a simulated score. Only the conversion changes -- it goes through the
    model's own points-per-net_epa slope, so a 1.7-point HFA moves the simulated line by
    1.7 points. Logged in DECISIONS.md.
    """
    slope = points_per_net_epa(model, cfg, expected_drives, start_fp=start_fp)
    if abs(slope) < 1e-9:
        return 0.0
    return points / slope


# --------------------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------------------

def simulate_game(
    home: str,
    away: str,
    net_home: float,
    net_away: float,
    pace_home: float,
    pace_away: float,
    drive_model: MultinomialModel,
    context_adj: "object",
    cfg: "object",
    start_fp: StartFieldPosition,
    endgame: "EndgameTable | None" = None,
    rng: "np.random.Generator | None" = None,
    neutral_site: bool = False,
) -> SimResult:
    """Play the game `n_sims` times, drive by drive.

    Vectorized over simulations: the loop below runs once per drive *index* (at most 17
    iterations), not once per simulation. Every random draw inside it is an array of
    `n_sims` values.
    """
    rng = rng or np.random.default_rng(cfg.simulation.seed)
    n = cfg.simulation.n_sims

    # --- drive counts ----------------------------------------------------------------
    mu = (
        cfg.pace.league_mean_drives_per_team
        + float(pace_home)
        + float(pace_away)
    )
    drives = np.rint(
        np.clip(rng.normal(mu, cfg.pace.drives_sd, n), _MIN_DRIVES, _MAX_DRIVES)
    ).astype(int)
    # Odd-possession games are real, but the extra possession belongs to whoever receives
    # first, which is a coin flip. §6c says to "add one drive to the home team with
    # probability 0.5"; read literally that gives the home team +0.5 possessions every
    # game -- worth about a point of phantom home-field on top of the HFA that context.py
    # already supplies. The coin flip decides *which* team gets the extra drive, not
    # whether the home team does. Logged in DECISIONS.md.
    has_extra = rng.random(n) < 0.5
    extra_to_home = rng.random(n) < 0.5
    home_drives = drives + (has_extra & extra_to_home)
    away_drives = drives + (has_extra & ~extra_to_home)
    max_drives = int(max(home_drives.max(), away_drives.max()))

    # --- context, converted into net_epa units ---------------------------------------
    expected_drives = float(np.mean(drives))
    spread_shift = net_epa_shift_for_points(
        context_adj.spread_points, drive_model, cfg, expected_drives, start_fp
    )
    # The spread adjustment is split evenly: it helps the home offense and hurts the away
    # offense, which moves the margin by the requested amount while leaving the total
    # essentially unchanged.
    net_home_adj = net_home + spread_shift / 2.0
    net_away_adj = net_away - spread_shift / 2.0

    total_shift = net_epa_shift_for_points(
        context_adj.total_points / 2.0, drive_model, cfg, expected_drives, start_fp
    )
    net_home_adj += total_shift
    net_away_adj += total_shift

    # --- per-field-position probability tables ---------------------------------------
    # net_epa and home/away are constant within a team, so drive-outcome probabilities
    # depend only on the sampled field position. Precomputing all 99 rows turns each
    # drive step into an array index instead of a softmax.
    # HOME FIELD (PATCH 01 §0, Bug A). The drive multinomial's own `is_home_offense` term
    # is the league baseline -- it is estimated from data at the drive level, so it is the
    # one to keep. context.py no longer contributes a league HFA at all; it supplies only a
    # per-venue *deviation* centered on zero. A neutral site gets 0.5 on both sides, which
    # is how the league baseline is removed for a game that has no home team.
    home_flag = 0.5 if neutral_site else 1.0
    away_flag = 0.5 if neutral_site else 0.0
    fp_support = start_fp.values.astype(float)
    cum_home = np.cumsum(
        drive_model.probabilities(
            np.full_like(fp_support, net_home_adj), fp_support, home_flag
        ),
        axis=-1,
    )
    cum_away = np.cumsum(
        drive_model.probabilities(
            np.full_like(fp_support, net_away_adj), fp_support, away_flag
        ),
        axis=-1,
    )

    # Whether each simulated game gives a team a second scoreboard-driven drive.
    second_endgame_prob = cfg.simulation.endgame_second_drive_prob
    home_second_endgame = rng.random(n) < second_endgame_prob
    away_second_endgame = rng.random(n) < second_endgame_prob

    home_pts = np.zeros(n, dtype=np.int32)
    away_pts = np.zeros(n, dtype=np.int32)
    prev_home_turnover = np.zeros(n, dtype=bool)
    prev_away_turnover = np.zeros(n, dtype=bool)

    for i in range(max_drives):
        # Possessions alternate: away drive i, then home drive i. A team's final drive is
        # played from the scoreboard rather than from team strength (see EndgameTable).
        away_active = i < away_drives
        pts, away_to = _play_drive(
            rng, cum_away, start_fp, prev_home_turnover, away_active, cfg,
            endgame=endgame,
            endgame_mask=away_active & (
                (i == away_drives - 1)
                | ((i == away_drives - 2) & away_second_endgame)
            ),
            score_diff=away_pts - home_pts,
        )
        away_pts += pts
        prev_away_turnover = away_to

        home_active = i < home_drives
        pts, home_to = _play_drive(
            rng, cum_home, start_fp, prev_away_turnover, home_active, cfg,
            endgame=endgame,
            endgame_mask=home_active & (
                (i == home_drives - 1)
                | ((i == home_drives - 2) & home_second_endgame)
            ),
            score_diff=home_pts - away_pts,
        )
        home_pts += pts
        prev_home_turnover = home_to

    # --- rare events, outside the drive multinomial -----------------------------------
    s = cfg.simulation
    home_pts += _nonoffensive_td_points(rng, s.defensive_td_lambda, n, cfg)
    away_pts += _nonoffensive_td_points(rng, s.defensive_td_lambda, n, cfg)
    home_pts += _nonoffensive_td_points(rng, s.special_teams_td_lambda, n, cfg)
    away_pts += _nonoffensive_td_points(rng, s.special_teams_td_lambda, n, cfg)
    # A safety scores 2 for the team whose defense makes it, i.e. it is charged against
    # the opposing offense.
    home_pts += 2 * rng.poisson(s.safety_lambda, n).astype(np.int32)
    away_pts += 2 * rng.poisson(s.safety_lambda, n).astype(np.int32)

    home_pts, away_pts = _play_overtime(
        rng, cum_home, cum_away, start_fp, home_pts, away_pts, cfg, endgame,
        max_exchanges=int(getattr(cfg.simulation, "overtime_max_exchanges", 3)),
    )

    return SimResult(
        margins=(home_pts - away_pts).astype(float),
        totals=(home_pts + away_pts).astype(float),
        home_scores=home_pts.astype(float),
        away_scores=away_pts.astype(float),
        home=home,
        away=away,
        context=context_adj,
    )


def _nonoffensive_td_points(
    rng: np.random.Generator, lam: float, n: int, cfg: Config
) -> np.ndarray:
    """Points from defensive and return touchdowns, with the conversion actually drawn.

    A flat 7 per score is wrong in a way GATE_KEY_NUMBERS can see: extra points miss 5.75%
    of the time, and those misses are a meaningful share of the 6-point final margins the
    gate checks. Teams effectively always kick after a non-offensive touchdown, so only the
    extra point is modelled here -- no two-point branch.
    """
    count = rng.poisson(lam, n).astype(np.int32)
    pts = 6 * count
    for k in range(1, int(count.max()) + 1) if count.size else ():
        good = (rng.random(n) < cfg.simulation.xp_make_prob) & (count >= k)
        pts += good.astype(np.int32)
    return pts


def _play_overtime(
    rng: np.random.Generator,
    cum_home: np.ndarray,
    cum_away: np.ndarray,
    start_fp: StartFieldPosition,
    home_pts: np.ndarray,
    away_pts: np.ndarray,
    cfg: Config,
    endgame: "EndgameTable | None",
    max_exchanges: int = 3,
) -> tuple:
    """Resolve tied games.

    Both teams get a possession, which is the modern rule and also the reason overtime
    contributes to the 3-point spike rather than smearing it: the most common way a tied
    game ends is one field goal. Without this step the simulator leaves ~3.3% of games
    tied against a real 0.4%, and all of that misplaced mass is stolen from |margin| = 3.

    `max_exchanges` IS SPORT-SPECIFIC and comes from `cfg.simulation`. The NFL's 3 is right
    for the NFL: a game still tied after three exchanges stays tied, which lands close to
    its real regular-season tie rate of 0.4%. College football has NO ties -- overtime
    repeats until somebody wins -- and its measured tie rate over 4,464 FBS-vs-FBS games is
    exactly 0.00%. Leaving it at 3 left 0.63% of simulated college games tied, and that
    mass is stolen from the small margins the key-number gate cares about.

    NOTE: this still plays college overtime under NFL rules -- possessions from ordinary
    field position rather than first-and-10 at the opponent's 25 with no clock. The number
    of exchanges is now right; the scoring environment inside them is not. See the college
    overtime note in ncaa-model/config/ncaa.yaml.
    """
    n = len(home_pts)
    no_turnover = np.zeros(n, dtype=bool)
    for _ in range(max_exchanges):
        tied = home_pts == away_pts
        if not tied.any():
            break
        pts, _ = _play_drive(
            rng, cum_away, start_fp, no_turnover, tied, cfg,
            endgame=endgame, endgame_mask=tied, score_diff=away_pts - home_pts,
        )
        away_pts = away_pts + pts
        pts, _ = _play_drive(
            rng, cum_home, start_fp, no_turnover, tied, cfg,
            endgame=endgame, endgame_mask=tied, score_diff=home_pts - away_pts,
        )
        home_pts = home_pts + pts
    return home_pts, away_pts


def _play_drive(
    rng: np.random.Generator,
    cum_probs: np.ndarray,
    start_fp: StartFieldPosition,
    prev_opponent_turnover: np.ndarray,
    active: np.ndarray,
    cfg: Config,
    endgame: "EndgameTable | None" = None,
    endgame_mask: "np.ndarray | None" = None,
    score_diff: "np.ndarray | None" = None,
):
    """One drive for one team, across all simulations at once.

    Returns (points, ended_in_turnover). Simulations where this drive does not exist (the
    team has fewer drives in that sim) score zero and report no turnover.
    """
    n = len(prev_opponent_turnover)
    active = np.asarray(active)
    fp = start_fp.sample(rng, prev_opponent_turnover)
    fp_idx = np.clip(fp - 1, 0, cum_probs.shape[0] - 1)
    cum = cum_probs[fp_idx]

    if endgame is not None and endgame_mask is not None and np.any(endgame_mask):
        bucket = endgame.bucket_index(np.asarray(score_diff))
        cum = np.where(np.asarray(endgame_mask)[:, None], endgame.cum[bucket], cum)

    u = rng.random(n)
    outcome = (u[:, None] > cum).sum(axis=1)
    outcome = np.clip(outcome, 0, cum.shape[1] - 1)

    is_td = (outcome == _TD) & active
    is_fg = (outcome == _FG) & active

    # Post-touchdown conversion. Late in a game the decision is dictated by the scoreboard
    # rather than by the league-average rate; see _TWO_POINT_CHART.
    go_for_two = rng.random(n) < cfg.simulation.two_point_attempt_rate
    if score_diff is not None and endgame_mask is not None:
        post_td = np.asarray(score_diff) + 6
        chart = np.asarray(cfg.simulation.two_point_chart)
        on_chart = np.asarray(endgame_mask) & np.isin(post_td, chart)
        follow = rng.random(n) < cfg.simulation.chart_follow_rate
        go_for_two = np.where(on_chart, follow, go_for_two)
    two_good = rng.random(n) < cfg.simulation.two_point_success_prob
    xp_good = rng.random(n) < cfg.simulation.xp_make_prob
    conv = np.where(go_for_two, np.where(two_good, 2, 0), np.where(xp_good, 1, 0))

    points = np.zeros(n, dtype=np.int32)
    points[is_td] = 6 + conv[is_td]
    points[is_fg] = 3
    return points, (outcome == _TURNOVER) & active
