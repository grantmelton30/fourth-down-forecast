"""NCAA binding for the shared drive simulator.

The simulator lives in `../shared/sim_core.py` and is the same one nfl-model uses. Nothing
about the Monte Carlo is sport-specific: it works in `net_epa` units on top of an empirical
field-position distribution and an empirical drive-outcome multinomial. What is
NCAA-specific, and stays here, is the step that turns this repo's ratings table into the
two `net_epa` numbers and two pace numbers the simulator wants.

Margins are never sampled from a normal distribution and points are never Poisson. Each
simulated game is played out drive by drive, which is the only way the lumpy structure of
football margins -- the spikes at 3 and 7 -- survives into the output.

CONVENTION: margins are `home_score - away_score`, positive = home favored, matching the
`spread_line` convention used throughout this repo.

THE MEAN PATH DOES NOT GO THROUGH HERE. `backtest.py` projects E[margin] and E[total] with
a direct linear map off the ridge ratings, because on the NFL build the simulator scored
worse than that map while contributing all of its excess dispersion (DECISIONS D4). This
module exists to produce a DISTRIBUTION -- cover probabilities, key numbers, the shape of
the margin -- not a better mean. Do not quietly promote `mean_margin` into the mean path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import sim_core
from .config import Config
from .context import NULL_CONTEXT, ContextAdjustment
from .drive_model import EndgameTable, MultinomialModel
from .drives import StartFieldPosition
from .ratings import net_epa, resolve_team_ratings

# Re-exported so importers match the NFL binding's surface.
SimResult = sim_core.SimResult
expected_points_per_drive = sim_core.expected_points_per_drive
points_per_net_epa = sim_core.points_per_net_epa
net_epa_shift_for_points = sim_core.net_epa_shift_for_points


def simulate_game(
    home: str,
    away: str,
    ratings: pd.DataFrame,
    drive_model: MultinomialModel,
    cfg: Config,
    start_fp: StartFieldPosition,
    context_adj: "ContextAdjustment | None" = None,
    endgame: "EndgameTable | None" = None,
    rng: "np.random.Generator | None" = None,
    neutral_site: bool = False,
) -> "sim_core.SimResult":
    """Play the game `cfg.simulation.n_sims` times, drive by drive.

    `ratings` is indexed by team and carries off_rating / def_rating / pace_rating.
    Lookups go through `ratings.resolve_team_ratings`, not `ratings.loc[team]`, so an FCS
    opponent resolves to the fitted `__FCS__` bucket instead of raising KeyError -- college
    schedules an FBS-vs-FCS game roughly every week of September.

    `context_adj` defaults to a null adjustment rather than being required, so a caller can
    get a distribution without having wired up weather. A null context is honest: it means
    no adjustment was applied, not that one was applied and came to zero.
    """
    h = resolve_team_ratings(ratings, home, cfg)
    a = resolve_team_ratings(ratings, away, cfg)

    return sim_core.simulate_game(
        home=home,
        away=away,
        net_home=net_epa(h["off_rating"], a["def_rating"], cfg),
        net_away=net_epa(a["off_rating"], h["def_rating"], cfg),
        pace_home=float(h.get("pace_rating", 0.0) or 0.0),
        pace_away=float(a.get("pace_rating", 0.0) or 0.0),
        drive_model=drive_model,
        context_adj=context_adj if context_adj is not None else NULL_CONTEXT,
        cfg=cfg,
        start_fp=start_fp,
        endgame=endgame,
        rng=rng,
        neutral_site=neutral_site,
    )
