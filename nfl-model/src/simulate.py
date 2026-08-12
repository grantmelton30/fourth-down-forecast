"""NFL binding for the shared drive simulator. NON-NEGOTIABLE #3.

The simulator itself now lives in `../shared/sim_core.py` and is used unchanged by
ncaa-model. Nothing about the Monte Carlo is NFL-specific: it works in `net_epa` units on
top of an empirical field-position distribution and an empirical drive-outcome
multinomial. What *is* NFL-specific, and stays here, is the step that turns this repo's
ratings table into the two `net_epa` numbers the simulator wants.

Margins are never sampled from a normal distribution and points are never Poisson. Each
simulated game is played out drive by drive, which is the only way the lumpy structure of
NFL margins -- the spikes at 3 and 7 -- survives into the output.

CONVENTION: margins are `home_score - away_score`, matching nflverse's `result`, and the
model spread is `mean_margin` on the same scale as `schedules.spread_line` (positive =
home favored).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import sim_core
from .config import Config
from .context import ContextAdjustment
from .drive_model import EndgameTable, MultinomialModel
from .drives import StartFieldPosition
from .ratings import net_epa

# Re-exported so existing importers keep working unchanged.
SimResult = sim_core.SimResult
expected_points_per_drive = sim_core.expected_points_per_drive
points_per_net_epa = sim_core.points_per_net_epa
net_epa_shift_for_points = sim_core.net_epa_shift_for_points


def simulate_game(
    home: str,
    away: str,
    ratings: pd.DataFrame,
    drive_model: MultinomialModel,
    context_adj: ContextAdjustment,
    cfg: Config,
    start_fp: StartFieldPosition,
    endgame: "EndgameTable | None" = None,
    rng: "np.random.Generator | None" = None,
    neutral_site: bool = False,
) -> "sim_core.SimResult":
    """Play the game `n_sims` times, drive by drive.

    Resolves this repo's ratings into `net_epa` and hands off to the shared simulator.
    """
    h_row, a_row = ratings.loc[home], ratings.loc[away]
    return sim_core.simulate_game(
        home=home,
        away=away,
        net_home=net_epa(h_row["off_rating"], a_row["def_rating"], cfg),
        net_away=net_epa(a_row["off_rating"], h_row["def_rating"], cfg),
        pace_home=float(h_row["pace_rating"]),
        pace_away=float(a_row["pace_rating"]),
        drive_model=drive_model,
        context_adj=context_adj,
        cfg=cfg,
        start_fp=start_fp,
        endgame=endgame,
        rng=rng,
        neutral_site=neutral_site,
    )
