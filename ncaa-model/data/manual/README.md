# Manual prospective NCAA inputs

CFBD supplies returning production, portal, recruiting, talent, coaches, plays, drives,
ratings and lines within its free allowance. Confirmed quarterback and coordinator news
does not have a dependable structured free feed, so `preseason_features.csv` is accepted
with these columns:

`season,team,qb_continuity,head_coach_continuity,offensive_coordinator_continuity,defensive_coordinator_continuity,observed_at,source`

Missing values remain unknown. Each observation must predate the first affected kickoff.

## `qb_status.csv` -- in-season starter availability

`season,week,team,starter_out,observed_at,source`

Supplies the one thing the quarterback adjustment (`src/qb.py`, DECISIONS.md D17) cannot
get anywhere else on the live path. Historically an absence is read off the box score --
the incumbent threw zero passes -- but an upcoming game has no box score, so a human or an
ESPN pull says so here instead. A row present for a (season, week, team) **overrides** the
box-score reading in both directions, so it can also correct a false positive.

`starter_out` is a boolean. Only full absences belong here: a quarterback who starts and is
pulled mid-game is worth about -6.5 points and is knowable to nobody in advance, so it is
deliberately not modelled (see the module docstring for the measurement).

No file, or no row for a game, means **no adjustment** -- the same answer as "the starter is
playing". That is what makes a missing or failed feed degrade to the model's prior
behaviour rather than quietly moving a line.

CFBD has no injury or depth-chart endpoint (`/injuries`, `/depth`, `/depthchart`,
`/player/injuries` all 404), which is why this is manual rather than ingested.
