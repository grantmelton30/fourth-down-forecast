# Manual prospective NCAA inputs

CFBD supplies returning production, portal, recruiting, talent, coaches, plays, drives,
ratings and lines within its free allowance. Confirmed quarterback and coordinator news
does not have a dependable structured free feed, so `preseason_features.csv` is accepted
with these columns:

`season,team,qb_continuity,head_coach_continuity,offensive_coordinator_continuity,defensive_coordinator_continuity,observed_at,source`

Missing values remain unknown. Each observation must predate the first affected kickoff.
