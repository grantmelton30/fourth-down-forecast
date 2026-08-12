# Manual prospective inputs

Manual files exist for information that has no dependable, documented free live feed.
They are evidence, not overrides of historical results. Never enter a row after kickoff.

`availability.csv` columns:

`season,week,team,full_name,position,report_status,observed_at,source`

`coaching_changes.csv` columns:

`season,team,head_coach_changed,offensive_coordinator_changed,defensive_coordinator_changed,observed_at,source`

The pipeline treats a missing row as **unknown**, not healthy/unchanged.
