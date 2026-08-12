# Audit snapshot — grant-claude

Snapshot of the NFL/NCAA football modelling workspace, taken 2026-08-10 for external
review. Three model repositories plus the shared viewer and cross-repo docs.

    nfl-model/     NFL ratings, drive simulator, backtest, gates
    ncaa-model/    college equivalent; own measured constants
    shared/        sim_core.py (one simulator, both sports), sport.py adapter, Streamlit viewer
    GATES.md       gate registry — what each gate asserts and whether it passes
    NEXT_SESSION.md  current state, known-wrong list, working rules
    COMMIT_LOG.txt full history of all four repositories, as text

## What was REMOVED, and why

**Credentials.** `.env`, `CFBD_API_KEY=.txt` (both repos) and `Odds API.txt` are excluded.
The archive was then scanned for the literal key strings; see the verification counts
printed when it was built.

**Git directories.** Excluded deliberately, not for size: the CFBD key is committed in
`nfl-model` history at `9fdc876` and shipping `.git` would ship the key. `COMMIT_LOG.txt`
carries the full history as text instead, so the reasoning behind each change survives.

**Data and caches.** `data/`, `output/`, `.venv/`, `__pycache__/`, `*.parquet`. These are
regenerable and account for ~60 MB of the 64 MB working tree. The code that builds them is
all present.

## Reading it

Start with `NEXT_SESSION.md`, then `GATES.md`. Both are written for exactly this purpose.

The headline finding is that **neither model beats the market** — RMSE ratios against the
closing line are ~1.07 (NFL spread) and ~1.08 (NCAA spread), both above 1.0 — and both
models therefore refuse to publish picks. `shared/sport.py::bets_allowed` is where that is
enforced. That is the intended result, not an unfinished state.

## If you rotate nothing else

The CFBD key is in `nfl-model` git history and was in HEAD. It should be rotated regardless
of what this archive contains.
