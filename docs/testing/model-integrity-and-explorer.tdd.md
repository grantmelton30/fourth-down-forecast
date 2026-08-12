# Model integrity and explorer — TDD record

## Contract first (RED)

The behavior was defined before production edits in:

- `shared/tests/test_model_integrity.py`
- `ncaa-model/tests/test_live_mean.py`
- `shared/tests/test_explorer_feed.py`

The initial focused run failed because the maturity phase, preseason matchup
features, independent per-market calibration permissions, quality assessment,
validated NCAA live mean, and explorer artifact did not exist. That failure was
committed as `ebf4304` before implementation.

## Smallest implementation (GREEN)

The production implementation added:

1. A football-only NCAA live mean chosen by held-out-season RMSE.
2. Historical-support clipping to prevent linear extrapolation beyond the fit data.
3. Prior-season power plus normalized free preseason inputs, separated by maturity phase.
4. Independent spread and total calibration permissions.
5. Raw book-level market evidence with three-source consensus required for that label.
6. Game-specific evidence quality, missing-input reasons, and disagreement suppression.
7. A prebuilt static league/team explorer artifact, so the website never fits models.

The core implementation reached GREEN in `4f8bb16`. Final verification uses the
root `run_tests.py` entry point because NFL and NCAA intentionally have conflicting
top-level `src` package names and therefore run in isolated interpreters.

## Publication rule

Prospective records remain append-only. A changed source digest creates a new
prediction identity; older snapshots are never overwritten. Missing free inputs
remain explicit blockers and are never imputed as current facts.
