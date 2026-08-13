# Automatic refresh and NCAA turnover — TDD evidence

Date: 2026-08-12

## Contract

- A scheduled run refreshes source data, rebuilds both league models, regenerates the
  prediction ledger and league explorer, validates both leagues, and only then promotes
  public artifacts.
- A failed betting gate remains a betting blocker but does not falsely classify a valid
  projection build as a crashed refresh.
- NCAA roster turnover may widen the forecast interval; it cannot add an unvalidated
  point penalty to the independent mean.
- Roster and missing-preseason uncertainty fade through Week 5 and are gone after that,
  when current-season observations should control.

## RED evidence

Initial focused run:

```text
3 failed, 13 passed
TypeError: game_uncertainty_multiplier() got an unexpected keyword argument
TypeError: forecast_from_projection() got an unexpected keyword argument
AssertionError: scheduled workflow did not restore/cache/rebuild/validate
```

Decay/durability follow-up:

```text
2 failed, 5 passed
assert game_uncertainty_multiplier(... week=8, preseason_available=False) == 1.0
assert "if: always()" in workflow
```

The failing specifications were committed before production changes in `73b40e1`,
`5c4db65`, and `913df76`.

## GREEN evidence

Focused contract after implementation:

```text
16 passed
```

League-isolated suites before the final durability refinement:

```text
shared: 57 passed
NFL: 52 passed, 1 skipped
NCAA: 16 passed, 6 skipped
```

The skipped tests require external historical cache artifacts; the scheduled workflow
creates and persists those artifacts before publication. Production implementation was
committed in `2b0216b`, `5bf0fdb`, and `456d315`.
