# PATCH 04 — NCAA GO PACKAGE

Final instruction set. Read alongside `NCAA_PLAYBOOK.md`, `DATA_SOURCES.md` and
`PATCH_03_DECISION.md`. Where they conflict, **this file wins.**

Two stops only. Run everything else straight through.

---

## STEP 0 — PRE-FLIGHT: DOES THE PIVOT'S PREMISE HOLD?

**Run this before writing ingest, priors, ratings, or anything else.** It costs 2 API
calls and it can invalidate the entire NCAA plan. Do not build on an unverified premise
twice in a row.

PATCH_03 §4 justified the college pivot on one fact: CFBD carries opening lines free, so
the model can be graded against the opener rather than the close — the test the NFL build
could not run. That claim has two failure modes and neither has been checked.

Pull `BettingApi.get_lines` for 2019–2025 and measure, **on the restricted universe only**
(Group of Five plus cross-tier games, FBS vs FBS, per PATCH_03 §5.5):

```
GATE_OPENER_COVERAGE
  (a) coverage:   >= 70% of games have non-null spread_open AND over_under_open
  (b) separation: mean |spread_open - spread_close| >= 0.5 points
  (c) same two checks on the full FBS slate, reported for contrast
```

Check (b) matters as much as (a) and is easier to miss. If CFBD's "opening" line is in
fact captured late in the week, `spread_open` will sit almost on top of `spread_close`,
the two grading standards collapse into one, and you are running the NFL test again under
a different name. Report the full distribution of `|open - close|`, not just the mean —
a healthy opener shows a median around 1 to 2 points with a real tail.

**Stop and report after this.** If either check fails, say so plainly and do not proceed
to Step 1; the pivot needs rethinking, not more code.

Also report: total API calls consumed, and calls remaining in the month's budget.

---

## STEP 1 — BUILD, STRAIGHT THROUGH

If Step 0 passes, build the whole NCAA repo without stopping. Order per PATCH_03 §5 as
amended:

```
git init
cfbd_client.py  ->  ingest.py  ->  ratings.py  ->  [VIEWER LIGHTS UP]
    ->  drives.py / drive_model.py / simulate.py  ->  backtest.py  ->  gates
```

Non-negotiables, all already specified — this is a checklist, not new instruction:

1. **Budget wrapper on every call.** Persisted counter, permanent parquet cache, raise on
   exhaustion. Cold backtest under 250 calls (`GATE_API_BUDGET`).
2. **Spread sign convention.** CFBD quotes negative-means-home-favored; nflverse is the
   opposite. Normalize to "positive = home favored" at ingest and write the assertion test
   against a known historical blowout. This is the silent killer — a sign flip produces
   entirely plausible numbers while being exactly backwards.
3. **Simulator out of the mean path from day one.** L1 is a direct linear projection off
   the ridge ratings. The simulator produces totals shape, the margin/total joint, and
   factor sensitivity — never the mean. This is the one hard-won finding from the NFL
   build; do not rediscover it.
4. **Free intercept in every residual fit**, `GATE_UNBIASED` (|a| < 0.5) live from the
   first backtest.
5. **Grade against `spread_open` / `over_under_open`.** Retain the close solely to measure
   CLV: did the market move toward the model's side. Report both, separately.
6. **Totals before spreads.** Evaluate totals first. Build the spread path only if totals
   show signal.
7. **Restricted universe.** Fit and report the blend on G5 and cross-tier games. Report
   the full slate alongside for contrast, but the kill criterion is judged on the
   restricted set.
8. **Garbage-time filter live before any efficiency number is computed.** Log the share of
   plays dropped; expect 12–18% league-wide. Under 5% means it is broken, over 30% means
   it is too aggressive.
9. **Pre-register the hyperparameter grid at ≤ 12 cells** in `DECISIONS.md` *before* the
   first sweep runs. Never 720. Tune on held-out `t(b_model)`, and remember that scale-only
   changes are inert — only re-ranking moves `t`.
10. **The viewer is shared and already verified.** `view.py` and `export.py` need no
    changes; write the NCAA adapter against the `shared/sport.py` seam. `bets_allowed()`
    stays computed inside the adapter.

---

## STEP 2 — THE VERDICT

Run the full gate set and report. The kill criterion from PATCH_03 §5, restated as the
single number that decides this:

```
held-out t(b_model) on college TOTALS, restricted universe, graded vs OPENING lines
    >= 2.0   ->  signal exists. Build the calibration layer, then the bet sheet.
    <  2.0   ->  stop. Do not sweep further.
```

Report at this stop:

- `GATE_OPENER_COVERAGE` numbers carried forward from Step 0
- Full gate table
- For totals and spreads separately, restricted and full slate: `a`, `t(a)`, `b`, `t(b)`,
  SD ratio, RMSE vs opener, RMSE vs close
- CLV: share of bets where the close moved toward the model's side
- The pre-registered grid as declared, with results for every cell — all 12, not the best
- Bootstrap CI on any filtered record

If the criterion fails, write it into `NCAA_PLAYBOOK.md` as Appendix A the way the NFL null
was recorded, including "do not resume tuning against this backtest." A recorded null is
the deliverable in that branch.

---

## WHAT NOT TO DO

- Do not build the bet sheet before Step 2 passes.
- Do not widen the restricted universe to improve a result.
- Do not sweep beyond the pre-registered 12 cells.
- Do not proceed past Step 0 on a failed pre-flight.
