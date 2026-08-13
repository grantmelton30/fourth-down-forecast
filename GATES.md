# GATE REGISTRY

What each gate asserts, how a gate result becomes trustworthy, and what is currently
measured. Any claim about gate status must read from here — and this file must read from
`data/evidence/`, never from memory.

Created because **"all seven gates pass" was once reported for a gate set missing five
required gates.** Rewritten 2026-08-13, when the previous version was found to be
describing a different codebase entirely.

---

## Current status: NO GATE HAS A CURRENT READING

`data/evidence/manifest.json` is the authoritative record of what has been measured. As of
2026-08-13 it reports, for both leagues:

| league | `gate_results.json` | `blend_weights.json` |
|---|---|---|
| nfl | absent | absent |
| ncaa | absent | absent |

**No gate in this repository currently has a measured pass/fail value.** Nothing has been
lost — the artifacts were never in version control. They were written into
`{league}-model/data/cache/`, which is gitignored and persisted only by the GitHub Actions
cache, so they existed solely inside CI.

Consequences, stated plainly:

- `bets_allowed()` is **False** for both leagues, because a missing gate artifact fails
  closed. This is the correct and expected result.
- Calibration is **closed** for both leagues, because missing calibration evidence fails
  closed.
- The prediction slate published before 2026-08-13 asserts
  `calibration_status.total: true` for NCAA. **That claim is not reproducible from this
  repository.** It rests on a `blend_weights.json` that no longer exists here and carried
  no record of which build produced it. It should not be treated as validated.

To produce current readings, run a full refresh (`scripts/refresh_publication.py`, which
runs both backtests). It writes both artifacts and snapshots them into `data/evidence/`,
where they are committed.

---

## What makes a gate result trustworthy

Two artifacts carry every promotion decision, and both are bound to a **model version** —
a SHA-256 over the league config, `run_backtest.py`, every file in `{league}-model/src/`,
every file in `shared/`, and a hash of the backtest evidence frame itself
(`shared/model_identity.py`).

| artifact | decides | writer | reader |
|---|---|---|---|
| `gate_results.json` | `bets_allowed()` | `shared/gate_artifact.py` | `SportAdapter.gate_artifact()` |
| `blend_weights.json` | `calibration_status` | `shared/calibration_evidence.py` | `SportAdapter.blend_weights()` |

An artifact is refused unless it names this exact model version. Editing a config value,
a source file, or rebuilding on different data changes the version and repudiates every
artifact fitted against the old one. Both readers fail closed on a missing, unreadable,
mis-versioned, or wrong-league artifact.

`SportAdapter.calibration_block_reason()` reports *why* calibration is closed, and that
reason is published into `quality_reasons`. A build that never fitted weights and a build
whose weights were repudiated both show no calibration; only one of them is routine, and
the reader is told which.

**Before 2026-08-13 the calibration side had none of this.** `blend_weights.json` was a
bare coefficient dict read with a raw `json.loads`, with no version, no schema, and a
bare `except` that turned every failure into an empty dict. The betting decision beside it
was rigorously bound. That asymmetry is what let an orphaned artifact authorize a public
"validated" label.

### Known gap: NCAA calibration weights are not produced by the pipeline

`ncaa-model/run_backtest.py` — the only NCAA script the scheduled refresh runs — **does not
fit or write blend weights**. Only `ncaa-model/run_paper.py` does, and nothing schedules
it. So NCAA calibration cannot legitimately open through the automated pipeline; it can
only be opened by a manual paper run. This is recorded, not fixed: wiring a fitter into
the refresh is a modelling decision, and NCAA's `fit_blend` defaults to grading against the
**opener**, which is precisely the construction that produced the totals false positive.
Do not wire it in without deciding the anchor first.

---

## What each gate asserts

Assertions are read from source and are current. **Pass/fail columns are deliberately
absent** — see the status section above.

### NCAA — `ncaa-model`

| gate | implemented | file | asserts |
|---|---|---|---|
| `GATE_OPENER_COVERAGE` | yes | `src/backtest.py` | ≥70% of restricted games carry both openers; mean \|open−close\| ≥ 0.5 |
| `GATE_NO_LOOKAHEAD` | yes | `src/backtest.py` | On a 200-row sample, no rating dated after its game's kickoff |
| `GATE_GARBAGE_FILTER` | yes | `src/backtest.py` | Garbage-time play drop share within 5–30% |
| `GATE_UNBIASED` | yes | `src/backtest.py` | \|intercept a\| < 0.5, full sample |
| `GATE_UNBIASED_BY_WEEK` | yes | `src/backtest.py` | \|a\| < 0.5 in every week bucket with n ≥ 40 |
| `GATE_SCALE` | yes | `src/backtest.py` | \|SD ratio − 1\| < 0.15 vs market |
| `GATE_BLEND_INFORMATIVE` | yes | `src/backtest.py` | best \|t(b)\| > 2.0 |
| `GATE_API_BUDGET` | yes | `src/backtest.py` | Cold build under 250 CFBD calls |
| `GATE_RMSE_SPREAD` / `GATE_RMSE_TOTAL` | yes | `src/backtest.py::rmse_gate` | Model RMSE / market RMSE ≤ 1.0, close-anchored |
| `GATE_CALIBRATED` | required for promotion | — | No probability bucket with 100+ obs off by >6 points |
| `GATE_KEY_NUMBERS` | required for promotion | — | Simulated margin PMF matches historical at 3/7/10/14/17/21 and the >28 tail |
| `GATE_BEATS_SRS` | **NOT BUILT** | — | Model spread RMSE beats an SRS baseline (`src/srs.py` never written) |
| `GATE_COMPETITIVE_WITH_SPPLUS` | **NOT BUILT** | — | Within 0.5 of SP+ (SP+ never ingested) |
| `GATE_SPEED` | **NOT BUILT** | — | Weekly refresh under 240s on cached data |

`GATE_CLV_CONTROL` was **deleted as a promotion metric**: a zero-information projection
earned CLV 0.5846 raw / 0.5607 side-adjusted over 200 seeds, so CLV measured something
other than skill. Do not reinstate it.

### NFL — `nfl-model`

| gate | implemented | asserts |
|---|---|---|
| `GATE_KEY_NUMBERS` | yes | Pooled simulated margin PMF within 2pp of historical at 3/7/6/10/14/4 |
| `GATE_NO_LOOKAHEAD` | yes | 200-row sample, no game used at or after the predicted kickoff |
| `GATE_BEATS_ELO` | yes | Model spread RMSE < Elo baseline RMSE |
| `GATE_VS_NFELO` | **REMOVED** | Deleted, not unbuilt: it compared model RMSE against a series that does not exist, so it passed vacuously. See `COMMIT_LOG.txt:221`. Do not reinstate without an nfelo feed. |
| `GATE_BLEND_INFORMATIVE` | yes | \|t\| > 2.0 on b_model |
| `GATE_CALIBRATED` | yes | No 100+ obs bucket off by >6pp |
| `GATE_UNBIASED` | yes | \|bias\| < 0.5 pts per market |
| `GATE_SCALE` | yes | SD ratio vs market within [0.85, 1.15] |
| `GATE_RMSE_SPREAD` / `GATE_RMSE_TOTAL` | yes | Model RMSE / market RMSE ≤ 1.0 |
| `GATE_SPEED` | yes | Weekly refresh < 180s |

The promotion sets enforced in code are in
`shared/gate_artifact.py::REQUIRED_PROMOTION_GATES`. A required gate that is never
produced is recorded as `passed: null` and blocks promotion — silence is not a pass.

---

## Three things not to "fix"

**The model is 5–10% worse than the market, and the failing gates are correct.** Walk-forward
readings from the predecessor build: NFL spread 13.88 vs market 12.69, NFL total 13.85 vs
13.21, NCAA spread 16.89 vs 15.43, NCAA total 16.73 vs 15.98. `GATE_BEATS_ELO` and
`GATE_BLEND_INFORMATIVE` measure signal the model does not have. Failing them is the
system working.

**`GATE_SCALE` is mis-specified for a model weaker than the market.** It demands the
model's predictions vary as much as the market's, but regressing outcomes on the model's
own projections gave slopes of 0.605 (spread) and 0.753 (total) — supporting sd 4.87 and
2.07 against market sds of 6.42 and 4.48. Inflating the model to pass would make it over
twice as confident on totals as its information justifies. Read the failure as a
measurement of the signal gap.

**`GATE_KEY_NUMBERS` can be made to pass by breaking the margin distribution.** Applying
the endgame table to a flat two drives per team reached 1.98pp but collapsed simulated
margin sd to 12.46 against a real 14.2, breaking
`test_simulated_margin_spread_is_realistic`. The shipped setting matches the measured 1.34
endgame drives per team instead. Do not trade the margin distribution for this gate.

And: do not re-add the NCAA totals edge. It exists only against the opener and vanishes
against the close.

---

## Historical readings (superseded — do not quote as current)

The previous version of this file carried a pass/fail table measured **2026-08-04**, with
corrections dated 2026-08-07, against the predecessor repository at `~/dev/grant-claude`.
Those readings predate this repository, which is a clean rebuild begun 2026-08-11. They
were carried over wholesale and described code that no longer exists here.

The numbers quoted in the section above are retained from that work because they document
*why* certain gates must not be chased. They are not current measurements, and no gate
should be reported as passing or failing on their basis. Full detail remains in git
history and in `COMMIT_LOG.txt`.
