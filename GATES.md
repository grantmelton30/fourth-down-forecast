# GATE REGISTRY

What each gate asserts, how a gate result becomes trustworthy, and what is currently
measured. Any claim about gate status must read from here — and this file must read from
`data/evidence/`, never from memory.

Created because **"all seven gates pass" was once reported for a gate set missing five
required gates.** Rewritten 2026-08-13, when the previous version was found to be
describing a different codebase entirely.

---

## Current status — measured 2026-08-13 (NCAA), NFL still unmeasured

A cold NCAA build ran on 2026-08-13 (173 CFBD calls against a 250 cap) and produced the
first reproducible gate artifact this repository has ever had. `bets_allowed` is **False**,
and it is now False for the right reason:

| gate | reading |
|---|---|
| `GATE_OPENER_COVERAGE` | **PASS** — 99.7% coverage, mean \|open−close\| 1.40 |
| `GATE_NO_LOOKAHEAD` | **PASS** — 0 of 200 sampled rows use ratings dated after kickoff |
| `GATE_GARBAGE_FILTER` | **PASS** — 9.6% of plays dropped |
| `GATE_UNBIASED` | **PASS** — largest \|a\| 0.214 |
| `GATE_BLEND_INFORMATIVE` | **PASS** — best positive t(b) 3.01 (total), vs the opener |
| `GATE_API_BUDGET` | **PASS** — 173 calls |
| `GATE_UNBIASED_BY_WEEK` | **FAIL** — worst \|a\| 3.401 (spread wk4, n=167) |
| `GATE_SCALE` | **FAIL** — worst SD-ratio deviation 0.245 (total) |
| `GATE_RMSE_TOTAL` | **FAIL** — 16.665 vs market 16.223 (ratio 1.027, n=1543) |
| `GATE_RMSE_SPREAD` | **FAIL** — 16.932 vs market 15.514 (ratio 1.091, n=1543) |
| `GATE_CALIBRATED`, `GATE_KEY_NUMBERS` | **NOT PRODUCED** — required for promotion, so recorded `passed: null` and blocking |

These reproduce the previously reported figures closely (spread 16.89/15.43, total
16.73/15.98), which is itself a useful consistency check across a full rebuild.

**Read the blend gate carefully.** `GATE_BLEND_INFORMATIVE` passes at t = 3.01 on totals
*against the opener*, while `GATE_RMSE_TOTAL` fails at 1.027 *against the close*. That is
the NCAA totals false positive, behaving exactly as documented: the apparent signal is
against the opening number and does not survive the closing one. It is a diagnostic, not
an edge.

### NFL — measured 2026-08-13 by the scheduled run

The first NFL artifact this repository has ever committed, produced by CI rather than by
hand. `bets_allowed` is **False**, blocked by six promotion gates:

| gate | reading |
|---|---|
| `GATE_NO_LOOKAHEAD` | **PASS** — 200 sampled rows, none used a game at or after kickoff |
| `GATE_UNBIASED` | **PASS** — worst bias 0.460 pts (total), tolerance 0.5 |
| `GATE_RMSE_SPREAD` | **FAIL** — 13.452 vs close 12.692 (ratio 1.060) |
| `GATE_RMSE_TOTAL` | **FAIL** — 13.627 vs close 13.214 (ratio 1.031) |
| `GATE_BEATS_ELO` | **FAIL** — 13.452 vs Elo 13.208 |
| `GATE_BLEND_INFORMATIVE` | **FAIL** — spread t = −1.16, total t = +1.41 |
| `GATE_CALIBRATED` | **FAIL** — worst bin off by 13.88pp (bin (0.35, 0.40], n=124) |
| `GATE_KEY_NUMBERS` | **FAIL** — worst gap 2.83pp at \|margin\|=3, tolerance 2pp |

**Do not read this as an improvement over the 13.88 spread RMSE quoted in the project
brief.** That figure came from the predecessor build under a different configuration, so
the two are not a controlled comparison — this is simply the first NFL measurement that is
reproducible from committed evidence. The model remains worse than the market on both
markets, and `GATE_BLEND_INFORMATIVE` reads negative on spreads, which is the same null
result recorded throughout.

### The verification path was broken until 2026-08-13

`bets_allowed()` had been returning False for a reason nobody could see. `run_backtest.py`
stamped the artifact with a model version hashed from its **in-memory** frame, while the
adapter recomputed the version from the **persisted parquet**. `walk_forward` returns a
working copy whose dtypes differ from the parquet it caches, so the two never matched and
every artifact either script wrote was silently repudiated on read.

Fail-closed made this invisible: a broken verification path and an honestly failing gate
both produce "no picks". Both backtests now hash the persisted evidence, and
`refresh_publication.py` warns loudly when an artifact exists but the adapter refuses it.

### Appendix A, revisited 2026-08-16 — decided, not tolerance-widened

`test_power.py::test_full_fbs_close_anchored_null_is_adequately_powered` failed on the
refreshed data: b = **+0.0911** (se 0.0631, t 1.44, n 2,985) against a recorded anchor of
b ≈ 0 with a ±0.05 band.

**Correcting the earlier read of this.** The previous version of this entry said "0.091 is
far below the 0.2 reference effect." That undersold the risk: the 95% CI is
**[−0.0326, +0.2148]**, and its upper bound sits *above* APPENDIX_A_OPEN_B (0.195, t=2.23)
— the magnitude Appendix A proved was a pure artifact of anchoring on the opener. The point
estimate is a null (t=1.44 < 1.96); the confidence interval no longer cleanly excludes the
thing this whole guard exists to catch.

**The decision:**

1. **Do not widen the tolerance.** A fixed ±0.05 band around zero fails on every
   legitimate change to `model_total`, which is most of what actually happened here — see
   below. It was never the right invariant.
2. **Do not cite the NCAA totals null as settled anymore.** Downgraded to *provisional*
   until re-validated. It does not change any live decision — `GATE_RMSE_TOTAL` already
   fails and `bets_allowed()` is already False for NCAA — but it does mean
   `DECISIONS.md`/`NCAA_PLAYBOOK.md` language claiming the totals question is closed is no
   longer accurate and should not be quoted as current.
3. **Most likely cause, not yet confirmed.** `ncaa-model/src/backtest.py::_fit_feature_challenger`
   promotes any `*_sum`-suffixed candidate into the totals model when it beats base RMSE by
   ≥0.05 on a single held-out season — a legitimate mechanism, unrelated to the
   opener-anchoring bug. Three commits landed exactly such candidates on
   **2026-08-12/13, immediately before this drift was first observed**:
   `4f8bb16` (validated NCAA projection integrity), `c8362ca`/`78a08c6` (coaching data /
   preseason merge-key fixes), `4b5ec98` (head-coach continuity from the nested coaches
   payload) — all touching `ncaa-model/src/features.py`, which emits both
   `head_coach_continuity_sum` and a turnover-uncertainty `*_sum` column, either of which
   `_fit_feature_challenger` would pick up as a totals candidate. This was not confirmed by
   inspecting `feature_total_promoted` on a live rebuild — no local CFBD-backed cache was
   available to run it — so it is a strong hypothesis, not a verified fact.
4. **The test now polices the actual invariant.** It asserts the 95% CI upper bound stays
   below the proven danger magnitude (0.195), not that the point estimate stays near zero.
   It is currently failing, correctly — that failure is the record of item 2 above, and it
   should keep failing until someone runs a fresh full rebuild, checks
   `feature_total_promoted` for the seasons in the full-FBS window, and either re-baselines
   the reference numbers with that provenance documented, or finds and fixes an actual bug
   if promotion turns out not to explain it.

Both affected tests are `@pytest.mark.integration` and excluded from CI, so this failure is
visible only to someone deliberately running them — which is also why it is written down
here rather than left to be rediscovered.

## Previously: NO GATE HAD A CURRENT READING

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
| `GATE_KEY_NUMBERS` | yes, added 2026-08-16 | `src/backtest.py::pooled_margin_pmf`, `gate_key_numbers` | Simulated margin PMF matches historical at 3/7/10/14/17/21 and the >28 tail. Pools a sample of historical games through the NCAA drive simulator (in-sample ratings; tests distribution SHAPE only, mirrors `nfl-model`'s `pooled_margin_pmf`) — does not touch the mean path (`model_spread`/`model_total` stay linear-projection-only, per the module docstring). **First real reading, 2026-08-17: FAILS**, worst gap 8.33pp at the >28 tail (sim 9.57% vs real 17.90% — the simulator badly underproduces blowouts); |3| also gaps -4.88pp. The first attempt at wiring this (2026-08-16) shipped without being run against real data and crashed the scheduled full rebuild in CI (`estimate_venue_hfa` needs `venue_id`, which `ingest.load_games` alone does not carry — fixed 2026-08-17 by attaching venues in `run_backtest.py`, the same one call `project_game.py` already made). Lesson applied, not just noted: this file's own "validate a method against a known answer before trusting it" rule was skipped once and cost a production rebuild. **Root-caused 2026-08-17, `ncaa-model/analysis/blowout_tail_diagnosis.py`.** The drive-clamp hypothesis from `NEXT_SESSION.md` was measured directly and REJECTED — the clamp binds on 1.6% of games and binds *less* often in blowouts (1.2%) than in close games (1.7%). The real cause: decomposing the pooled simulated margin SD (16.79) into within-game (Monte Carlo, 14.98) and across-game (spread of each game's own simulated mean, 7.56) shows the within-game component is approximately correct — the market's spread-vs-actual-margin gap implies real within-game SD of 15.2, matching the sim's 14.98 — but the across-game component is badly compressed: 7.56 vs. 13.085 for the market's own spread SD and 11.630 for this codebase's gain-corrected linear `model_spread`. The simulator isn't drawing games that are too noisy; it's failing to recognize which games are true mismatches. L2 regularization in `fit_drive_model` (`C`) was tested and rejected as the mechanism — swept 1/10/100, across-game SD held flat at 7.56/7.71/7.69. **Fix implemented and measured 2026-08-17.** `pooled_margin_pmf` now takes a `linear_frame` (the walk-forward `frame`/`model_spread`) and calls `sim.recentered(model_spread)` per sampled game before pooling — mirroring NFL's `SimResult.recentered()`, which NCAA previously had nowhere in the package. (`recentered()` reweights `sim.weights`, it does not shift `sim.margins`, so the pooling accumulator was rewritten to a weighted histogram — concatenating raw `.margins` after recentring would have silently discarded the fix.) `project_game.py`'s printed spread was left un-recentred — doing that properly needs the full gain-corrected walk-forward OLS pipeline, which that standalone CLI does not run; it now prints a caveat instead pointing at this entry.

**Result: confirms the diagnosis, does not clear the gate.** The >28 tail gap fell from -8.33pp to **-3.63pp** (sim 14.27% vs real 17.90%) — recentring recovered most of the missing across-game dispersion, as predicted. But mass that moved out to the tail came from the centre: `|margin|=3` is now the worst bin, sim 5.09% vs real 10.66% (**-5.57pp**, worse than the pre-fix -4.88pp). **FAIL** persists, now on a different bin. Full reading: |3| -5.57pp, |7| -2.28pp, |10| +0.13pp, |14| +0.82pp, |17| -0.80pp, |21| -0.23pp, >28 -3.63pp.

One caveat, checked and found pre-existing (not introduced by this fix): `pooled_margin_pmf`'s candidate pool is every completed game in `games` for the graded seasons — 17,144 rows, because `ingest.load_games` is not FBS-filtered — against which only 2,996 have a fitted rating pair AND (now) a `model_spread`. Of `n_games=400` sampled candidates, ~333 get skipped (printed at run time), so the gate reads off an effective sample of roughly 65-140 games, not 400. This inefficiency already existed via the `rt.index` ratings check before this fix added the `model_spread` check on top of it; not fixed here — a candidate pool pre-filtered to FBS-vs-FBS would give the gate a less noisy reading. |
| `GATE_CALIBRATED` | **not built — scoped, deliberately not attempted 2026-08-16** | — | No probability bucket with 100+ obs off by >6 points. Needs per-game `cover_prob_home`/`over_prob` from the simulator across the walk-forward backtest, which NFL gets by running `simulate_game` inside its `walk_forward` loop and recentring on the L1 mean (`probs = sim.recentered(model_spread)`). NCAA's `walk_forward` (`src/backtest.py`) has no such call — it is closed-form OLS + gain/week-bias correction only, and never touches `drives`/`drive_model`/`endgame`. Wiring the simulator into that loop is a real signature change (needs `drives`, `game_off`/`walkforward` threaded through `walk_forward`, called from both `run_backtest.py` and `run_paper.py`) across thousands of graded games — unlike `GATE_KEY_NUMBERS`'s few-hundred-game pooled sample, this is O(graded games), touches the live evaluation path, and cannot be responsibly shipped without running it end-to-end against real data first. No local CFBD-backed cache was available to do that validation. Left unbuilt rather than shipped unvalidated — see the project's own rule: "validate a method against a known answer before trusting it."|
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
