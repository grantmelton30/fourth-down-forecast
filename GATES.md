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

## Updated status — measured 2026-08-17 (NCAA): `GATE_CALIBRATED` implemented, all promotion gates now produced

First rebuild since the FCS-ratings fix and the first time `GATE_CALIBRATED` has ever
been produced for NCAA — it was declared as a required promotion gate in
`shared/gate_artifact.py::REQUIRED_PROMOTION_GATES` since this repo's gate schema was
written but never implemented, which meant `write_gate_artifact` silently inserted a
`passed: null` row on every single NCAA build and permanently blocked `bets_allowed()` for
a reason no gate table ever showed. Ported from nfl-model's working
`walk_forward`/`calibration_table`/`gate_calibrated` pattern: `attach_calibration`
(`ncaa-model/src/backtest.py`) walks forward one season at a time, refits the drive model,
endgame table, and venue HFA (with a proper `as_of` kickoff cutoff — `pooled_margin_pmf`'s
existing venue-HFA call has none, which is fine for that diagnostic-only gate but would
have been a real lookahead leak here), simulates every graded game with that season's
ratings, and reweights the result onto this build's own `model_spread`/`model_total`
(`SimResult.recentered().retotaled()`) before reading off `cover_prob_home`/`over_prob`.
Additive and separately cached from `walk_forward` — reads `model_spread`/`model_total`,
never recomputes them.

| gate | reading |
|---|---|
| `GATE_OPENER_COVERAGE` | **PASS** — 99.7% coverage, mean \|open−close\| 1.40 |
| `GATE_NO_LOOKAHEAD` | **PASS** — 0 of 200 sampled rows use ratings dated after kickoff |
| `GATE_GARBAGE_FILTER` | **PASS** — 9.6% of plays dropped |
| `GATE_UNBIASED` | **PASS** — largest \|a\| 0.151 (spread) |
| `GATE_BLEND_INFORMATIVE` | **PASS** — best positive t(b) 3.09 (total), vs the opener |
| `GATE_UNBIASED_BY_WEEK` | **FAIL** — worst \|a\| 2.703 (spread wk4, n=167) |
| `GATE_SCALE` | **FAIL** — worst SD-ratio deviation 0.259 (total) |
| `GATE_RMSE_TOTAL` | **FAIL** — 16.688 vs market 16.223 (ratio 1.029, n=1543) |
| `GATE_RMSE_SPREAD` | **FAIL** — 16.955 vs market 15.514 (ratio 1.093, n=1543) |
| `GATE_KEY_NUMBERS` | **FAIL** — worst gap 3.89pp at \|margin\|=\>28 (pre-existing softmax-saturation finding, unrelated to today's changes) |
| `GATE_CALIBRATED` | **FAIL (now produced)** — worst bin off by 34.33pp (bin 0.15-0.20, n=106, tolerance 6pp) |
| `GATE_API_BUDGET` | **FAIL** — 383 calls (budget 250) — **artifact of this session's retries, not the build's real cost**, see below |

`bets_allowed()` is still **False**, but for the first time every required NCAA promotion
gate in `REQUIRED_PROMOTION_GATES` has an actual measured reading — none are `passed: null`
anymore.

**RMSE/SCALE/UNBIASED/BLEND numbers are consistency-checked against the pre-FCS-fix
2026-08-13 reading above and are essentially unchanged** (e.g. RMSE_TOTAL 16.665→16.688,
RMSE_SPREAD 16.932→16.955) — confirming `attach_calibration` did not disturb the existing
measurement pipeline, exactly as designed, and confirming the FCS-ratings plan's own
prediction that these FBS-only-graded gates would be "materially unchanged" (they exclude
FCS games by construction). `GATE_KEY_NUMBERS` is a pre-existing, already-documented finding
(softmax saturation, `analysis/blowout_tail_diagnosis.py`) and did not change materially
either.

**`GATE_CALIBRATED`'s result is a genuinely new, clean, negative finding.** 5,621 graded
games (a materially larger population than the FBS-only gates above, because FCS teams now
carry real ratings and are simulable post-fix) carry a calibrated cover probability. The
calibration table is close to flat: bins predicting a 92% home cover realize ~50%; bins
predicting a 2.6% cover realize ~55%; nearly every bin, across the full 0-100% predicted
range, realizes somewhere in the 45-60% band. The model's own stated confidence in a cover
carries essentially no relationship to whether the cover actually happens — consistent with,
and now directly quantifying, this project's standing finding that no real edge has been
demonstrated.

**`GATE_API_BUDGET`'s 383-call reading is not a real regression.** This session hit CFBD's
429 four times in a row while rebuilding an empty local cache (the 2026-08-13 cache no
longer existed in this environment) — three of those retries each re-recorded the same
failed call against the monthly counter before the actual cause (an exhausted free-tier
quota, unrelated to any per-minute throttle) was found and the account upgraded to the
$1/5,000-call tier. `monthly_call_budget` in `config/ncaa.yaml` was bumped 900→4,500 to
match. A single clean cold build still costs ~173-350 calls, comfortably under the 250
`api_budget_max_cold` sanity check; this reading is the cumulative cost of one clean build
plus several failed retries within the same month, not a change in the build's own cost.

See "Appendix A, revisited again 2026-08-17" below for a second, unrelated finding
surfaced by this being the first real rebuild since the FCS-ratings fix: the totals
residual coefficient (`b`) has drifted upward, and the challenger-promotion hypothesis
that could explain the 2026-08-16 drift is now ruled out for this one.

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

### Appendix A, revisited again 2026-08-17 — challenger-promotion hypothesis ruled out, drift grew

The 2026-08-16 entry above ended with a specific instruction: run a fresh full rebuild,
check `feature_total_promoted`, and either re-baseline with that provenance documented or
find a bug. No local CFBD-backed cache existed to do that at the time. One now does (this
session bought CFBD's $1/5,000-call tier after the free tier's 1,000/month ran out mid-build
— see `monthly_call_budget` in `config/ncaa.yaml`), and a genuine cold rebuild ran
2026-08-17, the first since the FCS-ratings fix (`ingest.py::build_game_offense`, same day)
and the preseason-poll-points feature landed.

**`feature_total_promoted` is `False` for every graded row, every season 2021-2025.** The
challenger-promotion hypothesis from 2026-08-16 (three commits landing `*_sum`-suffixed
candidates on 2026-08-12/13) is ruled out directly, not circumstantially, as the cause of
*this* drift — no challenger has fired at any point in this rebuild.

**And the drift grew, in the same direction, not shrank:**

| universe | anchor | n | b | t / se | 2026-08-16 reading |
|---|---|---|---|---|---|
| restricted | close | 1,543 | +0.1549 | t=+1.745 | b=+0.076, t=+0.89 (Appendix A original) |
| restricted | open | 1,543 | +0.2808 | t=+2.859 | b=+0.195, t=+2.23 (Appendix A original) |
| full FBS | close | 2,985 | +0.1157 | se=0.0623, CI [−0.0065, +0.2378] | b=+0.0911, se=0.0631, CI upper +0.2148 |

Full-FBS close-anchored `b` moved from +0.091 to +0.116 and its CI upper bound from +0.215
to **+0.238** — further above APPENDIX_A_OPEN_B (0.195), not closer to it. The
restricted-close reading, the one `NCAA_PLAYBOOK.md` Appendix A quotes directly as
"decisive" at b=0.076/t=0.89, now reads b=0.155/t=1.75 — no longer the small, clearly-null
number the original table shows.

**CONFIRMED: the FCS-ratings fix, by direct counterfactual.**
`analysis/appendix_a_fcs_counterfactual.py` reverts `build_game_offense` to the old
single-bucket behavior in memory only (no source change), re-runs the walk-forward ratings
and backtest frame from that counterfactual `game_off` against the *same* cached
games/drives/plays (zero new CFBD calls), and recomputes the same three statistics:

| universe | anchor | counterfactual (old bucketing) | post-fix (current code) | 2026-08-16 reading |
|---|---|---|---|---|
| restricted | close | b=+0.1407, t=+1.573 | b=+0.1549, t=+1.745 | b=+0.076, t=+0.89 (original) |
| restricted | open | b=+0.2636, t=+2.665 | b=+0.2808, t=+2.859 | b=+0.195, t=+2.23 (original) |
| full FBS | close | b=+0.0911, se=0.0631, CI upper **+0.2147** | b=+0.1157, se=0.0623, CI upper +0.2379 | b=+0.0911, se=0.0631, CI upper +0.2148 |

The counterfactual's full-FBS-close reading (+0.0911, CI upper +0.2147) reproduces the
2026-08-16 reading almost to the fourth decimal — changing exactly one function back to its
old behavior, with everything else (cached data, walk-forward machinery, config) held
identical, restores the pre-drift numbers. This is as clean a single-variable isolation as
this kind of question gets. The FCS-ratings plan's invariance prediction was wrong for this
statistic.

**A plausible (not fully proven) mechanism for why spreads barely moved but totals did:**
`model_spread` is driven by `net_diff` (a *difference* of two teams' net efficiency);
`model_total` by `eff_sum` (a *sum*). A small, systematic shift in the ridge fit's rating
level from adding ~130 more teams to the joint solve — measured directly: mean FBS
`off_rating` shifted +0.0069, `def_rating` −0.0004, both small but one-directional rather
than noise, comparing the same season/week snapshot under both universes — cancels (mostly)
in a difference and compounds in a sum. `GATE_UNBIASED`'s spread \|a\| actually *improved*
slightly post-fix (0.214→0.151) while every totals reading moved the same direction as this
entry's drift, which is consistent with that asymmetry but was not proven down to the exact
mechanism inside `fit_ratings`'s ridge solve.

**The decision, updating 2026-08-16's:**

1. **Re-baseline the pinned constants in `test_power.py` to today's numbers.** The
   2026-08-16 entry conditioned re-baselining on ruling out challenger promotion; that
   condition is now met. `APPENDIX_A_CLOSE_B`/`_T`, `APPENDIX_A_OPEN_B`/`_T`, and the CI
   reading in `test_full_fbs_close_anchored_null_is_adequately_powered`'s docstring are
   updated to this entry's numbers.
2. **Do not upgrade "provisional" back to "settled," and do not downgrade it to "there is an
   edge" either.** The point estimates are still nulls by the pre-registered test (t=1.75
   and t=1.86 both clear neither 1.96 nor even reliably 2.0), but the CI-upper-bound guard
   that exists specifically to catch the opener-anchoring artifact reappearing is now
   failing by a wider margin than 2026-08-16, not a narrower one. Read that as "still
   cannot rule out re-contamination," not as "there is now a real edge" — nothing here
   changes `GATE_RMSE_TOTAL` (still fails) or `bets_allowed()` (still False).
3. **`NCAA_PLAYBOOK.md`'s Appendix A table is a historical record of the 2026-08-04 build
   and is left as written**, matching this file's own practice of appending dated findings
   rather than rewriting history. Its "standing conclusion" prose was already flagged
   2026-08-16 as not safe to quote as current; this entry is further reason not to.
4. **No action on the FCS-ratings fix itself.** The confirmed cause is a side effect on
   `model_total`'s residual coefficient, not a bug in what the fix was built to do (real
   FBS-vs-FCS spread/total projections, e.g. the NDSU sign-flip it fixed). The totals null
   was already provisional before this was traced; tracing it to a specific, understood
   cause is strictly more information than reverting or patching around it would be.

Both affected tests remain `@pytest.mark.integration`, excluded from CI, visible only to
someone deliberately running them against a real cache — which is why this is written down
here again rather than left for a third rediscovery.

## Pass-only EPA tested as a challenger feature, 2026-08-17 — nuanced result, not promoted

A competitor's methodology email (BTB Analytics) claimed a re-fit on 7,000+ FBS games shows
rushing EPA "never gets close" to significant while passing EPA carries the predictive
load — implying this repo's pooled `ppa_per_play` (pass and rush averaged together in
`ingest.py::build_game_offense`) is diluting a stronger signal with a weaker one. Tested
directly rather than assumed: `build_game_offense` gained an optional `play_type`
parameter, `ratings.py::split_net_epa_matchup` combines the resulting pass-only and
rush-only walk-forward ratings into `pass_net_epa_diff`/`_sum` and `rush_net_epa_diff`/`_sum`
candidates, and both were merged onto the existing `challenger` frame `run_backtest.py`
already builds — no changes to `walk_forward`/`build_features`/`project_walkforward`,
`_validated_challenger`'s existing suffix-driven promotion logic picked them up unchanged.

**Neither promoted.** `feature_spread_promoted`/`feature_total_promoted` are `False` for
every one of 5,621 graded rows (`None` only for the same 621 early-history rows every other
challenger candidate is also absent from). Confirmed this is a real result, not a wiring
bug: both candidate columns are 100% populated (6,242/6,242 rows) with sensible
mean-centered values.

**But the reason why is more interesting than a flat rejection.** Standalone (bivariate,
restricted, 2021-2025, n=1,543) correlation with the actual outcome:

| feature | spread: r vs `actual_margin` | total: r vs `actual_total` |
|---|---|---|
| `pass_net_epa_diff`/`_sum` (new, pass-only) | +0.4234 | +0.1985 |
| `rush_net_epa_diff`/`_sum` (new, rush-only) | +0.3749 | +0.1577 |
| `net_diff`/`eff_sum` (existing, pooled) | **+0.5090** | **+0.2405** |

**Pass does beat rush individually** — directionally replicating the claim's core
comparison, on this data, at this bivariate level. **But neither beats the existing pooled
feature**, which is the strongest of the three in both markets. This is not a contradiction:
averaging two noisy-but-correlated signals can out-predict either alone even when one of
them is individually weaker, provided each carries some independent variance relevant to
the outcome — which is exactly what these numbers show pass and rush both still do. This
is also why validated-ridge promotion correctly rejected both: `pass_net_epa_diff` is too
collinear with the already-present `net_diff` (both are largely the same underlying signal,
split differently) to add incremental value once `net_diff` is already in the model, even
though it explains real variance on its own.

**Read this as a partial replication with a specific, load-bearing caveat, not a clean
confirm or reject.** The claim that passing carries more signal than rushing holds up
directionally on this repo's own data. The architectural implication most naturally drawn
from it — split the core rating into pass/rush, or replace pooled EPA with pass-only EPA —
does not: on this evidence, that would *remove* information (rush's residual, independent
signal) rather than sharpen it. No core rating, the linear projection's base coefficients,
or the drive simulator were touched — this was always a Stage 1, challenger-only test by
design (`DECISIONS.md` D12), and stays that way given this result.

## Calibration gate no longer requires significance, 2026-08-18

Resolves a symptom flagged earlier this session ("why is calibrated always empty") that
the earlier fix only addressed on the frontend (removing a UI column that could never
populate). The actual cause was `shared/slate_builder.py::calibration_permissions`,
which required both a directionally positive blend weight *and* `t > 2` before
`calibration_from_weights` would publish a blended `calibrated` forecast at all. Checked
the real `blend_weights.json`: NCAA spread `b=+0.058, t=1.24`; NCAA total `b=+0.155,
t=1.75` — both positive, neither significant. Correcting an earlier claim in this
conversation: total does *not* clear `t > 2` here despite `GATE_BLEND_INFORMATIVE`
reading `t=3.09` in `run_backtest.py`'s printed table — that figure is opener-anchored
(the exact anchor Appendix A proved inflates `t`); `blend_weights.json` is close-anchored
(`CALIBRATION_ANCHOR = "close"`), the honest read, and on it neither market is
independently proven yet.

**Decision: drop the significance requirement, keep the sign requirement.** A blend
toward a more-informed market is the right move when the model's own edge isn't proven,
not something to withhold until it is — the "positive raw weight" check already excludes
the one case that would be nonsensical (an anti-predictive weight; NFL spread's raw
weight is `-0.095`, already clamped to `0.0`, and stays excluded with no new logic).
Verified directly against the real weights: `calibration_permissions` now returns
`{"spread": True, "total": True}` for NCAA (previously `{False, False}`), and
`calibration_from_weights` produces a real blended forecast — spread shrinks heavily
toward market (weight 0.058: a 2.0-point model/market gap collapses to ~0.1), total
retains more of the model's own signal (weight 0.155). `calibrated_forecast`'s `source`
string changed from `"validated spread and total blend"` to `"market blend, weight not
independently significant"` so it stops overclaiming now that significance isn't
required — `confidence_label`'s `"validated"` wording is unaffected, since it's gated on
`bets_allowed()`, not on this.

**NFL is unaffected in practice** (spread's own negative weight still blocks the
"both markets or nothing" coupling), and nothing in the acceptance-gate system
(`GATE_BLEND_INFORMATIVE`, `bets_allowed()`) changed — this is purely about what gets
published as `calibrated`, matching D7's separation between the gate-measurement and
live-publication paths.

**Won't appear on the live site until the next full publish cycle.** The live
`data/predictions.jsonl` currently shows `calibration_block_reason: "calibration
evidence was fitted by a different model build"` — `blend_weights.json`'s `model_version`
doesn't match what's currently deployed, the same fail-closed model-version check used
everywhere else in this repo. Self-resolves the next time `run_backtest.py` and
`build_slate.py` run against the same model version, same as every other
model-version-gated artifact here.

## `GATE_CALIBRATED` now costs 24 seconds on a routine rerun, not 40-45 minutes, 2026-08-18

Found by the user asking a simple, correct question: why would completed historical
seasons ever need re-simulating? They didn't -- `attach_calibration` cached its result as
one blob for every graded season combined, keyed to a signature that included the current,
still-in-progress season. Any routine change (a new week's games graded, a rerun during
development) invalidated the whole thing and re-simulated all 5,621+ games, including
years of permanently-static history. See `DECISIONS.md` D14 for the fix (per-season
caching, signature scoped to `season <= X`).

**Measured before/after, not assumed:** a same-day rerun with every per-season cache warm
completed in **24 seconds real time**, versus the ~40-45 minutes the calibration step alone
cost before. `GATE_CALIBRATED`'s reported numbers (5,621 games, worst bin off by 34.33pp,
bin (0.15, 0.2]) are byte-identical across three separate real runs spanning this change --
confirms the refactor moved nothing but the caching strategy.

## A genuine offense x defense interaction term, tested and not promoted, 2026-08-18

Direct follow-up to "Pass-only EPA tested as a challenger feature" above, prompted by a
user pushback worth taking seriously: that test showed rush-only ratings don't deserve
their own linear term, which is a different claim from "does this rushing style do better
against this specific front than the additive model predicts." Tested the second claim
directly -- `ratings.py::interaction_matchup` builds a genuine product term
(`home_off_rating * away_def_rating`, mirrored for the away side), the standard way to test
a statistical interaction, in both a pooled (general) and rush-only (the specific claim
raised) form. See `DECISIONS.md` D13 for the full controlled-regression numbers.

**Result: neither promoted**, matching the existing pass/rush finding. The closest signal
is the pooled spread interaction (t=+1.585 controlling for `net_diff`/`is_home`, n=1,543)
-- positive, in the theoretically expected direction, but not distinguishable from noise at
this sample size. Not disproven, not detected; recorded as a real negative result and a
possible direction for a wider search, not acted on.

**Follow-up same day: added pass-only interaction, and re-ran all three (pooled/rush/pass)
on the larger full-FBS window (n=2,985) alongside the original restricted window (n=1,543).**
Pass-only never approaches significance in either window (spread t=+0.398 / +0.940, total
t=-0.260 / +1.643). Pooled crosses conventional significance on the larger full-FBS window
(spread t=+2.422, total t=+2.075) but stays sub-significance on the restricted,
pre-committed window this project actually promotes on (t=+1.585 / +0.370) -- so still not
promotion-worthy by this repo's own standard of evaluating on the pre-committed set, not
whichever window looks best. Full numbers in `DECISIONS.md` D13.

## Checked whether pooling seasons hides a `GATE_CALIBRATED` trend -- it doesn't, 2026-08-18

Prompted by a direct parallel to `GATE_UNBIASED_BY_WEEK`'s own history: that gate exists
because a full-season aggregate hid a bias that was actually growing steadily week to
week, invisible until someone checked week-by-week instead of averaging. `GATE_CALIBRATED`
pools 5 seasons (2020-2025) into one calibration table the same way -- worth checking
whether it has the same blind spot across *seasons* instead of weeks, given how much about
college football changed inside this window (transfer portal maturing, NIL, conference
realignment). Cost nothing extra to check: the per-season caching fix above (D14) made
reconstructing the full calibrated frame from cache take under 20 seconds.

**Checked two ways. Neither shows a hidden trend the pooled number is covering up.**

Two-era split (2020-2022 vs. 2023-2025, roughly bracketing the realignment/portal
inflection): worst bin off by 32.47pp (n=2,188) in the early era, 28.99pp (n=3,428) in the
late era -- both badly miscalibrated, the later era very slightly *better*, not worse.

Season by season (widened to 20%-width bins so each season has enough observations per
bin to read at all): every single season fails badly, 13.30pp to 23.31pp off, with no
consistent direction -- 2023 reads best (13.30pp), 2025 (the most recent, fewest reasons to
expect it's stale) is back up to 20.96pp, roughly the same territory as 2020-2022.

**Read this as a different, arguably more informative finding than "hidden improving
trend," not a null result.** The concern was reasonable and worth checking directly rather
than assuming either way -- but what actually turned up is that the miscalibration isn't
concentrated in an older, out-of-date era being dragged along by a pool; it's persistently
bad in *every single season checked individually*. That argues against "the sport changed
and the model hasn't caught up" as the primary story, and argues for something more
structural in how the calibrated probabilities are produced -- a real, separate question
from this one, not investigated here.

## Homegrown O-line/D-line proxy features from CFBD, tested and not promoted, 2026-08-18

Prompted by PFF's own published numbers showing trench play is the strongest matchup
correlation found anywhere in the literature (pass-block vs. pass-rush grade R^2=0.66,
run-block vs. run-defense R^2=0.49) -- built a free substitute from CFBD's play-by-play:
`pass_block_success` (sack-avoidance per dropback) and `run_block_success` (non-stuffed-run
rate per carry), both new `PLAY_FEATURES` entries flowing through the existing rolling/
matchup-diff pipeline with no new fit. Confirmed directly against CFBD's raw payload that
no pass-breakup or tackle-for-loss field exists at all, so this covers only the sack/stuff
half of a full PFF-style Havoc Rate, not all of it.

**Result: neither promoted**, matching D12/D13. Closest signal: `run_block_success` on
spread, full-FBS window, t=-1.90 -- still short of the bar, and notably the *opposite*
sign from what the trench-matchup theory predicts (a better recent run-block edge weakly
associated with a *smaller* home margin, not larger). Reported as-is, not rationalized
away. Full numbers and cross-test reading in `DECISIONS.md` D15.

## Weather turned on, and the incremental-vs-market test retired as the default, 2026-08-19

After five straight negative feature tests the user pointed out that the *question* had
drifted from the goal: wind and trench play obviously change football games, so a framework
that keeps answering "no effect" is more likely mis-aimed than football is wrong. That is
correct and it exposed a repeated error. **An incremental-vs-market test subtracts the
market before looking, so anything the market already prices is removed by construction.**
Reporting that as "X does not matter" conflates *the market already knows* with *it isn't
real*. Full detail in `DECISIONS.md` D16; D15's trench conclusion is amended there too
(pass-block predicts margin at t=+2.45 on its own and flips sign only once `net_diff` --
which already contains sacks and stuffs -- is controlled for).

Measured on 2,985 graded games with real Open-Meteo readings, three questions separated:

| question | answer |
|---|---|
| does wind change the actual total? | **yes** -- b=-0.249 pts/mph, t=-3.07; 54.82 pts at 0-5mph falling to 43.86 at 20-25mph |
| does it beat the market? | **no** -- the market moves 3.09 points across that range against reality's 3.14 |
| does it improve our projection? | **yes** -- and this question had never been asked |

**Two near-misses worth recording.** The weather module could not have affected any shipped
number even if switched on -- its only consumer is the simulator, whose total mean
`_season_calibration` overwrites via `.retotaled(model_total)` -- so the adjustment had to
go into `project_walkforward`. And a slope-plus-intercept fit showed a -1.06% gain that
collapsed to **-0.01%** once the slope was mean-centred: the whole effect was an intercept
absorbing an unrelated -2.51 point level bias.

**Shipped**: the existing measured constants applied one-sided only. Wind suppresses
passing and kicking; the absence of wind is not a bonus, so the positive half of the centred
line was asserting a mechanism that does not exist -- and empirically made calm games 0.42%
and dome games 1.81% worse. Negative side only improves overall totals RMSE by **0.342%**,
harms no bucket, is **-3.26% on 15mph+ games**, and adds **no new fitted parameter**.
`dome_total_bump` is no longer applied for the same reason. Rain (152 games, t=+0.23) and
snow (9 games) are not used.

This creates no betting edge and is not evidence of any -- the second row of that table is
direct evidence against it. It makes the projection more accurate on windy games.

## Quarterback availability added, and the simulator decoupled, 2026-08-19

Two follow-ups to D16's reframe (judge an input on whether it improves the projection, not
on whether it beats the market). Full detail in `DECISIONS.md` D17.

**Quarterback (shipped).** The NCAA model previously had no notion of who was playing
quarterback. A starter change is worth -3.21 points against the close (t = -5.56), but that
decomposes into an **announced absence at -0.25, t = -0.37** -- the market prices it
correctly, there is no edge -- and a **mid-game change at -6.46, t = -7.56**, which nobody
can know in advance. Against *our* model the knowable half cost 1.37 points against a +0.61
baseline, and closing that is the point.

The version that mirrors `nfl-model` most closely -- scaling by quarterback QUALITY -- was
built first and **lost**: +5.12 per unit, t = +0.89, RMSE *worse* by 0.020%. A college backup
has too few prior attempts for shrinkage to leave anything but the incumbent's own rating in
disguise. The binary form wins: -1.758 per starter-out, t = -3.05, and the market's implied
adjustment (+1.82 home-out / -1.62 away-out, about 1.72) agrees to 0.04 points from a source
that never saw an outcome. Shipped at 1.75.

Spread RMSE improves in all three windows (restricted 16.955 -> 16.928, pre-committed
17.015 -> 16.985, full FBS 16.918 -> 16.899), by 0.560% on the 806 affected games, by exactly
0.000% on untouched games, and in every one of the five seasons individually.
`GATE_CALIBRATED` moved 34.33pp -> 29.31pp and `GATE_KEY_NUMBERS` 3.89pp -> 3.64pp alongside.

**A silent no-op worth remembering.** The first working build reported `0 with the incumbent
absent` across 11,441 team-games while all unit tests passed: CFBD returns only players who
appeared, so an absent starter has no row rather than a zero row, and the fixtures had
invented a shape real data never produces. Caught solely by a diagnostic count printed in
`run_backtest.py`, which is why that line stays.

**It reached nothing a user sees, and that was corrected the same day.** As first shipped
the adjustment moved `model_spread`, which is a backtest quantity -- but `project_game.py`
and the viewer both report the RAW SIMULATOR MEAN, whose only injection point is
`ContextAdjustment`. The table was also built from box scores, so a scheduled-but-unplayed
game had no row, no adjustment could fire on a future game, and a manual override had
nothing to attach to. Fixed by porting `nfl-model`'s `ContextAdjustment.with_qb`, applying it
in both live paths, and seeding the table from the SCHEDULE with a `_played` flag so an
unplayed game is never mistaken for a missing starter. **A general lesson worth keeping: an
input that only moves a backtest column changes no output, and "the tests pass" does not
establish otherwise.**

**Weather had the same disease, found and fixed the same day.** `bulk_game_weather` writes
a cache keyed on `game_id`; `weather_for_game` read a different file keyed on
lat/lon/date/hour. The viewer calls `build_context(allow_network=False)`, found nothing, and
**projected every outdoor game as though wind did not exist**. Fixed by consulting the
game-keyed cache first; a 17.4mph game now resolves to -1.94 points on the total offline,
where it previously resolved to nothing. Also stopped `bulk_game_weather` treating a
FORECAST as permanently cached -- one taken two weeks out would have been served at kickoff.

**Simulator decoupled (negative, but informative).** `_season_calibration` now records the
simulator's own mean before `recentered()`/`retotaled()` overwrite it. Its disagreement with
the OLS projection does **not** predict error -- correlation +0.020 (spread), +0.019 (total),
and RMSE by disagreement quintile is non-monotonic noise -- so no confidence filter was
built on it. The byproduct is worth more: the simulator's own mean scores **worse** than the
OLS projection (spread 16.785 vs 16.506, total 16.412 vs 16.396), so "the simulator is not in
the mean path" now rests on direct NCAA evidence instead of the NFL measurement
`backtest.py`'s header has always cited.

## NFL examined for the same effects, 2026-08-19/20 -- and it diverges on every one

Prompted by asking whether the day's NCAA work applied to NFL. It mostly does not, and two
of the three checks reversed a conclusion I had already stated.

**Weather: the NCAA one-sided fix must NOT be ported.** NCAA's positive (calm-game) half was
harmful -- it made calm games 0.42% and dome games 1.81% worse. On NFL the same half is
REAL: calm games (<8.17mph) come in **+1.014** above the market (se 0.416) against a config
that adds **+1.08**. Measured, not assumed, and almost exactly right. NFL keeps the
two-sided centred line.

**Dome constant retained, and my objection to it was wrong.** I measured domes at +1.395
against the MARKET and called the +3.16 config value overstated by 2.3x. That was a category
error: the constant is calibrated against the MODEL residual, which the config comment had
already done properly. Re-measured on the backtest frame against the model: indoor residual
**+0.451** (se 0.630), outdoor -0.874, gap +1.325 (t=+1.75, not significant). If anything
+3.16 is slightly small. Unchanged.

**No lean in either market.** Betting the side the model favours, vs the market: totals
**49.7%**, spreads **47.3%**, both below the 52.4% breakeven. The NCAA P1 rule shape applied
to NFL returns 49.1% / 46.0% -- a losing rule. It does not transfer.

**The market under-prices wind (t = -3.57) but it is not usable as measured.** Betting unders
at wind >= 10mph returns 57.7% over 603 games with a CI clearing breakeven, holding across
all three eras while the market's own calm-to-windy spread doubled (1.18 -> 2.35). BUT
nflverse's `wind` is the OBSERVED game-time reading, not the forecast available at bet time.
That is the same class of error as the opener anchoring that produced this repo's only false
positive, so 57.7% is an upper bound and the honest version needs forecast data.

## A post-2022 efficiency regime change in the NFL market, and NOT in college

Chased as a suspected bug in the `linear` spread estimator, which showed
`b = -0.364, t = -2.85` -- deviations from the market being reliably WRONG, not merely
uninformative. It is not a bug, and two intermediate diagnoses were also wrong (a sign
error, ruled out by `actual ~ model` slope +1.004; then compression, ruled out because the
model loses equally pulling toward even (46.0%) and away from it (46.2%)).

**The control settled it: Elo shows the same thing.** By season, our model and an
independently-built Elo move together:

| era | our model | Elo |
|---|---|---|
| 2019-2021 | +0.057 | +0.242 |
| 2022-2025 | **-0.316** (t=-2.64) | **-0.197** (t=-1.73) |

`linear` looked guilty only because it is confounded with era -- it covers 2021 and 2023-25,
and the negative years are 2023+. 2021 is also `linear` and reads -0.105; 2022 is the ridge
and reads +0.035. The estimator was never the variable. Since roughly 2023 the NFL market
appears efficient enough that ANY independent rating system's deviations from it are
systematically wrong.

**This kills the fade idea.** Fading our own NFL spread returns 52.7% [50.0, 55.4], not
clearing breakeven, inconsistent across seasons (4 of 7, 11.7pp range) -- and it would mean
betting on market efficiency, which is the one thing that cannot be exploited.

**College shows no such break**, which is why the same day's P1 registration survives the
question. NCAA's blend stays positive throughout (2021-22 +0.170, 2023-25 +0.108) where NFL
goes sharply negative. Plausible mechanism, and it matches the small-conference result from
the same session: P1 deliberately targets the `restricted` universe -- G5 and cross-tier
games, explicitly excluding marquee P5 matchups -- which is the least-watched segment of
college football, while the NFL is the most heavily bet market there is. **P1 is NOT
upgraded on this**: its by-era split (50.8% then 56.2%) is post-hoc and inside noise, and the
registered rule stays pooled exactly as declared. The objection is removed, not answered in
its favour.

## NFL improvement candidates, tested 2026-08-20 -- one survives, one fails validation

Everything below is stated as distance from the 52.38% breakeven at -110, because RMSE in
points does not translate obviously into whether a thing is worth betting.

**Where NFL actually stands.** Betting the side each estimator disagrees with the market on:

| | bets | win% | vs breakeven | units |
|---|---|---|---|---|
| our model, spread | 1,345 | 47.3% | **-5.1pp** | -143.9u |
| Elo baseline, spread | 1,327 | 48.9% | -3.5pp | -96.8u |
| our model, total | 1,360 | 49.7% | -2.7pp | -76.4u |
| *(NCAA totals, P1 rule, for contrast)* | *1,063* | *54.0%* | *+1.6pp* | *+36.1u* |

Everything NFL loses money, including the baseline. "The simulator does not beat Elo" is a
contest between two losing estimators, which is worth remembering before treating
`GATE_BEATS_ELO` as the thing standing between this build and a bet.

**WIND SURVIVES FORECAST ERROR -- the strongest candidate edge found in either league.**
The market misses wind (t = -3.51 with temperature controlled, so not a cold-weather effect
wearing a wind label). The obvious objection was that nflverse `wind` is the OBSERVED
game-time reading, not the forecast available at bet time -- the same class of error as the
opener anchoring behind this repo's only false positive. Tested by adding Gaussian noise to
the observed value, selecting on the noisy "forecast" and grading on the real outcome,
200 seeds per level:

| forecast error | wind >=10 | wind >=12 |
|---|---|---|
| 0 mph (observed) | +5.3pp | +5.7pp |
| 2 mph | +4.4pp | +4.7pp |
| 3 mph | +3.4pp | +4.0pp |
| 4 mph | +2.4pp | +3.2pp |

At realistic 1-3 day error (~2-4 mph RMSE) the edge holds at **+2.4 to +4.7pp**. It degrades
and does not die. NOT yet actionable: ~60 bets a season, the threshold came from a sweep,
and the noise model is unbiased Gaussian where real forecast error may be biased and worse
in precisely the high-wind games that carry the signal. A real forecast pipeline and a
pre-registration would be the next steps, not a bet.

**THE ELO BLEND FAILS WALK-FORWARD VALIDATION -- do not implement it.** Sweeping the blend
weight in-sample gave RMSE 13.153 at 25% model / 75% Elo, beating both our model (13.464)
and Elo (13.208). Refitting the weight on prior seasons only and applying it to the held-out
season gives **13.410** -- better than our model, WORSE than plain Elo, winning 2 of 5
seasons. The in-sample number was a lucky cell.

**What the blend weights do establish, and it is not flattering.** The walk-forward fit is
strikingly stable across every season it runs: **model 0.22-0.27, Elo 0.66-0.79.** Least
squares consistently wants about a quarter of our model and three quarters of a plain
538-style Elo. The entire NFL pipeline -- EPA ratings, drive simulator, context adjustments,
nfelo QB layer -- is worth roughly a quarter of the baseline this repo computes purely as a
control. That is the real answer to "why does the simulator not beat Elo", and it is more
fundamental than dispersion, which was the previous hypothesis and was wrong.

**Temperature: real, measured, and deliberately NOT implemented.** The config records
+0.072 pts/degF (t = +2.69) against the model residual and calls it the next weather term
worth having. Confirmed here that it is an ACCURACY item and not an edge item: against the
MARKET residual it reads +0.0086/degF, **t = +0.49** -- the market already prices temperature
correctly. Same shape as the NCAA weather result.

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

**Result: confirms the diagnosis, does not clear the gate.** The >28 tail gap fell from -8.33pp to **-3.63pp** (sim 14.27% vs real 17.90%) — recentring recovered most of the missing across-game dispersion, as predicted. But mass that moved out to the tail came from the centre: `|margin|=3` is now the worst bin, sim 5.09% vs real 10.66% (**-5.57pp**, worse than the pre-fix -4.88pp). **FAIL** persists, now on a different bin.

**Sample-size caveat found while reading that result, fixed the same day.** `pooled_margin_pmf`'s candidate pool was every completed game in `games` for the graded seasons — 17,144 rows, because `ingest.load_games` is not FBS-filtered — against which only 2,996 have a fitted rating pair. Of `n_games=400` sampled candidates, ~333 were getting skipped (this inefficiency predates today's `linear_frame` fix; it was already happening via the `rt.index` ratings check, `model_spread` just added a second reason to skip), so the gate was reading off an effective sample of roughly 65-140 games while reporting a "400 games" pool. **Fixed**: `pooled_margin_pmf`'s pool is now restricted to `homeClassification == "fbs" & awayClassification == "fbs"` before sampling. Re-measured: skips fell from 333/400 to **84/400**, effective sample now ~316 games. Reading on the larger sample: |3| -5.37pp, |7| -2.16pp, |10| +0.19pp, |14| +0.92pp, |17| -0.79pp, |21| -0.21pp, >28 **-5.07pp**. Same story as before (worst gaps at the tail and at |3|), now on a sample large enough that the two runs' small numeric differences (e.g. >28 moving from -3.63 to -5.07pp) are more likely genuine sampling variation than an artifact of the tiny post-recentring sample. **FAIL** persists either way — recentring narrowed the tail gap but did not close it, and GATE_KEY_NUMBERS was never expected to pass outright: the linear model's own SD (11.630) still trails the market's (13.085), so the target this gate implicitly asks the simulator to hit is itself short of the market.

**The actual dominant cause of the `\|3\|`/`\|7\|` gaps, found and fixed 2026-08-17.** Asked to work the `\|margin\|=3` gap fully. Isolated the recentring fix's own contribution first (controlled A/B on identical simulated draws, recentred vs raw): only -0.6pp of the gap. The other ~4.8pp was present even in the RAW, un-recentred simulator, which pointed at the drive-level mechanism itself, not the mean-calibration fix above. `shared/sim_core.py::EndgameTable` exists specifically to produce the 3/7-point spikes (its own docstring: independent-drive simulators structurally cannot clear this gate; the fix is conditioning late-drive outcomes on the current score). Inspecting the cached table directly (`endgame_2019_2026.json`) found **every bucket at n=0, every probability NaN** — the entire mechanism had been silently inert. Traced to `shared/sim_core.py::_play_drive`: `(u[:, None] > cum).sum(axis=1)` evaluates to 0 when `cum` is all-NaN (every comparison against NaN is False), and index 0 is `TD` in `DRIVE_CLASSES` — so every endgame-masked drive (a team's final 1-2 drives of every simulated game, ever) was resolving to a **guaranteed touchdown** instead of the scoreboard-conditioned outcome.

Root cause was one column-name typo, several layers upstream: `ncaa-model/src/ingest.py::DRIVE_KEEP` kept a column literally named `"startTime"`. CFBD's raw drive payload has no such column — `client.frame()` runs it through `pd.json_normalize`, which flattens the nested clock into `"startTime.minutes"`/`"startTime.seconds"`. Neither ever matched `"startTime"`, so `_drives_frame`'s `[c for c in DRIVE_KEEP if c in df.columns]` silently dropped the clock from every drive ever loaded through `ingest.load_drives` — for this repo's entire history. `drives.py::_game_seconds_remaining` then saw an all-null column, `start_gsr` was null for 100% of rows, and `fit_endgame_table`'s "last `late_seconds` of regulation" filter matched zero drives every single time it was ever called through the real pipeline — the NaN table wasn't a fluke of one bad run, it was the only possible outcome. The schema guard meant to catch exactly this (`load_drives`'s stale-cache check, right next to a comment about the identical failure class in `nfl-model`'s history) checked for the same wrong `"startTime"` name and so never fired either — it forced a full rebuild on every call instead, which is why the CFBD-call count never revealed the problem. `project_game.py` was unaffected: it reads `drives_<year>.json` directly, bypassing `DRIVE_KEEP`.

**Fixed**: `DRIVE_KEEP` now lists `"startTime.minutes"`/`"startTime.seconds"`; the schema guard checks the same two names. `shared/sim_core.py::fit_endgame_table` also hardened directly — raises loudly on 0 late-game drives instead of caching a degenerate table, and validates `counts.sum() > 0` on every cache load (not just `path.exists()`) so a corrupt table on disk self-heals instead of being served forever. Stale caches deleted (`endgame_2019_2026.json`, `drives_2019_2025.parquet`).

**Result, re-measured end to end on a real cold-cache-minus-network rebuild:** |3| **-5.37pp → -2.27pp**, |7| -2.16pp → -0.90pp, |10| +0.19 → +0.05pp, |14| +0.92 → +0.35pp, |17| -0.79 → -0.41pp, |21| -0.21 → -0.33pp. Every key number now inside or right at the 2pp tolerance. >28 tail: -5.07pp → -5.05pp, essentially unchanged, as expected — that gap is the separate, already-diagnosed across-game dispersion shortfall above, not an endgame-table problem. **GATE_KEY_NUMBERS still FAILS**, now cleanly on a single, already-understood cause (the >28 tail) rather than being muddied by a broken mechanism nobody knew was broken. 42 nfl-model tests, 31/33 ncaa-model tests (2 pre-existing Appendix A failures, unrelated) pass with the fix in. |
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

## Preseason poll points — a real, validated preseason signal, added 2026-08-17

Prompted by the week-1 review: the model's preseason candidate features (recruiting,
returning production, portal activity, continuity) all describe roster *inputs*, but none
of them are a human judgment about the *output* — how good the resulting team actually is.
A generic "new head coach" dummy applies the same average historical effect to every
coaching change; it cannot know that one specific hire is a proven program-builder and
another is a retread. The market prices that in immediately; the model had no path to it.

`ratings/sp` (CFBD's SP+) looked like the fix and was rejected after direct measurement:
`week=` had no effect on a past season's result (2024 full-season and week=1 pulls
returned the identical, final, descriptive rating), so a "preseason" SP+ pull for a
historical season is actually the whole season's outcome leaking in as a prior. Not
usable for a validated, leakage-safe feature.

`rankings?week=1&seasonType=regular` does not have that problem — polls are inherently
time-stamped, and the pulled 2024 week-1 AP Top 25 (Georgia 1, Ohio State 2, ... Florida
State 10) matches the real, well-documented preseason poll, not one contaminated by
FSU's actual historically bad season. Implemented as `preseason_poll_points` (mean of
AP Top 25 / Coaches Poll points, per `src/features.py::_preseason_poll_points`), zero-filled
for teams outside the top 25 — being unranked is the signal, not missing data — and wired
through the existing `PRESEASON_FEATURES` / validated-challenger-promotion machinery
exactly like every other preseason candidate, not force-fed in unvalidated.

**Result: promoted, and it measurably helps.** `preseason_poll_points_diff_preseason`
is in `fit_live_mean`'s spread feature set and `spread_preseason_promoted` stays `True`
with it included. Measured on the real week-1 slate (51 games with a market line): mean
spread bias -4.84pt → **-4.47pt**, SD 10.10 → 9.61, games off by >15pt 11 → 10. The
effect concentrates where you'd expect — teams the human polls actually rank move the
most (Texas State @ Texas: gap -18.6pt → -13.8pt; North Texas @ Indiana: -22.3pt →
-17.9pt), while games between two unranked teams (Oklahoma State @ Tulsa) are essentially
unaffected, since both sides contribute zero poll points either way. 6th CFBD call added
per season (192/250 budget, still comfortable). 32/34 ncaa-model tests, 114/114 shared
tests pass (2 pre-existing Appendix A failures, unrelated).

---

## GATE_UNBIASED_BY_WEEK is not really a week-4 problem, measured 2026-08-17

The gate's worst reading has consistently been "spread wk4: -3.40, n=167" -- read here
previously as an unexplained week-4 anomaly. It is not week-specific. Splitting that same
167-game restricted week-4 bucket by `cross_tier` (P5 vs G5): non-cross-tier bias is
-0.50 (n=101, clean); cross-tier bias is **-7.84** (n=66). Checking every other week the
same way, cross-tier bias is negative in 11 of 12 tested weeks, ranging -1.9 to -13.7 --
**worse than week 4 in most weeks** -- but no other week has 40+ cross-tier games, so
`min_n=40` never lets them register. Week 4 isn't where the bias is worst; it's the one
week with enough cross-tier sample size to prove statistically what every week already
shows. Pooled across the full season: cross-tier restricted games, n=170, bias **-7.89
points**; non-cross-tier restricted games, n=1,374, bias +0.20 (clean).

This is the same phenomenon the week-1 review and `preseason_poll_points` addition
above are about -- the model under-projecting games between mismatched tiers -- now
precisely quantified on the gate's own OLS pathway (`model_spread` from
`walk_forward`/`build_features`) rather than the live `fit_live_mean` ridge path.
**Important seam, not yet closed**: `preseason_poll_points` and the rest of
`PRESEASON_FEATURES` only feed `fit_live_mean`, the live-publication path. They are not
in `build_features`'s `net_diff`/`pace_sum`/`eff_sum` predictors, so they do not touch
`model_spread` and cannot move this gate, `GATE_RMSE_*`, or any other gate graded against
it. The two prediction pathways in this codebase are more separate than "gate reading"
vs. "published number" suggests -- fixing the live site's week-1 numbers, done above,
provably does not fix what the gates measure, and vice versa. Worth a real design
decision (unify the paths, or explicitly document why they diverge) rather than assuming
one pathway's fix reaches the other.

---

## NFL GATE_SCALE forensic pass, 2026-08-17 — confirmed honest, mechanism now precise

Item #6 from the project-status review: give NFL's `GATE_SCALE` (spread sd ratio 0.990,
total 0.707) the same scrutiny NCAA's got today, rather than taking its "don't fix, honest
signal gap" docstring on faith the way the previous pass did.

Traced the asymmetry to its exact mechanism instead of stopping at the docstring.
`config/nfl.yaml`'s `projection.spread_source: linear` means spread is on `_fit_l1_spread`
-- an actively-fit least-squares estimator, further enriched by `_fit_feature_challenger`
(rest days, coach continuity, injury/matchup burden, validated the same way NCAA's
`fit_live_mean` validates its candidates) -- while total is pinned to the raw drive
simulator. That split is not new; NEXT_SESSION.md already measured that the simulator
beats linear on total RMSE (13.445 vs 13.630) while linear beats the simulator on spread
RMSE (13.196 vs 13.668), and chose accordingly. What was not previously stated: that same
choice is *why* `GATE_SCALE` reads so differently on the two markets. Spread's ratio is
near 1.0 because it's on the actively-fit, richer-feature path; total's 0.707 is the raw
simulator's naturally compressed dispersion, undiluted -- the same mechanism behind
NCAA's blowout-tail underproduction, fixed there with `SimResult.recentered()` onto a
better-calibrated linear target.

That specific fix does not transfer here. Recentering NCAA's tail onto its linear mean
worked because the linear estimator is better-calibrated on dispersion without being
worse on accuracy. NFL's linear total estimator is *worse* on RMSE (that's the whole
reason total stays on the simulator) -- recentering onto it would trade a dispersion
problem for an accuracy problem, undermining the reason this design exists. No fix
applied. Confirms the docstring's conclusion holds, now for a stated mechanism rather
than as a blanket "trust the analysis" — worth knowing precisely instead of assumed.

---

## Root cause of the >28 tail's residual gap, 2026-08-17 — softmax saturation

Item #4 from the project-status review, left open after the recentring fix: why does
`fit_drive_model`'s multinomial produce a compressed across-game mean (SD 7.56) when the
drive-count clamp and L2 regularization (`C`) were both directly tested and ruled out?

Measured the mechanism directly: fed the fitted NCAA drive model a sweep of `net_epa`
across the real observed range (`off_rating`/`def_rating` extremes give roughly [-0.14,
+0.30]) at average field position, and read off expected points per drive
(`P(TD)*6.94 + P(FG)*3.0`) and its slope with respect to `net_epa`. The response is not
close to linear -- it is the S-curve `probabilities()`'s softmax guarantees by
construction. The marginal effect peaks at **13.4 pts/drive per unit net_epa** near
`net_epa≈+0.06` (a middling favorite) and falls to **6.8** at `net_epa=-0.14` (worst
offense vs. best defense) and **9.0** at `net_epa=+0.24` (best offense vs. worst
defense) -- roughly half the model's peak sensitivity spent exactly where real blowout
mismatches live. Over ~12 drives a team, that difference compounds into materially less
game-to-game spread than a model with constant marginal sensitivity would produce, with
no tunable knob responsible: it is a property of the multinomial-softmax parameterization
itself (`design = [net_epa, fp, fp², home, net_epa·fp]`, softmax over four classes),
present regardless of how well the coefficients are fit or how strongly they are
regularized -- which is exactly why sweeping `C` across two orders of magnitude
(recorded in `analysis/blowout_tail_diagnosis.py`) moved nothing.

**Not fixed in this pass, and deliberately so.** The candidate fix -- adding `net_epa²`
or a spline/piecewise term to the design matrix so the model has room to counteract its
own saturation at the extremes -- means refitting `fit_drive_model` and revalidating
every gate that reads from it (`GATE_KEY_NUMBERS`, `GATE_RMSE_*`, `GATE_SCALE`,
`GATE_UNBIASED_BY_WEEK`), the same scope discipline `GATE_CALIBRATED` was left unbuilt
under for the same reason. Recorded as the concrete next step, not attempted alongside
everything else today.

---

## Real, team-specific FCS ratings, 2026-08-17

Every non-FBS opponent used to collapse into one shared `__FCS__` rating bucket at
ingest (`ingest.py::build_game_offense`), so the model could not distinguish an elite,
perennial-contender FCS program from a bottom-tier one. Quantified the blast radius
before fixing it: FBS-vs-FCS games are only 3.5% of a full season (602/17,152,
2021-2025) but **18.5% of week 1** (231/1,252) — the exact week the model is already
weakest and the site gets the most attention. Directly caused the Jacksonville State @
North Dakota State sign flip from the week-1 review (model favored the FBS team, market
favored NDSU by 8.5).

Checked feasibility before touching anything: CFBD's cached drive/play data already
covers FCS-vs-FCS games in full (North Dakota State alone has 294 real drives cached for
2024, a normal full-season sample) — no new CFBD calls needed. The fix turned out to be
one function: stop collapsing `"fcs"`-classified teams alongside `"fbs"` in
`build_game_offense`'s `home_norm`/`away_norm` construction (still collapse
II/III/unclassified — verified directly that FBS never schedules them, 0 such games in
7 seasons). `team_universe`/`build_walkforward`/`fit_pace` already derive the fitted team
list dynamically from whatever's in the frame, and `resolve_team_ratings` already
prefers a real team name over the bucket (its own docstring: written to handle a wider
universe than it had ever actually been given) — none of that needed to change.
`fit_ratings`'s ridge is a connected graph (FBS-vs-FBS, ~120/season FBS-vs-FCS bridge
games, now FCS-vs-FCS too), so the joint least-squares solve places every team on the
same PPA-per-play scale automatically; no separate calibration step was needed.

**Validated end to end.** Cold rebuild: 30s, no crash, 266 teams now individually rated
(was ~131). New ratings pass a face-validity check directly: North Dakota State
(off +0.046, def -0.033, both strong) and Montana (+0.031/+0.004) rate clearly above the
residual bucket (-0.022/+0.045); Nicholls, a genuinely weaker program, rates below both.
Every currently-measured gate (`GATE_RMSE_*`, `GATE_SCALE`, `GATE_KEY_NUMBERS`,
`GATE_UNBIASED_BY_WEEK`) is computed on `fbs_only`/`restricted` populations that already
exclude FBS-vs-FCS games, so none of them should move on this fix's account — most read
flat, and two moved further than expected in a *good* direction anyway
(`GATE_UNBIASED_BY_WEEK` 3.401→2.703, `GATE_KEY_NUMBERS`'s >28 tail 5.05pp→4.13pp),
because FBS teams' own ratings get more precise too once their ~1 bridge game a season is
credited against the specific opponent they actually played instead of a blurred average.
The Jacksonville State @ North Dakota State sign flip is fully resolved: model now
favors NDSU by 15.6, matching the market's own +8.5 in direction, a real gap down from a
wrong-signed -19.3pt one. Whole week-1 slate: mean spread bias -4.47pt → **-1.09pt**,
games off by >15pt 10 → 6.

**One real edge case found and fixed along the way.** `fit_live_mean` fits spread and
total as two separate ridge models with nothing constraining their combination to be
physically possible. Real FCS ratings can now be extreme enough to break that assumption:
Georgia vs Tennessee State projected spread +47.6 and total +47.0, which implies an away
score of (47.0-47.6)/2 = -0.3 — impossible on any scoreboard. Each target was
individually inside the simulator's own range; the pair wasn't jointly achievable on its
score lattice, and `sim.recentered().retotaled()` correctly raised rather than return
something nonsensical. `scripts/build_slate.py::_independent_forecast` now catches that
specific `ValueError` and falls back to the raw simulator for that one game — the same
fallback already used when the validated mean doesn't exist at all, not a new code path,
and disclosed via the same `source` field either way. The real fix (constraining the two
ridge models jointly) is not attempted here; recorded as a candidate follow-on.

---

## The softmax-saturation candidate fix does not work, tested 2026-08-17

The root-cause entry above proposed a candidate fix for the >28 tail's residual gap: add
`net_epa**2` to `fit_drive_model`'s design matrix so the model has room to counteract its
own saturation at the extremes. Tested it directly on the real 196,371-drive training set
before shipping it, per this file's own rule. **It does not work.** Refit with and
without the squared term and compared the same marginal-sensitivity ratio the root-cause
measurement uses: 0.40 baseline vs. 0.41 with `net_epa**2` -- no meaningful change.

The reason is more useful than the negative result. `net_epa**2` is symmetric in
`net_epa`, so its fitted coefficient mostly rescales the curve's width around
`net_epa=0`, not the saturating shape at both extremes -- and no polynomial feature could
fix that shape, because softmax outputs are bounded in `[0, 1]` by construction, for any
input, for any design matrix. A team cannot have a >100% chance of scoring; that ceiling
is not a flexibility problem this model failed to solve, it is a property of predicting a
bounded probability at all. Correctly rules out "give the drive model more features" as a
path forward for this specific gap, rather than leaving it as an untested assumption.
What would actually need to change is the model family for the mean path itself, not its
feature set -- not attempted here. Reproducible via `analysis/blowout_tail_diagnosis.py`
PART 5, no new CFBD calls.

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
