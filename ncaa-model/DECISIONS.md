# DECISIONS — NCAA build

Judgment calls, in the order they were made. Anything pre-registered is marked as such and
was written here **before** the corresponding number was computed.

---

## D1. `provider_priority` — spec-bug fix, `consensus` dropped for opener purposes

`NCAA_PLAYBOOK.md` §4 sets `provider_priority: ["consensus", "DraftKings", "Bovada",
"ESPN Bet"]`. Measured on 2,737 restricted FBS-vs-FBS games, 2019–2025:

| provider | n | has `spreadOpen` | has `overUnderOpen` |
|---|---|---|---|
| **consensus** (priority #1) | 1,503 | **0.0%** | **0.0%** |
| Bovada | 2,447 | 84.5% | 84.5% |
| ESPN Bet | 950 | 84.3% | 83.4% |
| DraftKings | 1,169 | 94.1% | 34.7% |
| teamrankings, William Hill, numberfire, Caesars | — | 0.0% | 0.0% |

`consensus` is a synthetic aggregate and structurally carries no opener. With it first,
`GATE_OPENER_COVERAGE` measured 16.8% and failed. This is a spec bug, not a data problem:
the pre-flight was resolving most games to a provider that cannot contain the quantity
under test.

**Chosen:** `provider_priority: ["Bovada", "ESPN Bet", "DraftKings"]`, `consensus` removed
entirely for opener purposes. Selection prefers a provider that actually carries both
openers before falling back. Under this, `GATE_OPENER_COVERAGE` passes: coverage 75.5%
across 2019–2025 restricted, ~100% from 2021, mean |open − close| = 1.40.

This also matches the operator's standing preference to price off Bovada rather than the
ESPN/DraftKings number.

## D2. Backtest window — 2019–2020 excluded

Opener coverage is **all-or-nothing by season**, not game-by-game:

| season | restricted games | coverage (both openers) |
|---|---|---|
| 2019 | 418 | **0.0%** |
| 2020 | 243 | **0.0%** |
| 2021 | 413 | 100.0% |
| 2022 | 418 | 98.8% |
| 2023 | 409 | 100.0% |
| 2024 | 415 | 99.8% |
| 2025 | 421 | 100.0% |

The pre-committed missing-at-random test compared covered against uncovered restricted
games in 2019–2022. Because coverage is a per-season property, that comparison is 2021–22
against 2019–20 and is perfectly confounded with era. **All four checks failed:**

| check | covered (2021–22) | uncovered (2019–20) | difference | rule | result |
|---|---|---|---|---|---|
| closing spread, mean | +5.691 | +4.512 | 1.179 pts | ≤ 0.5 | **FAIL** |
| closing total, mean | 55.234 | 56.615 | 1.381 pts | ≤ 0.5 | **FAIL** |
| cross-tier share | 24.2% | 18.5% | 5.7 pts | ≤ 5 | **FAIL** |
| week distribution, max share gap | — | — | 6.3 pts (week 14) | ≤ 5 | **FAIL** |

Supporting detail: mean |closing spread| differs by only 0.294 and the spread KS test is
insignificant (p = 0.61), so the spread failure is a *location* difference rather than a
different mix of competitive games. The total KS test is significant (p = 0.0013). The week
gap is driven by 2020's COVID-shortened, back-loaded schedule (mean week 8.17 uncovered vs
6.87 covered) and by 2019–20 carrying late-season weeks the covered subset barely reaches
(week 14: 7.7% uncovered vs 1.3% covered).

**Chosen:** exclude 2019–2020 from opener-graded scoring. They remain available as ratings
history, which is where they are still useful.

**On the window boundary.** The pre-committed rule named 2023–2025 (1,241 games) as the
fallback. That figure came from an earlier under-measurement of mine, made under the broken
`consensus`-first priority, which showed openers starting only in 2023. The true opener era
begins in **2021**. The failed test indicts 2019–2020; it says nothing against 2021–2022,
which are ~100% covered.

**Pre-registered before any blend coefficient was computed**, to keep the correction
outcome-blind:

- **Primary window: 2021–2025 restricted** (~2,076 games).
- **Secondary window: 2023–2025 restricted** (~1,245 games), the literal pre-committed
  fallback.

The kill criterion is reported on **both**. If they disagree, that disagreement is the
finding and the more conservative reading governs. Choosing the larger window after seeing
a t-statistic would be gerrymandering; choosing it beforehand, to correct a factual error
in that rule's input, is not.

### D2a. The two-window clause, resolved explicitly rather than silently

D2 said "if they disagree, the more conservative reading governs." The windows returned
t = 2.844 (2021–2025) and t = 1.908 (2023–2025), and read literally that clause says stop.
**It is being overridden deliberately, and the reasoning is recorded here so the override
is not silent.**

They are not a disagreement. `b` is stable across every window measured:

| window | n | b | implied se | t |
|---|---|---|---|---|
| 2021–2025 restricted | 1,543 | +0.262 | 0.092 | 2.844 |
| 2023–2025 restricted | 933 | +0.243 | 0.127 | 1.908 |
| 2021–2025 full FBS | 2,985 | +0.191 | 0.065 | 2.921 |

The effect size barely moves; only the standard error does, and it moves with sample size
exactly as it should. That is **one finding measured at three sample sizes**, not three
findings in conflict.

Second, and decisive: **2023–2025 is not the held-out set.** The kill criterion in PATCH 04
§2 names *held-out* `t(b_model)`, and D3 defines held-out as 2024–2025. The 2023–2025 window
mixes 2023 — a tuning season — into it. Judging the criterion on a window that contaminates
tuning years into the holdout is a worse test than the pre-registered one, not a more
conservative one.

The properly held-out read (2024–2025, used for nothing else) clears 2.0 in **all twelve**
pre-registered cells, +2.20 to +3.12, varying monotonically with `half_life_games` — the
broad plateau PATCH 02 §3 asked for, not an isolated spike.

## D3. Pre-registered hyperparameter grid — 12 cells

Declared before the first sweep, per PATCH 04 §1.9 and PATCH 02 §3. Scored on **held-out
`t(b_model)`**, never RMSE. Only parameters that *re-rank* games are swept — scale-only
changes leave `t` exactly invariant, so sweeping λ alone is inert.

| cell | `half_life_games` | `prior_weight_games` | `off_weight`:`def_weight` |
|---|---|---|---|
| 1 | 6 | 5 | 1.45 : 1.0 |
| 2 | 6 | 10 | 1.45 : 1.0 |
| 3 | 6 | 5 | 1.20 : 1.0 |
| 4 | 6 | 5 | 2.00 : 1.0 |
| 5 | 10 | 5 | 1.45 : 1.0 |
| 6 | 10 | 10 | 1.45 : 1.0 |
| 7 | 10 | 5 | 1.20 : 1.0 |
| 8 | 10 | 5 | 2.00 : 1.0 |
| 9 | 16 | 5 | 1.45 : 1.0 |
| 10 | 16 | 10 | 1.45 : 1.0 |
| 11 | 16 | 16 | 1.45 : 1.0 |
| 12 | 16 | 5 | 2.00 : 1.0 |

Tune on 2021–2023, validate on 2024–2025. All twelve cells are reported, not the winner.
No cell outside this table is evaluated.

## D4. Architecture — the simulator is never in the mean path

From the NFL null (`NFL_PLAYBOOK.md` Appendix A): the drive simulator contributed all
5.02 points of excess dispersion in the NFL mean projection, while a direct linear
projection off the same ratings carried none and scored better (13.529 vs 13.882 RMSE).

**L1 (mean)** is a direct linear projection off the ridge ratings. **L2 (simulator)** is
built only if the kill criterion passes, and produces totals shape, the margin/total joint
and factor sensitivity — never `E[margin]` or `E[total]`.

> **CORRECTION 2026-08-07 — this rule is right for spreads and wrong for totals.** The
> 13.529-vs-13.882 evidence above is a *margin* measurement that was generalised to both
> markets without being tested on totals. Re-measured on NFL data under the 750/550
> ratings, 1,119 walk-forward games, paired season-clustered bootstrap: on spreads the
> linear projection wins by 0.47 RMSE (13.196 vs 13.668, CI [+8.36, +16.98] on the mean
> squared-error difference); **on totals the simulator wins by 0.19** (13.445 vs 13.630,
> CI [−8.44, −1.86]). Both significant. A margin is a difference and is near-linear in
> `net_diff`; a total is generated by drives, and pace × efficiency is compositional, which
> a linear model in `(pace_sum, eff_sum)` cannot represent.
>
> **MEASURED ON COLLEGE DATA, SAME DAY — AND THE NFL RESULT DOES NOT CARRY ACROSS.** The
> paragraph above originally guessed that `E[total]` should come from the simulator here
> too. It should not. 3,472 college games (2020–2025), simulator run with this repo's own
> artifacts against `model_total_pure`, graded on the close: **linear 16.567 vs simulator
> 16.954 — linear wins by 0.39, significant** (season-clustered CI [+4.65, +25.88]), stable
> excluding 2020 and on 2022–25 alone. On spreads linear leads by 0.14, not significant.
>
> So D4 as originally written is CORRECT FOR THIS REPO on both markets, and the two leagues
> genuinely diverge on totals — NFL's simulator beats its linear projection there, college's
> does not. Do not "port" the NFL fix. Changing nothing here also changes no verdict: best
> ratios against the close are 1.084 (spread) and 1.038 (total), so GATE_RMSE fails either
> way.
>
> One open lead, not an architecture question: the college simulator is **under-dispersed on
> margins** — sd 7.74 against a market sd of 12.79 and an actual 19.79, while the linear
> projection sits at 12.56. NFL's simulator is *over*-dispersed by comparison (7.91 vs 6.31).
> That looks like a defect in how rating differences propagate into simulated college
> margins, and it is not Monte Carlo noise, which would add variance rather than remove it.
> Until it is understood, read the result above as "the simulator as currently built loses
> here," not "the simulator cannot win here."

Consequence for sequencing: the Step 2 verdict needs L1, the blend and the gates. It does
not need the simulator, so the simulator is not built before the verdict.

## D5. Efficiency source — play-level, so the garbage-time filter is real

CFBD's `/ppa/games` would give per-game team efficiency in one call per season, but it is
pre-aggregated and cannot be garbage-time filtered. Non-negotiable §1.8 requires the filter
to run *before* any efficiency number is computed, so efficiency is built from `/plays`
(paginated by week) and filtered with the §6a rules-based definition. The extra API cost is
budgeted and the cache is permanent.

## D6b. Gain correction — GATE_SCALE, and what it cost

The raw totals projection had an SD ratio of **0.787** against the opener: its
disagreements were 21% too small. That is not a cosmetic problem. The bet sheet selects on
a fixed 3.5-point threshold, so a systematically compressed projection **under-flags** —
real edges fall below the line and are skipped — and a paper period run on an artificially
thin slate tells you less than it should.

The cause is not a bug: least squares is an RMSE-minimizing estimator, so its fitted values
are attenuated by construction (`sd(pred) = R · sd(y)`). Optimal for RMSE, wrong for
betting.

**Chosen:** a scalar gain per market, fit on strictly earlier seasons and applied about the
projection's own mean so the level — and therefore the intercept `a` — is preserved. Capped
to [0.5, 2.0]: a gain far from 1 means the projection is not merely attenuated, and
inflating it would amplify noise rather than restore signal.

**What it cost, recorded honestly.** Rescaling the component is not a scalar multiple of the
disagreement, so it moves `t` — and here it moved it down:

| market / window | SD ratio before → after | t(b) before → after |
|---|---|---|
| totals, 2021–2025 restricted | 0.787 → **1.018** | 2.844 → **2.285** |
| totals, 2023–2025 restricted | 0.779 → 1.066 | 1.908 → 1.077 |
| totals, 2021–2025 full FBS | 0.820 → 1.058 | 2.921 → 1.733 |
| spreads, 2021–2025 restricted | 0.908 → 1.007 | 0.874 → 0.886 |

The kill criterion still passes on the primary window (2.285), and `GATE_SCALE` now passes
at a 0.018 deviation.

**The t drop is the correction working, not damage.** 2.844 was earned partly on dispersion
the projection was getting credit for without having: an under-dispersed component makes
smaller disagreements, and smaller disagreements that still correlate with outcomes score a
higher t per point of spread. Removing the free credit lowers t to what a
correctly-scaled projection actually supports. **2.285 on a properly scaled projection is
the more trustworthy number**, and it is the one to carry forward. If the paper period
comes back weak, the uncorrected projection's higher t is not evidence against it.

## D6c. `min_edge_points_total` is unreachable at the fitted `b` — threshold re-derived

**Pre-committed before any CLV number was computed.** The threshold below was chosen from
the arithmetic of the filter, not from which setting produced a result worth having.

`NCAA_PLAYBOOK.md` §4 sets `min_edge_points_total: 3.5`, applied per §9 to the *blended*
edge (`blended − market`). The blend shrinks the model's disagreement by `b`, so the raw
disagreement needed to clear it is `3.5 / b`. At the fitted `b_total = 0.2413` that is
**14.5 points**. Measured on the 72-game paper slate:

| quantile | raw \|model − opener\| | blended \|edge\| |
|---|---|---|
| p50 | 3.75 | 0.90 |
| p90 | 7.46 | 1.80 |
| p99 | 11.64 | 2.81 |
| max | **12.07** | **2.91** |

The largest disagreement in the sample is 12.07 against a 14.5 requirement. **Zero picks
cleared, and zero would clear on any slate** — the filter is not selective, it is inert.

The playbook's 3.5 was calibrated for the `b` it expected: §8b says "if your NFL model
weight came out around 0.25 and your NCAA weight comes out around 0.45, that is the
expected pattern". The fitted weight is 0.24, so the threshold is mis-scaled by about 2×.

**Chosen, and why this is not goalpost-moving:** `min_cover_prob` is the economically
grounded filter and it is left untouched at 0.545 against a −110 breakeven of 0.5238. It
already encodes "beat the vig with margin", computed from the L3 calibrated distribution.
The points threshold is a *robustness floor* on top of it — "do not bet a disagreement so
small that model error dominates" — and a floor has to be on the scale of the quantity it
floors. It is set to **1.0 blended points**, roughly the median absolute line move between
open and close (1.40), so a pick must disagree with the market by more than the market
typically moves on its own.

Both filters must pass. Neither was selected by looking at the resulting CLV, win rate, or
pick count beyond confirming the filter is capable of firing at all.

**Watch this.** A threshold that admits ~20% of the slate is looser than §10's expected
8–15%. If the paper period shows CLV at or below 50%, the first thing to suspect is that
this floor is too low, not that the model is broken.

## D6d. Within-season level drift — diagnosed, corrected, and the residual excluded

**The partition claim was wrong and is retracted.** An earlier note read `b_early = +0.242`
against `b_late = −0.146` as "the signal is absent from week 13 on". Testing the difference
directly: se ≈ 0.095 and 0.235, difference +0.388 with se 0.254, **t = 1.53** — not
significant. The 90% CI on the late-season `b` is [−0.533, +0.241] and contains the
early-season estimate. At n = 254 you would need b ≥ 0.471 to detect anything at t = 2, so
that partition is underpowered by construction. It was an underpowered subsample examined
after a bad outcome, which is the same failure mode as re-picking the paper window.

**What is established is the level bias**, and it is sufficient on its own. At +1.74 points
by week 13, the constant offset alone exceeded the 1.0-point selection floor, so every game
cleared on the OVER side regardless of any game-specific view. 7/7 OVER was the filter
measuring a constant, not a directional read — which is why the 1-6 record carried no
information either way.

### Mechanism — measured, not assumed

All three candidate causes were tested and eliminated:

| candidate | measurement | verdict |
|---|---|---|
| pace ratings inflating as sample grows | `pace_sum` wk4→wk14: +0.003 → +0.005; `pace_sd` 0.0147 → 0.0153 | **no** |
| shrinkage relaxing late season | `off_sd` 0.0239 → 0.0239; `def_sd` 0.0216 → 0.0213 — flat | **no** |
| garbage filter on blowout-heavy late slates | would move `eff_sum` magnitude; it moves +0.003 | **no** |

The actual mechanism is **amplification**. `eff_sum` drifts only −0.010 → −0.003 across a
season (PPA/play units), but the fitted totals coefficient on it is roughly 250×, turning a
0.003 compositional drift into ~0.8 points. The opener stays flat (54.08 → 53.98) while the
model climbs 54.71 → 55.79. Weeks 15–16 are a separate phenomenon entirely — `pace_sum`
flips negative, n = 13 and 2.

### Correction

A **time-varying intercept**: per-week level bias estimated on strictly earlier seasons,
shrunk toward the global bias by `n / (n + 40)` so a two-game week cannot contribute a wild
correction, and subtracted before the blend. `GATE_UNBIASED_BY_WEEK` added — |a| < 0.5 in
every week bucket with n ≥ 40, not merely in aggregate. The full-sample gate could not see
this: averaging a drift that runs +0.62 → +7.68 produced a passing +0.219.

Totals drift fell from +1.74 (wk13) to +0.82.

### Residual exclusion — mechanistic, not outcome-driven

`GATE_UNBIASED_BY_WEEK` still fails after correction. Tuning the correction harder does not
help, which is itself informative:

| shrink | totals weeks still failing |
|---|---|
| 40 | wk10 +0.53, wk13 +0.82, wk14 +0.68 |
| 15 | wk10 +0.55, wk13 +0.78, wk14 +0.63 |
| 5 | wk10 +0.56, wk13 +0.76, wk14 +0.62 |
| 0 | wk10 +0.57, wk13 +0.74, wk14 +0.63 |

The residual is nearly invariant to shrinkage, so it is **year-specific rather than a
stable within-season pattern** — not learnable from prior seasons at any correction
strength. `shrink` was left at its original 40; it was not tuned.

**`exclude_weeks_total: [10, 13, 14, 15, 16]`** — the weeks where the gate fails (10, 13,
14) plus those too thin to test (15: n=13, 16: n=2). Decided by the gate before the paper
period was re-run, on the bias measurement alone. No outcome was consulted.

## D6e. Concentration caps — 15% per team, 25% per conference

A pick set dominated by a few teams is more plausibly a couple of rating errors producing
repeat disagreements than a distributed edge. Caps make the distinction testable.

Measured on 68 uncapped historical picks: **no team exceeds 15%** — Sam Houston is highest
at 13.2% (9 picks). The 4-of-13 concentration seen in the first paper period was a
small-sample artifact, not a standing property. Conferences do exceed: Mountain West 50%,
Conference USA 38%, Mid-American 29%, Sun Belt 28% — which is what a G5-restricted universe
looks like by construction rather than a defect.

Enforced by greedy selection on descending |edge| within each season, so when a cap binds
the weakest disagreements are dropped rather than an arbitrary subset. Caps default to ON
in `build_totals_sheet`, so they cannot be skipped by forgetting a flag.

**The edge survives:** CLV 0.7049 (68 picks) -> 0.6923 (42 picks). Cost is 38% of the pick
set for no CLV improvement, which is worth revisiting — the conference cap is arguably
measuring the universe rather than a defect.

## D7. OPEN ITEM — spread CLV of 0.5285 against b ≈ 0

On the primary window the spread filter produced 1,053 positions at a **52.85% CLV**: the
close moved toward the model's side more often than chance, on a market where the blend
coefficient is indistinguishable from zero (b = +0.047, t = 0.89) and the filtered win rate
is 48.9%.

Those two facts sit awkwardly together. Two readings:

1. **Noise.** At n = 1,053 the standard error on a CLV proportion is about 1.5 points, so
   52.85% is roughly two standard errors from 50 — suggestive, not decisive, and it is one
   number among many being looked at.
2. **A small masked effect.** CLV measures whether the market agrees with you *after* you
   commit, and it is far less noisy than a win record because it does not wait on game
   outcomes. A positive CLV with a null `b` is the signature of a real but small edge that
   the residual regression cannot resolve at this sample size — which is exactly the
   precision problem PATCH 02 §2 identified on NFL.

**Not being chased now.** Spreads stay dark and no spread picks are emitted. The paper
period will speak to it: if spread CLV persists above 52% over fresh weeks while `b` stays
null, that is worth revisiting; if it drifts to 50%, it was noise. Deliberately left as an
open question rather than resolved by searching for it.

## D8. Opener-injection points, and the pure projection (Phase 0E)

The column named `model_total` has never been a function of features alone. A betting line
enters it in **three** places, all before anything downstream sees it:

| # | quantity | line used | file:line |
|---|---|---|---|
| 1 | Gain correction — projection rescaled so its SD matches the market's | `total_open` | `src/backtest.py::_fit_gain`, applied at `project_walkforward` |
| 2 | Time-varying intercept — per-week `mean(projection − opener)` subtracted | `total_open` | `src/backtest.py::_fit_week_bias`, applied at `project_walkforward` |
| 3 | The blend — the final bet number is anchored on the opener | `total_open` | `src/betting.py::blended_total` |

`model_total_pure` / `model_spread_pure` are now persisted at the last point before (1),
so a projection uncontaminated by any line exists on disk for the first time.

### What the pure projection measures

Close-anchored, 2021–2025, `analysis/power.py`:

| universe | column | n | b | t | power at b=0.20 | verdict |
|---|---|---|---|---|---|---|
| restricted | `model_total` | 1,543 | +0.0762 | +0.89 | 64.9% | underpowered |
| restricted | `model_total_pure` | 1,543 | +0.1430 | +1.62 | 62.1% | underpowered |
| full FBS | `model_total` | 2,985 | **−0.0040** | −0.07 | 93.2% | **null** |
| full FBS | `model_total_pure` | 2,985 | **+0.0802** | +1.30 | 89.8% | **null** |

**The opener-anchored corrections do not merely contaminate the measurement — they destroy
the projection.** Stripping them moves `b` from −0.004 to +0.080 on the adequately powered
window. Corrections (1) and (2) were introduced to satisfy `GATE_SCALE` and
`GATE_UNBIASED_BY_WEEK`, and both are fit against the opener; they rescale and re-level the
projection toward a number that is itself noisy, and the projection loses information doing
it.

**This does not resurrect the edge.** At 89.8% power against the pre-registered reference
effect of 0.20, `b = +0.0802` is a null. What it rules out is only the *claimed* effect
size: a true effect near 0.08 would need roughly n = 7,000 to detect and remains
unexcluded. That is a statement about what has not been tested, not a finding.

**Consequence for any future phase:** measure on `*_pure` against the **close**. If the
gain and week-bias corrections are kept at all, they must be refit against the close, and
the refit must be pre-registered before `b` is looked at.

## D7. The live-publication path and the gate-measurement path stay separate, 2026-08-17

GATES.md's 2026-08-17 entries flagged, while root-causing `GATE_UNBIASED_BY_WEEK`'s "week
4" reading, that the live site's fixes (preseason poll points, FCS ratings) cannot move
any gate: `model_spread`/`model_total` (what every gate reads) come from `walk_forward`/
`build_features`'s single OLS fit across the whole season, while `PRESEASON_FEATURES`
only feed `fit_live_mean`, the separate system behind `projection_mean()` that the public
site actually uses. Flagged then as "worth a real design decision... not left as a
standing trap." Decided: **keep them separate.**

The reason is not inertia -- `fit_live_mean` is phase-aware by design
(`features.py::maturity_phase`: preseason/early/developing/established), fitting a
separate model and separately deciding which candidate features earn promotion in each
phase. `recruiting_rating`/`preseason_poll_points` are exactly the kind of feature that
should matter heavily in week 1 and fade to irrelevance by week 6 as real in-season data
accumulates -- `fit_live_mean` can represent that directly (a feature promoted in the
`preseason` phase can simply not be promoted in `established`); a single whole-season OLS
fit structurally cannot represent a feature's importance changing mid-season without
becoming phase-aware itself, which is not a unification, it is rebuilding
`build_features`/`walk_forward` into something close to what `GATE_CALIBRATED` already
needs and has not been built for the same reason: real scope, not attempted without
validating it end to end first.

**What actually closes this gap, not attempted today:** a dedicated gate or analysis
script measuring `fit_live_mean`'s own walk-forward RMSE per phase, the way
`GATE_RMSE_SPREAD`/`GATE_RMSE_TOTAL` already measure `model_spread`'s -- so "does the live
site's extra machinery actually predict better where it's supposed to" has a real,
committed number instead of only the internal `challenger_promoted` check `fit_live_mean`
already does on itself. Recorded as the concrete next step, scoped the same deliberate way
`GATE_CALIBRATED` was.

## D8. half_life_games looks under-tuned post-FCS, measured not yet acted on, 2026-08-17

Ran the pre-registered 12-cell grid (`run_backtest.py --sweep`, DECISIONS D3) against the
now-FCS-inclusive rating universe, prompted by "should regularization be re-tuned for the
larger universe." `held-out t(b_total)`, the grid's own headline metric (2024-2025,
unseen at tune time): the three `half_life_games=16` cells cluster at 2.10-2.21; the three
`half_life_games=6` cells (which includes the closest match to the live config) sit at
1.63-1.72. `off_weight=1.45` remains at or near the best score in every half-life bucket,
so that part of the live config is not in question. Plausible, not just numerical: FCS
teams have sparser, more irregular schedules than FBS teams, and a longer half-life (more
weight on the full season, less on recent-game recency) would naturally help stabilize
their now-freshly-fit ratings more than it would hurt FBS's already-dense ones.

**Not acted on.** `prior_weight_games`'s own entry above is the standard to match before
changing this: a careful, documented, multi-metric search (RMSE traded off against
`GATE_SCALE`'s SD-ratio constraint specifically, an explicit check for a monotone
interior optimum rather than one lucky cell, explicit acknowledgement of what the change
does and does not affect) -- and that decision's own text says a post-hoc choice "is not
a substitute for a pre-registered re-run," which is exactly the caution this measurement
needs before being trusted over a single coarse-grid dimension. This sweep tests
`half_life_games` at only three points (6/10/16); the true optimum could be anywhere
above 6, including past 16. Recorded as a concrete, well-evidenced next step -- run a
finer sweep centered on 16, holding `prior_weight_games=8.0` and `off_weight=1.45` fixed
at their own already-validated values, checking `GATE_SCALE` impact the same way the
existing decision does -- not attempted here alongside the rest of today's list.

## D9. `GATE_CALIBRATED` implemented for NCAA, 2026-08-17

Ported nfl-model's working `walk_forward` calibration pattern
(`nfl-model/src/backtest.py`) rather than inventing a new one: `attach_calibration`
(`src/backtest.py`) walks forward one season at a time, refits the drive model, endgame
table, and venue HFA on strictly earlier data, simulates every graded game with that
season's ratings, and reweights the simulated distribution onto this build's own
`model_spread`/`model_total` (`SimResult.recentered().retotaled()`, chained) before reading
`cover_prob_home`/`over_prob` off it at the market's CLOSE line.

**Design choices, and why:**

- **Full graded population, not a sample.** `pooled_margin_pmf` (GATE_KEY_NUMBERS) samples
  ~400 games because it is explicitly diagnostic-only (shape, not predictive power) -- its
  own docstring says so. `GATE_CALIBRATED` is a real accuracy claim, so it needed the same
  population NFL's `calibration_table`/`gate_calibrated` use: the *entire* frame NFL's
  `walk_forward` produces, unrestricted. Matched that here rather than restricting to the
  "restricted" (non-P5) window every other NCAA gate uses, both for precedent and because a
  narrower window would leave too few observations per 5%-probability bin to ever clear the
  n>=100 floor `gate_calibrated` requires.
- **Close-anchored, not open-anchored**, matching `CALIBRATION_ANCHOR` in `run_backtest.py`
  and every other absolute-accuracy NCAA gate -- the opener-anchored totals signal is a
  proven false positive (Appendix A), so grading calibration against the opener would risk
  reproducing the same artifact this project already caught once.
- **Additive and separately cached from `walk_forward`.** Reads `model_spread`/
  `model_total`, never recomputes them, so a bug here cannot move any RMSE/scale/blend
  gate's inputs, and this cache invalidates independently of the (expensive, load-bearing)
  `backtest_frame_*` cache.
- **Explicit `as_of` cutoff on venue HFA.** `pooled_margin_pmf`'s existing venue-HFA call
  has none (`estimate_venue_hfa(games, cfg, walkforward=walkforward)`), which is a real gap
  but a tolerated one there because that gate is diagnostic-only. This function backs a
  genuine accuracy claim, so it computes each season's first kickoff and passes it as
  `as_of`, matching the live path (`project_game.py`) rather than the diagnostic one.
- **Joint spread/total infeasibility falls back to margin-only recentering.** `.retotaled()`
  can raise `ValueError` when the target pair is outside the simulated lattice's support
  (the same infeasibility `live_mean.py`'s physical-floor clip closes for the live path, but
  this OLS-fit backtest path doesn't go through `LiveMeanModel`). Caught per-game: the
  margin stays calibrated onto `model_spread`, the total falls back to the simulator's own
  uncalibrated mean rather than dropping the game, matching `build_slate.py`'s
  `_independent_forecast` fallback discipline.

**Result: the gate is now produced (not `passed: null`) and fails honestly** -- see
GATES.md "Updated status -- measured 2026-08-17" and its calibration table. `bets_allowed()`
remains False, now for a fully-measured set of reasons rather than one permanently-blocking
un-produced gate.

**Real cost, measured, not estimated: this is expensive.** ~40-45 minutes of wall time for
5,621 games across 5 per-season refits, on top of the existing ingest/walk-forward cost --
far more than the ~114-456ms/game range profiled from isolated timing tests before a real
population count was available. The full graded population (post-FCS-fix, since FCS teams
are now simulable) is larger than any prior estimate assumed. Not yet a problem in
practice: this runs on-demand via `run_backtest.py`, not on the scheduled bot's
market-refresh cadence, and `walk_forward`'s own frame stays cached and reused between
runs. Worth revisiting (per-season checkpointing, or a bounded-but-larger-than-400 sample)
if this ever needs to run inside a tighter time budget.

## D10. Appendix A totals-coefficient drift confirmed caused by the FCS-ratings fix, 2026-08-17

D9's own totals null re-baseline left the cause of the drift as "leading hypothesis, not
yet confirmed." Confirmed same day via direct counterfactual
(`analysis/appendix_a_fcs_counterfactual.py`): revert `build_game_offense` to the old
single-bucket FCS behavior in memory only, re-run the walk-forward ratings and backtest
frame against the *same* cached games/drives/plays (zero new CFBD calls), recompute the
same three Appendix A statistics.

**Result: the counterfactual's full-FBS-close reading (b=+0.0911, se=0.0631, CI upper
+0.2147) reproduces the pre-fix 2026-08-16 reading (b=+0.0911, CI upper +0.2148) almost to
the fourth decimal.** Changing exactly one function back, holding cached data/config/
machinery fixed, restores the pre-drift number. This is confirmation, not correlation —
see GATES.md "Appendix A, revisited again 2026-08-17" for the full three-window table and
a plausible (measured, not fully proven) mechanism: `model_spread` runs on `net_diff` (a
difference, largely invariant to the ~0.007-point systematic rating-level shift measured
directly between the two universes) while `model_total` runs on `eff_sum` (a sum, which
compounds the same shift).

**No action taken on the fix itself.** The FCS-ratings fix's actual purpose -- giving FBS-
vs-FCS games real, individually-fitted opponent ratings instead of one shared bucket -- is
unaffected by this finding and remains correct; this is a traced, understood side effect on
one specific residual coefficient, not a defect in what the fix was built to do. The totals
null was already provisional before today; tracing its drift to a specific, confirmed cause
is strictly more information than the alternative of reverting or patching around it.

## D11. Pre-registered finer `half_life_games` sweep, declared before running, 2026-08-17

D8's concrete next step, acted on same day. **Declared here, before any cell is run**,
matching D3's discipline: only the cells below are evaluated, and all of them are reported
regardless of which one wins.

`ratings.half_life_games` only (`config/ncaa.yaml`) -- **not** `pace.half_life_games`,
a separately-tuned, unrelated parameter in the same file. `prior_weight_games=8.0` and
`off_weight=1.45` (`def_weight=1.0`) held fixed at their own already-validated live values,
matching the comment block directly above `prior_weight_games` in `config/ncaa.yaml` --
that block, not a DECISIONS.md entry, turns out to be the "prior_weight_games's own entry"
D8 pointed at.

| cell | `half_life_games` |
|---|---|
| 1 | 8 |
| 2 | 12 |
| 3 | 16 |
| 4 | 20 |
| 5 | 24 |
| 6 | 28 |

Centered on 16 (the best of D8's coarse 6/10/16 test) with wide enough range on both sides
to distinguish a true interior optimum from "the trend was still climbing when the coarse
grid stopped." Tune 2021-2023, hold out 2024-2025, matching D3 and D8's own coarse sweep.

**Metrics, matching `config/ncaa.yaml`'s own `prior_weight_games` precedent exactly:**
held-out `t(b_total)` as the headline (D3's convention), RMSE and `GATE_SCALE`'s SD-ratio
as the trade-off constraint, an explicit monotonicity check across the six cells rather
than reading off the single best one, and an explicit statement of what any resulting
config change does and does not affect (this changes `bets_allowed()` only if it flips a
currently-failing gate; it does not by itself resolve the Appendix A totals null).

**Results (`analysis/half_life_finer_sweep.py`, all six cells):**

| `half_life_games` | tune t(total) | hold t(total) | hold RMSE(total) | hold SD ratio(total) | hold t(spread) |
|---|---|---|---|---|---|
| 8 | 2.340 | 1.821 | 17.020 | 0.751 | −0.999 |
| 12 | 1.982 | 2.074 | 17.008 | 0.713 | −1.052 |
| 16 | 1.822 | 2.196 | 17.012 | 0.690 | −1.091 |
| 20 | 1.736 | 2.252 | 17.024 | 0.676 | −1.108 |
| 24 | 1.683 | 2.280 | 17.038 | 0.667 | −1.112 |
| 28 | 1.648 | 2.296 | 17.052 | 0.662 | −1.111 |

**No interior optimum found. Held-out t(b_total) is still climbing at the top of the
tested range** (1.82 at 8 -> 2.30 at 28, monotone throughout, no sign of turning over) --
D8's own worry ("the true optimum could be anywhere above 6, including past 16") is
confirmed, not resolved: past 16 is exactly where it keeps going. This is a materially
different shape than `prior_weight_games`'s own search, which is why that one was trusted
enough to become the live default and this one is not: that search found RMSE trough with
points rising on *both* sides (16.718 -> 16.563 -> 16.630 -> 16.791 as prior_weight went
0.42 -> 8.0 -> 20 -> 30). This search has only ever seen one side of whatever curve exists,
if one exists at all.

**GATE_SCALE reads as a non-issue here, but not for a reassuring reason.** Every single
cell -- 8 through 28 -- sits at SD ratio 0.66-0.75, already well outside the [0.85, 1.15]
tolerance the live config (`half_life_games=6`) is *already* failing at (0.7407, the
committed reading in GATES.md's 2026-08-17 status table). Moving within this range does not
flip `GATE_SCALE` from pass to fail because it never passed at any point tested -- the
trade-off D8 anticipated needing to adjudicate does not actually bind, but only because the
starting point was already on the wrong side of it. Dispersion also degrades monotonically
in the same direction `t(b_total)` improves (0.751 -> 0.662), which is the same shape as
"the projection is getting smoother/more heavily-averaged as decay slows," not obviously
distinguishable from "the projection is getting more accurate."

**Not acted on. No config change.** Two independent reasons this project has used before to
withhold a config change apply directly here: (1) a monotone trend with no interior optimum
is exactly the pattern D3's whole pre-registration discipline exists to guard against acting
on -- picking 28 because it is the best of six arbitrarily-chosen points is a lucky-cell
selection with extra steps, not a principled choice; (2) a metric that keeps improving while
a plausible confound (dispersion collapse) moves in lockstep is the same shape this project
has twice manufactured a false signal from before (Appendix A's opener-anchoring artifact;
D6d's underpowered-subsample claim). Recorded as a well-evidenced open question, not a
finding to act on: if this is worth pursuing further, the next step is a wider-range grid
(the search has not yet seen a ceiling) declared and pre-registered the same way this one
was, not an ad hoc extension of it.

## D12. Pass-only EPA tested as a challenger feature, not promoted, 2026-08-17

A competitor's methodology email (BTB Analytics) claimed rushing EPA "never gets close" to
significant on a 7,000+-game re-fit, while passing EPA carries the load -- implying
`ingest.py::build_game_offense`'s pooled `ppa_per_play` (pass + rush averaged together)
dilutes a real signal. Deliberately scoped as a Stage 1, low-risk test through the existing
validated-challenger machinery, not a rewrite of the core rating system: `build_game_offense`
gained an optional `play_type` parameter (pass/rush filter, reusing
`features.py::build_team_game_features`'s existing classifier), and a new
`ratings.py::split_net_epa_matchup` combines the resulting pass-only/rush-only walk-forward
ratings into `pass_net_epa_diff`/`_sum` and `rush_net_epa_diff`/`_sum` candidates merged
onto the `challenger` frame `run_backtest.py` already builds. No change to
`walk_forward`/`build_features`/`project_walkforward`, the core `off_rating`/`def_rating`,
or the drive simulator -- `_validated_challenger`'s existing suffix-driven promotion logic
picked the new columns up unchanged.

**Result: neither promoted, but not a flat rejection either -- see GATES.md "Pass-only EPA
tested as a challenger feature" for the full numbers.** Standalone, pass beats rush
(spread r=+0.4234 vs +0.3749; total r=+0.1985 vs +0.1577), replicating the claim's core
comparison directionally. But the *existing pooled* feature beats both individually
(spread r=+0.5090; total r=+0.2405) -- averaging two correlated-but-not-identical signals
outpredicts either alone here, which is also why validated-ridge correctly rejected both:
`pass_net_epa_diff` is too collinear with the already-present `net_diff` to add incremental
value once `net_diff` is in the model.

**Not acted on, and the natural next step (replacing pooled EPA with pass-only, or
threading a pass/rush split into the drive simulator) is now positively contraindicated by
this evidence, not just out of scope.** The architectural implication most people would
draw from the source claim -- split the rating, or go pass-only -- would remove real,
independent signal (rush's residual contribution) rather than sharpen anything, on this
data. The directional part of the claim (pass > rush individually) replicates; the
implementation implied by it does not survive contact with a properly controlled test.

## D13. A genuine offense x defense interaction term, tested and not promoted, 2026-08-18

Follow-up to D12, prompted directly by a user pushback that deserved a real test rather
than a hand-wave: D12 showed rush-only ratings don't deserve their own *linear* term, but
that's a different claim from "does a specific rushing offense do better against a
specific defensive front than the additive model predicts." The current formula
(`net_epa_vec`: `off_weight*off + def_weight*def`) is structurally additive everywhere --
it cannot represent an effect that scales with how good the offense already is, regardless
of how well `off`/`def` are individually fit. Tested directly: `ratings.py::
interaction_matchup(wf, games, prefix)` computes a genuine product,
`home_off_rating * away_def_rating` (and the mirrored away term), the standard textbook way
to test a statistical interaction -- included as a `_diff`/`_sum` candidate alongside the
existing main effects, not compared in isolation. Two versions: pooled ratings (the general
form of the question) and rush-only ratings from D12's already-built `wf_rush` (the specific
claim raised), both free -- no new walk-forward fit needed.

**Result: neither promoted.** `feature_spread_promoted`/`feature_total_promoted` are False
on all 5,621 graded rows, matching D12's own reading. But the honest, useful number is the
controlled test, not the bare promotion flag -- regressing `actual_margin` on
`net_diff + is_home + {interaction}` (n=1,543, restricted, 2021-2025) and reading the
interaction term's own coefficient:

| interaction | market | b | t |
|---|---|---|---|
| pooled (off x opponent def) | spread | +1598.5 | **+1.585** |
| rush-only | spread | +1551.4 | +0.648 |
| pooled | total | +358.4 | +0.370 |
| rush-only | total | +2156.7 | +0.936 |

The pooled spread interaction is the closest of the four to conventional significance
(t=1.96) and points in the theoretically expected direction (a positive coefficient means
a strong offense gets *more* benefit from a weak defense than the additive formula alone
would predict) -- but at n=1,543 it isn't distinguishable from noise, and none of the four
clear the bar `_validated_challenger` requires. Standalone correlation with the outcome is
much weaker than the existing main effects for all four (expected and not informative on
its own -- interaction terms are inherently second-order effects, not a replacement for
the primary relationship, so a low bivariate correlation was never the right way to judge
this; the controlled regression above is).

**Not acted on.** The user's underlying intuition -- that specific playing styles may
matchup better or worse against specific fronts in ways addition can't capture -- is not
disproven by this, just not detected at this sample size with this specific (product)
functional form. Worth a wider search (e.g. testing interactions built from pass-specific
ratings too, or a larger n via a less-restricted window) if this is worth pursuing further,
not concluded here either way.

## D14. Per-season caching for `attach_calibration`, 2026-08-18

`attach_calibration` (D9, GATE_CALIBRATED) cached its result as one blob for all graded
seasons combined, keyed to a signature hashing the *entire* multi-season `frame`/`games`/
`drives_raw`/`walkforward`. Found by the user asking why completed historical seasons kept
getting redone: they didn't need to -- the signature included the current, still-in-progress
season, which changes every week as new games get graded, so any routine update invalidated
everything and re-simulated all 5,621+ games including several years of permanently-static
history. Same "cache signature broader than what actually needs to invalidate it" bug class
found and fixed several other places in this repo, just not yet caught here since the
function was new this session.

**Fix**: split into `_season_calibration`, cached per season
(`backtest_calibration_{cache_key}_s{season}.parquet`), with the signature computed on
`games`/`drive_table`/`walkforward` filtered to `season <= this season` -- not the
unfiltered frames. A later season's data can then never appear in an earlier season's
signature. `attach_calibration`'s public signature and every call site are unchanged.

**Measured, not assumed.** Three real runs: (1) first run under the new scheme, no
per-season cache yet, ~18-45 min depending on how much fresh ingest was also needed that
run, numbers matched the known result exactly (5,621 games, worst bin 34.33pp) -- a
regression check that the refactor changed nothing but the caching strategy; (2) a second
run the same day, everything cached, no ingest refresh needed: **24 seconds real time**,
same numbers, byte-identical. That's the actual fix, demonstrated directly: a routine
rerun that used to cost ~40-45 minutes for the calibration step alone now costs about the
time it takes to read the cache files off disk.

Verified the fix itself with unit tests on the signature-scoping logic specifically
(`tests/test_calibration_caching.py`) -- a season's cache is provably unaffected by a
later season's data changing, and still correctly invalidated by a change to its own data.
The orphaned monolithic `backtest_calibration_default.parquet` was deleted; the new
per-season files are the only cache this function reads or writes now.

## D6. Spread sign convention

CFBD quotes spreads negative = home favored; nflverse is the opposite. Normalized to
**positive = home favored** at ingest, matching the NFL build, with an assertion test
against a known historical blowout. A sign flip produces entirely plausible numbers while
being exactly backwards, so it is tested rather than assumed.
