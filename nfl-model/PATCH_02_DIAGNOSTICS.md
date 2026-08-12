# PATCH 02 — DIAGNOSTIC BLOCK

Read alongside `NFL_PLAYBOOK.md` and `PATCH_01_ARCHITECTURE.md`. Where they conflict,
**this file wins.**

**Answer to your question: do not proceed to step 3 as written.** Steps 3 and 5 cannot
produce what you need, and I can show that formally. Run the diagnostic block in §4
instead. Your work in steps 1–2 was correct throughout, including the three places you
caught errors in my spec.

---

## 1. THINGS I GOT WRONG THAT YOU CAUGHT

**I predicted model SD would fall after Bug A. It didn't, and your explanation is right.**
An HFA double-count is a location error: it adds a constant to every game, moving the mean
and leaving dispersion untouched. Model mean went 2.773 → 1.561 against an actual 1.607,
which is exactly the fix working. I should not have predicted a dispersion change from a
location bug. Your `before/after` table is the correct read.

**The residual regression had no intercept. That is a spec bug and your catch is correct.**
With the market coefficient constrained to 1.0 and no intercept, any constant level bias in
a component loads onto its disagreement slope. Your totals result is the textbook
demonstration: +1.43 points of systematic over-projection presented as `b = +0.060`, and
de-biasing flipped it to −0.050.

**Fix, in both playbooks:** the residual regression takes a free intercept.

```
actual_margin - market_spread = a + Σ_k b_k * (component_k - market_spread)
```

Report `a` alongside every `b_k`. A significant `a` means a component is systematically
biased and must be de-biased before its `b` means anything. Add:

```
[NEW] GATE_UNBIASED   |a| < 0.5 points for every component, spreads and totals
```

**My rescaling instruction was ambiguous and invited the error you initially made.**
Rescaling the *disagreement* is a scalar multiple: `b` scales by `1/c`, `se` scales by
`1/c`, and `t` is exactly invariant. Verified — 1.0×, 0.5×, and 2.0× all return t = +2.990
on identical data. Only rescaling the *component* changes which part of the disagreement
survives. You caught this and corrected it yourself; the spec should have said "rescale the
component about its own mean," and now does.

**§5c's tuning objective is wrong, and this one is load-bearing** — see §3.

---

## 2. THE NUMBER THAT REFRAMES EVERYTHING

Your SD ratio is not a scale defect. Decompose it. The market's spread SD of 6.42 is
approximately the true game-to-game signal, since the market is close to unbiased. Then:

```
model noise = sqrt(8.148² − 6.42²) = 5.02 points
```

**Your spread projection carries about 5 points of noise. Your bet threshold is 1.5 to 2.5
points.** The noise is roughly twice the smallest edge you are hunting for.

That is the entire finding. `b = 0` is not a scale bug, a shape bug, a calibration bug, or a
blend bug. It is a **precision** bug. A projection with 5 points of error cannot resolve
2-point disagreements, and no downstream transformation changes that.

Two corollaries worth stating plainly:

- **Your sample is fine.** With `se(b) ≈ 0.060` you could have detected `b ≥ 0.12` at t = 2.
  You measured −0.066. This is not a power problem — the effect really is zero.
- **The target is quantifiable.** Reaching t ≈ 2 needs roughly a **50% cut in projection
  noise**, which shows up as an SD ratio near 1.05–1.10 rather than 1.27. That is the
  number to optimize against, and it is a much clearer target than "fix scale."

---

## 3. WHY STEPS 3 AND 5 CANNOT WORK, AND WHY §5c IS THE REAL CULPRIT

**Steps 3 and 5 are monotone transformations of the same projection.** Fitting a `net_epa`
gain rescales it; quantile-mapping reshapes it. Verified on controlled data: a 0.79 gain and
a quantile reshape move `b` from +0.189 to +0.239 and +0.230 respectively, while `t` stays
pinned at +2.99 in all three cases. **They move the coefficient and never its significance.**

They will make `GATE_KEY_NUMBERS` and `GATE_CALIBRATED` pass. They will not create edge.
Both remain worth building — later, once there is a signal worth shaping.

**The deeper problem is what §5c told you to optimize.** It says tune λ by out-of-sample
margin RMSE. That objective is wrong for a betting model, and it actively works against you:
actual margin is roughly 13 points of mostly irreducible noise around the true spread, so
minimizing RMSE against it mostly rewards getting close to the market's number. Optimize
that hard enough and you converge on an expensive market clone with extra variance — which
is a fair description of what you have.

**Tune on incremental information over the market instead**, because that is the quantity
that is actually edge:

```
objective = t-statistic of b_model in the residual regression, on held-out seasons
```

This is a different optimization surface and it will select different hyperparameters —
generally much heavier shrinkage, because shrinkage cuts noise and noise is what is killing
you.

**Overfitting guard, mandatory.** Tuning directly on a t-statistic is easy to overfit with
1,535 games. Three requirements:

1. Tune on 2019–2023. Validate on 2024–2025. Report both.
2. The held-out t must clear 2.0 independently. In-sample t is not evidence.
3. **The grid must be smooth.** Plot t across the hyperparameter grid. A broad plateau is a
   real effect; an isolated spike is noise, and you must not select it. If the surface is
   spiky everywhere, there is no signal and no amount of searching will find one.

---

## 4. THE DIAGNOSTIC BLOCK — RUN THESE THREE, IN ORDER

All three use data you already have. This is roughly a day, not a week.

### Diagnostic A — Is the ratings layer or the simulator losing the signal?

Compute a margin projection **directly from the ridge ratings**, linearly, bypassing the
simulator entirely:

```
direct_spread = gain * [(off_A - def_B) - (off_B - def_A)] + hfa
```

with `gain` fit by least squares on training seasons. Score it exactly as you scored the sim
output: RMSE, SD ratio, `b`, `t`, and the intercept `a`.

This partitions the blame, and the two outcomes point in opposite directions:

- **Direct is meaningfully better than sim** → the drive simulator is a noise source in the
  mean path. Adopt the L1/L2 split from PATCH_01 immediately: the sim stops producing the
  mean and is used only for shape, the joint, and factor sensitivity.
- **Direct ≈ sim** → the ratings themselves carry ~5 points of noise, and the simulator is
  faithfully transmitting it. All remaining work belongs in `ratings.py`.

Run this first. It determines where everything else goes.

### Diagnostic B — What is `b_elo`? You never measured it.

This is the missing number in your report. Elo sits at 13.208 RMSE against the market's
12.692 — far closer than the ridge's 13.882. Fit the residual regression for Elo alone,
with intercept, after de-biasing and SD-matching.

- **`b_elo` significant** → the ensemble path in PATCH_01 §2 is alive and worth building,
  even if the ridge contributes nothing. Elo carries decorrelated information the market has
  not fully absorbed.
- **`b_elo` ≈ 0 too** → neither of your two independent architectures beats this market. That
  is a strong and genuinely informative result. Go to §6.

Cheap, and it changes the recommendation either way.

### Diagnostic C — Re-tune the ratings against the right objective

Sweep, scoring on held-out `t(b_model)` per §3, not RMSE:

| Parameter | Current | Grid |
|---|---|---|
| `half_life_games` | 9 | 9, 14, 20, 28, 40 |
| `prior_weight_games` | 6 | 6, 10, 16, 24 |
| `lambda_off` / `lambda_def` | 220 / 260 | ×1, ×2, ×4, ×8 |
| `offseason_regression` | 0.28 | 0.28, 0.45, 0.60 |
| `off_weight` : `def_weight` | 1.6 : 1.0 | 1.2, 1.6, 2.0 : 1.0 |

Expect the optimum to sit at **much heavier shrinkage than the current settings** — longer
half-life, stronger prior, larger λ. Every one of those reduces variance at some cost in
bias, and variance is what is killing this model. Report the SD ratio at each grid point
alongside t; you are looking for the region where the ratio approaches 1.05–1.10.

Report the grid as a table and a plot. Do not silently select a winner.

---

## 5. DECISION RULE — DO NOT SKIP THIS

After the diagnostic block, one of three outcomes. Follow it mechanically.

**Outcome 1 — held-out `t(b_model) > 2.0` for any component.** Signal exists. Proceed to
PATCH_01 steps 3–5 as written: fix scale, build the ensemble, build the calibration layer,
then the report layer. Everything downstream now has something real to shape.

**Outcome 2 — `b_elo` clears but ridge does not.** Build the ensemble with Elo carrying the
weight and ridge at zero. Keep the ridge and the simulator for **totals, shape, and factor
sensitivity**, which is what they are actually good at. State plainly in the report that the
spread edge is Elo's.

**Outcome 3 — nothing clears on held-out data.** Stop work on NFL spreads. This is a real
result, not a failure: the NFL closing line is the sharpest number in American sports, and a
from-scratch model on public data failing to beat it is the *expected* outcome, not the
surprising one. Your market RMSE of 12.692 is a very good number to be measured against.

In Outcome 3, do not tune further. Continued searching against a fixed backtest after a null
result is how you manufacture a false positive.

**Time-box the whole block to one working session.** If the sweep in Diagnostic C is still
running after that, you are in Outcome 3 whether the grid says so or not.

---

## 6. IF IT IS OUTCOME 3 — THE PIVOT IS ALREADY WRITTEN

Worth saying now so it does not read as consolation later: **NFL was always the harder
market, and I sequenced it first for the wrong reason.**

I put NFL first because it is the easier *build* — 32 teams, cleaner data, faster iteration.
That was right about engineering and wrong about edge. The NFL closing line absorbs enormous
professional attention. College football does not: 130+ teams across a sprawling Saturday
slate, thin coverage on Group of Five games, and totals that move on pace information the
market prices lazily. That is where a model like this has room, and it is why
`NCAA_PLAYBOOK.md` sets `model_weight_cap` at 0.70 against the NFL's 0.60.

None of the NFL work is wasted in that pivot. The ridge machinery, the drive simulator, the
calibration layer, the residual blend, the gate harness, and every bug you have found
transfer directly. The NCAA build is the same skeleton pointed at a softer market.

If Diagnostic B comes back null, build NCAA next rather than tuning NFL further.

---

## 7. UPDATED GATE SET

```
GATE_NO_LOOKAHEAD        unchanged
GATE_UNBIASED            NEW      |intercept a| < 0.5 pts, every component, both markets
GATE_SCALE               REVISED  target SD ratio 1.05-1.10, not merely within 15%
GATE_ENSEMBLE_DOMINATES  as PATCH_01
GATE_BLEND_INFORMATIVE   REVISED  held-out t > 2.0 required; in-sample t is not evidence
GATE_KEY_NUMBERS         deferred until a signal exists to shape
GATE_SIM_REALISM         deferred likewise
GATE_CALIBRATED          deferred likewise
GATE_SPEED               unchanged
```

Report Diagnostics A and B before starting C. A may make C unnecessary.
