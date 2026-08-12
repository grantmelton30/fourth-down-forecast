# PATCH 01 — ARCHITECTURE RESTRUCTURE

Read this alongside `NFL_PLAYBOOK.md`. Where the two conflict, **this file wins.**

Your gate report was correct on every substantive point, including the two findings that
invalidate parts of the original spec. Do not revisit those diagnostics. This patch tells
you what to change and in what order.

---

## 0. TWO SPEC BUGS WERE MINE — FIX THEM IN THE SPEC, NOT JUST THE CODE

You found three bugs. Two were errors in `NFL_PLAYBOOK.md` itself, and they will recur if
the text stays as written.

**Bug A — double-counted HFA.** §6b puts `is_home_offense` in the drive multinomial while
§6c/§13 also apply a context HFA of 1.7 points. Both are live, producing your ≈4.4-point
home edge. **The multinomial's `is_home_offense` term is the correct one to keep** — it is
estimated from data at the drive level. Delete the additive context HFA for the *league
baseline*, and keep `context.hfa_*` only as a **per-venue deviation from the multinomial's
fitted league HFA**, centered on zero. Assert in a test that mean per-venue HFA deviation
across all 32 stadiums is within 0.1 of zero.

**Bug B — phantom home possession.** §6c says "with probability 0.5 add one drive to the
home team." That hands the home side +0.5 expected drives *every game*. Correct behavior:
draw the odd possession, then assign it to home or away with probability 0.5 each.

```python
extra = rng.random(n_sims) < 0.5          # does an odd drive exist at all
to_home = rng.random(n_sims) < 0.5        # who gets it
home_drives = d + (extra & to_home)
away_drives = d + (extra & ~to_home)
```

Bug C (drive start read off the kickoff row) was your own implementation; your fix is
right. Add a test asserting mean drive start is near own-27, not midfield.

Patch both bugs into `NFL_PLAYBOOK.md` and note them in `DECISIONS.md`.

---

## 1. THE CORE PROBLEM, NAMED PRECISELY

The simulator was being asked to do two incompatible jobs at once:

- **(a) Predict the mean margin.** This is a regression problem. Elo, which optimizes on
  margins directly, does it better than EPA-ridge mapped through a drive sim. Your
  numbers say so: 13.21 vs 13.88, and 13.48 even after optimal rescaling.
- **(b) Produce the distribution shape around that mean.** Key numbers, totals, the
  margin/total joint. No regression gives you this. This is what a simulator is *for*.

The original spec welded them together, so failing at (a) blocked (b). Separate them.

**The simulator is not being demoted.** It is being pointed at the job it is uniquely good
at, and it gains a new first-class output (§6) that the welded design could not produce.

---

## 2. NEW ARCHITECTURE — THREE LAYERS

### L1 — Mean layer: an ensemble, not a single model

Predicts `E[margin]` and `E[total]`. Components:

- `ridge_spread` — the existing EPA-ridge projection
- `elo_spread` — your existing Elo baseline, **promoted from baseline to component**
- `nfelo_spread` — when available; zero-weight otherwise
- `market_spread` — always included

**Rescale every component to market SD before it enters.** This is not optional and it is
upstream of everything else — see §4.

Fit by non-negative least squares on walk-forward output, in residual form with the market
coefficient constrained to 1.0:

```
actual_margin - market = Σ_k b_k * (component_k - market)     s.t. b_k >= 0
```

Fit **spreads and totals separately**. Elo cannot produce a total, so the totals ensemble
is ridge + market only.

### L2 — Simulator: unchanged in spirit, repurposed in role

Keep everything you built, including the final-drive score-state conditioning and
overtime. That work is correct and it makes the sim internally honest. Its jobs are now:

1. The **total** distribution — pace × efficiency is genuinely the right mechanism here,
   and nothing else you have can produce it.
2. The **margin/total joint dependence** — the copula, which L3 preserves.
3. **Factor sensitivities** (§6) — the deliverable the original spec never asked for.
4. The **ordering** of outcomes that L3 maps onto the empirical support.

Stop trying to make the sim reproduce key numbers through mechanics. §3 explains why that
is unreachable and what replaces it.

### L3 — Calibration layer: rank-preserving quantile map

New module, `src/calibrate.py`. This is the piece that makes `GATE_KEY_NUMBERS` reachable.

```python
def calibrate_margins(sim_margins: np.ndarray, blended_spread: float,
                      hist_pmf: ConditionalPMF) -> np.ndarray:
    """Map simulated draws onto the empirical conditional margin distribution,
    preserving rank order (and therefore the sim's game-specific information)."""
    v, cdf = hist_pmf.support_and_cdf(blended_spread)
    q = (np.argsort(np.argsort(sim_margins)) + 0.5) / len(sim_margins)
    return v[np.searchsorted(cdf, q).clip(0, len(v) - 1)]
```

`ConditionalPMF` is built from historical games: bucket by spread in 0.5-point bins,
pooling ±1.5 points to keep bin counts healthy, and store the empirical PMF of integer
margins within each bucket. **Bucket by the blended spread from L1, not the model spread.**

Constraints:

- Build the PMF from games **strictly before `as_of`**. Shape is stable, so a rolling
  10-season window is fine and keeps sample sizes large. It is still a lookahead violation
  if you use future games — the existing assertion must cover this module too.
- Do the same for totals, bucketed by blended total.
- **Map margins and totals jointly** to preserve the copula: rank the sim's `(margin,
  total)` pairs, map each marginal separately, and keep the pairing intact. Do not
  independently resample.

Verified behavior: the map hits every key number at ~0.00pp gap, preserves the sim's mean
to within 0.1 points, and retains Spearman correlation of 0.998 with the raw sim ordering.
Game-specific information survives; only the shape is corrected.

---

## 3. YOUR QUESTION 1 — PUSH ON THE ENDGAME MODEL?

**No. Stop where you are.**

Your diagnostic is right and it generalizes further than you stated: an independent-
possession model *cannot* produce key numbers at any level of refinement, because the
spikes are generated by teams optimizing against the scoreboard, which is a property of
the joint state, not of the scoring process. You closed 6.16% → 10.19% with final-drive
conditioning, which is real progress and worth keeping. The remaining 4.35pp requires a
full clock/field-position/timeout endgame model — a research project with a long tail and
no guarantee of landing inside 2pp.

L3 achieves the same result by construction, in roughly 40 lines, and is verifiable.

Keep the final-drive conditioning you built. It improves the sim's internal realism, which
matters for the factor-sensitivity work in §6 even though L3 now handles the shape.

---

## 4. YOUR QUESTION 2 — SHOULD GATE_BEATS_ELO STAY A HARD BLOCK?

**Keep it hard. Change what it gates.** Do not lower the bar — absorb the competitor.

The gate was wrong in framing, not in strictness. It treated Elo as a rival to defeat when
Elo is a free, decorrelated signal you should be consuming. A component that loses
head-to-head can still add real value in an ensemble: with plausible NFL noise levels, a
worse-RMSE ridge earns roughly a +0.27 weight alongside Elo and the pair beats both
individually. That is the normal case, not an edge case.

Replace it:

```
[REMOVED] GATE_BEATS_ELO      model RMSE < Elo RMSE
[NEW]     GATE_ENSEMBLE_DOMINATES
          L1 ensemble RMSE < min(RMSE of every single component, including market)
```

This is strictly harder to game and always achievable if any component carries independent
signal. It stays a hard block.

**But first, fix the scale.** Your model SD is 8.17 against the market's 6.42. That is a
27% overdispersion, and it contaminates the `b_model` read directly: an overdispersed
component's disagreement with the market is dominated by scale noise, which suppresses the
t-statistic badly. Under a controlled test, the same underlying signal produced t = +3.47
raw and t = +5.73 after rescaling to market SD.

So `b_model = −0.041, t = −0.68` is **not yet a clean verdict on the model.** Rescale, then
refit, then judge.

New gate, upstream of the blend:

```
[NEW] GATE_SCALE   |sd(component) / sd(market) - 1| < 0.15 for every L1 component
```

Where the overdispersion comes from, in likely order: the multinomial's `net_epa` link was
fit using in-sample ratings, so it learned a sensitivity that overshoots on noisier
out-of-sample ratings; the double-counted HFA (Bug A) inflated spread on every game; and λ
was tuned on ridge margin RMSE without accounting for the amplification the drive sim
applies downstream. Fit a single scalar gain on `net_epa` by walk-forward minimization of
margin RMSE *after* the sim, and let it shrink below 1.

---

## 5. THE THING YOU DID NOT TEST, AND IT MAY BE THE WHOLE ANSWER

**You reported spread RMSE only. Totals were never separately evaluated.**

This matters more than anything else in this patch. Elo cannot produce a total at all —
it has no offense/defense split and no pace term. Your architecture has both. EPA-ridge
plus a pace model plus a drive simulator is a *natural* totals engine and a somewhat
unnatural spread engine.

It is entirely plausible that this model has zero edge on spreads and real edge on totals.
That would not be a disappointing outcome; it would be the model telling you which market
it belongs in. Totals are on the same app as spreads.

**Run this before anything else in §7.** It is one backtest pass over data you already
have, and it determines whether the rest of the work is worth doing.

Report, separately for spreads and totals: RMSE vs market, `b_model` with t-stat after
rescaling, filtered ATS/OU record with bootstrap CI, and calibration.

Amend the blend gate accordingly:

```
[REVISED] GATE_BLEND_INFORMATIVE
          Evaluated independently for spreads and totals.
          PASS if EITHER market clears |t| > 2.0 after rescaling.
          Bets emit ONLY in the market(s) that pass.
          If spreads fail and totals pass: build the totals bet sheet, and state
          plainly in the report that the model has no demonstrated spread edge.
```

---

## 6. NEW DELIVERABLE — FACTOR SENSITIVITY

This is the output the simulator is uniquely positioned to produce, and the reason to keep
it rather than collapse to a regression.

Add `src/sensitivity.py` and a **`Factors` sheet** to the workbook. For each game, re-run
the sim under perturbations and report the effect on the projected spread and total:

| Factor | Perturbation | Report |
|---|---|---|
| QB | starter → backup (nfeloqb value delta) | Δ spread, Δ total |
| Pace | each team ±1.5 drives | Δ total |
| Offensive efficiency | each team ±0.05 EPA/play | Δ spread, Δ total |
| Defensive efficiency | each team ±0.05 EPA/play | Δ spread, Δ total |
| Wind | 0 / 10 / 20 / 30 mph | Δ total |
| Temperature | 20°F / 50°F / 75°F | Δ total |
| Rest | ±3 days differential | Δ spread |
| HFA | neutral site | Δ spread |

Two reasons this earns its place. First, it is the honest interface for information the
model does not have — you see a late scratch, you read the row, you know what it is worth
in points instead of guessing. Second, it is a standing sanity check: if wind at 30mph
moves the total by 0.3 points, or a backup QB moves the spread by 12, the sim is broken and
you will see it immediately.

Report these as **points**, not as percentages or EPA units.

---

## 7. ORDER OF WORK

Strictly sequential. Do not skip ahead — each step changes the inputs to the next.

1. **Fix Bugs A and B** (§0) and patch the playbook text. Re-run the backtest. Report the
   new model SD; expect it to fall.
2. **Run the totals evaluation** (§5). One pass. Report before continuing.
3. **Fix scale** — fit the `net_epa` gain, add `GATE_SCALE`.
4. **Rebuild L1 as an ensemble** (§2), NNLS in residual form, spreads and totals separately.
   Replace `GATE_BEATS_ELO` with `GATE_ENSEMBLE_DOMINATES`.
5. **Build L3** (`src/calibrate.py`). `GATE_KEY_NUMBERS` now applies to calibrated output.
   Add `GATE_SIM_REALISM` on the *raw* sim at a looser 5pp tolerance, so the simulator is
   still held to a standard and cannot silently rot behind the calibration layer.
6. **Re-run all gates.** Report the full table.
7. **Only then** build `report.py`, `sensitivity.py`, `app.py`, `run_week.py`.

Stop and report after steps 2 and 6.

---

## 8. FINAL GATE SET

```
GATE_NO_LOOKAHEAD          unchanged — must cover calibrate.py
GATE_SCALE                 NEW    |sd(component)/sd(market) - 1| < 0.15, all components
GATE_ENSEMBLE_DOMINATES    REPLACES GATE_BEATS_ELO — ensemble beats every component
GATE_BLEND_INFORMATIVE     REVISED — spreads and totals judged separately, either passes
GATE_KEY_NUMBERS           REVISED — applies to calibrated output, 2pp
GATE_SIM_REALISM           NEW    — raw sim vs historical, 5pp
GATE_CALIBRATED            unchanged — applies post-calibration
GATE_SPEED                 unchanged
GATE_VS_NFELO              unchanged — informational
```

---

## 9. THE OUTCOME THAT IS STILL ON THE TABLE

After all of this, it remains possible that no component carries information beyond the
market on spreads. If so, the correct output is a spread bet sheet with zero rows and a
report that says so — and, if §5 comes back positive, a totals sheet that has rows.

That is not a failed build. A model that reliably declines to bet a market it has no edge
in is doing the single most valuable thing a betting model can do. The failure mode to
fear is the one that always finds bets.
