# PATCH 03 — DECISION: OUTCOME 3, AND THE PIVOT

Read alongside the earlier files. Where they conflict, **this file wins.**

**Do not run Diagnostic C.** Your read is correct and the decision rule resolves to
Outcome 3. §2 explains why C cannot produce evidence, §3 corrects an error of mine that
your Diagnostic A exposed, and §5 specifies the NCAA transfer.

---

## 1. THE CALL

`GATE_BLEND_INFORMATIVE` fails terminally on NFL spreads. Three independent architectures —
ridge-direct, drive simulator, Elo — produce `b` of −0.107, −0.044, and −0.018, none
distinguishable from zero, none positive, in any split. Held-out 2024–25 is negative for all
three. The joint fit returns R² = 0.002.

Stop work on NFL spreads. Per §9 of PATCH_01, this is a completed build with a correct
negative result, not a failed one.

---

## 2. WHY DIAGNOSTIC C CANNOT PRODUCE EVIDENCE

Your instinct to run it "for the record" is reasonable and I want to close it off properly
rather than wave at it.

The grid in PATCH_02 §4 is 5 × 4 × 4 × 3 × 3 = **720 cells**. Simulated under a true null
with realistic correlation between neighbouring cells:

- expected cells showing |t| > 2 by chance: **36**
- P(at least one cell shows |t| > 2): **52.5%**
- Bonferroni-corrected threshold for that grid: **|t| > 3.99**

So a coin-flip chance of surfacing something that looks like a discovery and is not, and a
real bar so high that nothing in your measured range approaches it. **Running C after a null
result cannot confirm the null and can only manufacture a false positive.** PATCH_02 §5 said
not to tune further after a null; I am holding to that rather than being argued out of it by
a desire for completeness.

If you want closure on the record, the only defensible version is a **pre-registered
three-cell check** — `half_life_games` at 20, `off_weight:def_weight` at 1.2:1.0, and
`lambda ×4` — declared before running, evaluated on held-out 2024–25 only, at |t| > 2.
Three tests, not 720. My recommendation is to skip even that; the prior is too low to
justify the session.

---

## 3. WHERE I WAS WRONG, AND YOUR DIAGNOSTIC A IS WHAT SHOWED IT

**PATCH_02 §2 claimed that cutting projection noise ~50% would drive t toward 2. That was
wrong, and your data refuted it before I did.**

Your direct projection is *more* shrunk than the simulator (SD ratio 0.808 vs 1.269) and its
`b` is *more* negative (−0.107 vs −0.044). That is the opposite of what my noise theory
predicted, and you flagged the contradiction in your own report.

The mechanism, stated correctly: shrinking a projection toward the market is a **scalar
multiple of its disagreement term**. `b` scales by `1/c` and `se` scales by `1/c`, so **`t`
is exactly invariant** — the same algebra you identified when you corrected my rescaling
instruction two patches ago. I applied the rescaling insight to the blend and then failed to
apply it to my own noise argument. Shrinkage cannot move significance. Only changes that
**re-rank which games the model disagrees with the market about** can move `t`.

This also disposes of the one argument for C that had merit. You were right that C varies
`half_life`, `prior_weight`, `offseason_regression` and the off/def ratio, which do re-rank
rather than merely rescale — so C is not purely a shrinkage search. But the multiple-
comparisons arithmetic in §2 kills it regardless of that.

Two further things your report established that I want on the record:

- **The simulator is the noise source, definitively.** All 5.02 points of excess dispersion
  originate there; the ratings carry none. RMSE 13.529 direct vs 13.882 through the sim.
  This is a real, transferable finding and it validates the L1/L2 split.
- **Ensembling was never going to rescue this.** At a disagreement correlation of 0.685, two
  components buy only 1.09× the effective independence of one. Two near-zero, highly
  correlated signals combine to zero.

---

## 4. THE CAVEAT I OWE YOU — YOU GRADED AGAINST THE HARDEST POSSIBLE BENCHMARK

There is an inconsistency in my original spec that matters for how you read this result.

`00_START_HERE.md` says the achievable edge is against the **opening** line, before the
market has processed it. Then `NFL_PLAYBOOK.md` §8 grades everything against
`schedules.spread_line`, which is the **closing** line. You did exactly what the spec said,
and the spec asked for the wrong test.

Beating the closing line is the hardest benchmark in American sports betting — it is the
aggregate of every public model plus professional money. Your market RMSE of 12.692 is a
very sharp number. Failing to beat it is the expected outcome, not a surprising one.

The test the model was designed to pass — beat the opener, then measure CLV against the
close — was never run, because free historical NFL opening lines do not exist. nflverse
carries closing only.

**CollegeFootballData carries `spread_open` and `over_under_open` on the free tier.** So the
college build can run the test the NFL build could not. That is a concrete, material
difference between the two markets, not a consolation prize.

---

## 5. THE NCAA TRANSFER — WHAT CHANGES

Build NCAA next. Everything you have written transfers: ridge machinery, drive simulator,
calibration design, residual blend, gate harness, budget wrapper, and every bug you found.

Apply these seven changes from day one rather than rediscovering them.

1. **Simulator out of the mean path.** Build L1/L2 split from the start. The mean comes from
   a direct linear projection off the ridge ratings. The simulator produces totals shape, the
   margin/total joint, and factor sensitivity — never the mean.
2. **Free intercept in every residual fit**, with `GATE_UNBIASED` (|a| < 0.5) from day one.
   This is the fix you already landed in `src/backtest.py`; port it.
3. **Grade against OPENING lines.** `market_spread` = `spread_open` from CFBD. Then measure
   CLV separately: did the close move toward the model's side. Report both. This is the
   headline change and the reason the pivot is worth making.
4. **Totals before spreads.** College pace varies far more than NFL pace, the ridge produces
   a genuine pace term, and totals on non-marquee games are the least-watched number on the
   board. Run the totals evaluation first and only build spreads if totals show signal.
5. **Restrict the bet universe before measuring.** Group of Five and cross-tier games, off
   marquee windows. Partition the blend fit per PATCH_01 §8c and let the data say which slice
   carries information. Do not fit one blend across all 800 games.
6. **Pre-register the hyperparameter grid** at ≤ 12 cells, declared before the first run,
   evaluated on held-out seasons. Never 720.
7. **Tune on held-out `t(b_model)`, not RMSE** — but understand from §3 that this only helps
   where hyperparameters re-rank games. Scale changes are inert.

**Kill criterion, declared now:** if held-out `t(b_model)` fails to clear 2.0 on college
totals, on the restricted universe, against opening lines — stop. Do not sweep further. That
is the last well-posed test available on free data, and a null there is a real answer about
what is reachable from public inputs.

---

## 6. PLAYBOOK PATCHES TO LAND BEFORE YOU MOVE

Write these into both playbook files so the NCAA build inherits them:

- Free intercept in the residual regression; `GATE_UNBIASED` added to both gate sets.
- Component-level rescaling (about its own mean), with a note that rescaling the
  *disagreement* leaves `t` exactly invariant.
- §5c tuning objective changed from margin RMSE to held-out `t(b_model)`, with the caveat
  from §3 that scale-only changes cannot move it.
- NFL §8: `market_spread` grading against opening lines where available; NCAA §8a:
  `spread_open` / `over_under_open` as the primary, close retained for CLV.
- L1/L2 split promoted from PATCH_01 into the main architecture section of both files.
- A results appendix in `NFL_PLAYBOOK.md` recording this null: the three architectures,
  their `b` and `t`, the held-out splits, and the finding that the simulator contributes all
  excess dispersion. Future-you will otherwise rebuild this.

Then commit, and start the NCAA repo clean.
