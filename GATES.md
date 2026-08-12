# GATE REGISTRY

Authoritative record of every gate named in either playbook: whether it exists in code,
where, what it asserts, and whether it currently passes.

Created because **"all seven gates pass" was reported for a gate set missing five required
gates.** Any future claim about gate status reads from this file.

Status as of 2026-08-04, with corrections applied 2026-08-07 (marked inline). `NOT BUILT`
gates are recorded, not implemented — building them is out of scope for the current phase.

**STALENESS WARNING.** The pass/fail readings below were measured on 2026-08-04 and predate
the shared simulator, the measured weather constants, the NFL injury burden and the NFL
lambda retune to 750/550. Treat them as a record of what was asserted on that date, not as
current readings. The NCAA rows are the least trustworthy: those numbers come from a
backtest frame dated Aug 5 that has not been rebuilt since, for the reasons in
`NEXT_SESSION.md`. Re-measure before quoting any of this.

---

## NCAA — `ncaa-model`

Playbook names **12** gates. **8 implemented, 5 not built.** One implemented gate
(`GATE_GARBAGE_FILTER`) is not named in the playbook.

| gate | implemented | file:line | asserts | passes |
|---|---|---|---|---|
| `GATE_OPENER_COVERAGE` | yes | `src/backtest.py:225` | ≥70% of restricted games carry both openers; mean \|open−close\| ≥ 0.5 | **PASS** (99.7%, 1.40) |
| `GATE_NO_LOOKAHEAD` | yes | `src/backtest.py:242` | On a 200-row sample, no rating dated after its game's kickoff | **PASS** (0 violations) |
| `GATE_GARBAGE_FILTER` | yes | `src/backtest.py:260` | Garbage-time play drop share within 5–30% | **PASS** (9.6%) |
| `GATE_UNBIASED` | yes | `src/backtest.py:269` | \|intercept a\| < 0.5, full sample | **PASS** (0.449) |
| `GATE_UNBIASED_BY_WEEK` | yes | `src/backtest.py:280` | \|a\| < 0.5 in every week bucket with n ≥ 40 | **FAIL** (worst 1.853, spread wk4) |
| `GATE_SCALE` | yes | `src/backtest.py:328` | \|SD ratio − 1\| < 0.15 vs market | **PASS** (0.022 deviation) |
| `GATE_BLEND_INFORMATIVE` | yes | `src/backtest.py:341` | best \|t(b)\| > 2.0 | **PASS as measured — INVALID.** Measured on an opener-anchored, opener-contaminated column. Close-anchored it reads t = +0.89. See Appendix A. |
| `GATE_API_BUDGET` | yes | `src/backtest.py:354` | Cold build under 250 CFBD calls | **PASS** (120) |
| `GATE_RMSE` | yes (Phase 0D) | `src/backtest.py::rmse_gate` | Model RMSE / market RMSE ≤ 1.0, close-anchored | **FAIL** (totals 1.035, spreads 1.104) |
| `GATE_CLV_CONTROL` | yes (Phase 0C) | `analysis/clv_control.py` | Zero-information projection earns CLV 0.50 ± 0.03 | **FAIL** (0.5846 raw, 0.5607 side-adjusted, 200 seeds) — **CLV deleted as a promotion metric** |
| `GATE_BEATS_SRS` | **NOT BUILT** | — | Model spread RMSE beats an SRS baseline | n/a — **no SRS baseline exists** (`src/srs.py` never written) |
| `GATE_KEY_NUMBERS` | **NOT BUILT** | — | Simulated margin PMF matches historical at 3/7/10/14/17/21 and the >28 tail | still unbuilt, but **the stated reason is now false (2026-08-07)**: NCAA adopted `shared/sim_core.py` with its own measured constants, so a simulator does exist and this gate is buildable. Early validation: 419 pooled 2024 games, P(\|margin\|=3) 9.26% simulated vs 8.59% actual. |
| `GATE_COMPETITIVE_WITH_SPPLUS` | **NOT BUILT** | — | Model spread RMSE within 0.5 of SP+ | n/a — SP+ never ingested |
| `GATE_CALIBRATED` | **NOT BUILT** | — | No probability bucket with 100+ obs off by >6 points | n/a |
| `GATE_SPEED` | **NOT BUILT** | — | Weekly refresh under 240s on cached data | n/a |

**Accurate NCAA statement:** of 12 required gates, 8 are implemented; of those, 7 pass and 1
fails (`GATE_UNBIASED_BY_WEEK`). `GATE_BLEND_INFORMATIVE` passes only as measured against a
contaminated column and is superseded by the Appendix A null. Five required gates have
never been built.

---

## NFL — `nfl-model`

Playbook names **9** gates. **All 9 now implemented** (2026-08-05). 4 pass, 5 fail.

| gate | implemented | asserts | passes |
|---|---|---|---|
| `GATE_KEY_NUMBERS` | yes | Pooled simulated margin PMF within 2pp of historical at 3/7/6/10/14/4 | **FAIL** (worst 2.72pp at \|3\|; was 4.33pp) |
| `GATE_NO_LOOKAHEAD` | yes | 200-row sample, no game used at or after the predicted kickoff | **PASS** |
| `GATE_BEATS_ELO` | yes | Model spread RMSE < Elo baseline RMSE | **FAIL** (13.763 vs 13.208; was 13.877) |
| `GATE_VS_NFELO` | yes | Informational comparison vs nfelo | **PASS — vacuously**, nfelo series unavailable so nothing is compared |
| `GATE_BLEND_INFORMATIVE` | yes | \|t\| > 2.0 on b_model or b_nfelo | **FAIL** (t = −0.67) |
| `GATE_CALIBRATED` | yes | No 100+ obs bucket off by >6pp | **FAIL** (14.81pp; was 20.52pp) |
| `GATE_UNBIASED` | yes (new) | \|bias\| < 0.5 pts per market | **PASS** (spread −0.149, total −0.337) |
| `GATE_SCALE` | yes (new) | SD ratio vs market within [0.85, 1.15] | **FAIL** (spread 1.186, total 0.592) |
| `GATE_SPEED` | yes | Weekly refresh < 180s | **PASS** (3.7s) |

**REBUILT 2026-08-07** after `projection.spread_source: linear` landed (the L1/L2 split; see
`config/nfl.yaml`). Frame `backtest_frame_ad1e7bdee2.parquet`, 1,535 games — 1,119 linear,
416 simulator-fallback in 2019–20 before the fit has enough history. **No gate changed
state**, but the spread numbers moved substantially:

| gate | was | now | direction |
|---|---|---|---|
| `GATE_BEATS_ELO` | 13.763 vs 13.208 | **13.431** vs 13.208 | better, still FAIL |
| `GATE_SCALE` | spread 1.186 / total 0.592 | **spread 0.983** / total 0.707 | spread now inside tolerance; gate still FAIL on totals alone |
| `GATE_CALIBRATED` | 14.81pp | **13.87pp** | better, still FAIL |
| `GATE_UNBIASED` | −0.149 / −0.337 | worst 0.463 (total) | still PASS, less headroom |
| `GATE_KEY_NUMBERS` | 2.72pp | 2.83pp | slightly worse, still FAIL |
| `GATE_BLEND_INFORMATIVE` | t = −0.67 | **t = −1.18** | moved AWAY from significance |

Spread RMSE ratio vs market is 1.0582 overall and 1.0557 on linear rows — still above 1.0,
so `GATE_RMSE` fails and `bets_allowed()` stays False. Caveat: the "was" column is the
2026-08-04 reading under the old 220/260 lambdas, so it is not a clean A/B. The controlled
comparison — same games, same 750/550 ratings — is simulator 13.668 vs linear 13.196.

**Accurate NFL statement:** all 9 gates are implemented; 4 pass and 5 fail. The NFL build
remains a null result (Appendix A), so the failures on `GATE_BEATS_ELO` and
`GATE_BLEND_INFORMATIVE` are the expected and correct outcome — they measure signal the
model does not have, and must not be "fixed."

**Two cautions for whoever reads this next.**

`GATE_SCALE` is mis-specified for a model weaker than the market. It demands the model's
predictions vary as much as the market's, but regressing outcomes on this model's own
projections gives slopes of 0.605 (spread) and 0.753 (total) — the dispersion its signal
supports is sd 4.87 and 2.07, against market sds of 6.42 and 4.48. Inflating the model to
pass would make it over twice as confident on totals as its information justifies. Read the
failure as a measurement of the signal gap. Documented in `gate_scale`'s docstring.

`GATE_KEY_NUMBERS` can be made to pass (1.98pp) by applying the endgame table to a flat two
drives per team, but that collapses simulated margin sd to 12.46 against a real 14.2 and
breaks `test_simulated_margin_spread_is_realistic`. The shipped setting instead matches the
measured 1.34 endgame drives per team and leaves the gate red at 2.72pp. Do not trade the
margin distribution for this gate.

---

## Cross-cutting

| item | status |
|---|---|
| Zero-skill control alongside every metric | implemented for CLV only (`ncaa-model/src/backtest.py::zero_skill_projection`); not yet universal |
| `rmse_gate` precondition | Phase 0D |
| NCAA test suite | 3 tests (Phase 0.5C); previously zero |
| NFL test suite | 42 tests |
