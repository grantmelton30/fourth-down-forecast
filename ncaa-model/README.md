# NCAA totals model

> **STATUS: NULL. DO NOT TRADE.** The totals edge reported earlier in this file was an
> errors-in-variables artifact of grading against the **opening** line. Re-anchored on the
> close it collapses to `b = +0.076, t = +0.89`, and a zero-skill control (`close + noise`)
> reproduces the model's CLV exactly — 0.6923 against the model's 0.6923. See
> `NCAA_PLAYBOOK.md` Appendix A. The result tables below are retained as the record of what
> was measured and why it was wrong.

A college football totals model. It emits totals picks only, spreads are dark, and it sizes
no stakes. **It is not currently tradeable.**

This is the second of two builds. The first — `../nfl-model` — returned a null; this one
returned a null too, after first returning a false positive that the close-anchored test
caught.

---

## The standing rule

**Stakes go live only on positive CLV.** Nothing here sizes a bet: there is no `stake`
column in any output, and adding one is a deliberate act rather than a default. Track `b`
and CLV, not the win record.

That rule is not caution for its own sake. The model's absolute totals RMSE is **worse**
than the opener's (16.79 vs 16.38). It does not predict totals better than the market. It
carries incremental information the market has not priced, which is what `b` measures and
the only thing being bet. A win record over a short sample says almost nothing; CLV does
not wait on outcomes and is far less noisy.

---

## What the NFL build established (the null)

1,535 walk-forward games, 2019–2025, graded against the closing line. Three independent
architectures, none carrying information beyond the market:

| architecture | spread RMSE | b | t(b) |
|---|---|---|---|
| ridge direct | 13.529 | −0.107 | −1.40 |
| drive simulator | 13.882 | −0.044 | −0.74 |
| Elo | 13.208 | −0.018 | −0.20 |
| **market** | **12.692** | — | — |

Joint fit R² = 0.002. Full write-up in `../nfl-model/NFL_PLAYBOOK.md` Appendix A. Four
findings transferred directly into this build:

1. **The simulator never produces the mean.** It contributed all 5.02 points of excess
   dispersion in the NFL mean path while scoring worse than a plain linear projection off
   the same ratings. Here L1 *is* that linear projection; the simulator is not used for
   `E[margin]` or `E[total]` at all.
2. **The residual regression takes a free intercept.** Without it, constant level bias
   loads onto the disagreement slope — on NFL totals a +1.43-point bias masqueraded as
   `b = +0.060`, and de-biasing flipped it to −0.050.
3. **Rescale the component, not the disagreement.** Rescaling the disagreement is a scalar
   multiple: `b` and `se` both scale by `1/c` and `t` is exactly invariant. Shrinkage can
   never change significance; only re-ranking which games you disagree on can.
4. **Grade against the opener.** The NFL build could not — nflverse publishes closing lines
   only — which is the single biggest reason college was worth building.

---

## The NCAA result

Restricted universe (G5 + cross-tier, FBS vs FBS), graded against `spread_open` /
`over_under_open`, 2021–2025.

| market | n | a | b | t(b) | SD ratio | model RMSE | opener RMSE |
|---|---|---|---|---|---|---|---|
| **totals** | 1,543 | +0.449 | **+0.195** | **+2.23** | 1.022 | 16.79 | 16.38 |
| spreads | 1,543 | −0.140 | +0.042 | +0.80 | 1.011 | 17.13 | 15.56 |

Held out on 2024–2025, totals `t` clears 2.0 in **all twelve** pre-registered grid cells
(+2.20 to +3.12), varying monotonically with `half_life_games` — a broad plateau, not an
isolated spike.

### Historical CLV — the evidence that matters

Pick generator run across the full walk-forward backtest, with the blend and calibration
fit on strictly earlier seasons for every pick:

| | picks | CLV | movable lines | binomial p |
|---|---|---|---|---|
| **uncapped** | 68 | **0.7049** | 61 | 0.0019 |
| **capped** | 42 | **0.6923** | 39 | 0.0237 |

By season, uncapped: 2023 **0.675**, 2024 **0.700**, 2025 **0.818** — consistent across all
three. The record over those picks was 32–36, which is the expected pattern when the edge
is in the blend rather than in absolute prediction.

**The edge survives the concentration cap** (0.705 → 0.692), so it is distributed rather
than carried by a handful of teams.

---

## Exclusions, and why each was decided

Every exclusion below was decided on a measurement, before the outcome it might affect was
looked at. Full reasoning in `DECISIONS.md`.

| exclusion | reason | decided by |
|---|---|---|
| `consensus` as a line provider | carries **0.0%** openers across 1,503 games; it is a synthetic aggregate | measurement, D1 |
| seasons 2019–2020 | zero opener coverage; the missing-at-random test failed all four checks | pre-committed rule, D2 |
| weeks 10, 13, 14 | `GATE_UNBIASED_BY_WEEK` still fails after the time-varying intercept correction | the gate, D6d |
| weeks 15, 16 | n = 13 and n = 2 — too thin to test for bias at all | sample size, D6d |
| spreads entirely | `b` = +0.042, t = 0.80 — indistinguishable from zero | the blend, D7 |
| >15% of picks from one team | concentration is more likely repeat rating error than edge | cap, D6e |
| >25% of picks from one conference | same | cap, D6e |

Two candidate exclusions were **retracted** after testing rather than kept:

- **"The signal is absent from week 13 on"** — wrong. Testing the difference directly gives
  t = 1.53 (diff +0.388, se 0.254), and the 90% CI on late-season `b` contains the early
  estimate. That partition was underpowered by construction: at n = 254 you would need
  b ≥ 0.471 to detect anything. It was an underpowered subsample read after a bad outcome.
- **The 2023–2025 window as a veto** — it is not the held-out set; it mixes a tuning season
  into the holdout. `b` is stable at 0.262 / 0.243 / 0.191 across windows with SEs of
  0.092 / 0.127 / 0.065. One finding at three sample sizes, not a disagreement.

---

## Scope ledger — specified but NOT BUILT

Read this before any status report. Omissions surface here continuously, not in a final
inventory. `NCAA_PLAYBOOK.md` §3 specifies these modules; none exist.

| module | purpose per §3 | status |
|---|---|---|
| `src/priors.py` | Preseason prior construction (returning production, coaching change, recruiting) | **not built** — priors are last-season ratings regressed by 0.42, nothing else |
| `src/weather.py` | Open-Meteo joined to venue coordinates | **not built** — the model is blind to weather |
| `src/context.py` | HFA, rest, travel, altitude, rivalry variance | **not built** — no context adjustments of any kind |
| `src/srs.py` | SRS baseline to beat | **not built** — `GATE_BEATS_SRS` therefore cannot run |
| `src/drives.py` | Drive-level table from CFBD drives | **not built** — drives used only for pace counts |
| `src/simulate.py` | Monte Carlo engine | **not built** — no simulator exists in NCAA |
| `src/drive_model.py` | Multinomial drive-outcome model | **not built** |

Consequences that must be stated whenever results are reported:

- **No baseline.** `GATE_BEATS_SRS` and `GATE_COMPETITIVE_WITH_SPPLUS` cannot run, so the
  model has never been compared against anything except the market line.
- **No distribution.** There is no simulator; `E[total]` is a closed-form linear
  projection. The only distributional object is an empirical `ConditionalPMF`.
- **5 of 12 required gates do not exist.** See `../GATES.md`.

## Known defects and open items

- **Gate accounting was wrong.** "All seven gates pass" was reported for a set missing five
  required gates. `../GATES.md` is now the authoritative registry and every status claim
  reads from it.
- **Within-season level drift.** The projection runs progressively high through a season
  (+0.62 at week 4, +1.74 by week 13). Corrected with a walk-forward per-week intercept;
  the residual is year-specific and not learnable at any shrinkage, which is why three
  weeks are excluded rather than corrected. Mechanism: `eff_sum` drifts ~0.003 PPA/play
  against a ~250× fitted coefficient. It is **not** pace inflation, relaxing shrinkage, or
  the garbage filter — all three were tested and eliminated.
- **Conference concentration is structural.** Mountain West supplies 50% of uncapped picks
  and C-USA 38%, which is what a G5-restricted universe looks like by construction. The
  25% conference cap therefore binds hard, cutting the pick set by 38% without improving
  CLV. Worth revisiting whether that cap measures a defect or just the universe.
- **Spread CLV of 0.5285 against b ≈ 0.** Probably noise at n = 1,053, but also the
  signature of a small masked effect. Deliberately not chased — see D7.
- **No weather module.** Late-season college totals are weather-sensitive and this model is
  blind to it, a plausible contributor to the drift above.

---

## Setup

```bash
python -m venv ~/.venvs/ncaa-model
~/.venvs/ncaa-model/bin/pip install -r requirements.txt
echo "CFBD_API_KEY=your_key" > .env    # free key: https://collegefootballdata.com/key
export NCAA_MODEL_CACHE_DIR=~/.cache/ncaa-model   # keep the cache off iCloud-synced paths
```

## Commands

```bash
python run_backtest.py            # walk-forward backtest + all gates
python run_backtest.py --sweep    # the pre-registered 12-cell grid, all cells reported
python run_clv_history.py         # historical CLV + concentration, capped vs uncapped
python run_paper.py               # paper picks, caps live, no stakes

# viewer (shared with the NFL repo; sport is a config argument)
NCAA_MODEL_CACHE_DIR=~/.cache/ncaa-model \
  ~/.venvs/nfl-model/bin/streamlit run "/Users/annajenson/Desktop/Grant - Claude/ncaa-model/app.py"
```

`run_paper.py` writes a CSV and `output/ncaa_totals_{season}_wk{a}-{b}.xlsx` — sheets
**Bets** (capped picks), **All Games**, **Diagnostics**.

## API budget

Cold build: **120 calls** of a 900/month budget. Historical seasons are immutable and
cached permanently, so re-runs cost zero. `data/api_budget.json` tracks usage and
`BudgetedCFBD` raises rather than silently exceeding the cap.

## Attribution

- **CollegeFootballData.com** — games, drives, plays, betting lines. Free tier, personal
  use; do not redistribute bulk data.
- **nflverse** (CC-BY 4.0) and **nfelo** — used by the sibling NFL build.
