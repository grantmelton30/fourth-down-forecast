# CLAUDE.md — NFL Spread & Total Simulator

Working rules for any agent or human touching this repo. Source of truth for the build is
`NFL_PLAYBOOK.md` (with `DATA_SOURCES.md` as its data appendix). This file carries the
five constraints from §0 that override convenience at every layer.

---

## THE FIVE NON-NEGOTIABLES (§0)

### 1. No lookahead, ever.
Any rating used to predict game `g` must be computed only from games that kicked off
strictly before `g`. The backtest must be walk-forward. Every function that builds ratings
takes an `as_of` cutoff argument and filters on it. If you find yourself computing
season-long ratings and applying them to games inside that season, you have broken the
model. An assertion catches this (`src/ratings.py`, `tests/test_no_lookahead.py`).

### 2. The model output is never used raw.
It is blended with the market using a weight estimated from out-of-sample backtest results
(§8). If the estimated model weight is below 0.10, the system must print a loud warning and
emit zero bets.

### 3. Simulate drives, not margins.
Do not sample the final margin from a normal distribution. Do not use Poisson for points.
See §6 — drive-level multinomial, key-number spikes must survive.

### 4. Every number in the output workbook must be traceable
to a function that produced it. No hardcoded fudge factors sprinkled in the report layer.

### 5. Prefer boring, correct, and fast over clever.
This runs weekly on a laptop. Target: full weekly refresh in under 3 minutes on cached
data.

---

## Practical consequences

- `as_of` is a **kickoff timestamp**, not a week number. Filter with `<` (strictly before),
  never `<=`.
- Ratings sign convention: higher `off_rating` = better offense; higher `def_rating` =
  **worse** defense. Asserted in `tests/test_ratings.py`.
- `spread_line` from nflverse is **positive when home is favored** and aligns with
  `result = home_score - away_score`. Never flip this silently.
- Context adjustments enter as EPA-per-drive on `net_epa` *before* the multinomial, never
  as points added to a simulated score.
- Blend is fit in **residual form** against the market, never as a weighted average of
  levels.
- Cover probabilities condition on the bet resolving: `P(win) / (P(win) + P(loss))`.
- Never use `nfl_data_py` (deprecated). Use `nflreadpy`; `.to_pandas()` at the ingest
  boundary and stay in pandas thereafter.
- Out of scope, permanently: parlays, teasers, same-game parlays, live betting,
  public-betting features, H2H-history features, neural networks, auto bet placement, a
  database.
- **Player props were on that list and were removed from it on 2026-08-21.** The reason is
  recorded rather than assumed: this model does not beat the market on sides or totals and
  cannot be made to by being more accurate — Elo is *more* accurate and still returns 48.9
  per 100 — while the two things that do clear breakeven (the wind rule, the restricted NCAA
  universe) both sit where the market is not looking. Props are the largest remaining market
  of that kind. Scoping them in is a deliberate reversal of a written constraint, not an
  oversight.
  **It does not relax anything else.** Props live in their own `prop-model/` sibling, must
  not modify `shared/sim_core.py`, `PBP_COLUMNS`, the market ledger, or `track-rules.yml`,
  and are bound by every rule in this file — §1 no-lookahead on kickoff timestamps above all.
  Parlays and SGPs stay permanently out: pricing them needs high-dimensional dependence
  against a very high-hold market, and single props have to demonstrate forward value first.

## Environment

Python 3.12 in `.venv/` (system Python is 3.9, which the spec's floor of 3.11 rules out).
Run everything as `.venv/bin/python …`.

## Gates

`run_backtest.py` must print all six gates from §8b. A failed gate blocks bet-sheet
generation — write the diagnostic report instead. A model that says "no bets" is working
correctly; a model that always finds bets is broken.
