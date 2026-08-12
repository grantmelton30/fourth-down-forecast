# COMBINED VIEWER — BUILD SPEC

One local web page, both sports, sport toggle. **A projected-score-and-why explorer, not a
pick engine.** Written to disk 2026-08-05 because the previous build spec existed only in a
chat window and was lost.

---

## 1. What the operator asked for, verbatim

> "I want the ncaa and nfl models to be combined to one webpage. I want to toggle between
> the two. I want it to be predicting outcomes of upcoming games."

> "It doesn't necessarily need to pick games. I just want it to essentially show all factors
> that made teams win games that normal people wouldn't see. I will then decide what games
> to pick. There should be something that says 'our model says the score should be 40-38
> which is higher than the predicted 25-24 that vegas shows' or something like that. That
> would then lean into taking the over + looking at spread."

> "I want all errors to be worked out. No more failing."

### Decisions locked

| question | answer |
|---|---|
| NFL codebase | `nfl-model` for the numbers, lift only the odds ingest from `nfl-spread-terminal` |
| what to display | Full numbers + banner. Projected score vs Vegas score, factor breakdown. **No stake sizing.** |
| hosting | Local now (`localhost:8501`), structured so deploying later is a step, not a rewrite |
| live lines | The Odds API for NFL, CFBD for NCAA |

---

## 2. The one thing this spec must not get wrong

**"No more failing" means no more errors. It does not mean making gates pass.**

Two different things were conflated once and must not be again:

- **Errors** — crashes, stale hardcoded numbers, code committed but never run, documented
  decisions never implemented. These are defects. Eliminate all of them. See §7.
- **A failing gate is a measurement, not a bug.** `GATE_RMSE` fails because the model's
  totals RMSE is 16.79 against the market's 16.22. That is the model being less accurate
  than Vegas. It cannot be fixed in code, only by breaking the measurement.

The resolution that makes the page honest *and* useful: the page shows the divergence **and
what divergences of that size have historically been worth**. See §4.

---

## 3. Architecture

Already sport-agnostic — this is wiring, not a rebuild.

```
shared/view.py      the Streamlit UI (exists)
shared/sport.py     NFLAdapter / NCAAAdapter (exists)
shared/export.py    Excel export (exists)
```

Build `shared/app.py` as the single entrypoint holding the sport toggle
(`st.sidebar.radio`), which selects an adapter and hands it to `view.py`. The two existing
per-repo `app.py` files stay as thin shims so nothing that references them breaks.

**Cache dirs differ and both must be live in one process:**

```
NFL_MODEL_CACHE_DIR=~/.cache/nfl-model     (19 MB)
NCAA_MODEL_CACHE_DIR=~/.cache/ncaa-model   (1.1 GB)
```

`src.config.CACHE_DIR` is read at import time in each repo, so the combined app must resolve
both without one clobbering the other. Resolve per-adapter, not globally.

Python is `~/.venvs/nfl-model/bin/python` (3.12). System python is 3.9 and unusable.
Streamlit is not on PATH — use `~/.venvs/nfl-model/bin/streamlit`.

---

## 4. The headline display

### Score derivation

Both repos normalise `spread` to **positive = home favored**, and `spread` equals home
margin. So for either the model or the market:

```
home_score = (total + spread) / 2
away_score = (total - spread) / 2
```

Render as the operator described it:

```
  Model     Ohio St 34.5  —  Michigan 31.2      total 65.7   spread  HOME -3.3
  Vegas     Ohio St 31.0  —  Michigan 28.5      total 59.5   spread  HOME -2.5
  ────────────────────────────────────────────────────────────────────────────
  Model is +6.2 on the TOTAL  →  leans OVER
  Model is +0.8 on the SPREAD →  leans HOME (weak)
```

**Do not invent large divergences.** The operator's "40-38 vs 25-24" example is a 29-point
gap; real divergences run 2–6 points. Show the true number.

### Required honesty line, attached to every divergence

A divergence is only meaningful relative to the model's accuracy. Directly under the lean,
print the measured context:

```
  Historically, this model's totals RMSE is 16.79 vs the market's 16.22 (n=1,543).
  A 6.2-point divergence has NOT been shown to predict which side wins.
```

Those numbers come from `rmse_gate`; compute them live, never hardcode. `report.py:71-72`
already hardcodes two stale CLV literals and that is a defect to remove, not a pattern to
copy.

**No CLV anywhere on the page.** It failed its zero-skill control (0.5846 against a required
0.50 ± 0.03) and is deleted. See `ncaa-model/PREREG.md`.

---

## 5. The factor breakdown — the actual feature of interest

Per game, show what moved the model's number away from a league-average matchup. This is
the "things normal people wouldn't see" the operator asked for.

**Frame it as a decomposition of the model's own number, not as hidden market edges.**
`calibrate.py` measured situational indicators against the market residual at
**R² = 0.0008**, and per-team HFA is below the pure-sampling-noise floor. The market has
already priced these. What the page honestly offers is *why this projection says what it
says*, which is genuinely useful for a human deciding a bet.

### NCAA factors (`ncaa-model`)

Model is linear, so the decomposition is exact:

```
model_total  = bt0 + bt1*pace_sum + bt2*eff_sum
model_spread = bs0 + bs1*net_diff + bs2*is_home
```

| factor | column | what to tell the user |
|---|---|---|
| offensive efficiency | `h_off`, `a_off` | PPA per play, **garbage time removed** — the differentiator vs box-score stats |
| defensive efficiency | `h_def`, `a_def` | PPA/play allowed. **Higher = worse defense.** |
| pace | `h_pace`, `a_pace` | drives per game above/below league mean — drives the total independently of efficiency |
| neutral site | `is_home` | 0.0 at neutral, else 1.0 |

**BLOCKER:** `project_walkforward` fits `bt`/`bs` per season and then discards them. Without
the coefficients the contributions cannot be computed. **Persist `bt0..bt2`, `bs0..bs2` per
season** alongside the frame — this is a prerequisite for the whole factor panel.

### NFL factors (`nfl-model`)

Context enters as EPA-per-drive on `net_epa` *before* the multinomial, so each adjustment is
separable and reportable in points:

| factor | source | why a casual bettor misses it |
|---|---|---|
| `qb_points` | nfeloqb `qb1_adj`, capped ±7 | biggest single swing in the model |
| `hfa_venue_deviation` | per-venue, centred on zero | not the generic 2.5 everyone assumes |
| `rest` | `0.12 × (home_rest − away_rest)`, capped ±1, short-week penalty | Thursday games |
| `travel` | great-circle miles × 0.15/1000 | west-coast road trips |
| `timezone` | 0.25 penalty, west→east early kickoff | body-clock 10am starts |
| `wind_mph`, `temp_f` | schedules + Open-Meteo | totals only |
| `dome_bump` | +0.5 indoors | |

Sort by absolute point impact and show the top 5, each with its sign and magnitude in
points. That list is the product.

---

## 6. Live lines for upcoming games

**NCAA** — CFBD carries Bovada. Budget: 120 of 900 monthly calls used. `BudgetedCFBD`
raises rather than silently exceeding; keep it.

**NFL** — The Odds API. `nfl-spread-terminal/main.py:154-155` has the pattern:

```python
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
```

**The key is NOT on this machine.** Only `TURSO_*` are in `nfl-spread-terminal/.env`. It
lives in the Render dashboard for the deployed terminal. Copy it to `nfl-model/.env` as
`ODDS_API_KEY=...` (gitignored). Free tier is 500 req/month — wrap it in a budget guard
mirroring `BudgetedCFBD` before the first live call.

**Bovada only.** No line shopping, no multi-book comparison. The line, not the price;
assume −110 for any arithmetic. ESPN's free feed returns DraftKings, which is the wrong
number and routinely differs by a half point — decisive at 3 and 7.

### Calendar reality
CFB Week 1 ≈ Aug 29, NFL Week 1 ≈ Sep 10. Until then there are no meaningful upcoming
games. **Build and verify against the 2025 season replayed as if live**, then switch the
date filter to "upcoming" when real slates appear.

---

## 7. Known defects — all must be fixed, none deferred

| # | defect | location |
|---|---|---|
| 1 | CLV declared deleted but still printed as "the number that matters" | `ncaa-model/run_paper.py:120`, `:135`; `run_backtest.py:179` |
| 2 | Hardcoded stale CLV literals `0.7049` / `0.6923` in the Excel Diagnostics sheet | `ncaa-model/src/report.py:71-72` |
| 3 | `NCAAAdapter.simulate` returns `None` unconditionally — Game explorer tab inert for NCAA | `shared/sport.py` |
| 4 | `_sheet_bets_placeholder` writes a placeholder instead of real picks; never exercised | `shared/export.py` |
| 5 | `run_backtest.py` has `rmse_gate` wired but was never run end to end | `ncaa-model/run_backtest.py` |
| 6 | Phase 0 findings never written into `NCAA_PLAYBOOK.md` Appendix A | required by standing rules |
| 7 | Projection coefficients discarded — blocks the factor panel (§5) | `ncaa-model/src/backtest.py::project_walkforward` |
| 8 | `nfl-model` totals/spread path has no live-odds ingest at all | new work |

---

## 8. Acceptance criteria

Every one must be demonstrated, not asserted. Today's session shipped a commit containing
two files that did not parse, because "it committed" was treated as "it worked."

1. `~/.venvs/nfl-model/bin/streamlit run shared/app.py` serves HTTP 200 on `localhost:8501`.
2. Toggling NFL ↔ NCAA renders both without exception, on a cold process.
3. Every game row shows model score, Vegas score, both divergences, and the top-5 factors.
4. Every displayed number is computed live. `grep` for hardcoded metric literals returns
   nothing in the render path.
5. The RMSE honesty line (§4) appears on every projection.
6. No CLV appears anywhere in the UI.
7. `pytest tests/` green in **both** repos.
8. **Commit integrity verified** — every tracked file byte-compared against its committed
   blob, and every committed `.py` parses. The Desktop is iCloud-synced and silently
   truncated two files during a commit on 2026-08-05:

```bash
for f in $(git ls-files); do git show HEAD:"$f" > /tmp/_c; cmp -s /tmp/_c "$f" || echo "MISMATCH $f"; done
```

---

## 9. Standing constraints that still apply

- **DO NOT TRADE banner stays.** Nothing on this page authorises a bet.
- **No stake sizing.** No `stake` column, anywhere.
- Report, don't score: a factor with no measured coefficient is displayed with its value and
  **no point value attached**.
- Every metric shows its zero-skill control beside it, or is not shown.
- Any claim that an effect differs between subgroups requires
  `ncaa-model/analysis/power.py::compare` **before** the claim is written.
- No tuning against the backtest. Both models are recorded nulls; that is settled and is not
  what this page is trying to overturn.
