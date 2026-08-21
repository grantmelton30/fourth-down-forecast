# PRE-REGISTRATION LEDGER — nfl-model

Created 2026-08-21, on the same rules as `ncaa-model/PREREG.md`. Read that file's scoring
key first; it governs here unchanged.

The standing rule, restated because everything below depends on it: **a decision made after
seeing its own outcome may not be used to support a positive result. It may still be used to
kill one.** Asymmetric on purpose — that asymmetry is what stops a search from converging on
a false positive.

---

## W1. FORWARD TRACKING RULE — wind, totals, UNDER only. Declared 2026-08-21.

Declared **before the 2026 NFL season has played a single game**, so every graded result
under it is genuinely out of sample.

### Why this rule and not a better model

The NFL model does not beat the market and cannot be made to by being more accurate. The
Elo control is **more** accurate than it (13.208 vs 13.464 RMSE) and still returns 48.9% ATS,
3.5 points under the 52.38% breakeven. Since roughly 2023 the NFL market is efficient enough
that any independent rating system's deviations from it are systematically wrong — measured
on our model (−0.316, t = −2.64) and reproduced on Elo (−0.197) as a control, so it is a
property of the market and not of this build. `GATES.md` carries the full pass.

Wind is the single exception found in either league: a specific, physical, one-sided effect
the market appears not to price. This rule bets that one thing and nothing else.

### The rule, stated so December cannot reinterpret it

* **Market:** totals only.
* **Side:** **UNDER, always.** Never the over. The measured effect is one-sided — wind
  suppresses scoring — and a rule that can fire either way is a different and untested
  hypothesis.
* **Universe:** NFL regular-season and postseason games played **outdoors**. A dome or a
  closed retractable roof is excluded, decided by the roof status at record time. An open
  retractable roof counts as outdoors.
* **A retractable roof whose game-day state is not yet published does NOT qualify.**
  Added 2026-08-21, before any bet was recorded, on discovering the case rather than
  reasoning about it: on the real 2026 week 1 slate, `schedules.roof` is null for BUF at HOU
  and BAL at IND, and it is null for *every* unplayed game by construction. Five venues are
  retractable (ARI, ATL, DAL, HOU, IND) — about 42 games, 15% of a season. An unresolved
  roof is **unknown, not open** — betting wind at a stadium that may be sealed shut is
  betting on nothing. And the contamination would be ADVERSARIAL rather than merely
  diluting: a roof is most likely to be closed in bad weather, which is precisely when a
  high wind forecast fires this rule, so the nullified games would concentrate exactly
  where the rule bets. Same rule `src/availability.py`
  already runs on: absence of a report is unknown, never healthy. Flag: `outcome-blind` —
  the trigger was a schema property of the fixture list, inspected before any 2026 game
  existed, and it cannot correlate with the win rate.
* **Trigger:** the **forecast** wind speed at the venue for the hour nearest kickoff,
  retrieved at record time, is **>= 10.0 mph**.
* **Rejected readings:** a forecast above **40.0 mph** is treated as absent and the game does
  not qualify, matching `config/nfl.yaml: wind_max_plausible_mph`. nflverse carries a game
  recorded at 71 mph; an uncapped linear term moved that total by 19.6 points.
* **No model input.** The trigger is wind alone. The rule deliberately does **not** require
  the model to disagree with the total, because the model's disagreements with the NFL market
  are measurably worse than useless (above). Adding a model filter would be a different rule.
* **Stakes:** flat. One unit per qualifying game, no sizing, no Kelly, no discretion.
* **Grading:** at the total actually available when the bet is recorded. The closing total is
  recorded alongside for comparability. Pushes are no action.
* **No mid-season changes.** Any adjustment to the threshold, the side, the universe or the
  stake voids this test and starts a new one under a new id.

### What is being tested, and what may NOT be cited

Backtest baseline, from the 2026-08-20 forecast-error pass in `GATES.md`:

| assumed forecast error | wins per 100 | vs. 52.38% breakeven |
|---|---|---|
| 0 mph (kickoff observation) | 57.7 | +5.3 |
| 2 mph | 56.8 | +4.4 |
| 3 mph | 55.8 | +3.4 |
| 4 mph | 54.8 | +2.4 |

**That table is not evidence and may not be cited as support.** Three reasons:

1. **The 10 mph threshold came from a sweep.** Flag: `false`. Thresholds were tried and this
   is one of two cells reported. It is deliberately the **worse-looking** of the two — 12 mph
   measured higher at every error level — chosen so the registration cannot be accused of
   taking the best cell, and because the extra volume resolves the season faster. Choosing
   the weaker cell is still choosing after seeing the table.
2. **THE CENTRAL WEAKNESS: the forecast error is simulated, not real.** Every row below the
   first adds *unbiased Gaussian noise* to the kickoff observation. Real forecast error is
   very unlikely to be unbiased, and there is specific reason to expect it is **worst in
   exactly the high-wind games this rule selects on** — gusty, unsettled conditions are
   harder to forecast than calm ones. If real error is biased or fatter-tailed than Gaussian,
   the true rate is below this table and possibly below breakeven. **No part of this rule has
   ever been tested against a real forecast.** That is the single largest open risk and it is
   why the tracker records the forecast rather than reusing the observation.
3. **The selection is on the noisy value and the grading on the real outcome**, which is the
   correct construction, but it inherits assumption 2 entirely.

Pre-registering now is what converts a post-hoc threshold into a legitimate test: the choice
is frozen, and the 2026 games it will be graded on did not exist when it was made.

### Why this cannot be settled this season

To establish with 95% confidence that the true rate exceeds 52.38%, given an observed 55%,
requires roughly **1,400 bets — about 22 seasons at this volume**. At the pessimistic 54.8%
it is roughly **1,650 bets, about 25 seasons**. The margin under test is 2-3 percentage
points on a ~65-bet-per-season rule; **no realistic sample resolves it.**

This tracking exercise therefore **cannot prove the rule works.** What it can do is detect a
rule that is clearly broken, and accumulate honest out-of-sample record at a rate no backtest
slice can fake.

### The byproduct that may matter more than the record

Every graded bet stores the **forecast** wind captured at record time and the **observed**
wind from nflverse after the game. Across a season that is a direct measurement of real NFL
forecast error at real bet-time horizons — bias, spread, and specifically whether error is
worse in high-wind games. That is the untested assumption the entire edge rests on, and one
season of ~65 paired readings measures it far better than it settles the win rate.

`track_w1.py report` prints it. Treat it as the primary output of season one.

### Pre-committed evaluation

* **First checkpoint:** end of the 2026 regular season, or 60 graded bets, whichever comes
  later.
* **Kill condition:** below **50.0%** at the checkpoint. Below breakeven out of sample does
  not get a second season on the argument that the sample was small. Consistent with the
  ledger's asymmetry: a post-hoc choice may kill a result even though it may not support one.
* **This kill test is weak and that is acknowledged, not hidden.** At n = 60 the standard
  error is about 6.4 points, so 50% sits only ~0.8 SE below the 55% baseline. It catches a
  badly broken rule, not a marginally bad one. Widening it would mean killing live rules on
  noise, which is the opposite error.
* **Continue condition:** at or above 50.0%. Continuing is **not** a claim that it works; the
  arithmetic above says one season cannot establish that. It means the rule has not yet
  disqualified itself.
* **Reported either way**, in full, whatever it says.

### How it is recorded

`track_w1.py` implements the rule and owns the log (`data/w1_log.csv`):

    python track_w1.py record      # log this week's qualifying games, at today's number
    python track_w1.py grade       # settle finished bets
    python track_w1.py report      # running record + forecast-error measurement

The log is **append-only**: `record` refuses to rewrite a game already present, and
settlement uses the total the bet was RECORDED at, not the close. Both properties are pinned
by tests, for the same reason they are in `ncaa-model/track_p1.py` — a line remembered later
is the same class of error as the opener anchoring that produced this repo's only false
positive.

**The forecast is fetched live and never read from the weather cache.** `CACHE_DIR/weather.parquet`
is keyed on `(lat, lon, date)` with no record of when a forecast was issued, so a reused row
would silently turn a bet-time forecast into whatever was fetched days earlier — the cache
bug class that has already hit this repo three times in one day (`NEXT_SESSION.md`). Every
recorded row carries `forecast_issued_at` and `hours_before_kickoff` so the horizon is a
measured quantity rather than an assumption.

The rule constants live in `track_w1.py` as module-level values with a test asserting they
still match this registration, so a silent edit cannot quietly redefine what is being tracked.

### When to record, and the horizon that forces it

**Open-Meteo forecasts reach about 16 days out.** Verified 2026-08-21: every week 1 venue
returns HTTP 400 at a 19-23 day horizon, and `forecast_at_bet_time` correctly reports
`unavailable`, which does not qualify. There is therefore no way to record a season in
advance, and no temptation to try.

**Recorded inside 48 hours of kickoff, checked daily.** `track_w1.MAX_HOURS_BEFORE_KICKOFF`
is 48; the GitHub Actions job runs every day in season and logs each game on the first pass
that finds it inside its own window. The window, not the calendar, decides what records.

**Why not a fixed weekday.** The first version ran Thursdays, which gave Thursday-night games
a ~9-hour horizon and Monday-night games ~4 days — worse on average and inconsistent across
the slate. Forecast error grows with horizon and this rule's edge shrinks with it: roughly
2-3 mph at three days (~55.8-56.8 wins per 100) against ~1-2 mph at one to two days
(~56.8-57.7). The uneven version was leaving about two points of win rate unclaimed for no
reason but scheduling convenience.

Changed 2026-08-21, **before a single bet had been recorded and with no outcome visible**, so
it cannot be outcome-driven. Flag: `outcome-blind`. It is an **operating parameter, not a
rule constant** — it changes when the forecast is taken, never the trigger, side, universe or
stake, none of which moved.

**The horizon is stored per bet, not assumed.** `hours_before_kickoff` is written on every
row because the window is an intention and the realised horizon is the measurement. If runs
are missed and bets land at mixed horizons, that is visible afterwards rather than hidden.

**Not recorded later than this deliberately.** A tighter window (say 24h) would buy a
slightly sharper forecast, but it bets into a line that has had longer to price the weather,
and it leaves no slack: one failed run would permanently lose that game. At 48 hours most
games get two daily passes, so a single red X is recoverable.

**A game already under way is never recorded**, and a game missed entirely is never
back-filled. Back-filling would mean choosing a forecast in hindsight, which is the entire
failure mode this file exists to prevent.

### Status of the model itself, unchanged by this

`bets_allowed()` remains **False** — six promotion gates fail (`GATE_KEY_NUMBERS`,
`GATE_BEATS_ELO`, `GATE_BLEND_INFORMATIVE`, `GATE_CALIBRATED`, `GATE_RMSE_SPREAD`,
`GATE_RMSE_TOTAL`). This is a measurement protocol for tracking a declared rule, not an
authorisation to stake money, and it does not alter any gate or promotion decision.

W1 also does not touch the model: it reads a forecast and a posted total, and consults no
projection. Nothing in `src/` changes behaviour because this file exists.
