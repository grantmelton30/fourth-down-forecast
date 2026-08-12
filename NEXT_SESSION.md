# Next session

Rewritten 2026-08-07. Read this first, then `GATES.md`.

## Where everything lives now

**`~/dev/grant-claude/`** — moved off the iCloud-synced Desktop, which was truncating
files mid-write and corrupted a round of commits before it was caught. The Desktop copy
still exists and is STALE; delete it or you will edit the wrong tree.

Run everything with the cache dirs set — the repo-local `data/cache` folders are stale
partial copies and reading them silently produces wrong answers:

    NFL_MODEL_CACHE_DIR=~/.cache/nfl-model
    NCAA_MODEL_CACHE_DIR=~/.cache/ncaa-model
    python = ~/.venvs/nfl-model/bin/python

Test state: **nfl-model 42/42, ncaa-model 12/12** (with those env vars set; without them
the NFL lookahead test fails spuriously on the stale local cache).

Combined viewer: `streamlit run shared/app.py`, then http://127.0.0.1:8501 — or the
machine's LAN IP from another device.

## The one thing to do first

**Re-run the NFL lambda grid search against the widened grid.** The 2026-08-05 search
selected lambda_off=750 — the ceiling of the grid it was given — so the search was
truncated, and the true optimum is at least that high and has never been located. The grid
was widened afterwards (1100, 1600, 2400) and has not been searched since.

CORRECTION, verified 2026-08-07: an earlier version of this section said "the config still
runs 220/260." **It does not.** `config/nfl.yaml` carries a `tuned:` block at
lambda_off=750 / lambda_def=550, and `Config.effective_lambdas` (`src/config.py:194`)
prefers that block over the `ratings:` seed values, which are inert. `src/ratings.py:429`
resolves it and bakes it into the cache tag; the built artifact is on disk as
`walkforward_ratings_default_o750_d550_p10_2016_2026.parquet`. The model is therefore NOT
under-regularized at 220/260 today. The gain from this run is the distance from 750 to the
true optimum — worth having, but not the windfall the old wording implied. **Do not "fix"
the seed values by hand; the tuned block is what runs.**

    NFL_MODEL_CACHE_DIR=~/.cache/nfl-model python run_backtest.py --tune-lambdas

It was started and killed for time. NOTE: it ends by calling `write_tuned_block()`, which
rewrites `config/nfl.yaml` — run it when you can review the diff, not unattended. It is
slow (a full walk-forward ratings build per grid point) and prints nothing until the end.

## What changed on 2026-08-07

### Both models now correctly refuse to bet

`bets_allowed()` tested only the blend t-statistics, so ncaa-model showed a green
"picks are permitted" banner while its own GATE_RMSE failed on both markets. It now also
enforces the accuracy precondition, and both sports report False.

Do not try to reverse this. The NCAA signal that cleared |t| > 2 only exists against the
OPENER. On the uncontaminated `*_pure` projection against the CLOSE — which
`backtest.py` names as the only honest test — totals fall to t = +1.63 and spreads to
+1.87. Beating the opener but not the close is capturing the market's own move.

Both models are LESS ACCURATE than the market: RMSE ratios ~1.08 on spreads, ~1.03 on
totals, for both sports. That is the headline finding and it has survived every change.

### NCAA got the shared simulator

Measured constants (not copied from the NFL — two run opposite to intuition: college
converts extra points MORE often, 0.973 vs 0.940, and goes for two LESS often, 0.070 vs
0.095). Own drive model, own `net_epa`, `project_game.py`, and weather via Open-Meteo.
Validated by pooling 419 real 2024 games: P(|margin|=3) 9.26% simulated against 8.59%
actual.

### Weather is real on both sides

NFL wind was a 15 mph hinge fitted on 143 games (t = -1.82). Re-measured as linear from
zero: **-0.3339 pts/mph, t = -5.03**, centred on the 8.17 mph league mean. Implausible
readings above 40 mph are rejected — nflverse records one game at 71 mph, which was
moving that total by 19.6 points. Dome bump was 0.50, now 3.16 (see the caveat in the
config: the strength-controlled estimate is noisier than the raw one).

NCAA wind: -0.1867 pts/mph, t = -2.89, centred on 7.0.

### NFL injuries

Snap-weighted burden — each player listed Out or Doubtful weighted by the share of snaps
he had been taking, so one unit is roughly one starter. **-0.518 pts of margin per unit,
t = -2.15**, spread only (against the total it is -0.011, t = -0.05).

End-to-end it buys 0.017 of RMSE. Real, measured, and nearly irrelevant to accuracy;
kept because it is correct and it names a factor in the breakdown.

**There is no NCAA equivalent and there will not be.** CFBD's OpenAPI spec has no
injury, availability, or depth-chart endpoint, and no per-player snap counts — which was
the half that made the NFL version work.

### The recurring bug class: caches that ignore their own inputs

Hit THREE times in one day, all the same shape — a cache keyed on a name that does not
encode the config that produced it:

1. `ncaa-model` retuned `pace.lambda` 900 -> 15; the ratings artifact kept serving the
   lambda-900 fit for five days. The simulator projected Army and Air Force, the two
   slowest offenses in college football, at league-average pace.
2. `nfl-model`'s `backtest_frame.parquet` — the weather and injury constants would not
   have invalidated it, so the gates would have graded a model that no longer existed.
3. My own before/after measurement compared against a July artifact and I attributed the
   improvement to the wrong change. Caught by arithmetic that did not add up.

NFL is now hashed/tagged on config (`ratings.walkforward_path`,
`backtest.backtest_frame_path`) and is current: its config hashes to
`backtest_frame_5275f05be1.parquet`, which exists on disk.

**`ncaa-model`'s backtest frame is still untagged, and unlike the other three this
instance is live right now.** `NcaaSport.backtest_frame()` (`shared/sport.py:634`)
unconditionally reads `backtest_frame_default.parquet`. That file is dated Aug 5 07:37 —
older than the simulator NCAA adopted (`drive_model_2019_2025.json`, Aug 6 20:36), the
weather frame (Aug 6 09:15), the retuned ratings (Aug 6 21:06) and the drive rebuild
(Aug 7 12:47). It feeds `rmse_vs_market()`, which as of 2026-08-07 is a HARD PRECONDITION
inside `bets_allowed()`.

So every NCAA accuracy number quoted in this file, the Diagnostics panel, and the NCAA
bet/no-bet decision are all read from a model that no longer exists. The verdict is very
probably unchanged — a model that was worse than the market on Aug 5 is not likely to have
overtaken it — but right now that is an assumption, not a measurement. Fix by keying the
frame on a config hash the way `nfl-model` does, then rebuilding.

## Known-wrong, documented, not fixed

- **~~NFL uses the simulator for its mean; NCAA uses a linear projection.~~ RESOLVED
  2026-08-07 — and the old framing was wrong.** The question was posed as "simulator mean
  or linear mean," assumed to have one answer per sport ("at most one of the two is
  right"). It has one answer per MARKET, and both sports are half right.

  Re-measured on NFL data under the current 750/550 ratings — 1,119 walk-forward games,
  both estimators predicting identical games, paired season-clustered bootstrap on squared
  errors:

  | market | SIM RMSE | L1 RMSE | winner | 95% CI, mean sq-err diff |
  |---|---|---|---|---|
  | spread | 13.668 | **13.196** | linear, by 0.47 | [+8.36, +16.98] significant |
  | total | **13.445** | 13.630 | simulator, by 0.19 | [−8.44, −1.86] significant |

  The mechanism is not mysterious. A margin is a *difference*, close to linear in
  `net_diff` by construction, so OLS — which shrinks toward the mean — wins. A total is
  *generated* by drives, and pace × efficiency is compositional, so a linear model in
  `(pace_sum, eff_sum)` cannot represent it and the simulator wins. NFL is therefore right
  on totals and wrong on spreads; NCAA is right on spreads and, by the same mechanism,
  **probably wrong on totals — NOT YET MEASURED on college data.** College pace varies far
  more than NFL pace (`ncaa-model/src/backtest.py::build_features` says so in its own
  docstring), which if anything strains the linear totals model harder there.

  This falsifies the blanket rule in `nfl-model/PATCH_03_DECISION.md` §5.1 and
  `ncaa-model/DECISIONS.md` D4 — "the simulator is never in the mean path." It is right for
  spreads and wrong for totals. The 13.529-vs-13.882 evidence behind that rule was a
  *margin* measurement generalised to both markets without ever being tested on totals.

  **It changes no decision.** Best achievable ratios are 1.056 (spread) and 1.024 (total)
  against the market; GATE_RMSE fails both markets either way and `bets_allowed()` stays
  False. Worth doing for correctness, not for profit.

  **IMPLEMENTED on the NFL side, 2026-08-07.** `config/nfl.yaml` gained a `projection`
  block; `spread_source: linear` is live, totals stay on the simulator. The simulator still
  produces the whole margin distribution and `SimResult.recentered()` shifts it onto the L1
  mean, so cover probabilities describe the game the model actually predicts and the
  key-number spikes survive. The `projection` block is in the frame's hash, so flipping the
  knob rebuilds instead of serving a stale frame. 42/42 tests pass. On the rebuilt frame the
  linear rows reproduce the offline prediction exactly — 13.196 RMSE on n=1,119, ratio
  1.0557. No gate changed state; `GATES.md` carries the full before/after. One diagnostic
  moved the wrong way: `t(b_model)` went from −0.67 to −1.18, i.e. further from
  significance — consistent with a more accurate mean carrying no more information about
  *where the market is wrong*, which is the thing that gate measures.

  **NCAA MEASURED AND DELIBERATELY NOT CHANGED.** 3,472 college games: linear beats the
  simulator on college totals (16.567 vs 16.954, significant) — the OPPOSITE of the NFL
  result. NCAA ships correctly today. Full numbers, and an open lead on college margin
  under-dispersion, are in `ncaa-model/DECISIONS.md` D4.

  **What implementing cost, for the record.** (a) NCAA's half had to be measured on
  NCAA data first — carrying an NFL result across is the exact error this codebase keeps
  warning about. (b) `cover_prob_home` is read off the simulated distribution; if
  `model_spread` becomes L1 while the distribution stays centred on the sim mean, the two
  disagree — the sim margins must be re-centred on the L1 mean, which is the real L1/L2
  design and not a one-line change. (c) `backtest_frame_path()` hashes the *config*, so a
  pure code change does NOT invalidate the cached frame; it would keep serving the old
  projection. Same bug class as everything in the section above. (d) All nine NFL gates
  re-read afterwards, and the 42 tests are the regression net.
- **Option-vs-option matchups.** Two triple-option teams score 1.56 pts/drive against a
  league 2.27, and the model runs +14.1 points high across all 15 academy head-to-heads.
  Not a rivalry effect — rivalries measure to nothing (144 pairs, pair-level residual
  variance ratio 0.97, i.e. pure noise). Not explained by run rate either. It is 3 teams
  and ~3 games a season; the recommendation is to skip those games, not model them.
- **Temperature** is significant on NFL data (+0.072 pts/degF, t = +2.69) and unmodelled.
  The playbook's "temperature is noise" note is not supported.
- **College overtime runs under NFL rules** in `sim_core` — the count of exchanges is
  right, the scoring environment inside them is not. Affects the ~4% of games reaching OT.
- **The drive clamp `(8, 16)`** truncates college's top 1-2% of games.
- `nfl-model` declares `drive_count_dist` and never reads it.

## Housekeeping left

- Delete the stale Desktop copy at `~/Desktop/Grant - Claude`.
- `git init` at the workspace root so these docs are tracked (also needed before any
  Streamlit deploy). Currently only the three subdirectories are repos.
- Disposable clutter in the caches: two experiment frames in `~/.cache/nfl-model`, and
  the repo-local `data/cache` folders, which should probably be emptied so the only way
  to run is the correct way.
- The CFBD key is committed in `nfl-model` history at `9fdc876` and still in HEAD. The
  operator has seen the evidence and accepted the risk (internal-only, no remotes).

## Working rules

- **Measure, do not assume.** Every constant that turned out wrong was a plausible guess.
- **Validate a method against a known answer before trusting it.** The conversion MLE was
  only believable once it reproduced nfl-model's three configured constants.
- **Check the arithmetic of your own result.** The weather comparison looked like a win
  until the adjustment moved totals +0.84 while the model moved -0.99. Opposite signs
  meant the baseline was wrong, not the change.
- **A cache read must validate its inputs, not just its columns.**
- **A failing gate is a measurement.** A failing test that catches your own change is the
  system working.
- Run `python -m pytest tests/ -q` in `nfl-model` after touching `shared/sim_core.py`.
  Those 42 tests are the regression net for both sports.

## Do not do this

- Do not try to make `GATE_BEATS_ELO`, `GATE_BLEND_INFORMATIVE` or `GATE_SCALE` pass.
  They measure signal the models do not have.
- Do not re-add the NCAA "totals edge". Measured again on 2026-08-07: it exists only
  against the opener and vanishes against the close.
- Do not weaken `bets_allowed()` back to a t-statistic check.
