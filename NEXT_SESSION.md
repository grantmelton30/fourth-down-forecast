# Next session

**Rewritten 2026-08-23.** Read this, then `GATES.md`. The 2026-08-07 version described a
repo that no longer exists — a `~/dev/grant-claude/` path, a stale Desktop copy to delete, a
42/42 test count, and a "do this first" that has since been done and superseded. Replaced
rather than amended.

## Where everything lives

    C:\Users\grant\OneDrive\Documents\GitHub\fourth-down-forecast

One git repository, pushed to `github.com/grantmelton30/fourth-down-forecast`. Four
directories that matter: `nfl-model/`, `ncaa-model/`, `prop-model/`, `shared/`.

Run everything through the repo venv — the system Python has none of the dependencies:

    uv run python <script>                    # from the repo root
    .venv/Scripts/python.exe <script>         # equivalent

Test state, 2026-09-10: **shared 187, nfl-model 77, ncaa-model 135, prop-model 75.**
All green except one deliberate NCAA failure,
`test_power.py::test_full_fbs_close_anchored_null_is_adequately_powered`, which is the
provisional totals null documented in its own docstring and in GATES.md — a measurement,
not a break. Do not "fix" it by relaxing the assertion.

## The state of play, in one paragraph

Neither team model beats the market and neither can be made to by becoming more accurate —
Elo is MORE accurate than the NFL model and still returns 48.9 wins per 100. `bets_allowed()`
is False for both leagues and should stay that way. Two pre-registered rules are tracked
forward instead: **W1** (NFL, UNDER on forecast wind >= 10 mph) and **P1** (NCAA totals,
restricted universe, edge 0.5-6.0). Everything runs itself on GitHub Actions.

## What runs without you

| workflow | when | what |
|---|---|---|
| `track-rules` | daily 09:00 ET, Aug-Feb | records and grades P1 and W1 |
| `capture-props` | Thu + Sun in season | archives prop quotes (Odds API free tier) |
| `publish-ledger` | every 4h | refreshes the site |
| `integrity` | every push | the test suites |

## The only thing left is to wait

W1 starts logging around 9 September, when week 1 enters the 16-day forecast window. P1 has
28 bets logged and needs 200 settled to reach its first checkpoint. Both rules were designed
so a season of forward evidence is the output; no modelling task substitutes for it.

### Where that actually stands, 2026-09-10

**P1: 49 logged, 30 settled, 14-16 — 46.7% against a 52.4% breakeven, -3.6 units.** At
n=30 the interval is roughly +/-18 points, so this separates nothing from nothing; the
checkpoint is still 170 settled bets away. Two legacy cohorts carry it: `P1-legacy-opener`
12-14 over 26 settled, `P1-legacy-current` 2-2 over 4.

**W1: 0 logged, and correctly so.** The tracker runs daily and reports `0 qualifying
(forecast wind >= 10 mph)`. Nothing has met the rule yet — the machinery is fine.

**The active cohort is empty and the site scoreboard therefore reads 0-0.** Every one of
the 49 rows predates `protocol_version`, so `cohort_for()` files them under
`P1-legacy-*` while `ACTIVE_PROTOCOL` points at `P1-current-v2`. Nothing is hidden — all
49 bets and both cohort records are in `tracker.json` — but the headline a visitor sees
will stay 0-0 until v2 bets settle. `validate_payload()` passes on all-zeros because
0 = 0 + 0 reconciles. **Open decision: what the Record tab should show meanwhile.**

**Closing-line capture is the real bottleneck, not the win rate.** Same-book CLV resolved
on 8 of 286 graded rows (2.8%) up to 2026-09-10. Measured cause, in order: 138 rows were
discarded for surveying two books even though the entry book was named all along; ~48%
have no quote within six hours of the projection; ~47% have none inside the final hour
before kickoff. Fixed 2026-09-10 by resolving the entry book from the evidence label
(2.8% -> 4.2% on the existing ledger) and by moving market capture to hourly at :47, which
is what actually governs the remaining gap — kickoffs cluster at :00 and :30, and the old
fixed :17/:23 offsets structurally missed even-hour kickoffs. `grading_status.json` now
reports `clv_coverage` and warns below 25%, because it previously said `"ok"` while 97% of
rows had no CLV at all. **Expect the rate to climb through week 2; if it does not, the
capture schedule is still the thing to look at, not the grader.**

## Superseded: the lambda grid search

**DONE AND REVERTED, 2026-08-20. Do not act on this section; it is kept for the reasoning.**
The widened search ran, selected 2400/1600, and made the SHIPPED model worse on both
markets (spread RMSE 13.464 -> 13.495, total 13.627 -> 13.727). Reverted to 750/550. The
cause was the objective, not the search: `tune_lambdas` scores a ratings-only margin proxy
with no context adjustment, no simulator and no L1 projection, none of which the shipped
`model_spread` does without — and it scores MARGIN ONLY while the lambdas it writes feed
the totals path too. `ratings.tune_lambdas_shipped` was written to score the real
projection on both markets, and is deliberately NOT wired to the CLI: accuracy is not the
lever for this model, so it is parked rather than pursued.

The original text follows.

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

**~~`ncaa-model`'s backtest frame is still untagged~~ — RESOLVED 2026-09-10, and the
original diagnosis was half right.** The frame was never unvalidated: `walk_forward`
stamps every one with a `build_cache_signature` manifest over the full config and refuses
a mismatch. But that guard lives in the BUILDER, and `NCAAAdapter.backtest_frame()` read
the parquet with a bare `pd.read_parquet`, walking straight past it — so the leak was real
even though the tagging existed. It now validates the manifest's `config_sha256` against
the live config and returns None on a mismatch, which blocks picks rather than grading a
retired model. Config only, matching `nfl-model`; keying on the source tree would mark the
frame stale after every commit and leave the diagnostics permanently empty.

Checked rather than assumed at the time of the fix: the frame on disk was built 2026-08-19
from source that has since changed, but its **config hash still matches**, so the college
no-bet verdict standing today was read from the configuration that is actually shipping.
The guard is dormant until a config changes. The original text follows.

`NcaaSport.backtest_frame()` (`shared/sport.py:634`)
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

## Housekeeping — all four items resolved, 2026-08-23

Every entry that used to be here was stale. Checked rather than assumed:

- ~~Delete the stale Desktop copy~~ — the tree described no longer exists.
- ~~`git init` at the workspace root~~ — done. It is ONE repository now, pushed to
  `github.com/grantmelton30/fourth-down-forecast`, not three loose subdirectories.
- ~~Cache clutter~~ — moot; the repo-local caches are current and are what CI restores.
- ~~"The CFBD key is committed at `9fdc876` and still in HEAD... accepted the risk
  (internal-only, no remotes)"~~ — **that risk assessment no longer applies and the premise
  is false twice over.** There ARE remotes now. But commit `9fdc876` is not in this
  repository's history at all (it predates the consolidation), and a search of HEAD finds no
  key. The repository is also PRIVATE. Nothing to remediate — but the note mattered because
  "no remotes" was load-bearing in the original decision and stopped being true the day this
  was pushed. If it is ever made public, re-check before doing so.

Live secrets are GitHub Actions secrets (`CFBD_API_KEY`, `ODDS_API_KEY`), not files.

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
