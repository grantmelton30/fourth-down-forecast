# DECISIONS

Every judgment call the playbook left open, plus the four places where following the
playbook literally would have produced a broken model. Ordered by how much they affect the
output.

---

## A. Places where the spec contradicts itself or its own data

These are the ones worth reading. In each case the playbook states two things that cannot
both hold, and a choice had to be made.

### A1. `net_epa` uses `+ def_weight * def_B`, not `-` (§5a)

**The contradiction.** §5a defines the sign convention explicitly — "Higher `def_rating` =
**worse** defense" — and requires a unit test that "the previous season's best defense has
the lowest `def_rating`". It then gives the matchup formula as:

```
net_A = off_weight * off_A - def_weight * def_B
```

Under the stated convention those disagree. `def_B` is the coefficient on the opposing
defense in a model of offensive EPA, so a bad defense carries a *larger* value; subtracting
it would mean an offense projects **better** the better the defense it faces.

**Chosen:** keep the sign convention and its test — they are the load-bearing half, and §5a
itself flags this exact flip as "the single most common bug in this kind of build" — and use
`+`. `tests/test_ratings.py::test_net_epa_rewards_facing_a_worse_defense` pins the
behaviour. Implemented in `ratings.net_epa`.

### A2. Context adjustments convert through the model's points-per-net_epa slope (§6c)

**The contradiction.** §6c says to convert points to EPA and apply them to `net_epa` before
the multinomial, using `epa_per_drive = points / expected_drives`. But `net_epa` is EPA
**per play** — its league-wide spread is roughly ±0.15 — while `points / expected_drives`
for a 1.7-point HFA is ≈0.149. Applying the literal formula makes home-field worth about
ten times its real value, and distorts every other context adjustment identically.

**Chosen:** keep §6c's *intent* exactly — the adjustment is applied to `net_epa` before the
multinomial, never added to a simulated score, so the key-number structure survives — and
fix only the conversion, dividing by the model's own points-per-`net_epa` slope
(≈33 points per unit). A 1.7-point HFA now moves the simulated line by 1.7 points, verified
in `tests/test_simulate.py::test_context_moves_the_line_by_the_requested_points`.
Implemented in `simulate.net_epa_shift_for_points`.

### A3. ~~`is_home_offense` is held at 0.5 in the simulator~~ — SUPERSEDED BY PATCH 01

**Status: reversed.** The diagnosis below was accepted, but PATCH 01 §0 (Bug A) resolves the
double-count the other way round, and the patch overrides the playbook. The multinomial's
`is_home_offense` term is now the league baseline — it is estimated from data at the drive
level, which makes it the better of the two estimates — and `context.py` no longer supplies
any league HFA. What context contributes is a **per-venue deviation centered on zero**,
asserted by `tests/test_ratings.py::test_venue_hfa_deviations_are_centered`. Neutral sites
set the flag to 0.5 on both sides, which removes the baseline for games with no home team.
Both bugs are now patched into `NFL_PLAYBOOK.md` §6c and §13 so the text cannot reintroduce
them. The original reasoning is kept below for the record.

### A3 (original). `is_home_offense` is held at 0.5 in the simulator (§6b vs §13)

**The contradiction.** §6b puts `is_home_offense` in the drive multinomial, so the model
learns home-field itself (worth ≈2.7 points). §13 *also* specifies home-field as a context
adjustment — per-venue, shrunk toward 1.7 points, estimated from ratings residuals. Using
both gives a ≈4.4-point home edge against a market that prices it near 1.7, which §13 warns
"will generate a season of losing bets".

**Chosen:** §13's treatment is the richer one, so context owns home-field and the simulator
passes 0.5 for both teams. The feature stays in the model as §6b specifies; it simply is not
double-counted at predict time.

### A4. An endgame layer was added, because §6d is unreachable without one

**The contradiction, and it is measurable rather than a matter of taste.** §6c prescribes
independent drives. §6d requires the simulated margin distribution to match history at
|margin| = 3 within 2 points, i.e. ≈14.5%.

Independent possessions cannot produce that. Drawing two independent scores from the *real*
historical score distribution reproduces the real margin spread almost exactly (sd 14.06 vs
14.16) but yields P(|margin| = 3) = **6.4%** against an actual **14.7%**. The strict drive
simulator landed at 6.16% — i.e. it was already behaving exactly as a correct
independent-drive model must. The spikes at 3 and 7 are created by teams **playing to the
scoreboard**, not by lumpy scoring increments, and no amount of fixing the drive sampler
reaches them.

**Chosen:** keep §6's architecture and add the missing mechanism, measured from the data
rather than assumed:

1. **Score-state-aware final drive.** Each team's last drive draws from an empirical
   `P(outcome | score state)` table fitted on drives inside the last 5 minutes of the fourth
   quarter. The effect is large and unambiguous in the data:

   | Score state (offense) | P(FG) | P(TD) | P(no score) |
   |---|---|---|---|
   | Tied | 28.5% | 4.3% | 59.2% |
   | Trailing 1–3 | 26.2% | 12.9% | 37.5% |
   | Trailing 4–8 | **0.3%** | 28.1% | 43.9% |
   | Leading 1–3 | 6.5% | 3.9% | 87.0% |
   | *(baseline, Q1–Q3)* | 15.9% | 23.1% | 50.1% |

   Tied late, a drive ends in a field goal 28.5% of the time against a 15.9% baseline —
   that is literally where the 3-point spike comes from. Trailing by 4–8, teams essentially
   never kick, because a field goal is worthless to them.

2. **Overtime.** Both teams get a possession, repeated up to three exchanges. Without it the
   simulator left 3.3% of games tied against a real 0.4%, and that misplaced mass is stolen
   almost entirely from |margin| = 3.

Together these moved |3| from 6.16% → 10.84% and ties from 3.33% → 0.66%. Wider endgame
windows were tested (8, 10 and 15 minutes) and made |3| *worse*, by diluting the
concentrated tied/trailing states. See the gate report for the residual gap, which is
disclosed rather than tuned away.

Implemented in `drive_model.EndgameTable` / `fit_endgame_table` and
`simulate._play_overtime`.

### A5. The extra possession goes to a random team, not always the home team (§6c)

**The contradiction.** §6c step 1 says both teams get `d` drives, "then with probability 0.5
add one drive to **the home team**". Read literally, that hands the home team +0.5 expected
possessions in every game — worth about a point at ~2 points per drive — on top of the
home-field advantage `context.py` already supplies. It showed up as an unexplained bias:
the model's mean spread was 2.77 against a market mean of 1.51 and an actual mean of 1.61,
and the ≈0.95-point gap between the model's mean and its own context adjustment (1.82)
matched the phantom possession almost exactly.

**Chosen:** keep the odd-possession mechanic — it is real — but let the coin flip decide
*which* team gets the extra drive as well as whether there is one, since in a real game the
extra possession belongs to whoever receives first. Expected possessions per game are
unchanged; the systematic home tilt is gone.

**Confirmed by PATCH 01 §0 (Bug B)**, which prescribes exactly this and supplies the same
two-draw form (`extra` and `to_home`) now implemented in `simulate.simulate_game`. Patched
into `NFL_PLAYBOOK.md` §6c so the text cannot reintroduce it.

---

## B. Modelling choices the spec left open

- **Drive outcome mapping.** `fixed_drive_result` has nine values, mapped to §6a's four
  classes. Two need a decision: `Opp touchdown` (a pick-six) is a **TURNOVER** for the
  offense, with the 7 points generated by `defensive_td_lambda`; `Safety` is **NO_SCORE**
  for the offense, with the 2 points generated by `safety_lambda`. This keeps every rare
  scoring event in exactly one place.
- **Drives with no scrimmage play are dropped** (a kickoff returned for a touchdown, or a
  half expiring on the return). They have no starting field position, are not something the
  drive multinomial models, and their points already come from the special-teams Poisson.
- **Starting field position and drive ownership come from scrimmage plays only**
  (`special == 0`). nflverse attaches the kickoff to the receiving team's drive, so the
  first row of a drive is usually a kickoff at `yardline_100 == 35`; reading field position
  from it put the average drive start at midfield instead of a team's own 27, inflating
  every scoring rate and flattening the key numbers. This was a real bug, caught by the §6d
  table.
- **Prior-injection rows carry a 1 in one team column only** — intercept and `home_offense`
  are zero — so they act as a direct Gaussian prior on that coefficient rather than dragging
  the overall level. §5a says "one row with `off_t = 1` only", which this implements
  literally.
- **The pace ridge uses a single 32-column block**, with a 1 for the offense *and* a 1 for
  its opponent, because §5b defines `E[drives] = league_mean + pace_A + pace_B` — both teams
  in a game share a possession count. It does not reuse the efficiency ridge's separate
  offense/defense blocks.
- **Ratings history is decay-weighted across seasons, not truncated at the season
  boundary**, per §5a's definition of `games_ago` as the team's own games back from the
  cutoff. Games whose decay weight falls below 1e-3 are dropped as a pure speed
  optimisation.
- **Per-venue HFA is estimated walk-forward** — when projecting 2023, only 2022 and earlier
  inform the venue estimates. The strength-to-points scale inside that estimate is fit
  in-sample; it is a nuisance parameter, not a forecast.
- **The fitted `hfa_epa` coefficient is negative** (≈ −0.013 EPA/play) and is reported but
  not used for HFA. This is a real consequence of the garbage-time filter: home teams lead
  more often, so filtering blowout possessions removes proportionally more of their plays.
  §13's residual-based, points-scale estimate is used instead, which is what §13 asks for.
- **Null `wp` passes the win-probability leg of the garbage-time filter** rather than being
  dropped; the quarter and score legs still apply. Observed drop rate is 16.3%, inside the
  expected band.
- **The points-per-`net_epa` slope is averaged over the empirical field-position
  distribution**, not evaluated at a single mean position. Expected points is convex in
  field position, so the point estimate understated the true average slope and every
  context adjustment landed about 20% hot.
- **Tuned lambdas are `lambda_off = 750`, `lambda_def = 550`**, written into
  `config/nfl.yaml` by `--tune-lambdas` as §5c requires. Two caveats worth recording: the
  optimum sits at the *upper boundary* of the spec's grid, so the data wants more shrinkage
  than the grid allows; and the whole 9× range of the grid spans only 0.056 RMSE
  (13.529 best, 13.585 worst), so regularisation is not a material lever here.
- **The 1.6 / 1.0 offense-to-defense weighting was verified rather than assumed.** Grid over
  alternatives: 1.6/1.0 → 13.529 RMSE, 1.3/1.0 → 13.532, 2.0/1.0 → 13.535, 1.0/1.0 →
  13.552, 1.0/1.6 → 13.638. §5a's value is optimal, and only the *ratio* matters (1.6/1.6
  scores identically to 1.0/1.0), since the scale is absorbed downstream.
- **Elo baseline parameters** (K = 20, 25 Elo per point, 0.25 offseason regression, 538's
  margin-of-victory multiplier) are conventional; §8b only requires *a* baseline.
- **`bet_to_line` steps in half-point increments** away from the market number and returns
  the last line where both the probability and edge thresholds still hold.

## C. Data and environment

- **Python 3.12 in a virtualenv at `~/.venvs/nfl-model`.** The system interpreter is 3.9,
  below the spec's 3.11 floor, and no Homebrew or pyenv was present; the toolchain was
  installed with `uv` into the user directory, nothing system-wide.
- **The virtualenv is deliberately *outside* the project.** This checkout lives on an
  iCloud-synced Desktop, and iCloud actively corrupted an in-project `.venv` mid-session —
  numpy imported but had no `__version__` because the sync daemon was rewriting
  `scipy/_lib` underneath it. A 12,234-file venv inside a sync root is not viable.
- **`NFL_MODEL_CACHE_DIR` overrides the cache location**, defaulting to the spec's
  `data/cache/`. Same cause: parquet reads inside the sync root stalled for minutes against
  seconds of CPU, which alone would fail GATE_SPEED. Cache contents are disposable and
  gitignored either way.
- **Play-by-play requests are clamped to seasons nflverse has published.** `seasons.current`
  is 2026 and the 2026 slate exists in `load_schedules`, but no game has been played and
  `load_pbp` rejects the season outright. Schedules are deliberately exempt from the clamp —
  next season's slate is exactly what `run_week.py` needs.
- **Kickoff timestamps are naive US/Eastern.** Every comparison is between two nflverse
  kickoffs, so a consistent frame is all that is required, and tz-aware timestamps in
  parquet invite dtype friction. Missing `gametime` defaults to 13:00 ET, the modal kickoff.
- **Stadium roofs were cross-checked against `schedules.roof`** for all 32 teams since 2023,
  as §13 requires. Zero disagreements.

## D. Things degraded gracefully, as the spec permits

- **nfelo's historical spread series is not built**, so the blend runs in its two-term form
  with `b_nfelo = 0`. §7.5 and DATA_SOURCES.md §7b both explicitly permit this and instruct
  that "the build must not block on this". The high-value half of the nfelo integration —
  `qb_elos.csv`, which drives the QB module — *is* wired in and working. To add the spread
  series later, run `github.com/greerreNFL/nfelo` locally and drop a parquet with
  `[game_id, nfelo_spread]` into the cache; the blend picks it up with no code change.
- **The CLV proxy reports "unavailable"** rather than a number. §8a asks to grade the model
  against the closing line "where opening lines are available" — nflverse publishes only the
  closing number in `spread_line`, so grading it against itself would be circular. The hook
  is in place if an opening-line source is added.
- **`--n-sims 6000` for the backtest**, against the configured 20,000 for a weekly slate.
  Distribution shape is stable well below 20,000 and the backtest simulates ~1,900 games;
  the weekly path, which is what GATE_SPEED measures, uses the full configured count.
