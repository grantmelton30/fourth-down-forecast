# NCAA FOOTBALL SPREAD & TOTAL SIMULATOR — COMPLETE BUILD SPEC

You are building a production college football game simulator that outputs a projected
spread and total for every FBS game on a slate, compares them to market lines, and emits
a ranked bet sheet as an Excel workbook.

Build the whole thing. Do not ask clarifying questions. Every decision you would ask
about is specified below; where something genuinely is not specified, choose the simpler
option, write it down in `DECISIONS.md`, and keep going.

**Build the NFL model first if you have not.** This spec assumes the same architecture
and refers to it. The differences are large enough to warrant a separate codebase, but
the skeleton is identical and you should reuse the design.

---

## 0. WHY COLLEGE IS A DIFFERENT PROBLEM

Read this before writing code. Four structural differences drive every design choice.

**Connectivity.** 130-plus FBS teams play 12 games each, almost all inside their own
conference. The graph connecting the SEC to the Mountain West runs through a handful of
September non-conference games. A naive rating system has no way to know whether an
undefeated Sun Belt team is good or just isolated. This is why the ridge penalty must be
much heavier here than in the NFL and why conference-level effects need explicit
handling.

**Roster churn.** The portal turns over rosters annually rather than gradually. Last
season's final rating carries far less information into this season than it does in the
NFL. The preseason prior has to be constructed, not inherited.

**And the input weights have shifted hard.** Recruiting rankings used to be worth roughly
20-25% of a preseason projection. In the current portal environment that has fallen to
about 1-2%, because incoming transfers' *previous production* now gets folded directly
into the returning production calculation instead — which is a far more direct signal
than a high school star rating. Returning production, recent program history, and
coaching-change effects now carry the weight. If you build this model on recruiting
composites you will be modeling 2014.

**Blowouts and garbage time.** A 49-3 game contains maybe 40 competitive plays and 100
plays against backups. Unfiltered efficiency stats are dominated by garbage time in a way
they simply are not in the NFL. Filtering is not a refinement here; it is load-bearing.

**Pace variance.** NFL teams cluster tightly around 11-12 drives. College ranges from
sub-10 to 15-plus depending on tempo philosophy, and total lines range from the high 30s
to the 70s. Pace is a first-class modeling target, not a correction — and it is where a
disproportionate share of the available edge in totals lives.

---

## 1. NON-NEGOTIABLES

1. **No lookahead, ever.** Ratings used to predict game `g` come only from games before
   `g`. Walk-forward backtest. `as_of` cutoff on every fitting function. Assert it.
2. **API budget is a hard constraint.** The CollegeFootballData free tier allows 1,000
   calls per month. Build a call counter that persists to disk, logs every request, and
   **raises** when the monthly budget would be exceeded. Cache everything to parquet
   permanently. A season of historical data should cost well under 200 calls if you pull
   in bulk by season rather than by week.
3. **The model output is never used raw.** Blend with the market at a weight fit from
   out-of-sample backtest results. If the fitted model weight collapses to the floor,
   emit zero bets and say so.
4. **Simulate drives, not margins.**
5. **Garbage time is filtered everywhere ratings are computed.** No exceptions.

---

## 1b. DATA SOURCES

**`DATA_SOURCES.md` is attached alongside this playbook and is part of the spec.** It
carries every signup URL, API tier limit, licensing requirement, fallback path, and a
break-glass runbook — including the two CFBD gotchas that will silently corrupt this
build if you miss them (spread sign convention, and weather being behind the paid tier).
Read it before writing any ingest code.

---

## 2. ENVIRONMENT

Python 3.11+. `requirements.txt`:

```
cfbd>=5.18.0
pandas>=2.2
numpy>=1.26
scipy>=1.13
scikit-learn>=1.5
statsmodels>=0.14
xlsxwriter>=3.2
streamlit>=1.38
pyyaml>=6.0
requests>=2.32
python-dotenv>=1.0
pytest>=8.0
```

Get a free API key at collegefootballdata.com. Store it in `.env` as `CFBD_API_KEY`.
Never commit it.

The `cfbd` package v5.x targets **API v2**, which uses context-manager clients:

```python
import cfbd
configuration = cfbd.Configuration(host="https://api.collegefootballdata.com")
configuration.access_token = os.environ["CFBD_API_KEY"]
with cfbd.ApiClient(configuration) as client:
    games_api = cfbd.GamesApi(client)
    betting_api = cfbd.BettingApi(client)
    stats_api = cfbd.StatsApi(client)
```

If a method signature differs from what you expect, introspect the installed package
(`dir(cfbd.GamesApi)`) rather than guessing — the client is auto-generated from the
OpenAPI spec and method names track the API version. Do not stop and ask; inspect and
proceed.

---

## 3. FILE TREE

```
ncaa-model/
├── README.md
├── DECISIONS.md
├── requirements.txt
├── .env.example
├── config/
│   └── ncaa.yaml
├── data/
│   ├── cache/
│   ├── api_budget.json          # persisted call counter
│   └── manual/
│       ├── this_week_lines.csv
│       ├── coaching_changes.csv
│       ├── qb_starters.csv
│       └── bowl_opt_outs.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── cfbd_client.py           # budget-aware wrapper
│   ├── ingest.py
│   ├── drives.py
│   ├── ratings.py
│   ├── priors.py                # preseason prior construction
│   ├── weather.py               # Open-Meteo (CFBD weather is paid-tier)
│   ├── context.py
│   ├── drive_model.py
│   ├── simulate.py
│   ├── market.py
│   ├── betting.py
│   ├── backtest.py
│   ├── srs.py                   # baseline only
│   └── report.py
├── app.py
├── run_week.py
├── run_backtest.py
└── tests/
```

---

## 4. CONFIG — `config/ncaa.yaml`

```yaml
seasons:
  train_start: 2015
  backtest_start: 2019
  current: 2026
  # 2020 was structurally abnormal (partial conference schedules, no crowds).
  # Include it in ratings but EXCLUDE it from backtest scoring and from HFA estimation.
  exclude_from_scoring: [2020]

api:
  monthly_call_budget: 900        # 100 call safety margin under the 1000 free tier
  cache_permanent: true

teams:
  division: "fbs"
  # All non-FBS opponents collapse into one synthetic team with a fixed rating.
  fcs_bucket_name: "__FCS__"
  fcs_off_rating: -0.155          # EPA/play, seeded; re-estimated in backtest
  fcs_def_rating: 0.170

ratings:
  half_life_games: 6              # shorter than NFL: roster/scheme churn is faster
  # Heavier penalties than NFL. Unbalanced schedules demand more shrinkage.
  lambda_off: 850.0
  lambda_def: 950.0
  lambda_grid: [400, 600, 850, 1150, 1500, 2000]
  prior_weight_games: 5.0
  off_weight: 1.45
  def_weight: 1.0
  # Conference-level random effect strength. Prevents an isolated G5 team from
  # floating on three data points.
  conference_shrink_weight: 2.5

pace:
  league_mean_drives_per_team: 12.1
  drives_sd: 2.1                  # much wider than NFL
  half_life_games: 8
  lambda: 900.0

priors:
  # Weights for constructing a preseason rating. These reflect the current
  # portal-era reality, NOT pre-2021 conventional wisdom.
  prior_year_rating_regressed: 0.52
  returning_production: 0.26
  recent_history_3yr: 0.16
  coaching_change_effect: 0.04
  recruiting_composite: 0.02      # yes, two percent. See §0.
  offseason_regression: 0.42      # heavier than NFL's 0.28
  # Returning production sub-weights, offense (CFBD supplies these components)
  rp_offense_weights:
    ol_snaps: 0.40
    wr_te_yards: 0.35
    qb_pass_yards: 0.22
    rb_rush_yards: 0.03

context:
  hfa_league_mean: 2.6            # larger than NFL; longer travel, louder venues
  hfa_venue_shrink_games: 30
  hfa_neutral_site: 0.0
  hfa_g5_visiting_p5_penalty: 0.0 # do NOT add one; it double-counts the rating gap
  rest_point_per_day: 0.10
  rest_max_points: 1.2
  travel_points_per_1000mi: 0.30  # bigger than NFL: bus trips, no charters at G5
  altitude_threshold_ft: 4500
  altitude_points: 1.0
  wind_total_threshold_mph: 15
  wind_total_points_per_mph_over: 0.40
  dome_total_bump: 0.7
  bowl_opt_out_points_per_starter: 0.6
  bowl_interim_coach_points: 2.0
  rivalry_variance_multiplier: 1.08   # widens the sim, does NOT shift the mean

simulation:
  n_sims: 20000
  seed: 20260730
  xp_make_prob: 0.958
  two_point_attempt_rate: 0.075
  two_point_success_prob: 0.435
  defensive_td_lambda: 0.16
  special_teams_td_lambda: 0.070
  safety_lambda: 0.030
  # College blowouts have a fatter tail than a drive sim alone produces, because
  # a genuinely overmatched team's per-drive rate degrades as the game goes on.
  # Model this explicitly rather than by widening the variance.
  fatigue_enabled: true
  fatigue_onset_margin: 24        # once a team leads by this much
  fatigue_rate_shift: 0.12        # fractional shift toward the mean per-drive rate

market:
  devig_method: "multiplicative"
  # Blend coefficients are FIT in residual form (see 8b), never set by hand.
  model_weight_cap: 0.70          # higher cap than NFL: CFB lines are softer
  # Preferred line providers from CFBD, in priority order
  provider_priority: ["consensus", "DraftKings", "Bovada", "ESPN Bet"]

betting:
  breakeven_prob_110: 0.5238
  min_cover_prob: 0.545
  min_edge_points_spread: 2.5     # wider than NFL: bigger model error
  min_edge_points_total: 3.5
  # Big spreads are noisy in both directions. Cap where you will play.
  max_abs_market_spread: 24.5
  min_total: 38.0
  max_total: 78.0
  # Exclude games involving FCS opponents entirely
  exclude_fcs_games: true
  kelly_fraction: 0.20            # more conservative than NFL
  max_stake_pct_bankroll: 0.015
  bankroll: 10000
  default_price: -110

output:
  excel_path: "output/ncaa_week_{season}_{week}.xlsx"
```

---

## 5. THE CFBD CLIENT — `src/cfbd_client.py`

Wrap every API call. This module exists solely to keep you inside the free tier.

```python
class BudgetedCFBD:
    def __init__(self, budget_path: Path, monthly_budget: int): ...
    def call(self, api_name: str, method: str, cache_key: str, **kwargs) -> pd.DataFrame:
        """Check parquet cache -> return. On miss, check budget -> raise or call ->
        write parquet -> increment counter -> return."""
```

Budget file schema: `{"month": "2026-07", "calls": 143, "log": [...]}`. Reset the counter
when the month rolls over. On budget exhaustion, raise `APIBudgetExceeded` with a message
naming which cache keys are missing so the user knows exactly what is unavailable.

**Pull in bulk.** Fetch a whole season at a time, never week by week. Historical seasons
are immutable — once cached, never refetch them. Only the current season's most recent
weeks are ever re-pulled.

Endpoints you need, and roughly what they cost:

| Data | API | Calls per season |
|---|---|---|
| Games / results | `GamesApi.get_games` | 1-2 |
| Drives | `DrivesApi.get_drives` | 1-2 |
| Play-by-play | `PlaysApi.get_plays` | ~15 (paginate by week) |
| Betting lines | `BettingApi.get_lines` | 1-2 |
| Returning production | `PlayersApi.get_returning_production` | 1 |
| SP+ ratings | `RatingsApi.get_sp` | 1 |
| Recruiting | `RecruitingApi.get_recruiting_teams` | 1 |
| Teams / conferences | `TeamsApi.get_fbs_teams` | 1 |
| Venues | `VenuesApi.get_venues` | 1 (once, ever) |

That is roughly 25 calls per historical season. Seven seasons of history costs about 175
calls — a fifth of one month's budget, one time.

**Two things CFBD will not give you on the free tier, and what to use instead:**

*Weather.* CFBD's weather endpoint is behind the paid Patreon tier. Do not build against
it. Use **Open-Meteo** — free, no API key, no signup — joined to the venue coordinates
from `VenuesApi`. Forecast endpoint for upcoming games, archive endpoint
(`archive-api.open-meteo.com/v1/archive`, data back to 1940) to backfill the entire
backtest. Set `temperature_unit=fahrenheit` and `wind_speed_unit=mph` explicitly; the
defaults are metric and a 15 "mph" threshold compared against km/h will quietly suppress
totals on every calm Saturday. Skip the call for dome venues. Attribution is CC-BY 4.0.

*Player availability.* There is no reliable free college football injury feed, and this is
not a gap you can engineer around — the NCAA has no mandatory injury report, availability
lives in local beat reporting, and portal moves and opt-outs create questions no
structured source covers. The design response is to make the gap visible rather than
paper over it: the manual CSVs are genuinely manual, bowl games with an unfilled opt-out
file emit NO BET rather than defaulting to zero opt-outs, and `run_week.py` warns when a
manual file has gone stale. Practically, check availability news only for the five to
eight games the model already flagged — not all sixty. Model first, research second; the
reverse order is how you talk yourself into a game the model does not like.

**Note on SP+:** pull it, but use it only as a *backtest baseline to beat*, never as a
model input. If you feed SP+ into the model you are building a wrapper around someone
else's model and your edge is whatever lag exists between SP+ and the market, which is
nothing.

---

## 6. RATINGS — `src/ratings.py`

Same ridge machinery as the NFL build. The differences:

### 6a. Garbage-time filter

CFBD play-by-play does not ship a win-probability column in all versions. Compute your
own filter — a rules-based one is sufficient and more transparent:

```
garbage = (qtr == 2 and abs(margin) > 38)
       or (qtr == 3 and abs(margin) > 28)
       or (qtr == 4 and abs(margin) > 22)
       or (qtr == 4 and seconds_remaining < 300 and abs(margin) > 16)
```

Drop garbage plays before computing any efficiency number. Report in the run log what
percentage of plays got dropped — it should land somewhere around 12-18% league-wide. If
it is under 5% your filter is broken; if it is over 30% it is too aggressive.

### 6b. The FCS bucket

Every non-FBS opponent maps to `__FCS__`, one synthetic team with a rating fixed at the
config values. Include those games in the ratings fit — they carry real information about
the FBS team, especially early in a season when there is nothing else — but:

- Down-weight them to 0.45 of a normal game's weight.
- **Never bet a game involving an FCS team.** Config enforces this.
- Re-estimate the FCS bucket ratings once per backtest run from actual FBS-vs-FCS results
  and write them back to config.

### 6c. Conference shrinkage

Add a second-level shrinkage term. After the ridge solve, shrink each team's rating
toward its conference mean:

```
final_rating = (n_eff * raw_rating + conference_shrink_weight * conference_mean)
               / (n_eff + conference_shrink_weight)
```

where `n_eff` is the team's effective sample size (sum of its game weights, normalized).
Conference mean is computed from the raw ridge output.

Why: the ridge penalty shrinks toward the *global* mean, which is wrong for a team in a
weak conference — it makes a mediocre SEC team and a mediocre MAC team look more similar
than they are. Conference shrinkage restores the right target. Tune
`conference_shrink_weight` in the backtest alongside lambda.

### 6d. Prior construction — `src/priors.py`

This module is the biggest single difference from the NFL build. Preseason rating for
team `t` in season `s`:

```
prior(t, s) = w1 * regress(final_rating(t, s-1), offseason_regression)
            + w2 * returning_production_adjustment(t, s)
            + w3 * mean(final_rating(t, s-2), final_rating(t, s-3))
            + w4 * coaching_change_effect(t, s)
            + w5 * recruiting_z(t, s)
```

with weights from `priors:` in config: 0.52 / 0.26 / 0.16 / 0.04 / 0.02.

**Returning production** comes from CFBD's returning production endpoint. Use the
offensive and defensive percentages it supplies. Convert to an adjustment by fitting,
historically: `Δ rating year over year ~ returning_production_pct`. The relationship is
real and roughly linear — teams in the bottom decile of returning production regress
noticeably, teams in the top decile improve. Fit the coefficient, do not assume it.

**Coaching change effect** reads `data/manual/coaching_changes.csv`
(`team,season,change_type,prev_coach,new_coach`) where `change_type` is one of
`HC`, `OC`, `DC`, `HC+OC`, `none`. The effect is not a constant: the signal is that a team
which *underachieved relative to its own multi-year baseline* tends to improve after a
change, and one that overachieved tends to regress. Implement it as:

```
effect = -k * (rating(t, s-1) - baseline_5yr(t))    # only when change_type includes HC
```

Fit `k` on history. This is a mean-reversion term dressed up as a coaching term, which is
what it actually is.

**Recruiting** is a 2% weight. Use the CFBD team recruiting rating, z-scored. Do not
elaborate it. If you find yourself building a blue-chip-ratio calculator, stop — the
weight does not justify the code.

New FBS members and teams with no prior season: set prior to the mean of the conference
they are joining, minus 0.5 standard deviations.

### 6e. Lambda and shrinkage tuning

Grid search `lambda_off` × `lambda_def` × `conference_shrink_weight`. Write winners back to
config. This is a slow run — cache it and only rerun when the training window changes.

**Objective — PATCHED (PATCH 02 §3): held-out `t(b_model)`, not margin RMSE.** Actual
margin is mostly irreducible noise around the true spread, so minimizing RMSE against it
rewards getting close to the market's number, and optimizing that hard produces an
expensive market clone. Tune on incremental information over the market, because that is
what edge actually is.

**Two constraints on how you search:**

- **Scale-only changes are inert.** Shrinkage rescales the disagreement term and leaves `t`
  exactly invariant (§8b). λ moves scale; `half_life_games`, `prior_weight_games`,
  `offseason_regression` and the off/def ratio re-rank games. Only the latter can move `t`.
- **Pre-register the grid at ≤ 12 cells**, declared before the first run and evaluated on
  held-out seasons. A 720-cell grid under a true null yields ~36 cells at |t| > 2 by chance,
  a 52.5% chance of at least one false discovery, and a Bonferroni threshold of |t| > 3.99.
  Large grids after a null cannot confirm it and can only manufacture a false positive.

---

## 6.5 ARCHITECTURE — THE L1/L2 SPLIT, AND WHAT THE NFL BUILD PROVED

**Build this from day one. Do not rediscover it.** The NFL build returned a null on spreads
and produced seven transferable findings; these are them.

**The simulator does not produce the mean.** Two incompatible jobs were welded together
there, so failing at one blocked the other:

- **(a) `E[margin]`** is a regression problem. A direct linear projection off the ridge
  ratings beat the same ratings pushed through the drive simulator: 13.529 RMSE vs 13.882.
- **(b) Distribution shape** — key numbers, totals, the margin/total joint — is what a
  simulator is *for*, and nothing else you have produces it.

Measured on 1,535 NFL games: the simulator added `sqrt(8.148² − 6.419²) = 5.02` points of
noise to the mean path, while the direct projection off the same ratings added none. A
projection carrying 5 points of noise cannot resolve the 2.5-point disagreements these bet
thresholds hunt for.

**L1 — mean layer.** Ensemble of direct ridge projection + SRS + SP+ + market, fit by
non-negative least squares in residual form, free intercept, market coefficient at 1.0,
every component rescaled to market SD about its own mean. Spreads and totals fit separately.

**L2 — simulator.** Totals shape, the margin/total joint, factor sensitivities, outcome
ordering. Never the mean.

**L3 — calibration.** Rank-preserving quantile map onto the empirical conditional margin
distribution, bucketed by the blended line, built strictly from games before `as_of`. This
is how `GATE_KEY_NUMBERS` is reached. It cannot be reached by refining the simulator: two
independent draws from the *real* historical score distribution reproduce the real margin
spread but yield P(|margin| = 3) = 6.4% against an actual 14.7%, because the spikes come
from teams optimizing against the scoreboard, not from the scoring process. Keep the
final-drive score-state conditioning anyway — it improves the sim's internal realism, which
the factor-sensitivity output depends on.

### The seven transfer rules (PATCH 03 §5)

1. **Simulator out of the mean path** — L1/L2 from the start.
2. **Free intercept in every residual fit**, with `GATE_UNBIASED`.
3. **Grade against opening lines** (§8a), CLV measured separately.
4. **Totals before spreads.** College pace varies far more than NFL pace, the ridge yields a
   genuine pace term, and totals on non-marquee games are the least-watched number on the
   board. Run the totals evaluation first; build spreads only if totals show signal.
5. **Restrict the bet universe before measuring** — G5 and cross-tier, off marquee windows.
   Partition per §8c and let the data say which slice carries information. Do not fit one
   blend across all 800 games.
6. **Pre-register the hyperparameter grid at ≤ 12 cells**, held-out evaluation. Never 720.
7. **Tune on held-out `t(b_model)`**, understanding that scale-only changes cannot move it.

---

## 7. DRIVE MODEL & SIMULATION

Same structure as the NFL build. College-specific changes:

**Drive table** comes from CFBD's drives endpoint, which is cleaner than deriving it from
plays. Map `drive_result` into the same four classes: `TD`, `FG`, `NO_SCORE`, `TURNOVER`.
CFBD's result strings include values like `PUNT`, `DOWNS`, `MISSED FG`, `FG GOOD`, `TD`,
`INT`, `FUMBLE`, `END OF HALF`, `END OF GAME`. Write an explicit mapping dict and raise on
any unmapped value rather than silently bucketing it — new result strings appear.

**Multinomial predictors** — same five as NFL, plus one:

```
net_epa, start_fp, start_fp_sq, is_home_offense, net_epa_x_fp,
tempo_delta        # offense pace rating minus defense pace rating
```

**Drive count** uses `league_mean_drives_per_team: 12.1` and `drives_sd: 2.1`. The wider
spread is correct — college pace genuinely varies that much, and it is a major driver of
total-line edge.

**Fatigue module.** This is the one genuinely new mechanic versus the NFL build. In a
sim, once a team's lead exceeds `fatigue_onset_margin`, shift the trailing team's
per-drive scoring probabilities `fatigue_rate_shift` of the way toward the league mean and
shift the leading team's the same direction. This reproduces the real dynamic where
blowouts have both teams playing backups, without the crude hack of just inflating the
variance. It matters because CFB blowout margins are a meaningful fraction of the slate
and a symmetric-variance model misprices large spreads badly in both directions.

**Simulator validation gate** — same as NFL but the target distribution is different.
College margins are flatter and wider: the spike at 3 is real but smaller than the NFL's,
and the tail past 28 is much fatter. Compute the historical PMF from your own data rather
than assuming NFL numbers, and gate on matching margins 3, 7, 10, 14, 17, 21 within 2
percentage points, plus `P(margin > 28)` within 2 points.

---

## 8. MARKET — `src/market.py`

### 8a. Lines from CFBD

`BettingApi.get_lines(year=..., week=...)` returns each game with a list of lines by
provider. Each carries `spread`, `formatted_spread`, `over_under`, and where available
`spread_open` and `over_under_open`.

Take the first available provider by `provider_priority`. Store **both** open and current
where present.

**GRADE AGAINST THE OPENER — PATCHED (PATCH 03 §4). This is the headline change and the
reason the college build is worth making.**

```
market_spread = spread_open          # primary, what the backtest grades against
market_total  = over_under_open
market_close  = spread / over_under  # retained, for CLV only
```

The NFL build graded everything against the *closing* line and returned a null across three
independent architectures. That was the wrong test and the spec asked for it: beating the
close means beating the aggregate of every public model plus professional money, which is
the hardest benchmark in American sports betting. `00_START_HERE.md` had always said the
achievable edge is against the opener, before the market has processed it.

nflverse publishes closing lines only, so the NFL build *could not* run the right test.
**CFBD carries `spread_open` and `over_under_open` on the free tier**, so this build can.
Report both numbers every run: performance against the opener, and CLV — what fraction of
the time the close moved toward the model's side. Where `spread_open` is missing for a
game, fall back to the close and **flag the row**, so the two populations never get silently
pooled.

Sign convention: CFBD spreads are quoted from the **home team's** perspective (negative =
home favored). The NFL build's `spread_line` is the opposite (positive = home favored).
Normalize to one convention at ingest — pick "positive = home favored" to match the NFL
build — and write an assertion test that a known historical game has the sign you expect.
This will bite you otherwise, and it will bite you silently, because a sign-flipped model
still produces plausible-looking numbers.

### 8b. Blend

Same as the NFL build's §7c, in **residual form with a free intercept**, with SP+ playing
the role nfelo plays for the NFL:

```
actual_margin - market_spread = a
                              + b_model  * (model_spread  - market_spread)
                              + b_spplus * (spplus_spread - market_spread)
```

with the `market_spread` coefficient constrained to 1.0. Convert SP+ team ratings to a
projected spread the standard way — rating difference plus home-field adjustment — before
differencing.

**The intercept `a` is mandatory (PATCH 02 §1).** Without it, any constant level bias in a
component loads onto that component's disagreement slope. This is not hypothetical: on the
NFL build a totals projection running +1.43 points high reported `b = +0.060`, and
de-biasing flipped it to −0.050. Report `a` alongside every `b_k`; a significant `a` means
the component must be de-biased before its `b` means anything. `GATE_UNBIASED` enforces it.

**If you rescale a component, rescale the component about its own mean:**

```python
component_rescaled = component.mean() + (component - component.mean()) * (sd_market / sd_component)
```

Rescaling the *disagreement* instead is a scalar multiple — `b` and `se` both scale by
`1/c` and **`t` is exactly invariant**. The strategic corollary: shrinkage can never change
significance. Only changes that re-rank *which games* the model disagrees with the market
about can move `t`.

Residual form matters more here than it does in the NFL, not less. Levels regression on
two spread series correlated above 0.95 produces unstable coefficients and inflated
standard errors; differencing against the market leaves only the part of each model that
*disagrees* with the market, which is the only part that can add value, and the two
disagreement terms are close to orthogonal.

**This does not contradict §5's "SP+ is a baseline, not an input."** The distinction is
exactly where it enters. SP+ inside your ratings makes you a wrapper around SP+. SP+ as a
residual term in the blend is a testable ensemble hypothesis that the OLS fit either
confirms or rejects. Feed it to the blend; never to the ridge.

Read the two coefficients together:

| `b_model` | `b_spplus` | Read |
|---|---|---|
| significant | significant | Both carry independent signal. Use both. |
| ~0 | significant | You have rebuilt a worse SP+. Bet SP+ residuals and go find what it does that you do not. |
| significant | ~0 | You have signal SP+ lacks — plausible, since SP+ is not optimized for spread betting. Understand why before trusting it. |
| ~0 | ~0 | Neither beats the market here. Zero bets. |

Clamp each coefficient to `[0, model_weight_cap]`. A negative fitted weight is almost
always a sign error or a lookahead leak, not a discovery — investigate, do not deploy.

Expect a **higher** fitted model weight here than in the NFL — college lines carry more
noise, especially on non-marquee games. If your NFL model weight came out around 0.25 and
your NCAA weight comes out around 0.45, that is the expected pattern, not a bug. If NCAA
comes out *lower* than NFL, something is wrong with the ratings.

### 8c. Where the edge actually is

Fit and report the blend weight separately for these partitions, because they behave
differently:

- Power-conference vs Group of Five vs cross-tier games
- Totals vs spreads
- Early season (weeks 1-4) vs mid vs late
- Ranked-vs-ranked vs everything else

Do not use these partitions to build separate models — that is how you overfit. Use them
to decide **where to bet**, by restricting the bet universe to the partitions where the
model demonstrably carries information. Report a table.

---

## 9. BACKTEST — `src/backtest.py`

Walk-forward, weeks 4 onward, seasons `backtest_start` through `current - 1`, excluding
2020 from scoring.

Metrics table comparing: model / market / SP+ / your simple SRS baseline, on spread MAE,
spread RMSE, total MAE, total RMSE, Brier score, filtered ATS record.

### Acceptance gates

1. `GATE_KEY_NUMBERS` — simulated margin PMF matches historical at 3, 7, 10, 14, 17, 21
   and in the `>28` tail, each within 2 percentage points.
2. `GATE_NO_LOOKAHEAD` — assertion passes on a 200-row random sample.
3. `GATE_BEATS_SRS` — model spread RMSE beats the SRS baseline.
4. `GATE_COMPETITIVE_WITH_SPPLUS` — model spread RMSE within 0.5 points of SP+. You are
   not required to beat SP+; it is an excellent model with more data than you have. But
   if you are more than half a point worse, your ratings have a problem.
5. `GATE_BLEND_INFORMATIVE` — at least one of `b_model` or `b_spplus` (§8b) has a
   t-statistic above 2.0. If only `b_spplus` clears it, bets still emit, but the report
   must state plainly that the edge is SP+'s and not yours.
6. `GATE_CALIBRATED` — no bucket with 100+ observations off by more than 6 points.
7. `GATE_API_BUDGET` — a full backtest from cold cache uses fewer than 250 calls.
8. `GATE_SPEED` — weekly refresh on cached data under 240 seconds.
9. `GATE_UNBIASED` — **NEW (PATCH 02 §1).** `|a| < 0.5` points for every component, spreads
   and totals, where `a` is the free intercept of the residual regression (§8b).
10. `GATE_SCALE` — **NEW (PATCH 01 §4).** SD ratio of every L1 component against the market
    in `[0.85, 1.15]`, targeting 1.05–1.10.
11. `GATE_BLEND_INFORMATIVE` is **held-out**: the t-statistic must clear 2.0 on validation
    seasons the hyperparameters were not tuned on. In-sample `t` is not evidence.

**Kill criterion, declared in advance (PATCH 03 §5).** If held-out `t(b_model)` fails to
clear 2.0 on **college totals, on the restricted bet universe, graded against opening
lines** — stop. Do not sweep further. That is the last well-posed test available on free
data, and a null there is a real answer about what is reachable from public inputs, not an
invitation to keep searching.

Plus the bootstrap CI on filtered ATS% and the CLV proxy against closing lines, both
reported in plain language.

---

## 10. BET SELECTION — `src/betting.py`

Same structure as the NFL build — including the **residual-form blend** of §8b, not a
weighted average of levels — with these additional filters, all enforced:

- `exclude_fcs_games` — no game with `__FCS__` on either side.
- `abs(market_spread) <= max_abs_market_spread` (24.5). Beyond that, both the model and
  the market are largely guessing about how long starters play, and the variance swamps
  any edge.
- `min_total <= market_total <= max_total`.
- Both teams must have at least 3 completed FBS games in the current season, or the game
  is flagged `data_incomplete` and skipped. In weeks 1-3 this means the model produces
  numbers but bets almost nothing, which is correct.
- Bowl games: only bet if `data/manual/bowl_opt_outs.csv` has been filled in for both
  teams. An unfilled opt-out file on a bowl slate means NO BET, not a default of zero
  opt-outs. Bowl season is where models that ignore roster availability go to die.

Thresholds are wider than NFL (2.5 points on spreads, 3.5 on totals) and Kelly is more
conservative (0.20 fraction, 1.5% cap), because model error is genuinely larger here.

Expect the model to flag roughly 8-15% of a 60-game Saturday slate. If it is flagging 40%
of games, your threshold is too loose or your blend weight is too high — check the gate.

---

## 11. CONTEXT — `src/context.py`

- **HFA:** per-venue, shrunk toward `hfa_league_mean` of 2.6. Estimate from ratings-model
  residuals, exclude 2020 entirely from the estimation. Neutral sites get 0 — and note
  that CFBD flags neutral-site games, so use the flag rather than inferring it.
- **Do not add a separate "G5 visiting P5" penalty.** The talent gap is already in the
  ratings; adding a situational penalty double-counts it. Config has this at 0.0
  deliberately. Leave it there.
- **Altitude:** teams at venues above 4,500 feet (Wyoming, Air Force, Colorado, Utah
  State, New Mexico, BYU) get `altitude_points` added when hosting a team from below
  2,000 feet. This one is real and reasonably well documented.
- **Travel:** larger coefficient than the NFL because Group of Five programs travel far
  worse than NFL teams. Great-circle distance from `VenuesApi.get_venues()`, which returns
  latitude, longitude, **elevation**, dome flag, and city/state for every venue. Call it
  **once, cache it permanently, never call it again** — it is one API call that would
  otherwise recur weekly against a 1,000-call budget. Elevation from this same table
  drives the altitude adjustment above, so there is no second source to find.
- **Rivalry games:** widen the simulated distribution by `rivalry_variance_multiplier`,
  do **not** shift the mean. The common claim that "rivalry games are closer" does not
  survive controlling for team strength; what does survive is that they are more variable.
  Widening the distribution correctly reduces confidence rather than incorrectly moving
  the number. Maintain a rivalry list in config.
- **Bowls:** opt-outs and interim coaches from the manual CSVs.

---

## 12. REPORT, APP, WEEKLY RUN

Identical in structure to the NFL build. Differences:

- The `Ratings` sheet has additional columns: conference, conference rank, returning
  production percentile, and prior-vs-current rating delta. That delta column is your
  fastest sanity check — a team whose rating has moved 15 points from its prior by Week 6
  is either a genuine breakout or a data problem, and you want to look.
- The `All Games` sheet is long (60+ rows on a fall Saturday). Group by conference and
  add autofilter.
- Add a fifth sheet, `API Budget`, showing calls used this month and remaining. You will
  want this.
- `run_week.py --season 2026 --week 5` behaves as in the NFL build, plus it prints a
  warning if any manual CSV (QB starters, coaching changes, opt-outs) has not been touched
  in more than 14 days.

---

## 13. THINGS NOT TO BUILD

- SP+ or FPI as inputs to the **ratings**. They belong only as a residual term in the
  blend regression (§8b) and as backtest baselines. Never inside the ridge.
- Recruiting composite as anything more than a 2% prior term.
- Player props or any derivative market.
- Separate models per conference. One model, partitioned reporting.
- Any neural network. There are roughly 800 FBS games a season and 138 teams; the
  parameter count of the ridge model is already most of what the data can support.
- Head-to-head history, "revenge game" flags, or coach-versus-coach records.
- Automatic bet placement.

---

## 14. DELIVERABLES CHECKLIST

- [ ] `python run_backtest.py` runs end to end and prints every IMPLEMENTED gate.
      **See `../GATES.md` for the authoritative registry.** Of the 12 gates named in this
      playbook, 8 are implemented and 5 have never been built: `GATE_BEATS_SRS`,
      `GATE_KEY_NUMBERS`, `GATE_COMPETITIVE_WITH_SPPLUS`, `GATE_CALIBRATED`, `GATE_SPEED`.
      Any statement of the form "all gates pass" is invalid while that remains true.
- [ ] Implemented gates PASS, or failures are printed with observed values and explained.
      Currently 7 of 8 pass; `GATE_UNBIASED_BY_WEEK` fails.
- [ ] Cold-cache backtest uses under 250 API calls, verified in `data/api_budget.json`.
- [ ] `python run_week.py --season 2026 --week 5` produces a five-sheet Excel workbook.
- [ ] `streamlit run app.py` renders and the margin histogram works.
- [ ] `pytest` passes, including the spread-sign-convention test.
- [ ] `README.md` documents setup, the API key, the four manual CSVs and when to update
      each, and how to read `bet_to_line`.
- [ ] `DECISIONS.md` lists every open judgment call you resolved.
- [ ] The partition table from §8c is pasted into `README.md` — it tells you which slice
      of the slate your model actually has an edge on, and that is the most valuable
      output of the entire backtest.


---

## APPENDIX A — RESULTS OF THIS BUILD (2026-08-04): A NULL ON NCAA TOTALS

Recorded so it is not rediscovered. **This build is complete with a correct negative
result.** It is recorded in the same form as `NFL_PLAYBOOK.md` Appendix A.

### The headline

The apparent totals edge was an **errors-in-variables artifact of anchoring on the opener**,
not information. Two tests, either of which is decisive on its own.

**Test 1 — re-anchor the same model on the close.** Nothing else changed: same walk-forward
projection, same restricted universe, same free intercept. Only the market line the
regression is anchored on.

| anchor | a | t(a) | b | t(b) | n |
|---|---|---|---|---|---|
| opener (as shipped) | +0.449 | +1.08 | **+0.195** | **+2.23** | 1,543 |
| **close (decisive)** | +0.908 | +2.17 | **+0.076** | **+0.89** | 1,543 |

Close-anchored by season: +0.82, +0.38, −0.99, +0.90, +0.64. No season clears 2.0 and one
is negative. On the capped historical pick set alone, b = −0.090 (t = −0.32).

**Test 2 — the zero-skill control.** A projection defined as `close + gaussian noise`,
carrying no information whatsoever, run through the identical pick generator:

| projection | picks | CLV | movable lines |
|---|---|---|---|
| the model, uncapped | 68 | 0.7049 | 61 |
| the model, capped | 42 | 0.6923 | 39 |
| **zero-skill control** | 71 | **0.6923** | 65 |

**The control reproduces the model's CLV exactly.** CLV as measured against an
opener-anchored projection is a reading of line noise, not of skill.

### Why the opener anchor manufactures an edge

The opener is a noisy estimate of the true total. Anchoring on it puts that same noise into
both the regressor (`model − open`) and the response (`actual − open`), which correlates
them mechanically. The close, being a better estimate of the truth, then moves toward the
projection's disagreement on average — producing a positive `b` and a high CLV from a model
that knows nothing. At this sample size the simulated false-positive rate for the opener
anchor is ~68%, against ~0.8% for the close anchor.

**This was a specification error in the patch series, not an implementation error.**
PATCH_03 §4 and PATCH_04 both directed grading against the opener, on the reasoning that
the achievable edge is against the number before the market processes it. That reasoning is
sound for *execution* — you do bet the opener — but it is invalid as a *measurement* anchor,
because the quantity you are trying to estimate is contaminated by the anchor's own error.

### What still stands

Everything measured against the close is unaffected, and several findings are real and
transferable:

- **GATE_OPENER_COVERAGE and DECISIONS D1.** `consensus` genuinely carries 0.0% openers
  across 1,503 games; Bovada carries 84.5%. The provider fix is correct regardless.
- **The within-season level drift is real** (+0.62 at week 4 to +1.74 by week 13) and its
  mechanism was correctly identified: `eff_sum` drifting ~0.003 PPA/play against a ~250x
  fitted coefficient. Pace inflation, relaxing shrinkage and the garbage filter were each
  tested and eliminated.
- **The simulator does not belong in the mean path** — established on NFL, still true.
- **The free intercept, `GATE_UNBIASED_BY_WEEK`, and the zero-skill control** are all
  permanent harness improvements that came out of this build.

### The standing conclusion

**No edge has been demonstrated on NCAA totals against the closing line, and none on NCAA
spreads (b = +0.042, t = +0.80).** Together with the NFL null, three architectures across
two sports fail to beat the closing number on public data.

**Do not resume tuning against this backtest.** Continued searching after a null is how a
false positive is manufactured — and this build has now produced one and caught it, which
is exactly the failure mode the discipline exists to prevent.

Any future work must be judged **close-anchored**, with the zero-skill control reported
alongside every CLV figure. If the control does not read near 0.50, the metric is not
measuring skill.
