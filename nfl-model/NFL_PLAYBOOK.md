# NFL SPREAD & TOTAL SIMULATOR — COMPLETE BUILD SPEC

You are building a production NFL game simulator that outputs a projected spread and
total for every game on a slate, compares them to market lines, and emits a ranked bet
sheet as an Excel workbook.

Build the whole thing. Do not ask clarifying questions. Every decision you would ask
about is specified below; where something genuinely is not specified, choose the simpler
option, write down the choice in `DECISIONS.md`, and keep going.

---

## 0. NON-NEGOTIABLES

Read these first. They constrain everything downstream.

1. **No lookahead, ever.** Any rating used to predict game `g` must be computed only
   from games that kicked off strictly before `g`. The backtest must be walk-forward.
   Every function that builds ratings takes an `as_of` cutoff argument and filters on it.
   If you find yourself computing season-long ratings and applying them to games inside
   that season, you have broken the model. Add an assertion that catches this.
2. **The model output is never used raw.** It is blended with the market using a weight
   estimated from out-of-sample backtest results (§8). If the estimated model weight is
   below 0.10, the system must print a loud warning and emit zero bets.
3. **Simulate drives, not margins.** Do not sample the final margin from a normal
   distribution. Do not use Poisson for points. See §6.
4. **Every number in the output workbook must be traceable** to a function that produced
   it. No hardcoded fudge factors sprinkled in the report layer.
5. **Prefer boring, correct, and fast** over clever. This runs weekly on a laptop.
   Target: full weekly refresh in under 3 minutes on cached data.

---

## 0b. DATA SOURCES

**`DATA_SOURCES.md` is attached alongside this playbook and is part of the spec.** It
carries every signup URL, API tier limit, licensing requirement, fallback path, and a
break-glass runbook. Read it before writing any ingest code. Where this playbook names a
source, that file explains how to actually get it and what to do when it stops working.

---

## 1. ENVIRONMENT

Python 3.11+. Create `requirements.txt`:

```
nflreadpy>=0.1.5
polars>=1.0
pandas>=2.2
numpy>=1.26
scipy>=1.13
scikit-learn>=1.5
statsmodels>=0.14
xlsxwriter>=3.2
streamlit>=1.38
pyyaml>=6.0
requests>=2.32
nfelodcm>=0.2.2
pytest>=8.0
```

Notes you need:

- `nfl_data_py` is **deprecated**. Use `nflreadpy`. It returns **Polars** DataFrames.
  Call `.to_pandas()` at the ingest boundary and work in pandas everywhere else — do not
  mix the two APIs across modules, it creates constant friction.
- Key nflreadpy calls: `nfl.load_pbp(seasons)`, `nfl.load_schedules(seasons)`,
  `nfl.load_players()`, `nfl.load_snap_counts(seasons)`.
- `load_schedules()` is the single most important table in this build. It carries
  `spread_line` (positive = home favored, aligned with `result` = home_score -
  away_score), `total_line`, `home_score`, `away_score`, `result`, `total`, `roof`,
  `surface`, `temp`, `wind`, `away_rest`, `home_rest`, `div_game`, `location`,
  `overtime`, and `game_type`. This gives you free historical closing lines back to 1999
  and means **the entire backtest requires no paid odds API**.

Set up a disk cache at `data/cache/`. Every ingest function checks for a parquet file
first and only hits the network on a miss or when `--refresh` is passed.

---

## 2. FILE TREE

Create exactly this:

```
nfl-model/
├── README.md
├── DECISIONS.md
├── requirements.txt
├── config/
│   └── nfl.yaml
├── data/
│   ├── cache/            # parquet, gitignored
│   └── manual/
│       └── this_week_lines.csv   # fallback line input
├── src/
│   ├── __init__.py
│   ├── config.py         # loads yaml into a frozen dataclass
│   ├── ingest.py         # nflreadpy -> tidy tables
│   ├── drives.py         # pbp -> drive-level table
│   ├── ratings.py        # ridge team strength + pace
│   ├── qb.py             # QB adjustment (uses nfeloqb)
│   ├── nfelo.py          # nfeloqb QB values + nfelo spread for the blend
│   ├── stadiums.py       # 32-team lat/lon/tz/roof/elevation literal dict
│   ├── weather.py        # Open-Meteo forecast + archive
│   ├── context.py        # HFA, rest, travel, weather
│   ├── drive_model.py    # multinomial drive-outcome model
│   ├── simulate.py       # Monte Carlo engine
│   ├── market.py         # devig, blend, key numbers
│   ├── betting.py        # cover prob, Kelly, selection
│   ├── backtest.py       # walk-forward harness + gates
│   ├── elo.py            # baseline only
│   └── report.py         # Excel writer
├── app.py                # streamlit
├── run_week.py           # weekly CLI entrypoint
├── run_backtest.py       # backtest CLI entrypoint
└── tests/
    ├── test_no_lookahead.py
    ├── test_devig.py
    ├── test_simulate.py
    └── test_ratings.py
```

---

## 3. CONFIG — `config/nfl.yaml`

Write this file verbatim. Every tunable lives here; nothing is hardcoded in modules.

```yaml
seasons:
  train_start: 2016        # post-XP-move era; margin distribution is stable from here
  backtest_start: 2019     # 6 seasons of walk-forward
  current: 2026

ratings:
  # Exponential recency decay on game weight, in games
  half_life_games: 9
  # Ridge penalty on offense/defense coefficients. Tuned in backtest; this is the seed.
  lambda_off: 220.0
  lambda_def: 260.0
  # Prior injection strength, expressed as "equivalent games of information"
  prior_weight_games: 6.0
  # Plays per team per game, used to scale prior-injection weights into play units
  league_mean_plays_per_game: 63.0
  # How far last season's final rating regresses toward league mean before becoming
  # this season's prior. 0 = full carryover, 1 = full reset.
  offseason_regression: 0.28
  # Offense is more stable than defense; weight accordingly when forming net strength.
  off_weight: 1.6
  def_weight: 1.0
  # Cross-validation grid for lambda tuning
  lambda_grid: [80, 120, 160, 220, 300, 400, 550, 750]

pace:
  league_mean_drives_per_team: 11.4
  drives_sd: 1.7
  half_life_games: 12
  lambda: 400.0

qb:
  # Points of spread adjustment when a team's projected starter differs from the QB
  # who took the majority of dropbacks in the rating window.
  enabled: true
  # Cap on the absolute adjustment so a bad depth-chart read cannot wreck a line
  max_adjustment_points: 7.0
  # Minimum dropbacks for a QB to have a usable personal rating
  min_dropbacks: 150
  # Regression of a QB's personal EPA+CPOE composite toward positional mean
  regression_dropbacks: 320

context:
  hfa_league_mean: 1.7           # points; do not use 3.0, it is a decade out of date
  hfa_venue_shrink_games: 40     # shrink per-venue HFA toward league mean
  hfa_neutral_site: 0.0
  rest_point_per_day: 0.12       # capped
  rest_max_points: 1.0
  short_week_penalty: 0.5        # Thursday game on 4 days rest
  travel_points_per_1000mi: 0.15
  timezone_cross_penalty: 0.25   # west->east early kickoff
  wind_total_threshold_mph: 15
  wind_total_points_per_mph_over: 0.35   # subtract from total
  dome_total_bump: 0.5
  week18_rest_starters_points: 3.0       # applied only when flagged manually

simulation:
  n_sims: 20000
  seed: 20260730
  # Drive count sampling
  drive_count_dist: "normal_rounded"
  # Post-TD conversion
  xp_make_prob: 0.940
  two_point_attempt_rate: 0.095
  two_point_success_prob: 0.475
  # Rare-event processes, per team per game (Poisson means)
  defensive_td_lambda: 0.13
  special_teams_td_lambda: 0.045
  safety_lambda: 0.035

market:
  devig_method: "multiplicative"   # spreads/totals are near-balanced; see §7
  # Blend coefficients are FIT in residual form (see 7c), never set by hand.
  # This is only an upper clamp on each fitted coefficient.
  model_weight_cap: 0.60

betting:
  breakeven_prob_110: 0.5238
  min_cover_prob: 0.545        # required simulated cover prob to bet a side
  min_edge_points_spread: 1.5  # required |model - market| in points
  min_edge_points_total: 2.5
  kelly_fraction: 0.25
  max_stake_pct_bankroll: 0.02
  bankroll: 10000
  default_price: -110

output:
  excel_path: "output/nfl_week_{season}_{week}.xlsx"
```

---

## 4. INGEST — `src/ingest.py`

Functions, all cached to parquet, all returning pandas:

```python
def load_schedules(seasons: list[int], refresh: bool = False) -> pd.DataFrame
def load_pbp(seasons: list[int], refresh: bool = False) -> pd.DataFrame
def load_rosters(seasons: list[int], refresh: bool = False) -> pd.DataFrame
```

`load_pbp` must subset columns immediately — full PBP is huge and you only need:

```
game_id, season, week, season_type, posteam, defteam, home_team, away_team,
drive, fixed_drive, fixed_drive_result, drive_start_yard_line, play_type,
epa, wpa, success, yards_gained, down, ydstogo, yardline_100, qtr,
game_seconds_remaining, score_differential, passer_player_id, passer_player_name,
rusher_player_id, cpoe, qb_dropback, penalty, special, touchdown, field_goal_result,
interception, fumble_lost, safety, td_team, return_touchdown, extra_point_result,
two_point_conv_result, wp
```

**Garbage-time filter.** Build a boolean column `competitive` and use it everywhere
ratings are computed:

```
competitive = (wp between 0.05 and 0.95)
              AND NOT (qtr == 4 AND abs(score_differential) > 21)
              AND NOT (qtr == 4 AND game_seconds_remaining < 120 AND abs(score_differential) > 8)
```

This matters more than people expect. Unfiltered EPA rewards teams for running up the
score on backups and punishes teams for prevent defense, and both are noise.

**Team abbreviation normalization.** nflverse changed some abbreviations over time
(OAK→LV, SD→LAC, STL→LA). Write `normalize_team()` and apply it to every team column in
every table at ingest. Do not defer this; it silently corrupts joins.

---

## 5. RATINGS — `src/ratings.py`

This is the core. Two ridge regressions.

### 5a. Efficiency ridge (offense/defense strength)

**Unit of observation:** one row per (game, offensive unit). A 16-game week produces
32 rows.

**Response** `y`: mean EPA per play for that offense in that game, on `competitive`
plays only, excluding special teams (`special == 0`) and excluding plays nullified by
penalty where no yardage occurred.

**Weight** `w`: `n_competitive_plays * decay`, where
`decay = 0.5 ** (games_ago / half_life_games)` and `games_ago` counts that team's own
games back from the `as_of` cutoff.

**Design matrix** `X`, sparse:

- 32 columns `off_<TEAM>` — 1 if that team's offense is the unit in this row
- 32 columns `def_<TEAM>` — 1 if that team's defense is the opposing unit
- 1 column `home_offense` — 1 if the offensive unit is the home team, 0 if away,
  0.5 handling not needed (neutral sites get 0.5 on both; see below)
- intercept (unpenalized)

Neutral-site games (`location == "Neutral"`): set `home_offense = 0.5` for both rows.

**Objective:**

```
minimize  Σ w_i (y_i - x_i·β)²  +  λ_off Σ_j β_off,j²  +  λ_def Σ_k β_def,k²
```

The intercept and `home_offense` coefficient are **not** penalized.

**Implementation:** do not use `sklearn.Ridge` — it applies a single penalty to all
columns. Solve the weighted normal equations directly:

```python
# X: scipy.sparse csr, shape (n, p); w: (n,); y: (n,)
W = sparse.diags(w)
A = (X.T @ W @ X).toarray() + np.diag(penalty_vector)   # penalty_vector is per-column
b = X.T @ W @ y
beta = np.linalg.solve(A, b)
```

`penalty_vector` is `lambda_off` for offense columns, `lambda_def` for defense columns,
`0.0` for intercept and `home_offense`.

**Prior injection.** Before solving, append synthetic rows encoding preseason priors:

- For each team `t`: one row with `off_t = 1` only, `y = prior_off_t`,
  `w = prior_weight_games * ratings.league_mean_plays_per_game`
- Same for defense.

`prior_off_t` = last season's final offensive rating, regressed toward zero by
`offseason_regression`. For an expansion/relocated case or missing prior, use 0.

This is the clean Bayesian way to handle Week 1 through Week 5, and it is why the model
does not produce garbage in September. The prior decays naturally as real games
accumulate weight.

**Output:** a DataFrame indexed by team with columns `off_rating`, `def_rating` (both in
EPA/play units, centered on 0), plus the fitted `hfa_epa` coefficient.

**Sign convention — be explicit and assert it.** Higher `off_rating` = better offense.
Higher `def_rating` = **worse** defense (because it is the coefficient on "points/EPA
allowed"). Add a unit test that the previous season's best defense has the lowest
`def_rating`. This sign flip is the single most common bug in this kind of build.

**Net strength for a matchup:**

```
net_A = off_weight * off_A - def_weight * def_B
```

with `off_weight: 1.6`, `def_weight: 1.0` from config. That asymmetry is not arbitrary —
offensive efficiency is meaningfully stickier week to week than defensive efficiency, so
weighting offense more heavily produces better forecasts of future net EPA.

### 5b. Pace ridge

Same machinery, different response: `drives_by_that_offense_in_that_game`, weight =
`decay` only, single penalty `pace.lambda`. Output `pace_rating` per team such that
`E[total drives per team] = league_mean + pace_A + pace_B`.

### 5c. Lambda tuning

Do not ship the seed values. Write `tune_lambdas()` that runs a walk-forward grid search
over `lambda_grid` (cross of off and def).

**Scoring objective — PATCHED (PATCH 02 §3). Tune on held-out `t(b_model)`, not margin
RMSE.** The original text said to minimize out-of-sample margin RMSE. That objective is
wrong for a betting model and actively works against you: actual margin is ~13 points of
mostly irreducible noise around the true spread, so minimizing RMSE against it mostly
rewards getting close to the market's number. Optimize it hard enough and you converge on
an expensive market clone with extra variance.

```
objective = t-statistic of b_model in the residual regression, on HELD-OUT seasons
```

**Caveat that limits what this can buy you.** Per PATCH 02 §3, scale-only changes are
inert — shrinkage rescales the disagreement and leaves `t` exactly invariant. Tuning helps
only for hyperparameters that *re-rank* which games the model disagrees with the market
about (`half_life_games`, `prior_weight_games`, `offseason_regression`, the off/def ratio).
Pure λ changes mostly are not that.

**Overfitting guard, mandatory.** Tune on the earlier seasons, validate on the last two,
report both. The held-out `t` must clear 2.0 independently; in-sample `t` is not evidence.
**Pre-register the grid at ≤ 12 cells.** A 720-cell grid under a true null yields ~36 cells
at |t| > 2 by chance and a 52.5% probability of at least one false discovery, against a
Bonferroni threshold of |t| > 3.99. Searching a large grid after a null result cannot
confirm the null and can only manufacture a false positive. Write the winners back into
`config/nfl.yaml` under a `tuned:` block and print a table of the grid. This takes a few
minutes; cache the result.

---

## 5.5 ARCHITECTURE — THE L1/L2 SPLIT (PATCH 01 §2, PROMOTED)

**The simulator does not produce the mean.** This supersedes any reading of §6 in which the
simulated mean margin is the model's spread.

The original design welded two incompatible jobs together, so failing at one blocked the
other:

- **(a) Predict `E[margin]`.** A regression problem. A direct linear projection off the
  ridge ratings does it better than the same ratings pushed through a drive simulator.
- **(b) Produce the distribution shape** around that mean — key numbers, totals, the
  margin/total joint. No regression gives you this; it is what a simulator is *for*.

**Measured, not asserted.** On 1,535 NFL games the direct projection scored 13.529 RMSE
against the simulator's 13.882, and every bit of the simulator's excess dispersion was
noise it introduced itself: model SD 8.148 against a market SD of 6.419 implies
`sqrt(8.148² − 6.419²) = 5.02` points of added noise, while the direct projection off the
*same ratings* had SD 5.19 and no excess at all. A projection carrying 5 points of noise
cannot resolve the 1.5–2.5 point disagreements the bet thresholds are hunting for.

**L1 — mean layer.** `E[margin]` and `E[total]` from an ensemble, fit by non-negative least
squares in residual form with a free intercept and the market coefficient at 1.0.
Components: direct ridge projection, Elo, nfelo/SP+ where available, market always.
Rescale every component to market SD about its own mean before it enters. Fit spreads and
totals separately.

**L2 — simulator.** Totals shape, the margin/total joint dependence (the copula), factor
sensitivities, and the ordering of outcomes. Never the mean.

**L3 — calibration.** A rank-preserving quantile map onto the empirical conditional margin
distribution, bucketed by the *blended* line. Built from games strictly before `as_of`;
the no-lookahead assertion covers it. This is what makes `GATE_KEY_NUMBERS` reachable —
mechanically, rather than by refining the simulator, which cannot get there (see §6d).

---

## 6. DRIVE MODEL & SIMULATION — `src/drive_model.py`, `src/simulate.py`

### 6a. Build the drive table — `src/drives.py`

From PBP, produce one row per drive with:

```
game_id, season, week, offense, defense, drive_number, start_yardline_100,
n_plays, result, points_scored_on_drive, is_home_offense
```

Map `fixed_drive_result` into four classes:

- `TD` — offensive touchdown
- `FG` — successful field goal
- `NO_SCORE` — punt, downs, missed FG, end of half, end of game
- `TURNOVER` — interception or fumble lost

Handle defensive/return TDs and safeties **outside** the drive multinomial as separate
Poisson processes (config `defensive_td_lambda`, etc.). Folding them into the drive
outcomes makes the multinomial unstable because they are rare and belong to the other
team.

### 6b. Fit the multinomial — `src/drive_model.py`

```python
def fit_drive_model(drives: pd.DataFrame, ratings: pd.DataFrame) -> MultinomialModel
```

Multinomial logistic regression (`sklearn.linear_model.LogisticRegression`,
`multi_class='multinomial'`, `solver='lbfgs'`, `C` tuned lightly), predicting the
4-class drive outcome from:

```
net_epa        = off_weight*off_rating(offense) - def_weight*def_rating(defense)
start_fp       = start_yardline_100 (yards to opponent end zone)
start_fp_sq    = start_fp ** 2
is_home_offense
net_epa_x_fp   = net_epa * (start_fp / 100)
```

Fit on `train_start..as_of`. Five predictors, four classes. Keep it this small — this is
a calibration layer, not the place to add features.

**Starting field position** is itself sampled in the simulator, not taken from the
matchup. Fit an empirical distribution of `start_yardline_100` from historical drives,
conditioned on nothing but whether the previous drive ended in a turnover. Store it as a
lookup with two discrete distributions: `post_turnover` and `normal`. Sample from it.

### 6c. The simulator — `src/simulate.py`

```python
def simulate_game(
    home: str, away: str,
    ratings: pd.DataFrame,
    drive_model: MultinomialModel,
    context_adj: ContextAdjustment,
    cfg: Config,
) -> SimResult
```

Algorithm, per simulation `s` in `1..n_sims`:

1. Draw total drives per team:
   `d = round(clip(N(mu, sd), 8, 16))` where
   `mu = pace.league_mean_drives_per_team + pace_home + pace_away` and
   `sd = pace.drives_sd`. Both teams get `d`. Odd-possession games are real, so draw
   whether an extra drive exists and then assign it to either team with equal
   probability — **PATCHED (PATCH 01 §0, Bug B).** The original text said "with
   probability 0.5 add one drive to the home team", which hands the home side +0.5
   expected drives *every game*, worth about a point of phantom home-field advantage.

   ```python
   extra   = rng.random(n_sims) < 0.5     # does an odd drive exist at all
   to_home = rng.random(n_sims) < 0.5     # who gets it
   home_drives = d + (extra & to_home)
   away_drives = d + (extra & ~to_home)
   ```
2. For each team, for each of its `d` drives:
   - Sample starting field position from the empirical distribution (use `post_turnover`
     if the opponent's immediately preceding drive in this sim ended in `TURNOVER`).
   - Compute the multinomial probabilities from `net_epa` (with context adjustments
     already folded in — see below) and the sampled field position.
   - Draw the outcome. Points: `TD` → 6 plus a conversion sub-draw
     (`two_point_attempt_rate` → succeed at `two_point_success_prob` for 2, else kick and
     succeed at `xp_make_prob` for 1); `FG` → 3; otherwise 0.
3. Add rare events: `Poisson(defensive_td_lambda)` defensive TDs for each team (7 each),
   `Poisson(special_teams_td_lambda)` return TDs (7 each), `Poisson(safety_lambda)`
   safeties (2 to the opposing team).
4. Record `home_score`, `away_score`.

**Vectorize.** Do not write a Python loop over 20,000 sims. Build it as numpy arrays of
shape `(n_sims, max_drives)` and draw all outcomes at once with
`np.random.default_rng(seed).multinomial` / inverse-CDF sampling on a cumulative
probability matrix. A vectorized version runs 20,000 sims in well under a second; a
looped one takes minutes and you will stop rerunning it.

**Context adjustments.** `context_adj` supplies a points-scale adjustment (HFA, rest,
travel, weather, QB). Convert points to EPA-per-drive scale and apply it to `net_epa`
before the multinomial, rather than adding points to the final score. Adding points post
hoc breaks the key-number structure — it shifts a lumpy distribution by a continuous
amount and smears the spikes at 3 and 7, which defeats the entire purpose of simulating
drives. Use the conversion `epa_per_drive = points / expected_drives`.

**`SimResult`** exposes:

```python
margins: np.ndarray        # home - away, shape (n_sims,)
totals: np.ndarray
mean_margin: float         # this is your model spread (negated for display convention)
median_margin: float
mean_total: float
def cover_prob(self, line: float, side: str) -> float
def total_prob(self, line: float, side: str) -> float
def margin_pmf(self) -> dict[int, float]
```

`cover_prob` must handle pushes correctly: at a whole number, a push is neither a win nor
a loss, so return `P(win) / (P(win) + P(loss))` — the probability conditional on the bet
resolving. Getting this wrong systematically overstates edge on whole-number spreads,
which are exactly the lines sitting on 3 and 7.

### 6d. Simulator validation gate

Write `validate_simulator()` in `tests/test_simulate.py`. Simulate every historical game
from `backtest_start` onward using true (in-sample is fine here — this tests the
*distribution shape*, not predictive power) ratings, pool all simulated margins, and
compare the pooled distribution to the actual historical margin distribution.

**Gate:** the simulated frequency of margin == 3 must land within 2 percentage points of
the historical rate (roughly 14-15%), and margin == 7 within 2 points of roughly 9%. Also
check 6, 10, 14, and 4. Print a side-by-side table.

If the simulator does not reproduce the key-number spikes, stop and fix it before going
further. Every downstream number depends on this being right.

---

## 7. MARKET — `src/market.py`

### 7a. Devig

```python
def devig_two_way(price_a: int, price_b: int, method: str) -> tuple[float, float]
```

Implement three methods; default to `multiplicative` for spreads and totals.

- **multiplicative** — convert American to implied probability, divide each by the sum.
- **power** — solve for `k` such that `Σ p_i^k = 1` via `scipy.optimize.brentq`.
- **shin** — solve for the informed-money proportion `z` iteratively.

Why multiplicative is the default here: spreads and totals are priced near -110/-110, so
the book sum is only about 1.048 and all three methods converge to within a fraction of a
percent. The methods diverge on lopsided markets where favorite-longshot bias is real —
which is why `power` and `shin` are still built, for use on moneylines with heavy
favorites. Do not use multiplicative on a -2000 moneyline.

### 7b. Line acquisition

Two paths. Implement both; `run_week.py` tries live first, falls back silently.

**Live:** The Odds API, `americanfootball_nfl`, `markets=h2h,spreads,totals`,
`regions=us`. Store the API key in an env var `ODDS_API_KEY`. Cache the response with a
timestamp. If the key is absent or the call fails, fall through.

**Manual:** read `data/manual/this_week_lines.csv` with columns
`away_team,home_team,spread_home,total,spread_price_home,spread_price_away,total_price_over,total_price_under`.
Generate a pre-filled template of this file (teams and matchups from the schedule, blank
numbers) whenever it is missing, so the fallback is one paste away.

### 7c. The blend — the single most important function in the build

```python
def fit_blend_weights(backtest_frame: pd.DataFrame) -> BlendWeights
```

Input is the out-of-sample backtest output: one row per historical game with
`model_spread`, `nfelo_spread`, `market_spread` (the closing `spread_line` from
schedules), and `actual_margin`.

**Fit in residual form, not levels, WITH A FREE INTERCEPT — PATCHED (PATCH 02 §1):**

```
actual_margin - market_spread = a
                              + b_model * (model_spread - market_spread)
                              + b_nfelo * (nfelo_spread - market_spread)
```

Fit by OLS with the `market_spread` coefficient constrained to 1.0 (subtract it from the
response and regress the remainder on the disagreement terms) **and a free intercept `a`**.

**The intercept is not optional.** The original text omitted it. With the market
coefficient pinned at 1.0 and no intercept, any constant level bias in a component loads
onto its disagreement slope. This was demonstrated on live data: a totals projection
running +1.43 points high presented as `b = +0.060`, and de-biasing flipped it to −0.050 —
the apparent positive weight was entirely the level error, not game-level signal.

**Report `a` alongside every `b_k`.** A significant `a` means the component is
systematically biased and must be de-biased before its `b` means anything.

**Rescaling, if you rescale at all — PATCHED (PATCH 02 §1).** Rescale the *component*
about its own mean:

```python
component_rescaled = component.mean() + (component - component.mean()) * (sd_market / sd_component)
```

Rescaling the *disagreement* term instead is a scalar multiple: `b` scales by `1/c`, `se`
scales by `1/c`, and **`t` is exactly invariant**. Verified — 1.0x, 0.5x and 2.0x all
return the identical t-statistic. Only rescaling the component changes which part of the
disagreement survives. The corollary matters strategically: **shrinkage alone can never
change significance.** Only changes that re-rank *which games* the model disagrees with
the market about can move `t`.

**Why residual form and not levels.** Be precise about what this buys you, because it is
not what people usually claim. With the market coefficient left free, levels and residual
form are algebraically equivalent and give the *same point estimates*. The gains are
elsewhere, and they are real:

- **Standard errors.** `model_spread` and `market_spread` typically correlate above 0.85.
  Differencing against the market drops the correlation between regressors to roughly 0.2,
  which is what makes the t-statistics in `GATE_BLEND_INFORMATIVE` trustworthy rather than
  noise. This matters more once nfelo is a third term, because nfelo regresses to market
  by design and is therefore collinear with it.
- **Constraining the market coefficient to 1.0**, which is only expressible in residual
  form, and which is the right prior — the market is close to unbiased, so you want to
  estimate *deviations* from it, not re-estimate it.
- **Interpretability.** Each regressor is the part of a model that *disagrees* with the
  market, which is the only part that can possibly add value.

The coefficients are also directly interpretable: `b_model = 0.30` means "when my model
disagrees with the market by 3 points, about 0.9 points of that disagreement is real and
the rest is my error." That is a shrinkage factor on your edge, and it is exactly what you
need to size a bet.

**Application:**

```
blended_spread = market_spread
               + b_model * (model_spread - market_spread)
               + b_nfelo * (nfelo_spread - market_spread)
```

Report both coefficients, their standard errors and t-statistics, the R², and the residual
standard deviation. Clamp each coefficient to `[0, model_weight_cap]` — a negative fitted
weight means the model is anti-predictive on that sample, which is almost always a sign
error or a lookahead leak rather than a real finding. Investigate it, do not use it.

**Reading the two coefficients together** is the most informative diagnostic in the build:

| `b_model` | `b_nfelo` | What it means |
|---|---|---|
| significant | significant | Both carry independent signal. Use both. Best case. |
| ~0 | significant | Your model adds nothing nfelo does not already have. Bet nfelo, and go find out what it is doing that you are not. |
| significant | ~0 | Your model has signal nfelo lacks. Genuinely good, and worth understanding why before trusting it. |
| ~0 | ~0 | Neither beats the market on this sample. Emit zero bets. |

Run the same regression separately for totals. Note that nfelo does not publish a totals
model, so the totals blend has only the `b_model` term.

**Interpretation, and the hard gate:** `b_model` is a direct measurement of how much
independent information your model carries beyond the market. If `b_model` is not
statistically distinguishable from zero, and neither is `b_nfelo`, the
system must print:

```
*** NO EDGE DETECTED — model adds no information beyond the market. Emitting zero bets. ***
```

and write an empty bet sheet. Do not soften this. A model that says "no bets" is working
correctly; a model that always finds bets is broken.

Run the same regression separately for totals.

### 7d. Key-number awareness

```python
def key_number_value(model_line: float, market_line: float, pmf: dict) -> float
```

Using the simulated margin PMF, compute how much of the edge comes from crossing a key
number versus from a smooth shift. An edge of 1.5 points that moves you from -3.5 to -2
is worth substantially more than an edge of 1.5 that moves you from -8.5 to -7 is worth
less than one that moves -7.5 to -6. Surface this as a `key_cross` column in the output
so you can see it, and use it as a tiebreaker in ranking, not as a multiplier on stake.

---

---

## 7.5 NFELO INTEGRATION — `src/nfelo.py`

nfelo is an open-source NFL Elo model published at nfeloapp.com with all code on GitHub.
It is well built, it has a public backtest going back to 2009, and its picks are
independently tracked by PredictionTracker.com. It is used here in three distinct ways,
and the distinction between them matters a great deal.

### Use 1 — `qb_elos.csv` as a genuine model INPUT (do this)

**Source:** `https://raw.githubusercontent.com/greerreNFL/nfeloqb/main/qb_elos.csv`
Repo: https://github.com/greerreNFL/nfeloqb

`nfeloqb` recreates and maintains 538's QB Elo model, in 538's original CSV schema, with
documented improvements to seasonal regression, rookie initial values, and opponent
defensive adjustment. The file is refreshed **every Tuesday and Thursday morning during
the season**. Seasons through 2022 use 538's original model; 2023 onward uses nfeloqb's.

Relevant columns: `season`, `date`, `team1`, `team2`, `qb1`, `qb2`,
`qb1_value_pre`, `qb2_value_pre`, `qb1_adj`, `qb2_adj`, `qbelo1_pre`, `qbelo2_pre`.

**`qb1_adj` / `qb2_adj` are already a points-scale adjustment** for that starting QB
relative to the team's baseline. That is precisely what §12's QB module is trying to
compute, from a maintained source with a decade of history behind it.

This is a legitimate input rather than wrapper-ism because it is a **QB valuation model,
not a game prediction model** — it does not consume betting market data, so folding it in
does not double-count the market. Use it as the primary QB signal, keep your own
EPA+CPOE composite as a cross-check, and log any game where the two disagree by more than
3 points.

Simple HTTP GET, cache to parquet, refresh weekly. No key, no package needed.

### Use 2 — nfelo's game spread as a third term in the BLEND (do this, carefully)

Pull nfelo's projected spread for each game and feed it into `fit_blend_weights` as
`nfelo_spread` per §7c.

**Do not feed it into your ratings.** A model that consumes nfelo's line inside its own
rating system is a wrapper around nfelo, and its only edge is the lag between nfelo
publishing and the market absorbing it — which is close to zero, because nfelo is public
and widely read.

The blend is different, and it is the correct place for this. In residual form, `b_nfelo`
measures nfelo's disagreement with the market as an independent regressor. The OLS fit
decides how much it earns. If it earns a lot and your model earns nothing, that is real
information about your model, and you should want to know it.

**One caution you must handle.** nfelo itself regresses to market spreads — its author has
written about this explicitly as a design choice. So `nfelo_spread` is already partly a
market number. This is exactly why §7c uses residual form: `(nfelo_spread -
market_spread)` strips out the shared market component and leaves only nfelo's independent
view. If you regress on levels instead, the collinearity will make both coefficients
meaningless. Do not skip this.

**Getting the numbers.** The main model at https://github.com/greerreNFL/nfelo is open
source and can be run locally, which is the reliable path for building a *historical*
`nfelo_spread` series for the backtest. `nfelodcm` (`pip install nfelodcm`) is the
author's data-loading layer and is the right tool for assembling the inputs. Budget real
time for this — it is the most involved piece of the whole build. If running the full
model locally proves impractical, degrade gracefully: set `b_nfelo` to zero, run the
two-term blend, and note it in `DECISIONS.md`. **The build must not block on this.**

### Use 3 — methodology and benchmark (already partly done)

nfelo's published research is the best free writing on the specific problems this build
faces, and two of its findings are already baked into this spec:

- The **1.6 / 1.0 offense-to-defense weighting** in `ratings.off_weight` comes from
  nfelo's finding that offensive EPA is stickier than defensive EPA and that this ratio is
  near-optimal for predicting future net EPA.
- The **key-number-aware margin distribution** approach in §6 follows the same reasoning
  as nfelo's margin-probability work: a plain normal underfits 3 and 7 and misprices every
  line near them.

Also read their home-field-advantage tracker, which is the source of the 1.7-point default
rather than the stale 3.0.

Use nfelo as a **formal baseline** in the backtest table alongside your own Elo. If your
model's spread RMSE is worse than nfelo's, you have not built something worth betting —
you have built a worse version of a free public model, and the correct action is to bet
nfelo's numbers while you improve yours.

### What nfelo does not give you

- **No college football.** nfelo is NFL-only. The NCAA playbook's analog is SP+, used the
  same way — third term in the blend, never an input to ratings.
- **No totals model.** Your totals blend has one term.
- Attribution: credit nfelo in your README. It is free, open source, and it saved you
  weeks.

---

## 8. BACKTEST — `src/backtest.py`

The heart of the acceptance process.

```python
def walk_forward(cfg: Config) -> pd.DataFrame
```

For each season from `backtest_start` to `current - 1`, for each week `w` from 4 to the
end of the regular season:

1. Set `as_of` = kickoff of the first game in week `w`.
2. Fit ratings, pace, and the drive model using **only** data before `as_of`.
   Fit the drive model less often than weekly — refit it once per season — since it is a
   slowly-varying calibration layer and refitting weekly is wasted compute.
3. Simulate every game in week `w`.
4. Record: `model_spread`, `model_total`, `market_spread`, `market_total`,
   `actual_margin`, `actual_total`, plus all context flags.

**Which market line to grade against — PATCHED (PATCH 03 §4).** Grade against the
**opening** line where one is available, and retain the close separately for CLV. The
original text graded everything against `schedules.spread_line`, which is the *closing*
number — the aggregate of every public model plus professional money, and the hardest
benchmark in American sports betting. `00_START_HERE.md` had always said the achievable
edge is against the opener; §8 asked for the wrong test.

This is unfixable for NFL on free data: nflverse publishes closing lines only, so the test
this model was designed to pass was never run. It is fixable for college —
CollegeFootballData carries `spread_open` and `over_under_open` on the free tier — which is
a concrete, material difference between the two markets rather than a consolation.

Weeks 1-3 are excluded from backtest scoring because the prior dominates and the sample
is not representative. The model still *produces* numbers for weeks 1-3 in live use — it
just should not be graded on them, and the bet threshold is raised for them (see §9).

### 8a. Metrics to compute and print

| Metric | Model | Blended | Market | nfelo | Elo baseline |
|---|---|---|---|---|---|
| Spread MAE | | | | | |
| Spread RMSE | | | | | |
| Total MAE | | | | — | |
| Total RMSE | | | | — | |
| ATS record, all games at model side | | | — | | |
| ATS record, filtered bets | | | — | | |
| Brier score of cover probability | | | | | |

Also produce:

- **Calibration plot data** — bucket predicted cover probability into 5% bins, compute
  realized cover rate per bin, and print the table. A well-calibrated model has realized
  ≈ predicted along the diagonal. Systematic overconfidence here is the most common cause
  of a backtest that looks good and a live account that bleeds.
- **Bootstrap CI on filtered ATS%** — resample the filtered bet set 10,000 times with
  replacement and report the 5th and 95th percentile of win rate. If the 5th percentile
  is below 50%, the "edge" is not distinguishable from noise at that sample size, and the
  report must say so in plain language.
- **CLV proxy** — where opening lines are available, grade `model_spread` against the
  closing line: what fraction of the time did the market move toward the model's side.
  This is the most informative single number in the whole report.

### 8b. Hard acceptance gates

The build is not done until all of these print PASS:

1. `GATE_KEY_NUMBERS` — simulated margin PMF matches historical at 3, 7, 6, 10, 14, 4
   within 2 percentage points each.
2. `GATE_NO_LOOKAHEAD` — the assertion in `tests/test_no_lookahead.py` passes: for a
   random sample of 200 backtest rows, no game used in fitting has a kickoff timestamp
   >= the predicted game's kickoff.
3. `GATE_BEATS_ELO` — model spread RMSE is lower than your own Elo baseline's RMSE.
3b. `GATE_VS_NFELO` — model spread RMSE reported alongside nfelo's. Not a hard fail if
   nfelo wins, but if it does, the honest read is to bet nfelo's numbers until yours
   improve. Print this comparison prominently; do not bury it.
4. `GATE_BLEND_INFORMATIVE` — at least one of `b_model` or `b_nfelo` from §7c has a
   t-statistic above 2.0. If only `b_nfelo` clears it, the system still emits bets, but
   the report must state plainly that the edge is nfelo's and not yours.
5. `GATE_CALIBRATED` — no calibration bucket with 100+ observations is off by more than
   6 percentage points.
6. `GATE_SPEED` — a full weekly refresh on cached data completes in under 180 seconds.
7. `GATE_UNBIASED` — **NEW (PATCH 02 §1).** `|a| < 0.5` points for every component, spreads
   and totals, where `a` is the free intercept of the residual regression. A component that
   fails this is systematically biased, and its `b` is uninterpretable until it is
   de-biased.
8. `GATE_SCALE` — **NEW (PATCH 01 §4, revised by PATCH 02 §2).** SD ratio of every L1
   component against the market in `[0.85, 1.15]`, targeting 1.05–1.10.

If a gate fails, print exactly which one and what the observed value was. Do not proceed
to generating a bet sheet with a failed gate; write the diagnostic report instead.

---

## 9. BET SELECTION — `src/betting.py`

```python
def build_bet_sheet(games: list[GameProjection], cfg: Config) -> pd.DataFrame
```

For each game, for each of four candidate bets (home spread, away spread, over, under):

1. Blend in **residual form**, per §7c — do not use a weighted average of levels:
   ```
   blended_spread = market_spread
                  + b_model * (model_spread  - market_spread)
                  + b_nfelo * (nfelo_spread  - market_spread)
   blended_total  = market_total
                  + b_model_total * (model_total - market_total)
   ```
   `b_*` come from the most recent backtest artifact. There is no `w` in this build.
2. Re-center the simulated margin distribution on `blended_spread` — shift the array,
   do not re-simulate. Preserves the key-number shape.
3. `p = cover_prob(market_line, side)` with push handling.
4. `edge_points = blended - market`.
5. Bet only if **all** of:
   - `p >= min_cover_prob`
   - `abs(edge_points) >= min_edge_points_spread` (or `_total`)
   - the game is not flagged `data_incomplete` (missing QB, missing weather for an
     outdoor game with no forecast, a team on its first game of the season)
6. Stake: quarter-Kelly.
   `b = decimal_odds - 1`; `f = (p*b - (1-p)) / b`; `stake = min(kelly_fraction * f,
   max_stake_pct_bankroll) * bankroll`. Floor at 0.

**Week 1-4 handling:** multiply `min_edge_points_*` by 1.6 and `min_cover_prob` by adding
0.015. Early-season ratings lean on the prior and deserve a higher bar.

Output columns, in this order:

```
week, kickoff, away, home, market_spread, model_spread, blended_spread, edge_pts,
market_total, model_total, blended_total, edge_total, pick, pick_line, cover_prob,
key_cross, stake, kelly_pct, bet_to_line, confidence, notes
```

`bet_to_line` is the worst number at which the bet is still above threshold — i.e. "bet
this down to -4.5, no further." That is the column you will actually use when shopping
your app, so make sure it is computed and not left blank.

`confidence` is `HIGH / MED / NONE`, driven by cover probability bands, not by vibes.

---

## 10. REPORT — `src/report.py`

Write an Excel workbook with `xlsxwriter`. Four sheets:

**Sheet 1 — `Bets`.** Only rows with a live pick, sorted by cover probability descending.
Freeze the header row. Conditional formatting: green fill scaling with cover prob,
bold on `pick_line` and `bet_to_line`. Number formats: spreads to one decimal,
probabilities as percentages, stake as currency. Column widths set explicitly.

**Sheet 2 — `All Games`.** Every game on the slate, including NO BET rows, greyed out.
This is how you see the model's selectivity and sanity-check numbers it declined to bet.

**Sheet 3 — `Ratings`.** Every team: `off_rating`, `def_rating`, `net_rating`,
`pace_rating`, implied points per drive, rank in each, and the change from last week.
Sort by net rating. This is the sheet you will look at most often to sanity check whether
the model believes something absurd.

**Sheet 4 — `Backtest`.** Latest metrics table, calibration table, bootstrap CI, and the
gate results with PASS/FAIL. Regenerate on every backtest run so the workbook always
carries its own evidence.

Also write a one-line-per-bet CSV at `output/bets_{season}_{week}.csv` for logging.

---

## 11. STREAMLIT APP — `app.py`

Minimal. Four things and nothing else:

1. Week selector, defaulting to the current week.
2. The `Bets` table, rendered with the same formatting.
3. A per-game expander: pick a game, see the simulated margin histogram with the market
   line and the model line drawn as vertical rules, plus the total histogram. This
   visual is the fastest way to spot a broken projection.
4. A "what if" slider that lets you shift one team's rating by ±3 points and see the
   line move — useful for pricing an injury the model does not know about.

Do not build authentication, a database, or a deployment pipeline. This runs locally with
`streamlit run app.py`.

---

## 12. QB MODULE — `src/qb.py`

The largest single source of NFL model error is a quarterback change the ratings have not
absorbed. Handle it explicitly.

**Primary signal: `qb_elos.csv` from nfeloqb** (see §7.5). Its `qb1_adj` / `qb2_adj`
columns are already a points-scale adjustment for the starting QB relative to the team's
baseline, maintained back through the 538 era and refreshed twice weekly in season. Use
it as the main input. Steps 1-2 below build your own independent composite — keep it, but
as a cross-check rather than the primary, and log any game where the two disagree by more
than 3 points. Those disagreements are worth looking at; they are usually a depth-chart
read gone wrong on one side or the other.

1. For each team, identify the QB with the most dropbacks inside the rating window
   (the "embedded" QB).
2. Compute each QB's personal composite from PBP: a weighted combination of EPA per
   dropback and CPOE, regressed toward the positional mean by `regression_dropbacks`.
3. **Auto-detect the projected starter, then let the user override.** Do not make this
   fully manual — nflverse carries the data:
   - `nfl.load_depth_charts(seasons)` gives the weekly depth chart; take the QB1.
   - `nfl.load_injuries(seasons)` gives the official injury report back to 2009 with
     `report_status` (Out / Doubtful / Questionable) and `practice_status`.
   - If QB1's status is `Out` or `Doubtful`, promote QB2.
   - Write the result to `data/manual/qb_starters.csv`
     (`team,week,starter_id,starter_name,source`) as a **pre-filled default**, with
     `source` recording whether the pick came from the depth chart, the injury report, or
     a manual edit.
   - Never overwrite a row whose `source` is `manual`.

   Keep the manual override path. Injury reports are strategically vague, "Questionable"
   resolves both ways, and beat reporters know before the report does. Automation gets
   you most of the way; the override covers the cases where the money is.
4. If projected starter ≠ embedded QB, adjust the team's offensive rating by
   `(composite_new - composite_embedded)` converted to EPA/play, then to points, capped at
   `max_adjustment_points`.
5. If the projected starter has fewer than `min_dropbacks` career dropbacks, use the
   replacement-level prior rather than his tiny sample — a rookie with 40 great dropbacks
   is not an elite QB and the model must not think so.

Write the applied adjustment into the `notes` column of the bet sheet so you can see it.

---

## 13. CONTEXT MODULE — `src/context.py`

Returns a `ContextAdjustment` in points, home perspective.

- **HFA — PATCHED (PATCH 01 §0, Bug A).** The league-average home-field advantage is
  supplied by the drive multinomial's `is_home_offense` term (§6b), which is estimated
  from data at the drive level. **Do not also add a league HFA here** — the original text
  did, and running both produced a ≈4.4-point home edge against a market that prices it
  near 1.7. What this module contributes is only a **per-venue deviation from the
  multinomial's fitted league HFA, centered on zero**: shrink each venue's mean residual
  toward the league mean by `hfa_venue_shrink_games`, then subtract the league mean. A
  test asserts the mean deviation across all 32 stadiums is within 0.1 of zero. Neutral
  sites are handled inside the simulator by setting `is_home_offense = 0.5` on both sides,
  which removes the league baseline for a game with no home team.

  Estimate per-venue HFA from the residuals of the ratings
  model, not from raw home win rate. Neutral sites get 0. Do not use 3.0 — league-wide
  home-field advantage has been eroding for years and the market has been pricing it near
  1.5-2.0 for a while now; a stale 3.0 makes every home team look like value and will
  generate a season of losing bets.
- **Rest:** `rest_point_per_day * (home_rest - away_rest)`, capped at `rest_max_points`.
  Add `short_week_penalty` for a team on 4 days.
- **Travel:** great-circle distance between stadium coordinates from `src/stadiums.py` —
  a literal 32-entry dict of `lat`, `lon`, `tz`, `roof`, `elev_ft`. Do not scrape it, do
  not add a geocoding dependency for 32 rows that change once a decade. Cross-check each
  `roof` value against `schedules.roof` for a recent game, since several franchises have
  changed venues. Scale by `travel_points_per_1000mi`; apply `timezone_cross_penalty` for
  a West Coast team in a 1pm ET kickoff. International games (London, Munich, Mexico City,
  São Paulo) are identified from `schedules.location` and treated as neutral sites with
  zero HFA — do not attempt to model them.
- **Weather — `src/weather.py`.** Two paths, and you need both:
  - *Historical (backtest):* `schedules.temp` and `schedules.wind` are already populated
    for completed **outdoor** games. Use them.
  - *Upcoming games:* those columns are **null** until the game is played. Fetch from
    Open-Meteo — free, no API key, no signup:
    `https://api.open-meteo.com/v1/forecast?latitude=&longitude=&hourly=temperature_2m,wind_speed_10m,precipitation&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=`
    Use the archive endpoint (`archive-api.open-meteo.com/v1/archive`) to backfill any
    historical nulls.
  - **Set the units explicitly.** The Open-Meteo default is Celsius and km/h. A 15 "mph"
    threshold silently compared against km/h will suppress the total on every calm
    afternoon and you will not notice for a month.
  - Take the hourly reading nearest kickoff, not the daily mean.
  - **Skip the call entirely when `roof` is not `outdoors`.** About a third of the slate
    is indoors and those calls are pure waste.
  - Wind above `wind_total_threshold_mph` subtracts from the total at
    `wind_total_points_per_mph_over` per mph over. Wind has essentially no effect on the
    spread — apply it to the total only. Domes get `dome_total_bump`.
  - Attribution: Open-Meteo is CC-BY 4.0. One line in the README.
- **Week 18:** a manual flag file `data/manual/resting_starters.csv`. Not automated;
  the information is qualitative and arrives late.

---

## 14. WEEKLY RUN — `run_week.py`

```
python run_week.py --season 2026 --week 5
python run_week.py --season 2026 --week 5 --refresh    # force network pull
```

Sequence:

1. Refresh PBP and schedules through the last completed week.
2. Fit ratings, pace as of now.
3. Load or refit the drive model (refit once per season).
4. Load QB starters, resting-starters flags.
5. Acquire market lines (live, then manual fallback).
6. Simulate every game on the slate.
7. Load blend weights from the most recent backtest artifact. If none exists, refuse to
   emit bets and tell the user to run `run_backtest.py` first.
8. Build the bet sheet, write Excel, write CSV.
9. Print a compact terminal summary: N games, N bets, total stake, top three by edge.

---

## 15. THINGS NOT TO BUILD

Explicitly out of scope. Do not add them, do not suggest them mid-build:

- Player props, parlays, teasers, live betting.
- Public betting percentage or reverse-line-movement features.
- Head-to-head history features.
- Any neural network. With ~285 games a season, a gradient-boosted tree on 40 features
  will backtest beautifully and lose money. The ridge-plus-simulator structure is chosen
  specifically because it has few enough parameters to be estimable from the data that
  exists.
- Automatic bet placement of any kind.
- A database. Parquet files on disk are correct at this scale.

---

## 16. DELIVERABLES CHECKLIST

Before declaring done, confirm each:

- [ ] `python run_backtest.py` runs end to end and prints all six gates.
- [ ] Implemented gates PASS, or the failures are printed with observed values and
      explained. **See `../GATES.md`.** Of the 9 gates named in this playbook, 7 are
      implemented; `GATE_UNBIASED` and `GATE_SCALE` were added to §8b by PATCH 02 but never
      built. This build is a recorded null, so the 4 failures are the expected outcome.
- [ ] `python run_week.py --season 2026 --week 1` produces an Excel workbook with four
      populated sheets.
- [ ] `streamlit run app.py` opens and renders the margin histogram for a selected game.
- [ ] `pytest` passes.
- [ ] `README.md` documents: setup, the two commands, where to put manual overrides, and
      how to read the `bet_to_line` column.
- [ ] `DECISIONS.md` lists every judgment call you made that this spec left open.
- [ ] The key-number comparison table is printed and pasted into `README.md` as evidence
      the simulator is calibrated.

---

## APPENDIX A — RESULTS OF THIS BUILD (2026-07-31): A NULL ON NFL SPREADS

Recorded so it is not rediscovered from scratch. **This build is complete with a correct
negative result, not a failed build.**

**The finding.** Three independent architectures were fit and graded on 1,535 walk-forward
games, 2019–2025, weeks 4+, against the closing line. None carries information beyond the
market.

| architecture | spread RMSE | SD | SD ratio | added noise | a | b | t(b) |
|---|---|---|---|---|---|---|---|
| ridge direct (no sim) | 13.529 | 5.19 | 0.808 | 0.00 | +0.044 | −0.107 | −1.40 |
| drive simulator | 13.882 | 8.15 | 1.269 | **5.02** | +0.102 | −0.044 | −0.74 |
| Elo | 13.208 | 6.26 | 0.976 | 0.00 | +0.103 | −0.018 | −0.20 |
| **market (benchmark)** | **12.692** | 6.42 | 1.000 | — | +0.100 | — | — |

Held-out, measured independently per period: nothing clears +2.0 anywhere, and 2024–25 is
negative for all three (Elo −1.65, ridge −3.20, sim −1.21). Joint fit of Elo and ridge
disagreements together: **R² = 0.002**. Totals were evaluated separately and are also null —
the apparent `b = +0.060` was entirely a +1.43-point level bias, which is what motivated the
free-intercept fix in §7c.

**Why more search would not have helped.** `se(b) ≈ 0.060`, so `b ≥ 0.12` was detectable at
t = 2. The measured value was −0.066. This was never a power problem; the effect is zero.
Ensembling could not rescue it either: at a disagreement correlation of 0.685, two
components buy 1.09× the effective independence of one, and two near-zero correlated
signals combine to zero.

**Transferable findings, all confirmed on data:**

1. **The simulator is the noise source in the mean path** — all 5.02 points of excess
   dispersion originate there, none in the ratings. This is what §5.5's L1/L2 split exists
   to fix.
2. **An independent-possession simulator cannot reproduce key numbers at any refinement.**
   Two independent draws from the *real* historical score distribution reproduce the real
   margin spread (sd 14.06 vs 14.16) but yield P(|margin| = 3) = 6.4% against an actual
   14.7%. The spikes are generated by teams optimizing against the scoreboard. Final-drive
   score-state conditioning plus overtime closed 6.16% → 10.19%, which is real and worth
   keeping; the rest requires the L3 quantile map.
3. **Four spec bugs were found and are patched into this file:** double-counted HFA (§13),
   the phantom home possession (§6c), the missing residual intercept (§7c), and the wrong
   tuning objective (§5c). A fifth, reading drive start off the kickoff row, was an
   implementation bug — `tests/test_ratings.py` now asserts mean drive start near own-27.

**The benchmark was the hardest one available.** Grading against the closing line was what
§8 asked for and it is the sharpest number in American sports. A from-scratch model on
public data failing to beat it is the expected outcome. The test the model was designed for
— beat the opener, then measure CLV — could not be run, because free historical NFL opening
lines do not exist.

**Do not resume tuning this against the same backtest.** Continued searching after a null
is how a false positive gets manufactured. The next well-posed test is college football
against opening lines; see `PATCH_03_DECISION.md` §5.
