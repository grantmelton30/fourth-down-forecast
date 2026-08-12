# REPO INVENTORY

Read-only inventory of `nfl-model/`, `ncaa-model/` and `shared/` as of 2026-08-04. No model
or backtest was run to produce this; figures come from committed source and cached
artifacts.

---

## 1. FEATURE DICTIONARY

### 1a. NCAA — features fed to the projection

Assembled in `ncaa-model/src/backtest.py::build_features` (lines 40–70).

| variable | definition | source |
|---|---|---|
| `h_off`, `a_off` | Home/away offensive rating, ridge coefficient in PPA-per-play. Higher = better offense. | `walkforward_ratings_default.parquet` ← CFBD `/plays.ppa` |
| `h_def`, `a_def` | Home/away defensive rating, PPA/play. **Higher = worse defense.** | same |
| `h_pace`, `a_pace` | Per-team pace; `E[drives] = league_mean + pace_A + pace_B` | same, from CFBD `/drives` |
| `net_diff` | `(1.45·h_off + 1.0·a_def) − (1.45·a_off + 1.0·h_def)` — spread predictor | derived |
| `eff_sum` | `(1.45·h_off + 1.0·a_def) + (1.45·a_off + 1.0·h_def)` — totals predictor | derived |
| `pace_sum` | `h_pace + a_pace` — totals predictor | derived |
| `is_home` | 0.0 at neutral site, else 1.0 | CFBD `/games.neutralSite` |

Upstream inputs to the ratings: `ppa_per_play`, `n_plays`, `drives`, `is_home_offense`,
`game_weight` (0.45 for FCS opponents), `kickoff`, `competitive` (garbage-time flag). None
line-derived.

Two predictors reach the spread model (`net_diff`, `is_home`); two reach the totals model
(`pace_sum`, `eff_sum`).

### 1b. NFL — features fed to the projection

Assembled in `nfl-model/src/simulate.py::simulate_game` and `src/context.py::build_context`.

| variable | definition | source |
|---|---|---|
| `off_rating`, `def_rating` | Ridge team strength, EPA/play; same sign convention as NCAA | `walkforward_ratings_default_2016_2026.parquet` ← nflverse PBP `epa` |
| `pace_rating` | Per-team drives-per-game deviation | same |
| `net_epa` | `1.6·off_A + 1.0·def_B` per offense | derived |
| `start_yardline_100` | Drive start, yards to opponent end zone | nflverse PBP |
| `is_home_offense` | 1/0, or 0.5 at neutral sites | nflverse `schedules.location` |
| `hfa_venue_deviation` | Per-venue home-field deviation, centred on zero, from ratings residuals | derived from `schedules.result` |
| `rest` | `0.12 × (home_rest − away_rest)`, capped ±1.0, plus short-week penalty | nflverse `schedules.home_rest/away_rest` |
| `travel` | Great-circle miles × 0.15/1000 | `src/stadiums.py` literal dict |
| `timezone` | 0.25 penalty, west→east early kickoff | `src/stadiums.py` |
| `wind_mph`, `temp_f` | Weather; totals only | `schedules.temp/wind`, Open-Meteo |
| `dome_bump` | +0.5 to total indoors | `schedules.roof` |
| `qb_points` | Starting-QB adjustment, capped ±7.0 | nfeloqb `qb_elos.csv` `qb1_adj`/`qb2_adj` |
| `start_score_diff`, `start_qtr`, `start_gsr` | Score state for the endgame table | nflverse PBP |

### 1c. LINE-DERIVED FEATURES

**NCAA — three places a betting line enters the shipped `model_total` / `model_spread`:**

| # | quantity | line used | file:line |
|---|---|---|---|
| 1 | Gain correction — projection rescaled so its SD matches the market's | `total_open` / `spread_open` | `src/backtest.py:108–114`; `_fit_gain:166–190` |
| 2 | Time-varying intercept — per-week `mean(projection − opener)` subtracted | `total_open` / `spread_open` | `src/backtest.py:128–132`; `_fit_week_bias:147–164` |
| 3 | The blend — final bet number anchored on the opener | `total_open` | `src/betting.py:37` |

**NFL — one place:**

| # | quantity | line used | file |
|---|---|---|---|
| 1 | The blend | `schedules.spread_line` (closing) | `src/market.py::apply_blend`, applied in `src/betting.py::build_bet_sheet` |

The NFL *projection* (`sim.mean_margin`, `sim.mean_total`) contains no line-derived input.
The only line reference in `nfl-model/src/simulate.py` is a docstring describing the sign
convention.

Line-derived quantities used for evaluation/selection rather than as model inputs:
`spread_open`, `spread_close`, `total_open`, `total_close`, `has_opener`, `line_move`,
`clv_positive`, `blended_total`, `edge_pts`, `cover_prob` (computed against the opener).

---

## 2. DOES THE PROJECTION TOUCH THE MARKET LINE?

**NCAA: yes, in three separate places. The shipped output is not a pure feature-based
projection.**

The raw fit is pure — features only, response is the actual outcome:

```python
# ncaa-model/src/backtest.py:94-96
Xt = np.column_stack([np.ones(len(train)), train["pace_sum"], train["eff_sum"]])
bt = np.linalg.lstsq(Xt, train["actual_total"].to_numpy(float), rcond=None)[0]
test["model_total"] = bt[0] + bt[1] * test["pace_sum"] + bt[2] * test["eff_sum"]
```

It is then modified twice, both against the opener, before anything downstream sees it:

```python
# ncaa-model/src/backtest.py:108-114  — GAIN, scaled to the opener's dispersion
for kind, col, mkt in (
    ("spread", "model_spread", "spread_open"),
    ("total", "model_total", "total_open"),
):
    g = _fit_gain(train, col, mkt, kind)
    test[f"gain_{kind}"] = g
    test[col] = test[col].mean() + (test[col] - test[col].mean()) * g
```

```python
# ncaa-model/src/backtest.py:186-190  — the gain IS the market/model SD ratio
pred_sd = float(np.std(X @ beta))
mkt_sd = float(ref[mkt].std())
if pred_sd <= 1e-9 or mkt_sd <= 1e-9:
    return 1.0
return float(np.clip(mkt_sd / pred_sd, 0.5, 2.0))
```

```python
# ncaa-model/src/backtest.py:128-132  — WEEK BIAS, measured against the opener
train_pred = _project(train, bs if kind == "spread" else bt, kind)
train_pred = train_pred.mean() + (train_pred - train_pred.mean()) * g
bias = _fit_week_bias(train, train_pred, mkt)
test[f"week_bias_{kind}"] = test["week"].map(bias).fillna(0.0)
test[col] = test[col] - test[f"week_bias_{kind}"]
```

```python
# ncaa-model/src/backtest.py:156-164  — the bias is a residual against the line
ok = pred.notna() & train[mkt].notna()
if int(ok.sum()) < 300:
    return {}
resid = (pred[ok] - train.loc[ok, mkt])          # mkt == "total_open"
global_bias = float(resid.mean())
grp = resid.groupby(train.loc[ok, "week"])
n, mean = grp.size(), grp.mean()
return ((n * mean + shrink * global_bias) / (n + shrink)).to_dict()
```

The column named `model_total` in `backtest_frame_default.parquet` is therefore
`f(features)` rescaled to the opener's SD and level-shifted by a per-week opener residual.

The number actually bet is an explicit blend on top of that:

```python
# ncaa-model/src/betting.py:37
return market_total + w.b_model_total * (model_total - market_total)
```

**Exact weighting:** `blended_total = total_open + b_model_total × (model_total −
total_open)`. `b_model_total` is fit by `src/market.py::fit_blend` in residual form with a
free intercept, clipped to `[0, model_weight_cap = 0.70]`. Last fitted values:
`b_model_total = +0.1952` (full 2021–2025 window), `+0.2413` (paper-period training split).
`b_model_spread = +0.0421`; spreads are not emitted.

**NFL: the projection is pure feature-based; the bet number is a blend.**
`simulate_game` consumes only ratings and context and returns `mean_margin` / `mean_total`
with no market input. The line enters once, in
`nfl-model/src/betting.py::build_bet_sheet` → `market.apply_blend`:
`blended = market_spread + b_model·(model − market) + b_nfelo·(nfelo − market)`, `b_model`
clamped to `[0, 0.60]`. Last fitted: `b_model = 0.0` (clamped from −0.044), `b_nfelo = 0.0`
— the NFL blend currently returns the market line unchanged.

---

## 3. UNIVERSE CASCADE

### 3a. To the 1,543-game restricted NCAA universe

| step | filter | rows remaining |
|---|---|---|
| 0 | `market.parquet` — all CFBD games with a line record, 2019–2025 | **8,076** |
| 1 | `fbs_only` — both classifications == `fbs` | **4,973** |
| 2 | `restricted` — drop P5-vs-P5 (keeps G5-vs-G5 and cross-tier) | **2,737** |
| 3 | `completed` | **2,736** |
| 4 | `week >= 4` (`FIRST_GRADED_WEEK`) | **2,081** |
| 5 | `season ∈ 2021–2025` | **1,548** |
| 6 | `has_opener` — both `spread_open` and `total_open` non-null | **1,543** |
| 7 | model projections non-null after ratings join | **1,543** |

`backtest_frame_default.parquet` holds **4,068** rows (all seasons, week ≥ 4, completed,
FBS-vs-FBS including P5-v-P5). `select_window` applies steps 2, 5 and 6 at read time.

### 3b. To the 61–65 movable-line CLV set

| step | filter | rows remaining |
|---|---|---|
| 0 | restricted universe (above) | 1,543 |
| 1 | `season >= 2023` — walk-forward blend requires ≥400 prior graded rows, so 2021–22 are consumed as training | **933** |
| 2 | drop `exclude_weeks_total = [10, 13, 14, 15, 16]` | **676** |
| 3 | `total_open ∈ [38.0, 78.0]`, `abs(edge_pts) >= 1.5`, `cover_prob >= 0.545` | **68 picks** |
| 4 | `line_move != 0` — CLV undefined when the line did not move | **61 movable lines** |
| — | with concentration caps (15% team / 25% conference) applied at step 3 | 42 picks → **39 movable lines** |
| — | zero-skill control, same cascade, `close + noise` projection | 71 picks → **65 movable lines** |

---

## 4. DATA SCHEMA

### 4a. NCAA — `~/.cache/ncaa-model/`

| file | rows | seasons | cols |
|---|---|---|---|
| `games_2019_2025.parquet` | 19,292 | 2019–2025 | 16 |
| `drives_2019_2025.parquet` | 198,410 | 2019–2025 | 11 |
| `plays_2019_2025.parquet` | 1,441,860 | 2019–2025 | 21 |
| `lines_2019_2025.parquet` | 8,076 | 2019–2025 | 16 |
| `game_offense.parquet` | 16,483 | 2019–2025 | 24 |
| `market.parquet` | 8,076 | 2019–2025 | 27 |
| `walkforward_ratings_default.parquet` | 14,933 | 2019–2025 | 10 |
| `backtest_frame_default.parquet` | 4,068 | 2019–2025 | 43 |
| `lines_{year}.json`, `games_{year}.json`, `drives_{year}.json`, `plays_{year}_{wk}.json` | raw payloads | 2019–2025 | — |

Columns:

- **games**: `game_id, season, week, startDate, completed, neutralSite, conferenceGame, homeTeam, homeConference, homeClassification, homePoints, awayTeam, awayConference, awayClassification, awayPoints, kickoff`
- **drives**: `game_id, season, offense, defense, driveNumber, driveResult, startPeriod, startYardsToGoal, isHomeOffense, plays, scoring`
- **plays**: `game_id, driveId, driveNumber, playNumber, offense, defense, offenseConference, defenseConference, offenseScore, defenseScore, home, away, period, yardsGained, yardsToGoal, playType, ppa, season, week, garbage, competitive`
- **market**: `game_id, season, week, home_team, away_team, home_conference, away_conference, home_classification, away_classification, home_points, away_points, provider, total_close, total_open, spread_close, spread_open, kickoff, neutralSite, completed, home_p5, away_p5, fbs_only, restricted, cross_tier, actual_margin, actual_total, has_opener`
- **ratings**: `team, off_rating, def_rating, n_games, conference, pace_rating, season, week, as_of, hfa_ppa`

### 4b. NFL — `~/.cache/nfl-model/`

| file | rows | seasons | cols |
|---|---|---|---|
| `schedules_2016_2026.parquet` | 3,033 | 2016–2026 | 47 |
| `pbp_2016_2025.parquet` | 484,254 | 2016–2025 | 41 |
| `drives_main.parquet` | 60,715 | 2016–2025 | 16 |
| `game_offense_main.parquet` | 5,522 | 2016–2025 | 14 |
| `qb_elos.parquet` | 18,250 | 1920–2026 | 12 |
| `walkforward_ratings_default_2016_2026.parquet` | 6,176 | 2016–2026 | 10 |
| `walkforward_ratings_diag_2016_2026.parquet` | 6,176 | 2016–2026 | 10 |
| `backtest_frame.parquet` | 1,535 | 2019–2025 | — |

### 4c. Quarter-level / half-level scoring — coverage

**NCAA**

- **Raw JSON, present:** `games_{year}.json` carries `homeLineScores` / `awayLineScores`
  (per-quarter point arrays). Coverage in 2023: **3,557 of 3,595 games = 98.9%.**
- **Processed parquet, absent:** `ingest.load_games` subsets to a 16-column `keep` list
  omitting both. `games_2019_2025.parquet` has no `*LineScores` column.
- **Derivable at 100%:** `plays_2019_2025.parquet` carries `period` (100% non-null) and
  `offenseScore` / `defenseScore` (100% non-null), so quarter- and half-level scoring can be
  reconstructed play-by-play across all 1,441,860 plays.
- `drives` carries `startPeriod` but no `endPeriod` among retained columns.

**NFL**

- `schedules_2016_2026.parquet` has no quarter or half columns.
- `pbp_2016_2025.parquet` carries `qtr`, `score_differential`, `game_seconds_remaining` —
  quarter/half scoring derivable across all 484,254 plays, 2016–2025.
- `drives_main.parquet` carries `start_qtr` and `start_gsr`.

No table in either repo stores quarter or half scores as a materialised column.

---

## 5. HARNESS API

### 5a. `ncaa-model/src/backtest.py`

| line | signature | purpose |
|---|---|---|
| 26 | `class GateResult` | Gate name, pass/fail, observed value, detail. |
| 40 | `build_features(market, walkforward, cfg) -> DataFrame` | Join as-of ratings onto games; form `net_diff`, `eff_sum`, `pace_sum`, `is_home`. |
| 73 | `project_walkforward(feats, cfg) -> DataFrame` | Fit ratings→points on earlier seasons; apply gain and week-bias corrections. |
| 137 | `_project(rows, beta, kind) -> Series` | Apply a fitted projection to arbitrary rows. |
| 147 | `_fit_week_bias(train, pred, mkt, shrink=40.0) -> dict` | Per-week level bias vs the market line, shrunk `n/(n+40)`. |
| 166 | `_fit_gain(train, col, mkt, kind) -> float` | Scalar gain = `mkt_sd / pred_sd`, clipped `[0.5, 2.0]`. |
| 191 | `walk_forward(cfg, market, walkforward, cache_key="default") -> DataFrame` | Gradeable games with projections attached; caches to parquet. |
| 213 | `select_window(frame, seasons, restricted=True)` | Apply season / restricted / `has_opener` filters. |
| 224 | `gate_opener_coverage(market, cfg, seasons) -> GateResult` | Coverage ≥70% and mean \|open−close\| ≥0.5. |
| 241 | `gate_no_lookahead(feats, walkforward, sample=200) -> GateResult` | No rating dated after its game's kickoff. |
| 259 | `gate_garbage(share, cfg) -> GateResult` | Garbage-time drop share within 5–30%. |
| 268 | `gate_unbiased(fits, cfg) -> GateResult` | \|a\| < 0.5 full-sample. |
| 279 | `gate_unbiased_by_week(frame, cfg, min_n=40) -> GateResult` | \|a\| < 0.5 in every week bucket with n ≥ 40. |
| 327 | `gate_scale(fits, cfg) -> GateResult` | \|SD ratio − 1\| < 0.15. |
| 340 | `gate_blend_informative(fits, cfg) -> GateResult` | Best \|t(b)\| > 2.0. |
| 353 | `gate_api_budget(calls_used, cfg) -> GateResult` | Cold build under 250 calls. |
| 361 | `bootstrap(results, n_boot=10000, seed=20260801)` | Mean and 5th/95th percentile of a win rate. |
| 373 | `filtered_record(frame, cfg, kind, grade="open")` | Win/loss array for positions clearing the edge threshold. |
| 399 | `zero_skill_projection(frame, anchor="total_close", seed=20260804, noise_sd=None) -> DataFrame` | Control: replaces `model_total` with `anchor + N(0, σ)`. |

### 5b. `ncaa-model/run_clv_history.py`

| line | signature | purpose |
|---|---|---|
| 35 | `generate_history(cfg, frame, market, use_caps) -> DataFrame` | Walk-forward pick generation across every graded season. |
| 64 | `_conference_lookup(market) -> dict` | team → conference map. |
| 73 | `_attach_conferences(sheet, conf) -> DataFrame` | Add home/away conference columns. |
| 80 | `per_team(sheet) -> DataFrame` | Per-team pick count, CLV, hit rate, share. |
| 102 | `report(sheet, label) -> None` | CLV overall / by season / by week bucket, plus binomial test. |
| 142 | `report_concentration(sheet, cfg) -> None` | Team and conference concentration against the caps. |
| 172 | `main() -> int` | Runs uncapped, capped, and the zero-skill control. |

### 5c. Walk-forward split construction

Two distinct mechanisms:

1. **Ratings** — `src/ratings.py::build_walkforward` iterates `week_cutoffs(games, seasons)`,
   which sets `as_of` = the minimum kickoff of each `(season, week)`. `fit_ratings` and
   `fit_pace` filter `game_off["kickoff"] < as_of` and call `_assert_no_lookahead` on every
   fit. One snapshot per `(season, week)` — 14,933 rows.
2. **Projection / blend / calibration** — expanding window by *season*.
   `project_walkforward` trains on `feats["season"] < season`.
   `run_clv_history.generate_history` fits `fit_blend` and `build_conditional` on
   `graded["season"] < season`, requiring ≥400 prior rows.

The hyperparameter sweep (`run_backtest.py::run_sweep`) uses a fixed split: tune on
2021–2023, hold out 2024–2025.

### 5d. Where shrinkage and priors are applied

| mechanism | location | form |
|---|---|---|
| Ridge penalty | `ratings.py::fit_ratings` | `lambda_off = 850`, `lambda_def = 950` on team columns; intercept and `is_home_offense` unpenalised |
| Recency decay | `ratings.py::_decay` | `0.5 ** (games_ago / half_life_games)`; half-life 6 (ratings), 8 (pace); rows below `_MIN_DECAY_WEIGHT = 1e-3` dropped |
| Preseason prior | `ratings.py::fit_ratings` + `season_priors` | Synthetic rows at weight `prior_weight_games × league_mean_plays_per_game` = 5 × 70; prior = last season's rating × `(1 − 0.42)` |
| Conference shrinkage | `ratings.py::_conference_shrink` | `(n_eff·rating + 2.5·conf_mean) / (n_eff + 2.5)`, then re-centred |
| FCS down-weight | `ingest.py::build_game_offense` | `game_weight = 0.45` when either side is `__FCS__` |
| Pace ridge | `ratings.py::fit_pace` | Single team block, `lambda = 900`, intercept unpenalised |
| Week-bias shrinkage | `backtest.py::_fit_week_bias` | `n/(n+40)` toward the global bias |
| Blend clamp | `market.py::fit_blend` | `b` clipped to `[0, model_weight_cap = 0.70]` |
| L3 bucket pooling | `calibrate.py::build_conditional` | ±1.5 points pooled; widened until `min_bucket = 150` |

---

## 6. SIMULATOR

### 6a. NCAA — there is no simulator

`ncaa-model/src/` contains `backtest.py, betting.py, calibrate.py, cfbd_client.py,
config.py, ingest.py, market.py, ratings.py, report.py`. There is **no `simulate.py`,
`drives.py` or `drive_model.py`**. No Monte Carlo of any kind runs in the NCAA path.

`E[total]` is a closed-form linear projection: `bt0 + bt1·pace_sum + bt2·eff_sum`.

The only distributional object is `src/calibrate.py::ConditionalPMF` — an **empirical**
distribution of historical totals bucketed by line in 0.5-point bins with ±1.5-point
pooling and `min_bucket = 150`. It is not simulated. It exposes `quantiles`, `mean`,
`prob_over`, and `calibrate_draws` (rank-preserving quantile map). In the shipped path only
`calibrated_prob` → `prob_over` is called, from `betting.build_totals_sheet`, returning one
scalar per game.

### 6b. NFL — drive-level Monte Carlo; distribution produced, then collapsed

`nfl-model/src/simulate.py::simulate_game`, line 212.

- **Granularity: drive-level.** Per simulation it draws a drive count per team
  (`round(clip(N(mu, 1.7), 8, 16))`, plus a coin-flipped odd possession), then plays
  alternating possessions. Each drive samples a starting field position from an empirical
  distribution conditioned only on whether the previous drive ended in a turnover, then
  draws a 4-class outcome (`TD / FG / NO_SCORE / TURNOVER`) from a multinomial logit on
  `net_epa, start_fp, start_fp², is_home_offense, net_epa×fp`. Touchdowns draw a conversion
  sub-outcome. Each team's final drive uses a score-state-conditioned empirical
  `EndgameTable`. Rare events are per-team Poisson draws (defensive TD 0.13, special-teams
  TD 0.045, safety 0.035). Ties enter an overtime routine of up to three exchanges. It is
  **not** play-level, and it does **not** sample the margin from a parametric distribution.
- **Draws:** `cfg.simulation.n_sims = 20000` in config; the backtest was run with
  `--n-sims 6000`.
- **Output:** `SimResult` holding four full arrays — `margins`, `totals`, `home_scores`,
  `away_scores`, each of length `n_sims` — plus `cover_prob`, `total_prob`, `margin_pmf`,
  `recentered`, `retotaled`.

**Collapsed before downstream use in the backtest path.**
`nfl-model/src/backtest.py::walk_forward` persists four scalars and discards the arrays:

```
131:  "model_spread": sim.mean_margin,
132:  "model_total": sim.mean_total,
142:  "cover_prob_home": sim.cover_prob(float(g["spread_line"]), "home"),
144:  sim.total_prob(float(g["total_line"]), "over")
```

`backtest_frame.parquet` therefore holds point estimates plus two probabilities. Every gate,
metric, blend fit and CLV figure downstream reads that collapsed frame.

Two paths retain the distribution in memory:

- `src/betting.py::build_bet_sheet` receives a live `SimResult` on the `GameProjection` and
  calls `.recentered()`, `.retotaled()`, `.margin_pmf()`, `.cover_prob()`, `.total_prob()`.
- `src/backtest.py::pooled_margin_pmf` (line 469) concatenates `sim.margins` across ~400
  games for the §6d key-number validation.

The arrays are never written to disk. Any consumer reading a cached artifact sees point
estimates only.

---

## 7. OPEN TODOS / KNOWN-SHAKY CODE

No `TODO`, `FIXME`, `XXX` or `HACK` markers exist in any source file across
`nfl-model/src`, `ncaa-model/src`, `shared/`, or the entrypoints. The following are
observations from reading the code, not annotations in it.

| file:line | observation |
|---|---|
| `ncaa-model/src/report.py:71-72` | Historical CLV figures `0.7049` / `0.6923` are hardcoded as strings in the Diagnostics sheet; they do not recompute and are stale relative to the Appendix A null. |
| `ncaa-model/src/backtest.py:108-114, 128-132` | Gain and week-bias corrections take `spread_open` / `total_open` as reference, making `model_*` line-derived (§2). |
| `ncaa-model/src/backtest.py:241-257` | `gate_no_lookahead` compares `as_of` (week-minimum kickoff) against each game's kickoff; it validates the week-granularity split, not per-game rating provenance. |
| `ncaa-model/src/betting.py:3-5` | Module docstring quotes `b = +0.047, t = 0.89` for spreads and `16.68` totals RMSE; both drifted after the gain/de-bias changes (now `+0.042 / 0.80` and `16.79`). |
| `ncaa-model/run_paper.py:4` | Docstring reads "last 4 completed weeks"; the default is now `--weeks 0`, the pre-declared weeks 4–16. |
| `ncaa-model/config/ncaa.yaml` | `min_edge_points_spread`, `max_abs_market_spread`, `kelly_fraction`, `max_stake_pct_bankroll`, `bankroll`, `default_price` are loaded but unreferenced — spreads are dark and no stake is sized. |
| `ncaa-model/src/market.py::clv` | Hardcoded to `spread_close` / `spread_{grade}`; the totals CLV path is implemented separately inside `betting.build_totals_sheet`. |
| `ncaa-model/src/ingest.py::load_games` | The 16-column `keep` list drops `homeLineScores` / `awayLineScores`, discarding 98.9%-covered quarter scores at ingest (§4c). |
| `ncaa-model/tests/` | Contains only `__init__.py` — no tests. The spread-sign assertion required by `NCAA_PLAYBOOK.md` §14 is not implemented. |
| `ncaa-model/src/` | `drives.py`, `priors.py`, `weather.py`, `context.py`, `simulate.py`, `drive_model.py`, `srs.py` from the §3 file tree were never built. No weather module; no SRS baseline, so `GATE_BEATS_SRS` is not implemented. |
| `nfl-model/src/backtest.py::clv_proxy` | Returns `None` unconditionally — it looks for `spread_line_open`, which nflverse does not publish. |
| `nfl-model/` | `src/report.py` and `run_week.py` were never built; `app.py` delegates to `shared/view.py`. |
| `shared/sport.py::NCAAAdapter.simulate` | Returns `None` unconditionally — the Game explorer tab is inert for NCAA. |
| `shared/export.py::_sheet_bets_placeholder` | Writes a placeholder rather than real picks when the gate passes; never exercised (NFL gate fails; NCAA export goes through `ncaa-model/src/report.py`). |
| `~/.cache/ncaa-model/` | `walkforward_ratings_sweep{1..12}.parquet` and `backtest_frame_sweep{1..12}.parquet` persist from the hyperparameter sweep; `shared/sport.py::_wf_path` prefers `walkforward_ratings_default.parquet` to avoid selecting one. |
