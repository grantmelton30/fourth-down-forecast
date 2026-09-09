# NCAA totals research — September 9, 2026

Objective: determine whether NCAA total-points forecasts add information beyond the market. NCAA totals are the primary research focus. Existing P1 tracking stays frozen; this experiment is separate.

This is exploratory historical research. Seasons 2021–2025 have already been inspected. Earlier-season-only folds prevent fitting on evaluation outcomes, but do not turn repeatedly inspected history into an untouched holdout. No result here authorizes bets or automatically promotes a model.

Before running the new comparison, fix these choices:

- Population: completed FBS-vs-FBS games, 2021–2025. Train on earlier seasons only; evaluate 2022–2025. Require at least 300 training observations.
- Anchor: recorded closing total for a historical information diagnostic. It is not a known executable pregame quote. Do not use opener-relative transformations as evidence of independent model skill.
- Comparators: closing market, market plus earlier-season mean error, archived pure model, archived adjusted model, and frozen P1 selections.
- Candidate families: pure-model disagreement; tempo/efficiency (pace sum, efficiency sum, their product, absolute net-rating difference); play quality (success, explosive, havoc avoidance, red-zone touchdown matchup sums). Each candidate predicts residual total error relative to the market. No feature or threshold search after results.
- Fit: training-only median imputation and scaling; ridge penalty 45; unpenalized intercept. No weather candidate: historical observed weather is not a timestamped forecast.
- Selection: an additional adaptive candidate may choose among market, bias correction and the three families using only prior evaluated seasons, at least two. Otherwise use market. Include that complete selection procedure in evaluation.
- Metrics: paired RMSE/MAE, signed bias, squared-error improvement against both market and bias-only, season-week block-bootstrap intervals, and season-level results. Intervals are descriptive and not adjusted for candidate search.
- Fixed diagnostics: restricted/P5-vs-P5; weeks 1–3/later; absolute spread <=28/>28;
  pregame market total <=50/>50. These are diagnostics, not independently validated sub-rules.
- P1: 0.5 <= absolute model-market difference <6, restricted universe, pushes excluded from win rate. Report actual totals against the historical close, with hypothetical -110 returns, clearly separated from execution.
- Report coverage and source fingerprint. Missing inputs remain visible. No automatic model promotion; future timestamped evidence is required.

Next decision: prefer no correction unless performance improves against the market and bias-only across years. A historical candidate can earn a prospective shadow test, not a profitability claim.

## Result recorded after the specification was fixed

The comparison evaluated 2,410 FBS-vs-FBS games from 2022–2025. The cached backtest starts
at week 4, so it cannot answer the fixed weeks 1–3 diagnostic.

The closing market RMSE was 15.783. An earlier-season mean-bias correction scored 15.766.
The tempo/efficiency candidate scored 15.751: a 0.033-point gain over the market with a
descriptive 90% season-week block interval of +0.002 to +0.063, but only a 0.015-point gain
over bias correction with an interval of -0.015 to +0.046. The fully adaptive procedure
scored 15.759 and its market-gain interval touched zero. It selected the market in 2022–23
and tempo/efficiency in 2024–25 using prior evaluation seasons only.

The frozen P1 reproduction matched its registered historical-close result: 577–489 with 15
pushes, 54.13%, and +39.1 assumed units at -110. That remains a repeatedly inspected,
post-selected historical hypothesis rather than evidence of an executable edge.

Decision: do not modify the independent forecast or P1. Freeze tempo/efficiency and the
simple bias correction as prospective shadow forecasts anchored to the timestamped current
market quote. Their purpose is to test whether football features add anything beyond market
bias on untouched 2026 observations.

Prospective decision rule fixed before results: use only `T1-shadow-v1` grades at the one-hour
decision horizon, with one row per game and one unchanged shadow model version. Require at
least 300 games across at least eight season-week blocks. Resample season-week blocks 5,000
times and require the lower bound of the paired 90% RMSE-gain interval to exceed zero against
both the timestamped market total and the bias-only correction. Passing makes the candidate
eligible for human review; it never promotes the model or creates a bet automatically.

## V2 challenger registered before results

This is one fixed historical challenger pass. Its purpose is to decide whether another model
is worth a separate prospective shadow; it cannot change `T1-shadow-v1`, P1, the independent
forecast, or any public pick.

- Population and evaluation remain FBS-vs-FBS, 2022–2025 evaluation with strictly earlier
  seasons for fitted parameters. The close is the historical proxy for the one-hour current
  quote. The opener is allowed because it is known before that decision.
- Baselines remain the market, expanding earlier-season bias, and `T1-shadow-v1` tempo.
- `movement_context` uses opener-to-current total movement, current total level, squared
  current total level, absolute spread, indoor venue, and wind speed.
- `football_context` uses the frozen tempo features plus early-down efficiency, explosive
  rate, red-zone touchdown rate, starting field position, pass blocking, run blocking, and
  the absolute quarterback-adjustment gap. Offensive-coordinator continuity is excluded
  because coverage is zero in the cached frame.
- `team_memory` is calculated before each kickoff from each team's earlier games only. A team
  residual is its cumulative market error divided by its prior-game count plus 20; the game
  feature is the mean of the home and away values. Games sharing a kickoff cannot inform one
  another.
- `combined_v2` fits one ridge residual model with all three feature families. Continuous
  features are median-imputed and standardized from the training fold only. Missing opener
  movement is zero with a separate missingness flag. Ridge alpha remains 45; it will not be
  retuned after results.
- Primary comparison is paired RMSE gain against both market and expanding bias, with 5,000
  season-week block-bootstrap replicates and a 90% interval. Secondary metrics are MAE, bias,
  coverage, and per-season RMSE gain.
- A V2 candidate may enter a new prospective shadow only if both lower interval bounds exceed
  zero, coverage is at least 95%, and RMSE gain versus both baselines is positive in at least
  three of four evaluation seasons. Otherwise V2 stops without feature or threshold changes.
