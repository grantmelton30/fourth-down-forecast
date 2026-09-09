# NCAA totals walk-forward research

Historical exploratory analysis. No automatic promotion or betting authorization.

Evaluated 2,410 FBS-vs-FBS games across 2022, 2023, 2024, 2025.

| model | n | coverage | RMSE | gain vs market | 90% CI | gain vs bias | 90% CI | bias |

|---|---:|---:|---:|---:|---:|---:|---:|---:|
| market | 2410 | 100.00% | 15.783 | +0.000 | [+0.000, +0.000] | -0.018 | [-0.035, -0.001] | -1.014 |
| bias | 2410 | 100.00% | 15.766 | +0.018 | [+0.001, +0.035] | +0.000 | [+0.000, +0.000] | -0.572 |
| pure_residual | 2410 | 100.00% | 15.760 | +0.023 | [-0.005, +0.053] | +0.005 | [-0.013, +0.024] | -0.427 |
| tempo_efficiency | 2410 | 100.00% | 15.751 | +0.033 | [+0.002, +0.063] | +0.015 | [-0.015, +0.046] | -0.632 |
| play_quality | 2409 | 99.96% | 15.775 | +0.011 | [-0.012, +0.033] | -0.007 | [-0.021, +0.007] | -0.591 |
| adaptive | 2410 | 100.00% | 15.759 | +0.024 | [-0.000, +0.049] | +0.006 | [-0.016, +0.031] | -0.701 |
| archived_pure | 2410 | 100.00% | 16.403 | -0.620 | [-0.811, -0.431] | -0.637 | [-0.816, -0.458] | +2.310 |
| archived_adjusted | 2410 | 100.00% | 16.426 | -0.643 | [-0.815, -0.473] | -0.661 | [-0.826, -0.497] | +0.740 |

Adaptive choices by season: 2022=market, 2023=market, 2024=tempo_efficiency, 2025=tempo_efficiency

Weeks 1–3 diagnostic: unavailable; the cached backtest begins at week 4.

Frozen P1 historical-close diagnostic: 577-489-15 (54.13%), +39.1 assumed units.

The closing market is a historical information benchmark, not a recorded executable quote. Intervals are descriptive and do not correct for prior inspection of these seasons.

Lowest aggregate RMSE: tempo_efficiency (15.751).

The adaptive candidate did not clearly improve on the market.
