# Phase-2 statistical-significance tests

Generated offline from committed per-cell JSONs. Paired Wilcoxon + paired bootstrap (n=10,000 resamples). Cell statistic = median across seeds per ticker.

- Phase-2 main grid: 70 tickers per arm.
- Walk-forward: 32 (ticker, fold) pairs per arm.

## Phase-2 main grid (2022-2025 test window)

### final_portfolio_value

| Contrast | n pairs | Median diff (L-R) | 95% bootstrap CI | Hodges-Lehmann | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| B_vs_A | 70 | 6.344e+05 | [5.59e+05, 8.143e+05] | 7.795e+05 | 3.717e-13 |
| C_vs_A | 70 | 6.327e+05 | [5.789e+05, 9.043e+05] | 8.138e+05 | 3.881e-13 |
| B_vs_C | 70 | -8616 | [-2.278e+04, 4032] | -1.653e+04 | 0.07965 |

### sharpe_ratio

| Contrast | n pairs | Median diff (L-R) | 95% bootstrap CI | Hodges-Lehmann | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| B_vs_A | 70 | 0.6807 | [0.597, 0.809] | 0.6511 | 2.74e-12 |
| C_vs_A | 70 | 0.6712 | [0.5788, 0.7961] | 0.6394 | 3.237e-12 |
| B_vs_C | 70 | 0.0137 | [0.007008, 0.01865] | 0.01261 | 0.0006955 |

### max_drawdown

| Contrast | n pairs | Median diff (L-R) | 95% bootstrap CI | Hodges-Lehmann | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| B_vs_A | 70 | 0.1828 | [0.1721, 0.2151] | 0.206 | 1.803e-12 |
| C_vs_A | 70 | 0.2074 | [0.1808, 0.2387] | 0.226 | 1.4e-12 |
| B_vs_C | 70 | -0.006518 | [-0.01376, -0.00326] | -0.01113 | 2.122e-05 |

### capital_preservation_rate_95pct_hwm

| Contrast | n pairs | Median diff (L-R) | 95% bootstrap CI | Hodges-Lehmann | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| B_vs_A | 70 | -0.02015 | [-0.02603, -0.01032] | -0.02429 | 1.602e-08 |
| C_vs_A | 70 | -0.01992 | [-0.02633, -0.01112] | -0.02693 | 1.351e-08 |
| B_vs_C | 70 | 9.133e-06 | [-1.757e-07, 0.0001556] | 8.082e-05 | 0.1574 |

## Walk-forward (out-of-time, 2018-2025)

### final_portfolio_value

| Contrast | n pairs | Median diff (L-R) | 95% bootstrap CI | Hodges-Lehmann | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| WFB_vs_WFA | 32 | 2.052e+05 | [1.033e+05, 2.801e+05] | 2.287e+05 | 4.657e-09 |

### sharpe_ratio

| Contrast | n pairs | Median diff (L-R) | 95% bootstrap CI | Hodges-Lehmann | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| WFB_vs_WFA | 32 | 0.5247 | [0.231, 0.8188] | 0.5287 | 6.795e-06 |

### max_drawdown

| Contrast | n pairs | Median diff (L-R) | 95% bootstrap CI | Hodges-Lehmann | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| WFB_vs_WFA | 32 | 0.1591 | [0.1486, 0.1799] | 0.1605 | 4.657e-10 |

### capital_preservation_rate_95pct_hwm

| Contrast | n pairs | Median diff (L-R) | 95% bootstrap CI | Hodges-Lehmann | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| WFB_vs_WFA | 32 | 0.005155 | [-0.01406, 0.01411] | -0.002435 | 0.719 |
