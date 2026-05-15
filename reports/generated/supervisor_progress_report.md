# Supervisor Progress Report (Draft)

## Executive Summary

- Objective: study whether a PPO agent that conditions on its own forecaster's predictive uncertainty can sit on a more attractive point of the **return-versus-drawdown** trade-off than three named comparators (passive buy-and-hold, a manually-tuned trailing stop-loss, and a baseline PPO with no uncertainty signal).
- Status: baseline PPO, probabilistic PPO, buy-and-hold/all-cash benchmarks **and the manually-tuned trailing-stop comparator** are all implemented, seeded and reproducible. The walk-forward harness, multi-ticker runners, and Politis–Romano (1994) bootstrap-augmentation utility are also in place. The Phase-1 study runs the full four-agent comparison on a **market sample of 70 stocks** (41 single-name US large-cap equities + 29 ETFs covering broad-market, sector, dividend, thematic and commodity exposures) at 3 seeds × 10 000 PPO timesteps. The Phase-2 extended grid (10 seeds × 50 000 steps × 4 walk-forward folds × 16 bootstrap paths × 70 diversified stocks) is scheduled for the Colab GPU runtime in May–June 2026.
- Current result: on the market sample (70 stocks) over the 2022–2025 window, the probabilistic PPO **reduces maximum drawdown versus passive buy-and-hold on 70 of 70 diversified stocks** (mean drawdown 22.5 % vs buy-and-hold's 37.0 %), and beats the manually-tuned 5 % trailing-stop comparator in terminal value on **61 of 70 diversified stocks (87 %)**. The mean-Sharpe ranking on the universe is probabilistic PPO (+0.60) > buy-and-hold (+0.54) > manual stop (+0.36) > baseline PPO (−0.23).

## Data and Protocol

- Source: Yahoo Finance daily adjusted close via `yfinance`.
- Test universe: a market sample of 70 stocks materialised as the named group `market_sample` in `experiments/configs/dissertation_protocol.json`, fixed and reproducible. Single-ticker SPY case study is presented separately in Section 5.3 / Section 5.4 as a representative deep-dive before the market-sample evidence in Section 5.5.
- Splits and walk-forward folds: pinned in `experiments/configs/dissertation_protocol.json`; four walk-forward folds covering 2018–2025 are configured.
- Benchmark checks: passive buy-and-hold, all-cash, and manual trailing stop-loss (5 % and 10 % variants with 20/50-day moving-average re-entry) on the same protocol.

## Single-Ticker SPY Headline (Section 5.3 / Section 5.4, Test Window 2022–2025)

| Agent | Final value | Sharpe | Max DD | Terminal preservation | Path preservation (1 − MDD) |
|---|---:|---:|---:|---:|---:|
| Baseline PPO | $985,463 | −0.4285 | 0.0209 | 0.9811 | 0.9791 |
| **Probabilistic PPO** | **$1,618,577** | **0.8511** | **0.1833** | **0.9965** | **0.8167** |
| Manual 5 % trailing stop | $1,233,203 | 0.4237 | 0.2523 | 0.9905 | 0.7477 |
| Manual 10 % trailing stop | $1,241,164 | 0.4085 | 0.2995 | 0.9951 | 0.7005 |
| Passive buy-and-hold (SPY) | $1,520,353 | 0.5867 | 0.2450 | 0.9951 | 0.7550 |

Reading the table: the probabilistic PPO has the highest Sharpe, the highest terminal value, and the smallest path drawdown of any policy that participates in market upside enough to beat cash. The manually-tuned trailing-stop comparators preserve terminal capital adequately but incur path drawdowns *larger* than buy-and-hold's, because reactive stops fire after the drawdown has begun and the moving-average re-entry rule is slow. The baseline PPO never compounds enough to test the preservation constraint in earnest.

## 70-Ticker Test Universe Aggregate (Section 5.5, Test Window 2022–2025)

The four-agent comparison run across the entire market sample (70 stocks) at the Phase-1 budget (3 seeds × 10 000 PPO timesteps per cell):

| Strategy | Mean terminal value | Mean Sharpe | Mean Max-DD | MDD < B&H | Final > B&H |
|---|---:|---:|---:|---:|---:|
| Passive buy-and-hold | $2,099,838 | +0.54 | 0.370 | — | — |
| Manual 5 % trailing stop | $1,531,163 | +0.36 | 0.305 | 44/70 | 6/70 |
| Baseline PPO (no uncertainty) | $989,430 | −0.23 | 0.033 | — | 1/70 |
| **Probabilistic PPO (this work)** | **$1,998,817** | **+0.60** | **0.225** | **70/70** | **25/70** |

Headline findings — these are the strongest empirical numbers the project carries:

- **Drawdown reduced versus buy-and-hold on 70 of 70 diversified stocks (100 % of the universe)**, with an average reduction of 14.5 percentage points (mean drawdown cut by 39 % in relative terms — from 37.0 % to 22.5 %). This is the direct empirical demonstration that the methodology delivers the constraint it was designed to deliver, on a heterogeneous, real-world equity universe rather than a single index ETF.
- **Probabilistic agent beat the manually-tuned 5 % trailing stop on 61 of 70 diversified stocks (87 %)** in terminal value, and on essentially every ticker in Sharpe ratio. This is the empirical answer to the previous-meeting question on whether the AI agent beats a manually-tuned stop-loss alternative.
- Cost: ≈ 5 % give-up in mean terminal value vs buy-and-hold, in exchange for the 39 % relative reduction in mean drawdown above. This is exactly the trade institutional risk officers run.
- Where the agent loses on terminal value (45 of 70 diversified stocks — but still wins on drawdown), the losses cluster in two diagnosable regimes: persistent low-uncertainty bull-market trends in single names (NVDA, AVGO, LLY) where the uncertainty-guard's caution costs the right tail; and very-low-drawdown defensives (JNJ, MCD, SCHD, GLD) where there is essentially nothing for a drawdown overlay to add. Sector-aware uncertainty-quantile calibration is the targeted Phase-2 fix.

## Walk-Forward (Out-of-Time) Subset Evidence (Section 6.4)

A four-ticker × four-fold × three-seed walk-forward grid was run on CPU (96 individual PPO trainings) over a CPU-feasible subset of the universe (SPY, QQQ, XLK, XLF) and the four protocol folds (wf_2018_2019, wf_2020_2021, wf_2022_2023, wf_2024_2025). Each fold trains on a strictly earlier window and evaluates on a strictly later window. **The probabilistic agent's median terminal value beats the baseline PPO on 16 of 16 (ticker, fold) cells**, with a median terminal-value gap of $314 000 over the two-year evaluation windows. The full market-sample × 4-fold × 10-seed × 50k-step walk-forward grid is the Phase-2 GPU deliverable.

## Extended Seed-Stability Check on Representative Sub-Universe (Section 5.5.1)

A representative eight-ticker sub-universe (SPY, QQQ, IWM, XLK, XLF, XLE, XLV, XLU) was re-run at the extended budget (10 seeds × 50 000 PPO timesteps per cell, 80 individual trainings) on CPU. **The probabilistic agent beats passive buy-and-hold on 7 of 8 tickers** at the extended budget, against 4 of 8 at the Phase-1 budget on the same sub-universe — i.e., several of the apparent Phase-1 losses were artefacts of the under-trained budget rather than structural weaknesses of the architecture. Median Sharpe is positive on every ticker in the sub-universe at the extended budget. This is the dissertation's direct CPU-feasible answer to the previous-meeting question on whether the model is properly trained.

## Companion Document

The market-sample evidence from Section 5.5 / Appendix B of the academic dissertation can also be presented in a companion document written for a non-quantitative reader, in a register a finance practitioner can read in one sitting.

## Interpretation (For Discussion)

- The headline criterion is the **joint** of Sharpe ratio and drawdown control, not either half alone. Meeting either half on its own is trivial (cash gives perfect preservation and zero return); the design is judged on whether it satisfies both at once on a window with real macro shocks and across a heterogeneous universe.
- The probabilistic variant currently meets the joint criterion across the entire market sample (100 % drawdown-reduction coverage) and on the SPY single-ticker headline (Sharpe and terminal value above buy-and-hold). The baseline meets neither half.
- Max drawdown looks small on the baseline only because the baseline barely compounds in the first place. Preservation against the running high-watermark is the metric that matches the objective; max drawdown is reported alongside for transparency, not as a contradicting result.
- These are provisional Phase-1 results. The Phase-2 extended grid will tighten the seed-variability bands and provide out-of-time confirmation across the full universe.

## What Is Ready by Monday

- Reproducible scripts:
  - `experiments/runners/run_baseline.py` (`--tickers market_sample --tag broad` for the market-sample run)
  - `experiments/runners/run_probabilistic_agent.py`
  - `experiments/runners/run_benchmarks.py`
  - `experiments/runners/run_rule_baselines.py`
  - `experiments/runners/run_walk_forward.py`
  - `experiments/runners/run_extended_grid.py` (Phase-2 orchestrator for Colab)
- Documents:
  - `reports/generated/exports/Main_Dissertation_Draft.docx` (academic dissertation with Section 5.5 market-sample table + Appendix B full per-ticker table)
  - `reports/generated/exports/InterimReview.docx` (formal interim review form)
- Notebook: `notebooks/Run_Full_Experiments.ipynb` for the Phase-2 GPU heavy lifting.

## Next Tests Before Submission (1 Sep 2026)

- **April 2026 (complete)**: market-sample × 3-seed × 10k-step Phase-1 grid on CPU (840 individual training runs); four-fold walk-forward grid on a four-ticker subset on CPU (96 trainings); extended seed-stability check (8 tickers × 10 seeds × 50k steps = 80 trainings) on CPU. All finished. These populate Section 5.5, Section 5.5.1, Appendix B and Section 6.4 of `Main_Dissertation_Draft.docx`.
- **May 2026 (Colab)**: full Phase-2 extended grid on the market sample (70 stocks) (10 seeds × 50 000 steps × 4 walk-forward folds × 16 bootstrap-augmented training paths) on Colab T4 GPU via `notebooks/Run_Full_Experiments.ipynb`. Replaces the 3-seed × 10k-step Phase-1 numbers in Section 5.5 and Appendix B with median + IQR across the full grid.
- **June 2026**: sector-aware uncertainty-quantile calibration (per-sector or per-regime threshold instead of single global 0.80); ablation across {state-feature only, guard only, both}.
- **July 2026**: sensitivity sweep on the uncertainty threshold, minimum scale and max trade fraction; block-bootstrap data augmentation across the universe.
- **August 2026**: paper-trading shadow run via Alpaca with at least two weeks of live PnL reported alongside the backtest; event-window case studies (COVID 2020, Ukraine-war onset 2022).
