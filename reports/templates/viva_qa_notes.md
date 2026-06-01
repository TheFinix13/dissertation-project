# Viva Q&A Notes (Draft)

## 1) Why probabilistic RL instead of standard PPO?
- Standard PPO uses point estimates and can over-trust noisy signals.
- Probabilistic layer estimates uncertainty; high uncertainty triggers safer behavior.

## 2) How is uncertainty measured?
- Forecast model outputs mean and variance for next-step return.
- Variance (or standard deviation) is normalized into an uncertainty score.
- This score is injected into the RL state and used for risk guard decisions.

## 3) What is the main hypothesis?
- Adding uncertainty-aware signals and guardrails improves capital preservation relative to baseline PPO.

## 4) What are your baselines?
- PPO baseline (same environment, no uncertainty signal/guard).
- Buy-and-hold benchmark.
- All-cash benchmark.

## 5) What are your key metrics?
- Final portfolio value
- Max drawdown
- Sharpe ratio
- VaR violation rate
- Capital preservation ratio relative to high-watermark (target >= 0.95)

## 6) What are the limitations?
- Daily granularity and single-source market data in current phase.
- Limited assets in initial runs.
- Further robustness checks needed for multi-event and multi-sector generalization.

## 7) Why not full live trading?
- Dissertation scope prioritizes scientific evaluation and reproducibility over production execution.

## 8) How do you validate reproducibility?
- Fixed seeds and scripted runs.
- Artifact generation in `experiments/results` and `reports/generated`.
- The canonical Phase-2 grid is consolidated in
  `experiments/results/summary_20260530T122900Z_phase2.csv`; the LaTeX
  dissertation in `latex/` is reconciled to it.

---

## Results and defence (Phase-2, complete)

> All numbers below are the **completed Phase-2** results — the full grid
> across the `market_sample` universe (70 stocks), ten seeds, 50,000 PPO
> time-steps per cell, on the held-out 2022–2025 test window. They are the
> headline of the dissertation. The earlier single-ticker SPY result is an
> illustrative case study, **not** the headline.

## 9) What is the headline result?
- The uncertainty-aware (probabilistic) agents grow a $1M portfolio to a
  **median of ~$1.6M** ($1,612,478 aleatoric / $1,619,713 epistemic) with a
  **median Sharpe of +0.68 to +0.70** and a **win-rate of 89–92%** (fraction of
  cells finishing above the $1M starting capital).
- The **baseline PPO** (no uncertainty signal) finishes essentially flat at a
  **median of ~$999,063**, with a **median Sharpe of −0.04** and a **46.1%**
  win-rate. It learns to under-trade because, without a confidence signal, it
  cannot tell dangerous days from safe ones.
- The factor-of-1.6 gap in median terminal wealth is reproduced across ten
  seeds per ticker — it is not a single-run fluke.

## 10) Does the agent actually reduce drawdown? (Be honest here.)
- Median max drawdown across the grid: buy-and-hold ~25.9%, trailing stop-loss
  5% ~26.2%, trailing stop-loss 10% ~32.7%, **aleatoric guard ~22.1%**,
  **epistemic guard ~24.1%**, baseline PPO ~1.4%.
- **The correct benchmark is buy-and-hold, not the baseline PPO.** The
  baseline's tiny ~1.4% drawdown is an **under-trading artefact** (it barely
  invests, so there is almost nothing to lose) — not risk-management skill.
- On the **median / typical stock and on the index basket**, the guard takes
  **less drawdown than buy-and-hold (~22% vs ~26%)** *while* earning more.
- **Honest caveat — the tail.** The guard does **not** win on the worst case:
  on the most volatile single names its worst-case drawdown (~57%) exceeds
  buy-and-hold's worst (~35%). The claim is "lower drawdown on the typical stock
  and the basket", **never** "reduces drawdown on every ticker". State this
  proactively if asked — it is in the dissertation (Chapter 5).

## 11) Aleatoric vs epistemic — why both, and does it matter?
- Aleatoric (Gaussian-NLL LSTM) measures data noise: "how chaotic is the market
  now?". Epistemic (MC Dropout) measures model unfamiliarity: "how different
  from training does today look?". Their correlation on SPY is ~0.05 — they
  capture genuinely different quantities.
- At full scale the two arms are **within noise of each other** (median final
  values $1.61M vs $1.62M; Sharpe +0.70 vs +0.68; per-stock correlation ~0.99).
- The finding: the *specific source* of the uncertainty signal matters less
  than the fact that **any** calibrated uncertainty signal is present. A
  practitioner can use either.

## 12) Does the advantage hold out of sample / over time?
- Yes. A four-fold **walk-forward** validation (train on an earlier window, test
  on a strictly later one) spans 2018–2025, including the COVID crash and the
  2022 inflation shock — 320 cells.
- The probabilistic agent **wins on every fold**: overall **87.2%** win-rate,
  median final value **$1,167,050**, median Sharpe **+0.55**, versus the
  baseline's ~$997,591, −0.10, 46.9%.
- The narrowest margin is the 2022–2023 inflation-shock fold (still a positive
  median final value of ~$1,061,717 and a 78.8% win-rate); the largest is the
  2020–2021 COVID-recovery window.

## 13) Isn't the baseline PPO suspiciously safe? / Why is its drawdown so low?
- Because it barely trades. Its near-flat final value (~$999k) and tiny
  drawdown (~1.4%) are two views of the same under-trading behaviour. It is a
  fair, controlled baseline (identical environment minus the uncertainty
  coordinate and trade-size shrinkage), and its failure to take useful risk is
  itself the point: the uncertainty signal is what lets the agent take
  *informed* positions.
