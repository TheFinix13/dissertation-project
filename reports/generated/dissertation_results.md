# Dissertation Experiment Report (Phase-2, final)

> **Canonical results.** The figures below are the completed **Phase-2** full
> grid and match the LaTeX dissertation in `latex/` (the source of truth).
> They supersede the earlier Phase-1 single-window snapshot. Phase-2 trains
> every agent variant on the **market_sample** universe (70 stocks) with ten
> seeds and 50,000 PPO time-steps per cell. "Median" and "win-rate" are
> computed across all cells in each arm; win-rate is the fraction of cells
> that finish above the \$1,000,000 starting capital.

## Objective
Evaluate whether probabilistic uncertainty integration improves the joint of
risk-adjusted return and downside-risk behaviour compared to a baseline PPO
that sees no uncertainty signal.

## Protocol
- Config: `experiments/configs/dissertation_protocol.json`
- Baseline runner: `experiments/runners/run_baseline.py`
- Probabilistic runner: `experiments/runners/run_probabilistic_agent.py`
- Phase-2 grid orchestrator: `experiments/runners/run_extended_grid.py`
- Consolidated results: `experiments/results/summary_20260530T122900Z_phase2.csv`

## Headline grid — held-out test window 2022–2025 (market_sample, 70 stocks)

| Agent | Cells | Final value (median) | Sharpe (median) | Max drawdown (median) | Win-rate |
|---|---:|---:|---:|---:|---:|
| Baseline PPO (no uncertainty) | 700 | $999,063 | −0.04 | ~1.3% | 46.1% |
| **Probabilistic — aleatoric (Arm B)** | **697** | **$1,612,478** | **+0.70** | **20.9%** | **89.1%** |
| **Probabilistic — epistemic (Arm C)** | **697** | **$1,619,713** | **+0.68** | **22.9%** | **92.3%** |

The uncertainty-aware agents grow a \$1M starting portfolio to a **median of
about \$1.6M** with a median Sharpe of **+0.68 to +0.70** and a win-rate of
**89–92%**. The baseline PPO finishes essentially flat at a median of about
**\$999,063** with a slightly negative Sharpe and a **46.1%** win-rate — it
learns to under-trade because, without a confidence signal, it cannot tell
dangerous days from safe ones. The aleatoric and epistemic arms are within
noise of each other.

## Walk-forward validation (4 folds, 2018–2025, 320 cells)

| Agent | Cells | Final value (median) | Sharpe (median) | Win-rate |
|---|---:|---:|---:|---:|
| Baseline PPO | 320 | $997,591 | −0.10 | 46.9% |
| **Probabilistic PPO** | **320** | **$1,167,050** | **+0.55** | **87.2%** |

The probabilistic agent wins on **every one of the four folds**. The advantage
is largest in the COVID-recovery window (2020–2021) and smallest in the
2022–2023 inflation-shock fold, but it is positive throughout — evidence the
agent learned a general strategy, not a pattern specific to one period.

## Drawdown — the honest story (median max drawdown across the grid, 2022–2025)

| Strategy | Median max drawdown |
|---|---:|
| Baseline PPO (do-nothing) | ~1.4% |
| **Aleatoric uncertainty guard** | **~22.1%** |
| **Epistemic uncertainty guard** | **~24.1%** |
| Passive buy-and-hold | ~25.9% |
| Trailing stop-loss 5% | ~26.2% |
| Trailing stop-loss 10% | ~32.7% |

Reading this correctly matters:

- The **proper drawdown benchmark is passive buy-and-hold**, *not* the
  baseline PPO. The baseline's ~1.4% drawdown is an **under-trading artefact**
  (it barely takes positions, so there is almost nothing to lose) — it is not
  evidence of risk skill.
- On the **median / typical stock** and on the index basket, the uncertainty
  guard takes **less drawdown than buy-and-hold (~22% vs ~26%)** while also
  earning more — a genuine improvement.
- This does **not** hold on the worst-case tail: the guard's worst-case
  drawdown (~57%) is larger than buy-and-hold's worst case (~35%). The
  dissertation reports this honestly; the claim is "better on the typical
  stock and the basket", **never** "reduces drawdown on every ticker".

## Interpretation
- The probabilistic agents meet the **joint** objective — beating cash on
  risk-adjusted return while controlling typical-case drawdown versus
  buy-and-hold. The baseline meets neither half.
- The factor-of-1.6 gap in median terminal wealth, reproduced across ten
  seeds per ticker and confirmed out-of-sample by the walk-forward grid, is
  the single most informative result in the dissertation.
