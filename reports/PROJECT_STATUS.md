# Project Status Reference

> **Historical note:** Sections below the "Interim Review Session" heading describe
> the April 2026 interim-review workflow. The Phase-1 DOCX builders and
> `reports/generated/exports/` were removed in the v0.17.2 repo prune (June 2026).
> The canonical dissertation is LaTeX in `latex/`.

**Last updated:** 2026-06-16
**Programme:** MSc EEEM004 — Electrical & Electronic Engineering, University of Surrey
**Supervisor:** Dr Cuong Nguyen

---

## Current Status — Phase-2 complete

**Phase-2 is finished.** The full experimental grid has been run across the
`market_sample` universe (70 stocks), ten seeds and 50,000 PPO time-steps per
cell, plus a four-fold walk-forward validation. The LaTeX dissertation in
`latex/`, both notebooks, and the chart suite in `reports/generated/charts/`
are all reconciled to these results and are the source of truth.

**Headline (held-out 2022–2025 test window):** the uncertainty-aware agents
grow a $1M portfolio to a **median of ~$1.6M** (median Sharpe **+0.68 to
+0.70**, win-rate **89–92%**); the baseline PPO finishes flat at **~$999,063**
(Sharpe **−0.04**, win-rate **46.1%**). The walk-forward grid (320 cells)
confirms the advantage on **all four folds** (overall 87.2% win-rate, median
final ~$1,167,050).

**Drawdown — honest framing:** the proper benchmark is **buy-and-hold (~25.9%
median)**, not the baseline PPO (whose ~1.4% drawdown is an under-trading
artefact). The guard beats buy-and-hold on the median stock and the index
basket (~22–24%) but **not** on the worst-case tail (guard worst ~57% >
buy-and-hold worst ~35%).

The interim-review feedback and Phase-1 notes below are retained as a
historical record of how the project reached this point.

---

## Project Overview

Probabilistic deep reinforcement learning for portfolio risk management.
An RL agent learns to rebalance a multi-asset portfolio by optimising a
risk-adjusted objective (Sharpe-like ratio) under distributional uncertainty.

## Supervisor Feedback — Interim Review Meeting

Key points Nguyen raised:

- **Proposed work is unclear.** Needs explicit MDP background: states, actions,
  reward function. The objective function must appear as a math expression to
  optimise, not just prose.
- **Training setting must be explicit.** Specify datasets, splits, horizons,
  hyperparameters — not just "we used PPO."
- **Baselines need detail.** Each baseline described with how it was trained and
  how it was evaluated; side-by-side comparison criteria stated upfront.
- **Technical progress ≠ results.** He wants methodology walkthrough — how we
  trained, what we trained on, why we got the numbers we got. Step-by-step in
  plain English any reader can follow.
- **Diagrams/images, not just tables.** Visual pipeline, RL loop, data splits.
- **Future plan:** literature review should be a standalone chapter or folded
  into Chapter 2 (background), not scattered.

## What We Changed This Session

1. Added **"Description of Proposed Work"** section — MDP definition
   (S, A, R, T), objective J(π), training setting, baseline descriptions.
2. Rewrote **Technical Progress** multiple times balancing detail against the
   form's 2–3 page limit.
3. Created `plot_interim_methodology_diagrams.py` — generates 3 figures:
   data splits, RL training loop, full pipeline overview.
4. Added `render_equation` / `render_equation_stack` helpers for italic math
   PNG generation (used in the DOCX builder).
5. Added **results interpretation** paragraphs after performance tables.
6. Updated **Future Plan** — lit review positioned as Chapter 2.

## Key Files

| File | Role |
|------|------|
| `reports/builders/build_interim_review_docx.py` | Main DOCX builder |
| `reports/builders/plot_interim_methodology_diagrams.py` | Methodology figures |
| `reports/generated/exports/InterimReview.docx` | Final output |
| `experiments/configs/dissertation_protocol.json` | Experiment protocol |
| `experiments/common.py` | `StockEnv` + `compute_metrics` |

## Still Pending

- **Technical progress rewrite.** Current version is too summary-like. Nguyen
  wants a step-by-step plain-English walkthrough of methodology — training
  procedure, data pipeline, evaluation — that any reader can follow without
  domain expertise.
- **Diagram polish.** Arrow styling, label placement, and colour consistency
  across the three methodology figures.
- **Equation rendering.** Visual quality of math PNGs could be improved
  (font weight, spacing, alignment in DOCX).
