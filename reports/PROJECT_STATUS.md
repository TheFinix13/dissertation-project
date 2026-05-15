# Project Status Reference

**Last updated:** 2026-05-15
**Programme:** MSc EEEM004 — Electrical & Electronic Engineering, University of Surrey
**Supervisor:** Dr Cuong Nguyen

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
