# CHECKPOINT — current state of the dissertation implementation

A living snapshot, re-issued at every major divergence in the project's
path. The compact state summary — the file every fresh chat reads first —
lives at `../ai_context.md`; this file is the deeper, less-frequently-updated
counterpart. The Phase-1-era status text lives in
[`../reports/PROJECT_STATUS.md`](../reports/PROJECT_STATUS.md). Newest
checkpoint on top.

`ai_context.md` is the daily-truth file: numbers, deployed code paths,
next subgoal. `CHECKPOINT.md` is the deep-state file: methodology gates,
evidence summary, supervisor checkpoints, parked items with unpark
conditions. Update `ai_context.md` after every meaningful session;
update `CHECKPOINT.md` only at major divergences.

---

## Checkpoint 2026-06-16

### Headline state

Phase-2 complete. Held-out 2022–2025 test window: uncertainty-aware
agents grow $1M → median ~$1.6M (Sharpe +0.68 to +0.70, win-rate
89–92%); baseline PPO finishes flat at ~$999,063 (Sharpe −0.04, win-rate
46.1%). Walk-forward 320-cell grid: 87.2% win-rate across all four
folds. LaTeX dissertation in `latex/` is the canonical source; chapters
1–7 + abstract + appendices A/B/C are populated. Last `main.pdf`
compile: 2026-06-01.

Drawdown is honestly framed in the discussion: buy-and-hold ~25.9%
median is the proper benchmark, and the guard wins on the median stock
and basket but loses on the worst-case tail (~57% vs ~35%).

### Validation methodology gates

1. **Phase-1 ablation** — baseline PPO + probabilistic-PPO + benchmarks
   + rule baselines on the development universe.
2. **Phase-2 grid** — full `market_sample` (70 stocks) × 10 seeds ×
   50,000 PPO timesteps per cell.
3. **Walk-forward** — four-fold rolling-window validation; consistency
   across folds is the gate.
4. **MOBO** — multi-objective Bayesian optimisation over Sharpe vs MDD
   to surface the Pareto-optimal hyperparameter set.
5. **Honest benchmarking** — buy-and-hold AND a rule-based trailing
   stop-loss are reported alongside the agents.
6. **XAI** — SHAP + LIME post-hoc explanations on the deployed agent's
   action distribution.

### Parked

| Item | Why parked | Unpark condition |
|---|---|---|
| Full A100 walk-forward (~12–14 h GPU) | Phase-2 evidence already covers the headline | Dr Nguyen requests it as a robustness check |
| Promoting Phase-1 DOCX builders to current source of truth | Removed in v0.17.2 repo prune — LaTeX is canonical | N/A |
| Live broker integration | Forbidden in this repo | Never (lives in `eurusd-ai-agent` / `global-portfolio-assistant`) |

### Open questions

- Worst-case-tail drawdown — should the discussion chapter explore a
  hybrid (guard + small position-cap) as a future-work hook?
- Whether the abstract retune from "narrow-scope dissertation" framing
  to "capital preservation under regime stress" framing is locked, or
  needs another supervisor pass.

---

## THE ROUTINE — run this at every major divergence

A "divergence" is any change of methodology, scope, or deliverable: a
new evaluation window, a new algorithm, a chapter rewritten end-to-end,
a supervisor decision, an experimental gate added.

1. **Update the journey** — append the divergence to a dated review in
   this file or `reports/PROJECT_STATUS.md` (what changed, what was
   eliminated/added, why, with numbers).
2. **Snapshot current state** — add a new dated section at the TOP of
   this file: headline state, validation gates, parked list, open
   questions.
3. **Reconcile chapters + figures** — every number that changed in the
   snapshot must change in the LaTeX chapters AND the corresponding
   chart caption. Re-render charts via the brain-box `.mplstyle`.
4. **Sync the stale docs** — `reports/PROJECT_STATUS.md` and the
   chapter prose must not contradict the new snapshot.
5. **Prune stale artifacts** — Phase-1 DOCX pipeline and `reports/generated/exports/`
   were removed in the v0.17.2 repo prune (June 2026). LaTeX in `latex/` is the
   only document source of truth.
6. **Verify** — every notebook re-executes top-to-bottom in a clean
   kernel; `latex/build_docx.sh` runs clean; record the date here.
7. **Tell the user** — confirm in chat that the checkpoint moved and
   `ai_context.md` is fresh.
