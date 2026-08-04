# ai_context.md · simple-modelling-clean
Last updated: 2026-08-04

Compact state for the **Nguyen simple-modelling track only**.
Orphan branch `simple-modelling-clean` — no shared history with `main` / v0.17.

## 1) What is built and working

- **Phase 0 multi-algo games.** Random / tabular Q-learning / DQN /
  scratch-REINFORCE / A2C / PPO. Seed 42, 30-episode greedy eval.
  CartPole: random 28.8; REINFORCE 292.2; Q-table/DQN/A2C/PPO **500**.
  Flappy: PPO best **12.6**. LunarLander-v3: PPO **176.4**, DQN 177.6
  (solved ≈ 200). Scripts: `experiments/phase0_games/`. Charts:
  `reports/generated/charts/phase0_*_{comparison,curves}.png`.
  Viva: `docs/phase0_games_explained.md`.
- **Phase 1 Iteration 1 (3-D state).** SPY daily months, 58 train / 26 test,
  fee 5 bps, cash \$10k, PPO 80k. Mean ΔW: B0 0 · B1(one) \$8 · **B1b \$76.9**
  · B2 \$14 · **A1 PPO \$76.9**. Honest: PPO = buy-max on **26/26** months.
- **Phase 1 Iteration 2 (5-D state).**
  $s=[\Delta P,\mathrm{PnL},\tau,C/C_0,nP/C_0]$. Breaks buy-max collapse
  (0/26 identical) but converges mostly-flat: deterministic ΔW **0.0**,
  stochastic +\$1.87, beats B1b in 8/26 falling months. Next lever:
  reward $\Delta W-\lambda D_t$.
- **Chapters 4–5 (new drafts).** `latex/simple_modelling/` — LaTeX + PDF +
  Word. Do **not** overwrite the user's University Word skeletons.

## 2) Key paths

| Area | Path |
|---|---|
| Branch | `simple-modelling-clean` (orphan, on origin) |
| Phase 0 | `experiments/phase0_games/` |
| Phase 1 | `experiments/iteration1/` |
| Chapters | `latex/simple_modelling/` |
| Docs | `docs/{phase0_games_explained,tutor_sl_to_rl_training_loop,nguyen_meeting_memo_jul30,going_beyond_recorder}.md` |

## 3) Next immediate goal

1. Iteration 3: reward $r_t=\Delta W_t-\lambda D_t$; re-run SPY months; fill Ch5.
2. Multi-seed CIs + one minute-bar day ($T=390$).
3. Keep parked: live brokers; merging into `main`; restoring v0.17 dual-path as the meeting story.
