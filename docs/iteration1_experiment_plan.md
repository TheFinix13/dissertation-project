# Iteration-1 experiment plan — baselines, implement, results
Last updated: 2026-07-30 (updated after New Recording 47 / 23 July feedback)

Matches Nguyen's whiteboard Exp block:
Baselines (objective) → Implement → Results.

Also matches his explicit homework from the 23 July meeting:
**first learn policy-based RL on a simple game (Flappy Bird / similar), then transfer the same training loop to trading.**

---

## Phase 0 — correctness sandbox (THIS WEEK, non-negotiable)

Nguyen's words (paraphrased from transcript):

> Start with something everyone has done — Flappy Bird / a game.
> Learn a **policy** (not Q-learning).
> When the game works, you know the implementation is correct.
> Then move to trading — because in trading you do **not** know the correct outcome.

| Item | Choice |
|---|---|
| Environment | Flappy Bird clone **or** CartPole / LunarLander if Flappy setup is slow — but prefer a side-scroller with continuous play for 1-minute episodes |
| Algorithm | **Policy gradient / PPO** (policy-based). Do **not** start with tabular Q-learning |
| Episode | e.g. play for ≤ 1 minute (maps mentally to “one trading day”) |
| State | bird position / velocity / pipe gap (must be able to explain every component) |
| Action | flap / no-flap (or left/right/straight for a robot) |
| Reward | survival time / score — use the tutorial's standard reward first |
| Success test | agent visibly improves vs random; you can narrate predict→score→update |

**Deliverable for Phase 0:** a short note + 1 plot (reward vs episode) proving the policy learned.
Do not touch trading code until this works.

---

## Phase 1 — trading Iteration 1 (only after Phase 0)

### Scope lock

- One stock (SPY).
- Actions: Hold / Buy **one** / Sell **one**.
- State: \(s_t=[\Delta P_t, C_t, n_t]^\top\)  
  (Nguyen rejected compressing cash+shares into a vague \(H\in\{0,1\}\).)
- Reward: \(r_t=W_{t+1}-W_t\), \(W_t=C_t+n_tP_t\).
- Horizon: one trading day = one episode, \(T=390\) minute steps (state this in the report).

### A. Baselines and their objectives

| ID | Method | Decision rule | Explicit objective |
|---|---|---|---|
| B0 | Do-nothing | always Hold | preserve cash; no market exposure |
| B1 | Buy-and-hold | Buy once at first affordable bar, then Hold | capture full-day price move with 1 share |
| B2 | Random-masked | uniform over feasible actions | chance baseline |
| A1 | PPO (Alg 3.1) | learned \(\pi_\theta\) | maximise expected \(\sum_t (W_{t+1}-W_t)\) |

Same \(C_0\), fee \(c\), test days for every method.

### B. Implement

1. Chronological day split (no shuffle across days).
2. Modules: `env_iteration1.py`, `baselines_iteration1.py`, `train_ppo_iteration1.py`, `eval_iteration1.py`.
3. Tiny budget first: 3 seeds, 20–50 epochs, \(c=0.0005\).

### C. Results

- Table: mean ΔW, std, win-rate vs B1, mean trades/day.
- Fig: \(W_t\) vs minute for B1 vs A1 on one test day.
- Honesty checks: \(\sum r_t \approx W_T-W_0\); mask never allows illegal Buy/Sell; B0 with \(n_0=0\) has ΔW=0.

Losing to buy-and-hold on Iteration 1 is fine.
Not understanding why is not.

---

## Phase 2 — variants (only after Phase 1 numbers exist)

Nguyen: once the pipeline works, changing the reward/loss is like swapping MSE for cross-entropy.

Examples (later): drawdown penalty \(W_{t+1}-W_t-\lambda D_t\); different fee; continuous sizing.
Each variant is a new row in the experiment table — not a rewrite of the whole dissertation.
