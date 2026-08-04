# Action plan — understand, code, experiment (for next Nguyen meeting)
Last updated: 2026-07-30

Goal: walk into the next meeting able to **explain** every symbol and show
**evidence** that a policy-based RL loop works (Phase 0), plus a clear
Iteration-1 trading plan that matches his modelling board.

---

## Meeting pack (what we hand him)

| # | Item | Status target |
|---|---|---|
| 1 | SL → RL training-loop template (Word PNG) | ready |
| 2 | Algorithm 3.1 with \(W_t\), cash+shares state | ready |
| 3 | Phase 0: CartPole PPO reward curve + short explanation | **build now** |
| 4 | Phase 1: trading env smoke test (or clear next-step code) | build after 3 |
| 5 | 1-page memo: state/action/reward/baselines in his language | build with 3–4 |

---

## Track A — Understand (do before / while coding)

Spend 20–40 min per block. After each block, answer the checkpoint **out loud**.

### A1. Supervised template (Nguyen's green board)
- Read: `docs/tutor_sl_to_rl_training_loop.md` §§1–2
- Checkpoint: name the four beats; what is \(y_n\)?

### A2. RL vs supervised
- Read: same doc §3 + `latex/tikz/rl_training_loop_template_word.png`
- Checkpoint: why RL has no \(y_n\); what replaces the dataset?

### A3. Our Iteration-1 model
- Read: Algorithm 3.1 PNG; know \(s_t=[\Delta P_t,C_t,n_t]^\top\), \(W_t\), mask
- Checkpoint: why not \(H\in\{0,1\}\)? (Recording 47)

### A4. Experiments outline
- Read: this file Track B–C; `docs/iteration1_experiment_plan.md`
- Checkpoint: why Phase 0 game before trading?

---

## Track B — Phase 0 code (Nguyen's homework)

**Why CartPole first (not Flappy immediately):** same lesson he wants —
policy PPO, known success signal, fast to run — with Gymnasium built-in.
If he insists on Flappy visuals, we add it as Phase 0b after CartPole works.

| Step | Task | Done when |
|---|---|---|
| B1 | Install 3.11 env + gymnasium + SB3 | `python -c "import gymnasium, stable_baselines3"` |
| B2 | Script `experiments/phase0_cartpole/train_ppo.py` | trains PPO, logs episodic return |
| B3 | Script `experiments/phase0_cartpole/plot_learning.py` | PNG: return vs timestep |
| B4 | One-page `docs/phase0_cartpole_explained.md` | you can map CartPole state/action/reward to robot analogy |

**Success test for meeting:** curve rises above random; you can say
“predict → score PPO loss → backward → step” while pointing at the plot.

---

## Track C — Phase 1 trading (only after B succeeds)

| Step | Task | Done when |
|---|---|---|
| C1 | `experiments/iteration1/env.py` — Discrete(3), mask, \(W_t\) reward | unit tests / smoke rollout |
| C2 | Baselines B0/B1/B2 | CSV of ΔW on synthetic or SPY minutes |
| C3 | PPO train tiny budget | at least 1 seed finishes |
| C4 | Eval table + one wealth plot | in `reports/generated/` |

If time is short before the meeting: finish **B fully** + **C1 smoke** +
baselines on synthetic prices. Real SPY minute download can be next.

---

## Track D — Day-by-day (aggressive, “until tired”)

| Block | Focus |
|---|---|
| Now | Action plan + Phase 0 env + first CartPole train |
| Next | Phase 0 plot + explained.md; start Iteration-1 env |
| Then | Baselines smoke + meeting memo |
| Buffer | Rehearse viva answers A1–A4; optional Flappy 0b |

---

## Non-goals before the meeting (Nguyen would scold these)

- Uncertainty / LSTM / dual-path / 70-ticker grid
- Q-learning “because a blog used it”
- Vague state \(H\in\{0,1\}\)
- Claiming Phase-2 v0.17 numbers as Iteration-1 results
