# Nguyen meeting memo — simple modelling + Phase 0 evidence
**Student:** Fiyinfoluwa Akano · 6962514  
**Date:** 30 July 2026  
**Purpose:** Show understanding + a working policy-RL loop before trading claims

---

## 1. What I understood from your boards / Recording 47

**ML fundamental (image classification board):**  
predict → score loss → backward → step.  
Machine learning finds an unknown function \(f(x;\theta)\approx y\).

**RL is the same shape without labels:**  
state \(s\) plays the role of \(x\); action \(a\) plays the role of \(y\);  
reward replaces the supervised loss signal; trajectories replace a fixed dataset.

**Homework you set:** learn a **policy** on a simple game first (not Q-learning),  
then move to trading — because trading has no known correct outcome for debugging.

---

## 2. Iteration-1 trading model (explicit, as you requested)

| Piece | Definition |
|---|---|
| State | \(s_t=[\Delta P_t,\,C_t,\,n_t]^\top\) (cash + share **count**, not vague \(H\in\{0,1\}\)) |
| Actions | \(\{0:\mathrm{Hold},\,1:\mathrm{Buy\ one},\,2:\mathrm{Sell\ one}\}\) with feasibility mask |
| Wealth | \(W_t=C_t+n_tP_t\) |
| Reward | \(r_t=W_{t+1}-W_t\) |
| Episode | one trading day; \(T\) minute bars (390 on US cash equity session) |
| Learning | PPO on \(\pi_\theta\); value loss on \(V_\phi\) |

Algorithm figure: `latex/tikz/algorithm3_1_training_pipeline_word.png`  
SL↔RL template: `latex/tikz/rl_training_loop_template_word.png`

---

## 3. Phase 0 result (CartPole PPO — correctness sandbox)

| Metric | Value |
|---|---|
| Algorithm | PPO (policy-based) |
| Env | CartPole-v1 |
| Timesteps | 50,000 |
| Random eval mean return (30 eps) | **22.1** |
| PPO eval mean return (30 eps) | **500.0** (env cap — solved) |

Learning curve: `reports/generated/charts/phase0_cartpole_learning_curve.png`  
Explanation I can viva: `docs/phase0_cartpole_explained.md`

*(CartPole chosen as the fastest Gymnasium “everyone has done this” policy demo; Flappy Bird can follow as Phase 0b if you want the side-scroller analogy.)*

---

## 4. Experiment plan next (your Exp outline)

1. **Baselines (objective):** do-nothing, buy-and-hold, random-masked, PPO — each with an explicit objective.  
2. **Implement:** Iteration-1 env + train loop (smoke baselines already coded).  
3. **Results:** mean \(\Delta W\), trades/day, one-day wealth path — on chronological held-out days.

Synthetic baseline smoke: `experiments/iteration1/results/baseline_smoke.json`

---

## 5. Questions for you

1. Is CartPole acceptable as Phase 0, or do you require Flappy Bird specifically?  
2. For Iteration-1 data: minute bars preferred — is a short SPY window enough for the first results table?  
3. Should the first trading reward stay pure \(\Delta W\), with drawdown penalty only as a later variant?
