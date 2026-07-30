# Nguyen meeting memo — simple modelling + Phase 0/1 evidence
**Student:** Fiyinfoluwa Akano · 6962514  
**Date:** 30 July 2026  
**Purpose:** Show understanding + a working policy-RL loop + first real-data table  
**Git branch:** `simple-modelling-iteration1` (v0.17 chapter rewrite kept off this track)

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
| Episode (Phase 1 now) | **one calendar month of daily closes** (\(T\approx 18\)–\(23\)) — honest interim; minute \(T=390\) is next |
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

---

## 4. Phase 1 result (SPY daily months — first real experiment)

**Data:** SPY adjusted close 2018–2024 · chronological split · **58 train / 26 test** months · fee 5 bps · cash \$10k · 80k PPO steps.

| Method | Mean ΔW (\$) | Std | Mean trades/mo | Win vs buy-max |
|---|---:|---:|---:|---:|
| B0 do-nothing | 0.0 | 0.0 | 0.0 | 31% |
| B1 buy-one hold | 8.0 | 16.7 | 1.0 | 31% |
| **B1b buy-max hold** | **76.9** | **178.7** | **19.1** | — |
| B2 random-masked | 14.1 | 32.9 | 12.8 | 31% |
| A1 PPO | **76.9** | **178.7** | **19.1** | **0%** |

**Honest finding (doer, not recorder):** deterministic PPO matched B1b on **26/26** test months. With this state and \(\Delta W\) reward, the learned policy *is* fully invested buy-and-hold. We only saw that because we added the fair buy-max baseline (one-share BAH looks “beaten” for the wrong reason).

Charts: `reports/generated/charts/phase1_spy_daily_delta_w.png`, `…/phase1_spy_daily_wealth_path.png`  
JSON: `experiments/iteration1/results/phase1_spy_daily_results.json`  
Accounting tests: all pass (`test_env.py`). Illegal PPO action proposals ≈ **3.9%** (env forces Hold).

---

## 5. What I added beyond what you dictated

See `docs/going_beyond_recorder.md`. Short list already done:
- accounting identity unit tests  
- fair **buy-max** baseline (caught the PPO collapse)  
- chronological split + explicit “not 390 minutes” disclaimer  
- illegal-action rate as a constraint diagnostic  

Next doer extras if you want them: from-scratch REINFORCE (no SB3), reward \(=\Delta W-\lambda D_t\), fee sensitivity, one real minute-bar day.

---

## 6. Questions for you

1. Is CartPole acceptable as Phase 0, or do you require Flappy Bird specifically?  
2. Is daily-month Phase 1 acceptable as an interim table, with minute \(T=390\) as the next upgrade?  
3. Given PPO collapsed to buy-max under pure \(\Delta W\), should Iteration-2 start with a **drawdown / turnover penalty**, or with a **richer state** first?
