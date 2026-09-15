# Final dissertation outline — locked 19 Aug 2026

Deadline: **1 September 2026** (EEEM004; 60–80 pages; IEEE references;
report = 80% technical achievement + 20% quality).

Direction agreed after Recording 52 (transcript:
`docs/nguyen_recording52_transcript.txt`):

- **One complete model** — a ~9-feature state, divisible ("Bitcoin-style")
  asset, fixed Discrete(3) actions. No iteration ladder as the experiment
  structure (Nguyen: "Don't do that. It's not going to work" [78:41]).
- **The progression survives as design rationale**: a minimal-state pilot
  study (already run) collapses to buy-and-hold; that motivates the
  complete state. Reasoning appears in Ch3; pilot numbers appear once in
  Ch5 as the preliminary study.
- **Algorithms: from-scratch REINFORCE + Deep Q-learning** (the two
  priorities [77:56]). PPO = stretch row only if time remains.
- **Corrected mathematics**: objective J(θ) with θ in the *policy*, the
  likelihood-ratio (log) derivation, and the empirical Monte-Carlo loss —
  the exact homework set in the meeting [61:00].
- Write order: Ch3 → Ch4 → Ch5 → Ch2 → Ch1 → Ch6 → abstract
  (methodology/experiments first, intro last — Nguyen's advice).

---

## Chapter 1 — Introduction (~5 pages, write LAST)

1.1 Motivation: sequential decision-making under uncertainty; why trading
    is a natural but hard RL testbed.
1.2 Problem statement: single-asset intraday-style trading as an episodic
    MDP with a fully observable state.
1.3 Objectives (measurable, matched to what is delivered):
    O1 formulate the trading MDP with a complete observable state;
    O2 derive policy-gradient (REINFORCE) and value-based (Deep Q)
       learning rules from first principles;
    O3 validate both implementations on games with known outcomes;
    O4 evaluate both on held-out real market data against passive and
       random baselines;
    O5 analyse failure modes (reward-induced policy collapse) and state
       limitations (partial observability).
1.4 Contributions + dissertation structure.

## Chapter 2 — Background & Literature Review (~10–12 pages)

2.1 RL foundations: MDP, return, value functions, policy.
2.2 Value-based methods: Q-learning → DQN (replay, target nets); the
    table-size / continuous-state limitation (Nguyen's mining example).
2.3 Policy-gradient methods: REINFORCE, score-function estimator,
    variance; baselines/advantage; brief A2C→PPO (clipping) as context.
2.4 RL for trading: FinRL and related work; known pitfalls — overfitting,
    transaction costs, buy-and-hold as the hard baseline, non-IID data.
2.5 Partial observability: POMDP/HMM framing — why state completeness
    matters (sets up our design choice and the limitation discussion).
2.6 Gap summary: understanding-first, honest-baseline evaluation of the
    two canonical algorithm families on one carefully formulated MDP.

## Chapter 3 — Methodology & Mathematical Formulation (~12–15 pages)

3.1 Overview: MDP tuple; robot/game ↔ trading mapping table (keep from v1).
3.2 Episode definition: one calendar month of daily bars as one episode,
    T ≈ 18–23; episodes treated as independent (game sessions); the
    non-IID caveat stated honestly. (Fixes the old 390-minute mismatch.)
3.3 The complete state vector (9 features) + design rationale:
    market block [ΔP, momentum_k, volatility_k, price/MA − 1],
    clock [τ = t/T],
    balance-sheet block [C/C0, position value/C0, unrealized PnL, W/C0].
    Subsection: why the minimal 3-feature state is insufficient (the
    Flappy-Bird "see the whole world" test; the pilot collapse preview).
3.4 Action space: Discrete(3) with a divisible asset — Buy/Sell a fixed
    monetary slice; feasibility masking; forced liquidation at T.
3.5 Reward: r_t = ΔW_t net of proportional fees; explicit worked cases.
3.6 The objective function — CORRECTED:
    J(θ) = E_{τ~π_θ}[ G(τ) ],  G(τ) = Σ_t γ^t r_t.
    θ parameterizes the POLICY (not the reward).
3.7 The policy-gradient theorem / likelihood-ratio derivation:
    ∇θ J = E[ Σ_t ∇θ log π_θ(a_t|s_t) · G_t ] — full derivation, then the
    Monte-Carlo empirical loss L(θ) = −(1/N) Σ_n Σ_t log π_θ · G_t.
    Explain: exact gradient vs sample approximation; why the log appears;
    why PyTorch minimises −J. (The meeting homework, answered in full.)
3.8 Deep Q-learning formulation: Q*(s,a), Bellman target, TD loss,
    replay buffer, target network, ε-greedy with action masking.
3.9 Algorithm boxes: REINFORCE (Alg 3.1) and DQN (Alg 3.2) side by side.
3.10 Evaluation protocol: chronological split, greedy evaluation,
     metrics (ΔW, trades, win rate vs buy-and-hold, Sharpe, max drawdown),
     standing diagnostics (accounting identity, illegal-action rate).

## Chapter 4 — Implementation (~8–10 pages)

4.1 System architecture: data → environment → agents → evaluation
    (diagram); repo layout.
4.2 Data pipeline: SPY 2018–2024 daily bars, chronological 58/26 month
    split, no shuffling.
4.3 TradingEnv (Gymnasium): state computation, fractional trade
    execution, fee accounting, masking, forced liquidation.
4.4 From-scratch REINFORCE: network, masked categorical sampling,
    return computation, the loss line ↔ equation 3.7 correspondence.
4.5 From-scratch Deep Q: network, replay, target sync, masked ε-greedy,
    the TD-loss line ↔ equation 3.8 correspondence.
4.6 Verification: accounting-identity unit tests, mask tests, Phase-0
    game validation (CartPole/Flappy/Lunar — implementations learn).
4.7 Hyperparameters table + budgets; seeds.

## Chapter 5 — Empirical Results & Analysis (~10–12 pages)

5.1 Experimental setup recap + metrics.
5.2 Phase 0 (games): algorithms validated on known-outcome tasks
    (existing table: random floors cleared; REINFORCE/DQN/PPO curves).
5.3 Preliminary study (pilot, minimal 3-D state): all policy-gradient
    optimisers collapse to buy-and-hold; caught only by the fair buy-max
    baseline. Motivates the complete state. (Real numbers, one table.)
5.4 Main results: REINFORCE vs Deep Q on the complete 9-D state vs
    B0/B1b/random baselines on 26 held-out months. Wealth paths, action
    distributions, per-month win rates. (+ PPO row if time.)
5.5 Analysis: what each algorithm learned; reward-vs-state attribution;
    fee sensitivity if time permits.
5.6 Diagnostics: accounting identity, illegal-action rates, seeds.

## Chapter 6 — Conclusions & Future Work (~4–5 pages)

6.1 Achievements vs objectives O1–O5.
6.2 Critical appraisal: what worked, what didn't, honest limitations
    (single asset, daily bars, single/few seeds, non-IID episodes,
    state incompleteness → POMDP/HMM future work).
6.3 Future work: minute bars (T=390), risk-aware rewards (ΔW − λ·D),
    continuous actions, uncertainty-aware extensions.
6.4 Reflection on project management (handbook requires this).

Abstract: write absolutely last.

---

## Experiment work queue (gates Ch4/Ch5)

1. `experiments/final_model/env_full.py` — 9-D state, fractional units.
2. `experiments/final_model/dqn_trading.py` — scratch DQN with masking.
3. Adapt `reinforce_trading.py` to obs_dim=9 (already generic).
4. Runner: baselines + REINFORCE + DQN (+PPO stretch), seeds {42, 43, 44},
   80k-step budget each, JSON + charts out.
5. Reuse pilot results from `results/phase1_algo_ablation.json` for §5.3.
