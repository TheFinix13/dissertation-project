# Phase 0 games explained — viva-ready notes
Last updated: 2026-08-04 · branch `simple-modelling-iteration1`

Why games before trading (Nguyen, Recording 47): a game has a **known correct
outcome**, so if the RL loop fails we know the *code* is wrong, not the market.
Trading has no ground truth, so we validate the loop on games first.

## The three games and what each one tests

| Game | State | Actions | Reward | What it proves |
|---|---|---|---|---|
| CartPole-v1 | 4 numbers (cart pos/vel, pole angle/ang-vel) | 2 (push left/right) | +1 per step upright, cap 500 | The basic policy loop learns at all; solved bar ≈ 475 |
| FlappyBird-v0 | 12 features (pipes, bird pos/vel) | 2 (flap / nothing) | +0.1 alive, +1 pipe, −1 death | Sparse rewards + survival — harder exploration |
| LunarLander-v3 | 8-D (pos, vel, angle, leg contacts) | 4 discrete engines | shaped; +100 land / −100 crash; solved ≈ 200 | Multiple discrete actions managing a "position" — closest to trading |

## The five algorithms (one sentence each)

1. **Random** — no learning; the floor every method must beat.
2. **Tabular Q-learning** — a lookup table updated by the Bellman rule
   \(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'}Q(s',a') - Q(s,a)]\).
   Only works after hand-binning CartPole's continuous state — the table idea
   cannot scale to trading states, which motivates everything after it.
3. **DQN** — replaces the table with a neural network trained by MSE to the
   Bellman target (experience replay + frozen target net). Value-based.
4. **REINFORCE** — the simplest *policy* method; implemented from scratch here
   (`reinforce.py`, no SB3): loss \(= -\sum_t \log\pi_\theta(a_t|s_t)\,G_t\).
   High variance but the direct ancestor of what we use in trading.
5. **A2C / PPO** — actor–critic policy gradients; PPO adds the clipped ratio
   \(\min(\rho_t A_t,\ \mathrm{clip}(\rho_t,1\pm\epsilon)A_t)\) so one update
   can't destroy the policy. PPO is the algorithm carried into Phase 1.

## Value-based vs policy-based (the viva question)

Value methods (Q-learning, DQN) learn "how good is each action" and act
greedily; they struggle when the action set or state grows and cannot output
stochastic policies. Policy methods (REINFORCE, A2C, PPO) learn the action
distribution directly — which is why Nguyen locked the dissertation to
policy-based RL for trading.

## Mapping to the trading MDP (Iteration 1)

| Game concept | Trading equivalent |
|---|---|
| Pole angle / lander position | \(s_t = [\Delta P_t, C_t, n_t]\) |
| Push left/right, flap | Hold / Buy one / Sell one |
| +1 per step upright | \(r_t = W_{t+1} - W_t\) |
| Episode = one game | Episode = one trading period |
| Solved threshold 475 | Beat buy-and-hold honestly |

## Where the numbers live

- `experiments/phase0_games/results/{cartpole,flappy,lunarlander}/suite_metrics.json`
- `experiments/phase0_games/results/cartpole/qlearning_metrics.json`
- Charts: `reports/generated/charts/phase0_{cartpole,flappy,lunarlander}_{comparison,curves}.png`
- Scratch REINFORCE source: `experiments/phase0_games/reinforce.py`
