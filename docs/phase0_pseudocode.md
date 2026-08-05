# Phase 0 — Pseudocode you can defend to Nguyen
Last updated: 2026-08-05 · branch `simple-modelling-clean`

**Purpose:** show the *algorithms*, not library calls. If he asks “did AI write
this?”, you walk the `▷` comments line by line. Style matches his papers
(procedures + triangle comments).

**Code that implements this**
- Scratch REINFORCE: `experiments/phase0_games/reinforce.py` (no SB3)
- Tabular Q-learning: `experiments/phase0_games/run_cartpole_qlearning.py`
- A2C / PPO / DQN: Stable-Baselines3 wrappers in `suite.py` — same update
  maths as below; SB3 is the optimiser plumbing, not a different algorithm.

---

## 0. Shared game loop (every algorithm uses this)

```
procedure ROLLOUT(env, policy_or_q, max_steps):
  ▷ One episode: reset → act → observe → repeat until done
  s ← env.RESET()
  buffer ← ∅
  for t = 0 … max_steps − 1:
      a ← SELECT-ACTION(policy_or_q, s)     ▷ method depends on algorithm
      s′, r, done ← env.STEP(a)
      store (s, a, r, s′, done) in buffer
      s ← s′
      if done: break
  return buffer
```

The **environment** is what changes between CartPole and Flappy Bird.
The **update rule** is what changes between Q-learning / DQN / REINFORCE / PPO.

---

## 1. CartPole MDP (what you write on the board first)

```
CartPole-v1
  State  s ∈ ℝ⁴:
      s₁ = cart position
      s₂ = cart velocity
      s₃ = pole angle
      s₄ = pole angular velocity
  Actions A = {0: push LEFT, 1: push RIGHT}
  Reward  r_t = +1 every step the pole is upright
  Terminal: pole falls |cart| too large | t = 500 (cap)
  Solved ≈ mean return ≥ 475 over evaluation episodes
```

**Why this MDP:** continuous state, discrete actions, known success bar —
same *shape* as trading (numbers in → Hold/Buy/Sell out) without market noise.

---

## 2. Flappy Bird MDP

```
FlappyBird-v0   (use_lidar = False → 12-D feature vector)
  State  s ∈ ℝ¹²: bird y / velocity, next pipe positions, gaps, …
  Actions A = {0: do nothing, 1: FLAP}
  Reward:
      +0.1  each frame alive
      +1    each pipe passed
      −1    on death (terminal)
  Terminal: bird hits pipe or ground
```

**Same algorithms as CartPole.** Only `RESET` / `STEP` / reward change.
That is the point: if the update maths is right, swapping the env should still learn.

---

## 3. Tabular Q-learning (value-based; CartPole only after binning)

Continuous \(s\) cannot index a table → **discretize** each of the 4 coords into bins
(we used \(6×6×12×12\)). Trading cash/prices would need infinite bins → this is
why we do **not** use tabular Q for stocks.

```
procedure TRAIN-TABULAR-Q(env, episodes, α, γ, ε):
  ▷ Q is a table: Q[bin(s), a] ≈ expected return of action a in state s
  initialise Q[·,·] ← 0
  for episode = 1 … episodes:
      s ← env.RESET()
      while not done:
          ▷ ε-greedy: explore randomly or exploit best Q
          with probability ε:  a ← Uniform(A)
          else:                a ← argmax_a′ Q[bin(s), a′]
          s′, r, done ← env.STEP(a)
          ▷ Bellman / TD update (the whole algorithm in one line)
          target ← r + γ · max_a′ Q[bin(s′), a′] · (1 − done)
          Q[bin(s), a] ← Q[bin(s), a] + α · (target − Q[bin(s), a])
          s ← s′
  return Q
```

**Say this:** “The target is reward plus discounted best future Q. We pull the
current entry toward that target by learning rate α.”

---

## 4. DQN (Deep Q-Network) — table → neural net

Same Bellman target; \(Q_\phi(s,a)\) is a network. Needs replay buffer + target net
for stability (otherwise training oscillates).

```
procedure TRAIN-DQN(env, steps, γ, ε-schedule):
  initialise online net Q_φ and target net Q_φ̄ ← Q_φ
  initialise replay buffer D ← ∅
  s ← env.RESET()
  for t = 1 … steps:
      a ← ε-greedy from Q_φ(s, ·)
      s′, r, done ← env.STEP(a)
      store (s, a, r, s′, done) in D
      sample mini-batch B ⊂ D
      for each (s_i, a_i, r_i, s′_i, done_i) in B:
          ▷ Bellman target using the frozen target network
          y_i ← r_i + γ · max_a′ Q_φ̄(s′_i, a′) · (1 − done_i)
          ▷ MSE loss: push Q_φ(s_i,a_i) toward y_i
      φ ← φ − ∇_φ (1/|B|) Σ_i ( Q_φ(s_i,a_i) − y_i )²
      every C steps: φ̄ ← φ          ▷ freeze target for stability
      s ← s′ if not done else env.RESET()
  return Q_φ
```

**Say this:** “DQN is Q-learning with a network, a memory of past transitions,
and a slowly copied target network so the target doesn’t chase itself.”

---

## 5. REINFORCE (policy gradient — we implemented this from scratch)

No Q-table. Directly learn \(\pi_\theta(a\mid s)\). After an episode finishes,
increase probability of actions that led to high return \(G_t\).

```
procedure TRAIN-REINFORCE(env, episodes, γ, η):
  ▷ π_θ(a|s) = Categorical( softmax( network_θ(s) ) )
  initialise policy parameters θ
  for episode = 1 … episodes:
      ▷ Collect one full trajectory
      τ ← ROLLOUT(env, π_θ)
      ▷ Compute discounted returns G_t from the end of the episode
      G_T ← 0
      for t = T−1 … 0:
          G_t ← r_t + γ · G_{t+1}
      ▷ Optional: normalise {G_t} (zero mean, unit variance) to reduce noise
      ▷ Policy-gradient loss (ascent on expected return = minimise negative)
      L(θ) ← − Σ_t log π_θ(a_t | s_t) · G_t
      θ ← θ − η ∇_θ L(θ)            ▷ Adam / SGD
  return θ
```

**Say this (maps to Nguyen’s board):**
1. **Predict** — network outputs action probabilities  
2. **Score** — weight \(\log\pi\) by return \(G_t\) (not MSE to a label \(y\))  
3. **Differentiate** — \(\nabla_\theta L\)  
4. **Update** — step \(\theta\)

This is exactly `experiments/phase0_games/reinforce.py`.

---

## 6. A2C (Advantage Actor–Critic)

REINFORCE is noisy because \(G_t\) has high variance. A2C adds a **critic**
\(V_\phi(s)\) so we update the actor with an *advantage*
\(A_t = G_t - V_\phi(s_t)\) (“how much better than average was this action?”).

```
procedure TRAIN-A2C(env, steps, γ, η):
  initialise actor π_θ and critic V_φ
  while total_steps < steps:
      collect a short rollout of n steps (or until done)
      for each step t in rollout:
          A_t ← (Σ_{k≥0} γ^k r_{t+k}) − V_φ(s_t)     ▷ advantage
      ▷ Actor: increase log-prob of actions with positive advantage
      L_actor(θ)  ← − Σ_t log π_θ(a_t|s_t) · A_t
      ▷ Critic: regress V_φ toward the return
      L_critic(φ) ← Σ_t ( V_φ(s_t) − G_t )²
      update θ, φ by gradient descent on L_actor + L_critic
  return θ, φ
```

---

## 7. PPO (what we carry into trading)

A2C can take too-large policy steps and collapse. PPO keeps a copy of the
**old** policy and **clips** the probability ratio.

```
procedure TRAIN-PPO(env, steps, γ, ε_clip, K_epochs):
  initialise actor π_θ and critic V_φ
  while total_steps < steps:
      ▷ Collect rollout with current policy; store also log π_θ_old(a|s)
      B ← ROLLOUT-BATCH(env, π_θ)
      compute advantages A_t (GAE or Monte-Carlo − V_φ)
      for epoch = 1 … K_epochs:          ▷ reuse the same batch several times
          ρ_t ← π_θ(a_t|s_t) / π_θ_old(a_t|s_t)     ▷ probability ratio
          ▷ Clipped surrogate — do not trust huge policy changes
          L_CLIP(θ) ← − mean_t [
              min( ρ_t A_t ,
                   clip(ρ_t, 1−ε_clip, 1+ε_clip) · A_t )
          ]
          L_V(φ) ← mean_t ( V_φ(s_t) − G_t )²
          update θ, φ on L_CLIP + c_v L_V
      θ_old ← θ
  return θ, φ
```

**Say this:** “ρ is how much more (or less) likely the new policy is to take
the same action. The clip stops one update from destroying a working policy.
ε_clip ≈ 0.2 in our runs.”

---

## 8. How CartPole and Flappy share the same pseudocode

```
procedure PHASE0-SUITE(env_id ∈ {CartPole-v1, FlappyBird-v0}):
  ▷ Identical training procedures; only the env changes
  results ← ∅
  results[Random]    ← EVALUATE(env_id, random_policy)
  results[REINFORCE] ← EVALUATE(env_id, TRAIN-REINFORCE(env_id))
  results[A2C]       ← EVALUATE(env_id, TRAIN-A2C(env_id))
  results[PPO]       ← EVALUATE(env_id, TRAIN-PPO(env_id))
  results[DQN]       ← EVALUATE(env_id, TRAIN-DQN(env_id))
  if env_id = CartPole-v1:
      results[TabularQ] ← EVALUATE(env_id, TRAIN-TABULAR-Q(env_id))
      ▷ Tabular Q skipped for Flappy: state is continuous / high-dim
  return results
```

That is literally what `experiments/phase0_games/suite.py` orchestrates.

---

## 9. Whiteboard order if he asks you to derive one

Pick **REINFORCE** (shortest, and we wrote it ourselves):

1. Write \(\pi_\theta(a|s)\)  
2. Write return \(G_t = r_t + \gamma r_{t+1} + \cdots\)  
3. Write \(L = -\sum_t \log\pi_\theta(a_t|s_t)\,G_t\)  
4. Say: “gradient ascent on expected return; Adam steps on −L”  
5. Optional: “PPO = same idea + critic advantage + clip on ρ”

Then point to CartPole numbers: random 28.8 → REINFORCE 292 → PPO 500.

---

## 10. What is *not* pseudocode (don’t pretend otherwise)

| Piece | Honest status |
|---|---|
| REINFORCE | Full loop in our repo — you can open the file |
| Tabular Q | Full loop in our repo |
| A2C / PPO / DQN | Same maths; training loop executed via Stable-Baselines3 |
| Env physics | Gymnasium / flappy-bird-gymnasium (we do not re-derive gravity) |

If he pushes on SB3: “I treat SB3 as a tested optimiser for the PPO/A2C/DQN
objectives above; I derived and coded REINFORCE myself to show I own the
policy-gradient update.”
