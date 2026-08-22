# Phase 0 — Flappy Bird boards (same process as CartPole)
Last updated: 2026-08-06 · branch `simple-modelling-clean`

Same four beats: **predict → score → differentiate → update**  
Same algorithm boards as CartPole. Only `env` and the MDP table change.

---

## FlappyBird-v0 — the environment

| Piece | Value |
|---|---|
| State `s` | 12 numbers (bird y/vel, pipe positions, gap, …) |
| Actions | `0` do nothing · `1` flap |
| Reward | `+0.1` alive · `+1` pipe passed · `−1` death |
| Done | hit pipe or ground |

```text
env = make("FlappyBird-v0")   # use_lidar=False in our suite
# identical training boards to CartPole from here
```

---

## 1) REINFORCE on Flappy ✅

```text
policy = PolicyNet(obs=12, actions=2)   # π_θ
opt    = Adam(policy)

traj = ROLLOUT(env, policy)            # act until death
G = discounted_returns(traj.rewards)
L = - sum( log_prob(a|s) * G )         # SCORE

opt.zero_grad(); L.backward(); opt.step()
```

**Our result:** mean eval ≈ **7.1** (random ≈ **−7.4**).

Harder than CartPole: rewards are sparse (mostly small alive bonus until a pipe).

---

## 2) Actor–Critic (A2C) on Flappy ✅

```text
policy = PolicyNet(obs=12, actions=2)
critic = ValueNet(obs=12)
opt_pi, opt_V = Adam(), Adam()

B = ROLLOUT(env, policy)
A = returns(B) - critic(B.s)           # A = G − V(s)

L_pi = - sum( log_prob(a|s) * A )
L_V  = MSE( critic(s), returns )

opt_pi.zero_grad(); L_pi.backward(); opt_pi.step()
opt_V.zero_grad();  L_V.backward();  opt_V.step()
```

**Our result:** mean eval ≈ **4.6**.

Works, but under our fixed budget PPO did better (exploration / stability).

---

## 3) PPO on Flappy ✅

```text
policy = PolicyNet(obs=12, actions=2)
critic = ValueNet(obs=12)

B = ROLLOUT(env, policy)
A = advantages(B, critic)
L_pi = - PPO_CLIP(policy, B, A)
L_V  = MSE( critic(s), returns )

opt_pi.zero_grad(); L_pi.backward(); opt_pi.step()
opt_V.zero_grad();  L_V.backward();  opt_V.step()
```

**Our result:** mean eval ≈ **12.6** (best in our Flappy suite).

---

## 4) Tabular Q-learning on Flappy ❌ (as a plain table)

**Does not work in practice without heroic engineering.**

| Why | Explanation |
|---|---|
| State is 12 continuous numbers | No natural small bin grid like CartPole’s 4 numbers |
| Curse of dimensionality | Even 5 bins per number → 5¹² ≈ 244 million cells |
| Sparse reward | Most cells never visited → table stays empty |

```text
# NOT what we run:
Q = zeros(huge_bins, 2)    # memory / sample nightmare
```

**What we say to Nguyen:**  
“Tabular Q needs a finite index. Flappy’s 12 features make a table explode.  
That is exactly why DQN (network Q) or policy gradients are used.”

---

## 5) DQN on Flappy ✅

```text
Q, Q_bar = QNet(obs=12, actions=2), copy(...)
buf = ReplayBuffer()

a = eps_greedy(Q(s))
s2, r, done = env.step(a)
buf.add(...)
y = r + gamma * max(Q_bar(s2))     # Bellman target
L = MSE(Q(s,a), y)
opt.zero_grad(); L.backward(); opt.step()
```

**Our result:** mean eval ≈ **7.0**.

---

## Flappy summary

| Algorithm | Works? | Our mean return |
|---|---|---:|
| Random | baseline | −7.4 |
| REINFORCE | ✅ | 7.1 |
| Actor–Critic (A2C) | ✅ | 4.6 |
| PPO | ✅ | **12.6** |
| Tabular Q-learning | ❌ without huge binning | — |
| DQN | ✅ | 7.0 |

---

# Next: LunarLander (third game)

Open: `docs/phase0_lunarlander_boards.md`
