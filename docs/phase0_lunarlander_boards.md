# Phase 0 — LunarLander boards (third game, same process)
Last updated: 2026-08-06 · branch `simple-modelling-clean`

Same four beats. Same algorithm boards. New env only.

Closest game analogy to trading: **several discrete actions** managing a “position”
(the lander), not just binary left/right or flap.

---

## LunarLander-v3 — the environment

| Piece | Value |
|---|---|
| State `s` | 8 numbers: x/y, vel x/y, angle, ang-vel, left-leg contact, right-leg contact |
| Actions | `0` noop · `1` left engine · `2` main engine · `3` right engine |
| Reward | shaped landing score; **+100** safe land · **−100** crash |
| Solved | mean return ≈ **200** |

```text
env = make("LunarLander-v3")
# training boards identical to CartPole / Flappy
```

---

## 1) REINFORCE on LunarLander ✅

```text
policy = PolicyNet(obs=8, actions=4)   # note: 4 actions now
opt    = Adam(policy)

traj = ROLLOUT(env, policy)
G = discounted_returns(traj.rewards)
L = - sum( log_prob(a|s) * G )

opt.zero_grad(); L.backward(); opt.step()
```

**Our result:** mean eval ≈ **15.3** (random ≈ **−183**).  
Learns vs random, but high variance — needs more episodes / a critic.

---

## 2) Actor–Critic (A2C) on LunarLander ✅ *(works, but weak under our budget)*

```text
policy = PolicyNet(obs=8, actions=4)
critic = ValueNet(obs=8)

B = ROLLOUT(env, policy)
A = returns(B) - critic(B.s)           # A = G − V(s)
L_pi = - sum( log_prob(a|s) * A )
L_V  = MSE( critic(s), returns )

opt_pi.zero_grad(); L_pi.backward(); opt_pi.step()
opt_V.zero_grad();  L_V.backward();  opt_V.step()
```

**Our result:** mean eval ≈ **−41**.  
Still beats random (−183), but did not reach “solved” in 150k steps with our settings.  
**Honest:** A2C *can* solve LunarLander with more tuning/time; our fixed suite budget was not enough for A2C here.

---

## 3) PPO on LunarLander ✅

```text
policy = PolicyNet(obs=8, actions=4)
critic = ValueNet(obs=8)

B = ROLLOUT(env, policy)
A = advantages(B, critic)
L_pi = - PPO_CLIP(policy, B, A)
L_V  = MSE( critic(s), returns )

opt_pi.zero_grad(); L_pi.backward(); opt_pi.step()
opt_V.zero_grad();  L_V.backward();  opt_V.step()
```

**Our result:** mean eval ≈ **176** (near solved ≈ 200).

---

## 4) Tabular Q-learning on LunarLander ❌ (practical table)

Same reason as Flappy, worse:

| Why | Explanation |
|---|---|
| 8 continuous numbers in the state | Binning explodes (e.g. 6 bins⁸ ≈ 1.7M+ cells, still coarse) |
| 4 actions | Table width grows too |
| Shaped + sparse terminal rewards | Hard to fill the table |

**Say to Nguyen:** “Possible only with crude bins + huge memory. Not a serious method here. DQN or PPO instead.”

---

## 5) DQN on LunarLander ✅

```text
Q = QNet(obs=8, actions=4)
Q_bar = copy(Q)
buf = ReplayBuffer()

a = eps_greedy(Q(s))                   # among 4 engines
y = r + gamma * max(Q_bar(s2))
L = MSE(Q(s,a), y)
opt.zero_grad(); L.backward(); opt.step()
```

**Our result:** mean eval ≈ **178** (ties PPO within noise).

---

## LunarLander summary

| Algorithm | Works? | Our mean return |
|---|---|---:|
| Random | baseline | −183 |
| REINFORCE | ✅ (weak under budget) | 15 |
| Actor–Critic (A2C) | ✅ (weak under budget) | −41 |
| PPO | ✅ | **176** |
| Tabular Q-learning | ❌ practical table | — |
| DQN | ✅ | **178** |

---

## Master takeaway across all three games

| Algorithm | CartPole | Flappy | LunarLander |
|---|---|---|---|
| REINFORCE | ✅ | ✅ | ✅ |
| Actor–Critic | ✅ | ✅ | ✅ (needs more budget here) |
| PPO | ✅ | ✅ best | ✅ |
| Tabular Q | ✅ if binned | ❌ | ❌ |
| DQN | ✅ | ✅ | ✅ |

**Policy methods (REINFORCE → A2C → PPO) always fit.**  
**Tabular Q only fits the tiny binned CartPole.**  
That is the bridge to trading: continuous cash/prices → no table → **PPO**.
