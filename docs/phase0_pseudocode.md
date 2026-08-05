# Phase 0 — algorithms in Nguyen’s board style
Last updated: 2026-08-05 · branch `simple-modelling-clean`

Same four beats as his whiteboard and our trading figure:

**predict → score → differentiate → update**

Only the *data* and the *score* change.

Paste-ready figure (trading template): `latex/tikz/rl_training_loop_template_word.png`  
Paste-ready figure (games algorithms): `latex/tikz/phase0_board_style_word.png`

---

## The two games (env only — algorithms stay the same)

### CartPole-v1
| Piece | Meaning |
|---|---|
| State `s` | 4 numbers: cart pos, cart vel, pole angle, pole ang-vel |
| Actions | `0` = push left · `1` = push right |
| Reward | `+1` each step the pole stays up |
| Done | pole falls, or hit 500-step cap |
| Solved | mean return ≈ 475–500 |

### FlappyBird-v0
| Piece | Meaning |
|---|---|
| State `s` | 12 numbers: bird + next pipes |
| Actions | `0` = do nothing · `1` = flap |
| Reward | `+0.1` alive · `+1` pipe · `−1` death |
| Done | hit pipe / ground |

> Swap `env = make("CartPole-v1")` for `make("FlappyBird-v0")`.  
> The training boards below do not change.

---

## His board (classification) — reminder

**Objective**

\[
\min_\theta \frac{1}{N}\sum_n \ell\bigl(f(x_n;\theta),\, y_n\bigr)
\]

```text
net = ResNet18()
opt = Adam(...)

# data (x, y) fixed
preds = net(x)              # 1 PREDICT
loss  = MSE(preds, y) / N   # 2 SCORE

opt.zero_grad()
loss.backward()             # 3 DIFFERENTIATE
opt.step()                  # 4 UPDATE
```

---

## 1) REINFORCE — CartPole / Flappy  
*(we coded this from scratch — open `reinforce.py`)*

**Objective**

\[
\max_\theta \; \mathbb{E}_{\tau\sim\pi_\theta}\!\left[\sum_t \gamma^t r_t\right]
\qquad
a_t \sim \pi_\theta(a_t\mid s_t)
\]

```text
policy = PolicyNet()          # pi_theta(a|s)
opt    = Adam(policy)

# data created by acting (no labels y)
traj = []
s = env.reset()
while not done:                          # ── collect episode ──
    probs = policy(s)                    # 1 PREDICT
    a     = sample(probs)
    s2, r, done = env.step(a)
    traj.append(s, a, r)
    s = s2

G = discounted_returns(traj.rewards)     # how good was each step?
L = - sum( log_prob(a|s) * G )           # 2 SCORE  (like CE loss × G)

opt.zero_grad()
L.backward()                             # 3 DIFFERENTIATE
opt.step()                               # 4 UPDATE
```

**Board translation**

| Supervised | REINFORCE |
|---|---|
| fixed `(x, y)` | episode collected by acting |
| MSE / CE vs label `y` | `−log π(a\|s) · G` |
| minimise error | maximise return |

---

## 2) Tabular Q-learning — CartPole only (binned state)

**Objective (Bellman)**

\[
Q(s,a) \leftarrow Q(s,a) + \alpha\Bigl[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\Bigr]
\]

```text
Q = zeros(n_bins, n_actions)    # lookup table (not a net)

s = env.reset()
while not done:
    a = eps_greedy(Q[bin(s)])          # 1 PREDICT  (best / random)
    s2, r, done = env.step(a)

    target = r + gamma * max(Q[bin(s2)])   # 2 SCORE
    td_err = target - Q[bin(s), a]

    Q[bin(s), a] += alpha * td_err     # 3+4  “backward + step” in one line
    s = s2
```

**Why not for Flappy / trading?**  
State is continuous. We hand-bin CartPole into boxes. Trading cash/prices → infinite boxes → use a network (DQN) or a policy (REINFORCE/PPO).

---

## 3) DQN — same Bellman idea, network instead of table

```text
Q     = QNet()                 # online
Q_bar = copy(Q)                # frozen target
buf   = ReplayBuffer()
opt   = Adam(Q)

s = env.reset()
while training:
    a = eps_greedy(Q(s))               # 1 PREDICT
    s2, r, done = env.step(a)
    buf.add(s, a, r, s2, done)

    batch = buf.sample()
    y = r + gamma * max(Q_bar(s2))     # 2 SCORE  (target net)
    L = MSE( Q(s,a) , y )

    opt.zero_grad(); L.backward(); opt.step()   # 3 + 4
    every C steps: Q_bar ← Q
    s = s2 if not done else env.reset()
```

---

## 4) A2C — REINFORCE + a critic (less noisy)

```text
policy = PolicyNet()           # actor  pi_theta
critic = ValueNet()            # critic V_phi
opt_pi, opt_V = Adam(), Adam()

B = ROLLOUT(env, policy)               # data by acting
A = returns(B) - critic(B.s)           # advantage = “better than average?”

L_pi = - sum( log_prob(a|s) * A )      # 2 SCORE  (actor)
L_V  = MSE( critic(s) , returns )      # 2 SCORE  (critic)

opt_pi.zero_grad(); L_pi.backward(); opt_pi.step()   # 3 + 4
opt_V.zero_grad();  L_V.backward();  opt_V.step()
```

---

## 5) PPO — what we take into trading  
*(same board as the trading figure — only `env` changes)*

**Objective**

\[
\max_\theta \; L^{\mathrm{CLIP}}(\theta)
\;+\;
\min_\phi \; L^{V}(\phi)
\]

```text
policy = PolicyNet()           # pi_theta
critic = ValueNet()            # V_phi
opt_pi, opt_V = Adam(), Adam()

# data created by acting
B = ROLLOUT(env, policy)
A = advantages(B, critic)
L_pi = - PPO_CLIP(policy, B, A)        # 2 SCORE  (clipped)
L_V  = MSE( critic(s) , returns )      # 2 SCORE

opt_pi.zero_grad(); L_pi.backward(); opt_pi.step()   # 3 + 4
opt_V.zero_grad();  L_V.backward();  opt_V.step()
```

Compare to trading template: replace `ROLLOUT(prices, pi)` with `ROLLOUT(env, pi)`.  
**Same four beats.**

---

## Side-by-side cheat sheet

| Beat | His board | REINFORCE | PPO / A2C |
|---|---|---|---|
| **1 Predict** | `preds = net(x)` | `probs = policy(s)` | `a ~ policy(s)` in rollout |
| **2 Score** | `loss = MSE(preds,y)` | `L = -logπ · G` | `L_pi = -PPO_CLIP` · `L_V = MSE` |
| **3 Diff** | `loss.backward()` | `L.backward()` | `L_pi.backward()` / `L_V.backward()` |
| **4 Update** | `opt.step()` | `opt.step()` | `opt_pi.step()` / `opt_V.step()` |

---

## What to say if he asks “did AI write this?”

1. Point at **REINFORCE** board → open `experiments/phase0_games/reinforce.py`  
2. Say the four beats out loud on that file  
3. For PPO: “same board as trading; SB3 runs the Adam steps; I own the objective”  
4. For Q-learning: “Bellman one-liner; we bin CartPole; doesn’t scale to stocks”

---

## Files

| What | Where |
|---|---|
| This note | `docs/phase0_pseudocode.md` |
| Trading SL↔RL figure | `latex/tikz/rl_training_loop_template_word.png` |
| Games board figure | `latex/tikz/phase0_board_style_word.png` |
| Scratch REINFORCE | `experiments/phase0_games/reinforce.py` |
| Plain-English story | `docs/EXPLAIN_THE_EXPERIMENTS.md` |
