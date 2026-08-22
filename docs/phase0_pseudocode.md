# Phase 0 — algorithms in Nguyen’s board style
Last updated: 2026-08-06 · branch `simple-modelling-clean`

Same four beats as his whiteboard and our trading figure:

**predict → score → differentiate → update**

Only the *data* and the *score* change.

Paste-ready figure (trading): `latex/tikz/rl_training_loop_template_word.png`  
Paste-ready figure (games): `latex/tikz/phase0_board_style_word.png`

**Per-game walkthroughs (start here):**
1. [`phase0_cartpole_boards.md`](phase0_cartpole_boards.md)
2. [`phase0_flappy_boards.md`](phase0_flappy_boards.md)
3. [`phase0_lunarlander_boards.md`](phase0_lunarlander_boards.md)

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

```text
minimise over θ :   average loss between  f(x; θ)  and  label y

# in words: make the network’s guess match the answer key
```

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

```text
maximise over θ :   average total discounted reward of an episode

actions:   a  ~  π_θ(a | s)
```

```text
policy = PolicyNet()          # π_θ(a|s)
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

```text
Q(s, a)  ←  Q(s, a)  +  α · [ r  +  γ · max_a' Q(s', a')  −  Q(s, a) ]

# new Q = old Q + step_size × (target − old Q)
# target = reward + discounted best future Q
```

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
policy = PolicyNet()           # actor  π_θ
critic = ValueNet()            # critic V_φ
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

```text
maximise over θ :   clipped PPO score   L_CLIP(θ)
minimise over φ :   value error         L_V(φ)
```

```text
policy = PolicyNet()           # π_θ
critic = ValueNet()            # V_φ
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
| CartPole / Flappy / Lunar boards | `docs/phase0_*_boards.md` |
| Trading SL↔RL figure | `latex/tikz/rl_training_loop_template_word.png` |
| Games board figure | `latex/tikz/phase0_board_style_word.png` |
| Scratch REINFORCE | `experiments/phase0_games/reinforce.py` |
| Plain-English story | `docs/EXPLAIN_THE_EXPERIMENTS.md` |
