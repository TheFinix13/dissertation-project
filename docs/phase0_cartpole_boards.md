# Phase 0 — CartPole boards (one algorithm at a time)
Last updated: 2026-08-06 · branch `simple-modelling-clean`

Four beats every time: **predict → score → differentiate → update**

Paste figure: `latex/tikz/rl_training_loop_template_word.png`  
Games figure: `latex/tikz/phase0_board_style_word.png`

---

## CartPole-v1 — the environment (shared by all algorithms)

| Piece | Value |
|---|---|
| State `s` | 4 numbers: cart pos, cart vel, pole angle, pole ang-vel |
| Actions | `0` left · `1` right |
| Reward | `+1` every step pole stays up |
| Done | pole falls, cart off track, or t = 500 |
| Solved | mean return ≈ **475–500** (random ≈ **29**) |

```text
env = make("CartPole-v1")
s, info = env.reset()
# every algorithm only differs AFTER this shared loop shape:
a = ...                      # how we choose a
s2, r, done, trunc, info = env.step(a)
```

All five methods below **can** run on CartPole.  
The only special case: **tabular** Q-learning needs binning (see §4).

---

## 1) REINFORCE on CartPole ✅  
*(from scratch — `experiments/phase0_games/reinforce.py`)*

**Objective (read this like the whiteboard)**

```text
maximise over θ :   average of  [ sum of  γ^t · r_t  over the episode ]

actions:   a_t  ~  π_θ(a | s)     # sample from the policy
```

```text
policy = PolicyNet(obs=4, actions=2)   # π_θ
opt    = Adam(policy)

# --- one episode ---
traj = []
s = env.reset()
while not done:
    probs = policy(s)                  # 1 PREDICT
    a     = sample(probs)              # left / right
    s2, r, done = env.step(a)
    traj.append(s, a, r)
    s = s2

G = discounted_returns(traj.rewards)   # G_t from the end
L = - sum( log_prob(a|s) * G )         # 2 SCORE

opt.zero_grad()
L.backward()                           # 3 DIFFERENTIATE
opt.step()                             # 4 UPDATE
```

**Our result:** mean eval ≈ **292** (beats random 29; below PPO’s 500).

**Say to Nguyen:** “No labels. Score = −log π × return. Same four beats as his ResNet board.”

---

## 2) Actor–Critic (A2C) on CartPole ✅

REINFORCE is noisy because `G` varies a lot. Add a **critic** V_φ(s):

```text
advantage:   A_t  =  G_t  −  V_φ(s_t)
# “how much better was this action than average for this state?”
```

```text
policy = PolicyNet(obs=4, actions=2)   # actor
critic = ValueNet(obs=4)               # critic
opt_pi, opt_V = Adam(), Adam()

B = ROLLOUT(env, policy)               # data by acting
A = returns(B) - critic(B.s)           # advantage

L_pi = - sum( log_prob(a|s) * A )      # 2 SCORE (actor)
L_V  = MSE( critic(s), returns )       # 2 SCORE (critic)

opt_pi.zero_grad(); L_pi.backward(); opt_pi.step()   # 3+4
opt_V.zero_grad();  L_V.backward();  opt_V.step()
```

**Our result:** mean eval ≈ **500** (solved).

**Say to Nguyen:** “Actor = policy. Critic = baseline. Advantage = how much better than average.”

---

## 3) PPO on CartPole ✅  
*(same board we use for trading)*

```text
policy = PolicyNet(obs=4, actions=2)
critic = ValueNet(obs=4)
opt_pi, opt_V = Adam(), Adam()

B = ROLLOUT(env, policy)
A = advantages(B, critic)

L_pi = - PPO_CLIP(policy, B, A)        # 2 SCORE (clipped ratio)
L_V  = MSE( critic(s), returns )

opt_pi.zero_grad(); L_pi.backward(); opt_pi.step()
opt_V.zero_grad();  L_V.backward();  opt_V.step()
```

**Clip idea (one line):**

```text
ρ = π_new(a|s)  /  π_old(a|s)
# don’t trust huge policy jumps — clip ρ near 1 (e.g. between 0.8 and 1.2)
```

**Our result:** mean eval ≈ **500** (solved).

**Say to Nguyen:** “Trading uses this exact board; only `ROLLOUT(env)` becomes `ROLLOUT(prices)`.”

---

## 4) Tabular Q-learning on CartPole ✅ *with binning*

**Can it work?** Yes — **only after** we turn continuous `s` (4 real numbers) into bins.

**Why binning?** A Q-table needs a finite index. Raw cart position is continuous → infinite cells.

We used bins `6 × 6 × 12 × 12`.

**Bellman update (whiteboard form)**

```text
Q(s, a)  ←  Q(s, a)  +  α · [ r  +  γ · max_over_a' Q(s', a')  −  Q(s, a) ]

# in words:
#   new Q  =  old Q  +  step_size × (target − old Q)
#   target =  reward + discounted best future Q
```

```text
Q = zeros(n_bins, 2)                   # table, not a net

s = env.reset()
while not done:
    a = eps_greedy(Q[bin(s)])          # 1 PREDICT
    s2, r, done = env.step(a)

    y = r + gamma * max(Q[bin(s2)])    # 2 SCORE
    Q[bin(s), a] += alpha * (y - Q[bin(s), a])   # 3+4 in one line
    s = s2
```

**Our result:** mean eval ≈ **500** (solved).

**Say to Nguyen:** “Works on CartPole because we hand-bin 4 numbers. Trading cash/prices would explode the table — that motivates DQN / policy methods.”

---

## 5) DQN on CartPole ✅

Same Bellman target as §4; network replaces the table.

```text
Q     = QNet(obs=4, actions=2)
Q_bar = copy(Q)                        # frozen target
buf   = ReplayBuffer()
opt   = Adam(Q)

s = env.reset()
while training:
    a = eps_greedy(Q(s))               # 1 PREDICT
    s2, r, done = env.step(a)
    buf.add(s, a, r, s2, done)

    batch = buf.sample()
    y = r + gamma * max(Q_bar(s2))     # 2 SCORE
    L = MSE( Q(s,a), y )

    opt.zero_grad(); L.backward(); opt.step()   # 3+4
    every C steps: Q_bar ← Q
    s = s2 if not done else env.reset()
```

**Our result:** mean eval ≈ **500** (solved).

---

## CartPole summary (show this table)

| Algorithm | Works on CartPole? | Caveat | Our mean return |
|---|---|---|---:|
| Random | baseline | — | 28.8 |
| REINFORCE | ✅ | noisy | 292 |
| Actor–Critic (A2C) | ✅ | — | **500** |
| PPO | ✅ | — | **500** |
| Tabular Q-learning | ✅ | **must bin state** | **500** |
| DQN | ✅ | needs replay + target net | **500** |

**None of the five is impossible on CartPole.**  
The only “can’t work *as written*” case is tabular Q **without** discretization.

---

# Next: Flappy Bird (same boards, different env)

Open: `docs/phase0_flappy_boards.md`
