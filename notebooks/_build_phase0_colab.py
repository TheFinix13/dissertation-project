#!/usr/bin/env python3
"""Build notebooks/phase0_games_colab.ipynb (run once; do not import from Colab)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "phase0_games_colab.ipynb"


def md(text: str) -> dict:
    lines = text.strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": uuid.uuid4().hex[:8],
        "source": [ln + "\n" for ln in lines.split("\n")[:-1]] + ([lines.split("\n")[-1] + "\n"] if lines else []),
    }


def code(text: str) -> dict:
    lines = text.strip("\n") + "\n"
    src = [ln + "\n" for ln in lines.split("\n")]
    # drop final extra blank from trailing newline handling
    if src and src[-1] == "\n":
        src = src[:-1]
        if src:
            src[-1] = src[-1]  # keep last line newline
    # simpler: one string splitlines keepends
    src = [ln + "\n" for ln in text.strip("\n").split("\n")]
    if src:
        src[-1] = src[-1].rstrip("\n") + "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


cells: list[dict] = []

cells.append(md(r"""
# Phase 0 — Games before trading (Nguyen demo)

**Student:** Fiyinfoluwa Akano · URN 6962514 · EEEM004
**Branch:** `simple-modelling-clean`
**Purpose:** Show that a **policy-based RL training loop** works on games with a *known* outcome, before we trust it on trading (no ground truth).

Four beats every time (same as his classification board):

**predict → score → differentiate → update**

| Game | Why it is here |
|---|---|
| **CartPole-v1** | Fast, solved bar ≈ 475–500 — correctness sandbox |
| **FlappyBird-v0** | Harder / sparser rewards — closer to “survive + score” |

Algorithms: **Random**, **REINFORCE (from scratch)**, **A2C**, **PPO**, **DQN**, plus **tabular Q-learning** on CartPole only.

> **Meeting tip:** set `RUN_MODE = "cached"` below — charts and numbers appear in seconds from the repo metrics (no long train). Use `live_fast` if he wants to see training happen live.
"""))

cells.append(md(r"""
## 0 — Setup (Colab)

1. Runtime → Change runtime type → **CPU is fine** (GPU optional).
2. Run the install cell once.
3. Set `RUN_MODE` in the config cell.
"""))

cells.append(code(r"""
# Installs (Colab). Safe to re-run.
import sys, subprocess

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

IN_COLAB = "google.colab" in sys.modules
print("IN_COLAB =", IN_COLAB)

pip_install(
    "gymnasium==0.29.1",
    "stable-baselines3==2.3.2",
    "torch",
    "matplotlib",
    "pandas",
    "numpy",
    "flappy-bird-gymnasium",
    "pygame",
)

if IN_COLAB:
    # Headless display for Flappy / pygame
    subprocess.check_call(["apt-get", "-qq", "update"])
    subprocess.check_call(
        ["apt-get", "-qq", "install", "-y", "xvfb"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pip_install("pyvirtualdisplay")
    from pyvirtualdisplay import Display
    _disp = Display(visible=0, size=(800, 600))
    _disp.start()
    print("Virtual display started")

print("Installs OK")
"""))

cells.append(code(r"""
# Config — change RUN_MODE for the meeting
from pathlib import Path
import os, sys, json, time
import numpy as np
import matplotlib.pyplot as plt

# "cached"    → load dissertation metrics from GitHub (seconds) — USE THIS IN MEETING
# "live_fast" → retrain with small budgets (~5–15 min CartPole; Flappy longer)
# "full"      → budgets matching the dissertation numbers (can be 30–90+ min)
RUN_MODE = "cached"

SEED = 42
EVAL_EPS = 30

BUDGETS = {
    "live_fast": {
        "cartpole_sb3": 15_000,
        "cartpole_reinforce_eps": 300,
        "cartpole_q_eps": 5_000,
        "flappy_sb3": 40_000,
        "flappy_reinforce_eps": 400,
    },
    "full": {
        "cartpole_sb3": 50_000,
        "cartpole_reinforce_eps": 800,
        "cartpole_q_eps": 20_000,
        "flappy_sb3": 150_000,
        "flappy_reinforce_eps": 1_500,
    },
}

REPO_URL = "https://github.com/TheFinix13/dissertation-project.git"
BRANCH = "simple-modelling-clean"
CLONE_DIR = Path("/content/dissertation-project") if IN_COLAB else Path(".").resolve()

def ensure_repo() -> Path:
    here = Path(".").resolve()
    if (here / "experiments" / "phase0_games").exists():
        print("Using local repo:", here)
        return here
    # walk parents (when notebook is under notebooks/)
    for p in [here, *here.parents]:
        if (p / "experiments" / "phase0_games").exists():
            print("Using local repo:", p)
            return p
    if IN_COLAB:
        import subprocess
        if CLONE_DIR.exists():
            subprocess.check_call(["rm", "-rf", str(CLONE_DIR)])
        subprocess.check_call([
            "git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(CLONE_DIR)
        ])
        print("Using cloned repo:", CLONE_DIR)
        return CLONE_DIR
    raise FileNotFoundError("Run from the dissertation repo, or open this notebook in Colab.")

ROOT = ensure_repo()
GAMES = ROOT / "experiments" / "phase0_games"
sys.path.insert(0, str(GAMES))
OUT = Path("/content/phase0_colab_out") if IN_COLAB else (ROOT / "notebooks" / "_colab_out")
OUT.mkdir(parents=True, exist_ok=True)
print("RUN_MODE =", RUN_MODE)
print("GAMES    =", GAMES)
"""))

cells.append(md(r"""
## 1 — Why games first? (say this to Nguyen)

Trading has **no known correct action**. If the agent fails, you cannot tell whether:

- the *code / training loop* is wrong, or
- the *market* simply did not reward that policy.

A game has a known success signal (CartPole stays up; Flappy passes pipes).
If the loop fails on CartPole, the implementation is wrong — stop before trading.

That is Recording 47 homework: **policy-based RL on a game first**, then transfer the same four beats to the trading MDP.
"""))

cells.append(md(r"""
## 2 — The four beats + objectives (readable LaTeX)

### Supervised board (his classification example)

$$
\min_{\theta}\; \frac{1}{N}\sum_{n=1}^{N} \ell\!\bigl(f(x_n;\theta),\, y_n\bigr)
$$

### RL objective (no labels $y$ — collect trajectories by acting)

$$
\max_{\theta}\; \mathbb{E}_{\tau\sim\pi_\theta}\!\left[\sum_{t=0}^{T-1}\gamma^{t}\, r_t\right]
\qquad\text{with}\qquad
a_t \sim \pi_\theta(a\mid s_t)
$$

### REINFORCE score (what we coded from scratch)

$$
L(\theta) = -\sum_t \log\pi_\theta(a_t\mid s_t)\, G_t
\qquad
G_t = \sum_{k=0}^{T-t-1}\gamma^{k}\, r_{t+k}
$$

### Tabular Q-learning (Bellman — CartPole only, after binning)

$$
Q(s,a)\;\leftarrow\; Q(s,a) + \alpha\Bigl[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\Bigr]
$$

### PPO (what we take into trading)

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\!\left[\min\bigl(\rho_t A_t,\;\mathrm{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\,A_t\bigr)\right]
\qquad
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
$$

Same four beats as ResNet: **predict** (policy) → **score** (PPO / value loss) → **backward** → **Adam step**.
"""))

cells.append(md(r"""
## 3 — Shared helpers (same code as the repo)

We import the dissertation modules so the notebook is not a second “shadow” implementation.
"""))

cells.append(code(r"""
from common import EpisodeReturnCallback, evaluate_random, evaluate_sb3, make_env
from reinforce import train_reinforce, evaluate_reinforce
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import torch

print("Imports OK")
print("CartPole sample obs:", make_env("CartPole-v1").reset(seed=0)[0])
"""))

cells.append(code(r"""
# Plotting helpers

def moving_avg(x, w=20):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def plot_curves(results, title, ylabel="Episode return"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, blob in results.items():
        rets = blob.get("episode_returns") or []
        if not rets:
            continue
        ax.plot(moving_avg(rets, 20), label=name)
    ax.set_title(title)
    ax.set_xlabel("Episode (smoothed)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_bars(means, title, solved=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(means.keys())
    vals = [means[k] for k in names]
    colors = ["#888888" if k == "random" else "#2a6fdb" for k in names]
    ax.bar(names, vals, color=colors)
    if solved is not None:
        ax.axhline(solved, color="green", ls="--", lw=1, label=f"solved ≈ {solved}")
        ax.legend()
    ax.set_title(title)
    ax.set_ylabel("Mean eval return")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def load_cached_suite(game):
    path = GAMES / "results" / game / "suite_metrics.json"
    return json.loads(path.read_text())


def load_cached_q():
    path = GAMES / "results" / "cartpole" / "qlearning_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None

print("Helpers ready")
"""))

cells.append(md(r"""
---
# Part A — CartPole-v1

| Piece | Value |
|---|---|
| State $s$ | cart pos, cart vel, pole angle, pole ang-vel (4-D) |
| Actions | $0$ = push left · $1$ = push right |
| Reward | $+1$ each step upright (cap $500$) |
| Solved | mean return $\approx 475$–$500$ |
"""))

cells.append(code(r"""
# Peek one random CartPole episode (length ≈ how long the pole stayed up)
env = make_env("CartPole-v1")
obs, _ = env.reset(seed=0)
total, steps = 0.0, 0
done = False
while not done:
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    total += float(r)
    steps += 1
    done = term or trunc
env.close()
print(f"Random episode: return={total:.0f}, steps={steps}")
print("(Random mean over 30 eps in our suite ≈ 29 — far below solved 500.)")
"""))

cells.append(md(r"""
### A.1 REINFORCE — from scratch (open the loss)

This is the non-black-box policy method. Loss matches the LaTeX above:
$L = -\sum_t \log\pi_\theta(a_t\mid s_t)\, G_t$.
"""))

cells.append(code(r"""
# Show the core of reinforce.py (same file Nguyen can ask to open)
src = (GAMES / "reinforce.py").read_text()
start = src.find("def train_reinforce")
print(src[start:start + 1800])
"""))

cells.append(md(r"""
### A.2 Train or load CartPole suite

- **cached:** dissertation numbers (seed 42)
- **live_fast / full:** trains Random → REINFORCE → A2C → PPO → DQN → tabular Q
"""))

cells.append(code(r"""
def train_sb3_cartpole(algo_cls, name, timesteps, **kwargs):
    env = Monitor(gym.make("CartPole-v1"))
    model = algo_cls("MlpPolicy", env, verbose=0, seed=SEED, **kwargs)
    cb = EpisodeReturnCallback()
    model.learn(total_timesteps=timesteps, callback=cb)
    mean = evaluate_sb3(model, "CartPole-v1", n_episodes=EVAL_EPS, seed=SEED)
    env.close()
    return {
        "algorithm": name,
        "mean_eval_return": mean,
        "episode_returns": cb.returns,
        "episode_timesteps": cb.timesteps,
    }


def train_tabular_q(episodes):
    # Bellman Q-table on binned CartPole (same idea as run_cartpole_qlearning.py)
    N_BINS = (6, 6, 12, 12)
    LOWS = np.array([-2.4, -3.0, -0.21, -3.0])
    HIGHS = np.array([2.4, 3.0, 0.21, 3.0])

    def discretize(obs):
        clipped = np.clip(obs, LOWS, HIGHS)
        ratios = (clipped - LOWS) / (HIGHS - LOWS)
        idx = (ratios * (np.array(N_BINS) - 1)).round().astype(int)
        return tuple(idx.tolist())

    rng = np.random.default_rng(SEED)
    env = gym.make("CartPole-v1")
    q = np.zeros(N_BINS + (2,), dtype=np.float64)
    alpha, gamma = 0.1, 0.99
    returns = []
    for ep in range(episodes):
        eps = max(0.02, 1.0 * (1 - ep / (0.8 * episodes)))
        obs, _ = env.reset(seed=SEED + ep)
        s = discretize(np.asarray(obs))
        done, total = False, 0.0
        while not done:
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(q[s]))
            obs, r, term, trunc, _ = env.step(a)
            s2 = discretize(np.asarray(obs))
            q[s + (a,)] += alpha * (r + gamma * np.max(q[s2]) * (not term) - q[s + (a,)])
            s = s2
            total += float(r)
            done = term or trunc
        returns.append(total)

    eval_totals = []
    for ep in range(EVAL_EPS):
        obs, _ = env.reset(seed=SEED + 100_000 + ep)
        s = discretize(np.asarray(obs))
        done, total = False, 0.0
        while not done:
            a = int(np.argmax(q[s]))
            obs, r, term, trunc, _ = env.step(a)
            s = discretize(np.asarray(obs))
            total += float(r)
            done = term or trunc
        eval_totals.append(total)
    env.close()
    return {
        "algorithm": "tabular_q",
        "mean_eval_return": float(np.mean(eval_totals)),
        "episode_returns": returns[:: max(1, episodes // 200)],
    }


cartpole = {}
q_metrics = None

if RUN_MODE == "cached":
    payload = load_cached_suite("cartpole")
    cartpole = payload["results"]
    q_metrics = load_cached_q()
    print("Loaded cached CartPole suite from repo.")
else:
    b = BUDGETS[RUN_MODE]
    t0 = time.time()
    print("=== Random ===")
    cartpole["random"] = {
        "algorithm": "random",
        "mean_eval_return": evaluate_random("CartPole-v1", EVAL_EPS, SEED),
        "episode_returns": [],
    }
    print(" ", cartpole["random"]["mean_eval_return"])

    print("=== REINFORCE ===")
    rf = train_reinforce("CartPole-v1", total_episodes=b["cartpole_reinforce_eps"], seed=SEED)
    cartpole["reinforce"] = {
        "algorithm": "REINFORCE",
        "mean_eval_return": evaluate_reinforce(rf.policy, "CartPole-v1", EVAL_EPS, SEED),
        "episode_returns": rf.episode_returns,
    }
    print(" ", cartpole["reinforce"]["mean_eval_return"])

    print("=== A2C ===")
    cartpole["a2c"] = train_sb3_cartpole(
        A2C, "a2c", b["cartpole_sb3"], learning_rate=7e-4, n_steps=5, gamma=0.99
    )
    print(" ", cartpole["a2c"]["mean_eval_return"])

    print("=== PPO ===")
    cartpole["ppo"] = train_sb3_cartpole(
        PPO, "ppo", b["cartpole_sb3"],
        learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99,
    )
    print(" ", cartpole["ppo"]["mean_eval_return"])

    print("=== DQN ===")
    cartpole["dqn"] = train_sb3_cartpole(
        DQN, "dqn", b["cartpole_sb3"],
        learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000,
        batch_size=64, gamma=0.99, target_update_interval=500, exploration_fraction=0.3,
    )
    print(" ", cartpole["dqn"]["mean_eval_return"])

    print("=== Tabular Q ===")
    q_metrics = train_tabular_q(b["cartpole_q_eps"])
    print(" ", q_metrics["mean_eval_return"])
    print(f"CartPole live train wall time: {time.time() - t0:.0f}s")

rows = {k: float(v["mean_eval_return"]) for k, v in cartpole.items()}
if q_metrics is not None:
    rows["tabular_q"] = float(q_metrics["mean_eval_return"])
print("\nCartPole mean eval returns:")
for k, v in rows.items():
    print(f"  {k:12s}  {v:7.1f}")

plot_bars(rows, f"CartPole — mean eval return ({RUN_MODE})", solved=475)
plot_curves(cartpole, f"CartPole — learning curves ({RUN_MODE})")
"""))

cells.append(md(r"""
### A.3 What to say about CartPole

| Method | Family | Takeaway |
|---|---|---|
| Random | floor | ~29 |
| REINFORCE | policy (scratch) | learns (~292) — proves we own the update |
| A2C / PPO / DQN / tabular Q | AC / PG / value | hit **500** (solved) under our budget |

**Viva line:** “All learners beat random. PPO is the algorithm we carry into trading because it is policy-based and stable. Tabular Q only works after hand-binning — that does not scale to cash/price trading states.”
"""))

cells.append(md(r"""
---
# Part B — FlappyBird-v0

Same algorithms, harder exploration (sparse pipe reward + death).
**Tabular Q is impractical** here (12-D continuous features → table explodes) — same reason it fails for trading.

| Piece | Value |
|---|---|
| State | 12 features (bird + pipes; `use_lidar=False`) |
| Actions | $0$ = do nothing · $1$ = flap |
| Reward | $+0.1$ alive · $+1$ pipe · $-1$ death |
"""))

cells.append(code(r"""
flappy = {}

if RUN_MODE == "cached":
    payload = load_cached_suite("flappy")
    flappy = payload["results"]
    print("Loaded cached Flappy suite from repo.")
else:
    b = BUDGETS[RUN_MODE]
    env_id = "FlappyBird-v0"
    t0 = time.time()

    def train_sb3_flappy(algo_cls, name, timesteps, **kwargs):
        env = Monitor(make_env(env_id))
        model = algo_cls("MlpPolicy", env, verbose=0, seed=SEED, **kwargs)
        cb = EpisodeReturnCallback()
        model.learn(total_timesteps=timesteps, callback=cb)
        mean = evaluate_sb3(model, env_id, n_episodes=EVAL_EPS, seed=SEED)
        env.close()
        return {
            "algorithm": name,
            "mean_eval_return": mean,
            "episode_returns": cb.returns,
            "episode_timesteps": cb.timesteps,
        }

    print("=== Random ===")
    flappy["random"] = {
        "algorithm": "random",
        "mean_eval_return": evaluate_random(env_id, EVAL_EPS, SEED),
        "episode_returns": [],
    }
    print(" ", flappy["random"]["mean_eval_return"])

    print("=== REINFORCE ===")
    rf = train_reinforce(env_id, total_episodes=b["flappy_reinforce_eps"], seed=SEED)
    flappy["reinforce"] = {
        "algorithm": "REINFORCE",
        "mean_eval_return": evaluate_reinforce(rf.policy, env_id, EVAL_EPS, SEED),
        "episode_returns": rf.episode_returns,
    }
    print(" ", flappy["reinforce"]["mean_eval_return"])

    print("=== A2C ===")
    flappy["a2c"] = train_sb3_flappy(
        A2C, "a2c", b["flappy_sb3"], learning_rate=7e-4, n_steps=5, gamma=0.99
    )
    print(" ", flappy["a2c"]["mean_eval_return"])

    print("=== PPO ===")
    flappy["ppo"] = train_sb3_flappy(
        PPO, "ppo", b["flappy_sb3"],
        learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99,
    )
    print(" ", flappy["ppo"]["mean_eval_return"])

    print("=== DQN ===")
    flappy["dqn"] = train_sb3_flappy(
        DQN, "dqn", b["flappy_sb3"],
        learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000,
        batch_size=64, gamma=0.99, target_update_interval=500, exploration_fraction=0.3,
    )
    print(" ", flappy["dqn"]["mean_eval_return"])
    print(f"Flappy live train wall time: {time.time() - t0:.0f}s")
    print("Note: tabular Q skipped — continuous 12-D state (same scaling argument as trading).")

frows = {k: float(v["mean_eval_return"]) for k, v in flappy.items()}
print("\nFlappy mean eval returns:")
for k, v in frows.items():
    print(f"  {k:12s}  {v:7.1f}")

plot_bars(frows, f"FlappyBird — mean eval return ({RUN_MODE})")
plot_curves(flappy, f"FlappyBird — learning curves ({RUN_MODE})")
"""))

cells.append(md(r"""
### B.1 What to say about Flappy

Under the dissertation budget (cached): **PPO ≈ 12.6** was the strongest mean eval; Random is negative (~−7).
Harder than CartPole — that is the point. Policy methods still beat chance.

**Bridge to trading:** replace `ROLLOUT(Flappy)` with `ROLLOUT(prices)` — same PPO board, new MDP:

$$
s_t = [\Delta P_t,\, C_t,\, n_t]^\top,\qquad
W_t = C_t + n_t P_t,\qquad
r_t = W_{t+1}-W_t
$$

Actions: Hold / Buy one / Sell one (with a feasibility mask).
"""))

cells.append(md(r"""
---
# Part C — Side-by-side summary + viva lines
"""))

cells.append(code(r"""
import pandas as pd

crows = {k: float(v["mean_eval_return"]) for k, v in cartpole.items()}
if q_metrics is not None:
    crows["tabular_q"] = float(q_metrics["mean_eval_return"])
frows = {k: float(v["mean_eval_return"]) for k, v in flappy.items()}

all_keys = sorted(set(crows) | set(frows), key=lambda k: (k != "random", k))
df = pd.DataFrame(
    {
        "CartPole": [crows.get(k, float("nan")) for k in all_keys],
        "FlappyBird": [frows.get(k, float("nan")) for k in all_keys],
    },
    index=all_keys,
)
display(df.round(1))

print(
    "\nViva checklist (answer out loud):\n"
    "1. Why games before trading?\n"
    "2. What are the four beats?\n"
    "3. Why is REINFORCE in the suite (not only SB3 PPO)?\n"
    "4. Why tabular Q on CartPole but not Flappy / trading?\n"
    "5. Which algorithm goes into Phase 1 trading, and why?\n"
)
"""))

cells.append(md(r"""
## Optional — watch a short CartPole rollout

Requires `rgb_array` render. Skip if it fails on your runtime.
"""))

cells.append(code(r"""
# Optional: 1 greedy CartPole episode with a freshly trained tiny PPO (fast)
WATCH = False

if WATCH:
    from IPython.display import HTML, display
    import matplotlib.animation as animation
    from matplotlib import rc
    rc("animation", html="jshtml")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    model = PPO("MlpPolicy", env, verbose=0, seed=SEED)
    model.learn(total_timesteps=8_000)
    frames = []
    obs, _ = env.reset(seed=0)
    done = False
    while not done and len(frames) < 500:
        frames.append(env.render())
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(int(a))
        done = term or trunc
    env.close()

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=30)
    plt.close(fig)
    display(HTML(anim.to_jshtml()))
else:
    print("Set WATCH = True and re-run to animate a short CartPole episode.")
"""))

cells.append(md(r"""
---
## Repo map (if he asks “where is the code?”)

| What | Path |
|---|---|
| This notebook | `notebooks/phase0_games_colab.ipynb` |
| Scratch REINFORCE | `experiments/phase0_games/reinforce.py` |
| Shared helpers | `experiments/phase0_games/common.py` |
| CartPole / Flappy runners | `run_cartpole_suite.py`, `run_flappy_suite.py` |
| Cached metrics | `experiments/phase0_games/results/{cartpole,flappy}/suite_metrics.json` |
| Board-style docs | `docs/phase0_*_boards.md` |

**Open in Colab:** upload this `.ipynb`, or from GitHub (after push)

`https://github.com/TheFinix13/dissertation-project/blob/simple-modelling-clean/notebooks/phase0_games_colab.ipynb`

→ “Open in Colab”.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT} ({len(cells)} cells)")
