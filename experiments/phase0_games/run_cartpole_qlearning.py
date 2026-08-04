#!/usr/bin/env python3
"""Phase 0 — TRUE tabular Q-learning on discretized CartPole.

Demonstrates the Q-table update Nguyen drew:
  Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]

CartPole's state is continuous, so a Q-table only works after binning each
of the 4 state variables. That discretization requirement is exactly why
tabular Q-learning does NOT scale to trading states — the point we make in
Chapter 4 before moving to function approximation (DQN) and policy methods.

Run:
  .venv311/bin/python experiments/phase0_games/run_cartpole_qlearning.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "phase0_games" / "results" / "cartpole"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
EPISODES = 20_000
EVAL_EPS = 30

# Bin edges per state variable (position, velocity, angle, angular velocity)
N_BINS = (6, 6, 12, 12)
LOWS = np.array([-2.4, -3.0, -0.21, -3.0])
HIGHS = np.array([2.4, 3.0, 0.21, 3.0])


def discretize(obs: np.ndarray) -> tuple[int, ...]:
    clipped = np.clip(obs, LOWS, HIGHS)
    ratios = (clipped - LOWS) / (HIGHS - LOWS)
    idx = (ratios * (np.array(N_BINS) - 1)).round().astype(int)
    return tuple(idx.tolist())


def main() -> None:
    rng = np.random.default_rng(SEED)
    env = gym.make("CartPole-v1")
    q = np.zeros(N_BINS + (env.action_space.n,), dtype=np.float64)

    alpha, gamma = 0.1, 0.99
    eps_start, eps_end = 1.0, 0.02
    returns: list[float] = []

    for ep in range(EPISODES):
        eps = max(eps_end, eps_start * (1 - ep / (0.8 * EPISODES)))
        obs, _ = env.reset(seed=SEED + ep)
        s = discretize(np.asarray(obs))
        done, total = False, 0.0
        while not done:
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(q[s]))
            obs, r, term, trunc, _ = env.step(a)
            s2 = discretize(np.asarray(obs))
            # The Bellman TD update — the whole algorithm in one line
            q[s + (a,)] += alpha * (r + gamma * np.max(q[s2]) * (not term) - q[s + (a,)])
            s = s2
            total += float(r)
            done = term or trunc
        returns.append(total)

    # Greedy evaluation
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

    mean_eval = float(np.mean(eval_totals))
    payload = {
        "algorithm": "tabular_q_learning",
        "family": "value_based (true Q-table)",
        "env": "CartPole-v1 (discretized)",
        "bins": list(N_BINS),
        "episodes": EPISODES,
        "mean_eval_return": mean_eval,
        "train_return_last100_mean": float(np.mean(returns[-100:])),
        "episode_returns_every100": returns[::100],
        "note": (
            "Q-table on 6x6x12x12 bins. Works only because we hand-binned a 4-D state; "
            "a trading state with continuous cash/prices makes the table blow up — "
            "motivating DQN (network Q) and policy-gradient methods."
        ),
    }
    out = OUT / "qlearning_metrics.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("mean_eval_return", "train_return_last100_mean")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
