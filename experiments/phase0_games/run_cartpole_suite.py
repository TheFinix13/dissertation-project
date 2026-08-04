#!/usr/bin/env python3
"""Phase 0 — CartPole multi-algorithm suite.

Algorithms (Nguyen: prefer policy-based; DQN included as value-based comparator):
  - Random
  - REINFORCE (scratch policy gradient — not a black box)
  - A2C (actor–critic)
  - PPO (clipped policy gradient)
  - DQN (Q-learning family / value-based)

Run:
  .venv311/bin/python experiments/phase0_games/run_cartpole_suite.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import EpisodeReturnCallback, evaluate_random, evaluate_sb3  # noqa: E402
from reinforce import evaluate_reinforce, train_reinforce  # noqa: E402

OUT = ROOT / "experiments" / "phase0_games" / "results" / "cartpole"
OUT.mkdir(parents=True, exist_ok=True)

ENV_ID = "CartPole-v1"
SEED = 42
TIMESTEPS = 50_000
EVAL_EPS = 30


def train_sb3(algo_cls, name: str, **kwargs):
    env = Monitor(gym.make(ENV_ID))
    model = algo_cls("MlpPolicy", env, verbose=0, seed=SEED, **kwargs)
    cb = EpisodeReturnCallback()
    model.learn(total_timesteps=TIMESTEPS, callback=cb)
    path = OUT / f"{name}_cartpole.zip"
    model.save(str(path))
    mean = evaluate_sb3(model, ENV_ID, n_episodes=EVAL_EPS, seed=SEED)
    return {
        "algorithm": name,
        "family": kwargs.pop("_family", "sb3"),
        "mean_eval_return": mean,
        "episode_returns": cb.returns,
        "episode_timesteps": cb.timesteps,
        "model": str(path.relative_to(ROOT)),
    }


def main() -> None:
    results = {}

    print("=== Random ===")
    random_mean = evaluate_random(ENV_ID, n_episodes=EVAL_EPS, seed=SEED)
    results["random"] = {
        "algorithm": "random",
        "family": "baseline",
        "mean_eval_return": random_mean,
        "episode_returns": [],
        "episode_timesteps": [],
    }
    print(f"  eval mean: {random_mean:.1f}")

    print("=== REINFORCE (scratch policy gradient) ===")
    # ~800 episodes ≈ similar wall-time ballpark; CartPole episodes are short
    rf = train_reinforce(ENV_ID, total_episodes=800, seed=SEED)
    rf_mean = evaluate_reinforce(rf.policy, ENV_ID, n_episodes=EVAL_EPS, seed=SEED)
    torch_path = OUT / "reinforce_cartpole.pt"
    import torch

    torch.save(rf.policy.state_dict(), torch_path)
    results["reinforce"] = {
        "algorithm": "REINFORCE",
        "family": "policy_gradient (scratch)",
        "mean_eval_return": rf_mean,
        "episode_returns": rf.episode_returns,
        "episode_timesteps": rf.episode_timesteps,
        "model": str(torch_path.relative_to(ROOT)),
        "note": "Vanilla Monte-Carlo policy gradient; explicit loss = -log π(a|s) * G_t",
    }
    print(f"  eval mean: {rf_mean:.1f}")

    print("=== A2C (actor–critic) ===")
    results["a2c"] = train_sb3(
        A2C, "a2c", learning_rate=7e-4, n_steps=5, gamma=0.99,
    )
    results["a2c"]["family"] = "actor_critic"
    print(f"  eval mean: {results['a2c']['mean_eval_return']:.1f}")

    print("=== PPO ===")
    results["ppo"] = train_sb3(
        PPO, "ppo", learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99,
    )
    results["ppo"]["family"] = "policy_gradient (clipped)"
    print(f"  eval mean: {results['ppo']['mean_eval_return']:.1f}")

    print("=== DQN (Q-learning family — value-based comparator) ===")
    results["dqn"] = train_sb3(
        DQN, "dqn", learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000,
        batch_size=64, gamma=0.99, target_update_interval=500, exploration_fraction=0.3,
    )
    results["dqn"]["family"] = "value_based (Q-learning / DQN)"
    print(f"  eval mean: {results['dqn']['mean_eval_return']:.1f}")

    payload = {
        "env": ENV_ID,
        "state_meaning": {
            "s[0]": "cart position",
            "s[1]": "cart velocity",
            "s[2]": "pole angle",
            "s[3]": "pole angular velocity",
        },
        "actions": {0: "push left", 1: "push right"},
        "reward": "+1 per timestep the pole stays upright (max episode length 500)",
        "solved_threshold": 475,
        "timesteps_budget_sb3": TIMESTEPS,
        "seed": SEED,
        "why_cartpole": (
            "Classic Gym control task: continuous 4-D state, discrete 2 actions, "
            "fast episodes, known 'solved' threshold. Used as a correctness sandbox "
            "before trading (Nguyen: verify policy RL on a game with a known outcome)."
        ),
        "nguyen_note": (
            "Primary dissertation path remains policy-based (REINFORCE / A2C / PPO). "
            "DQN is included only as a Phase-0 comparator so we can explain "
            "value-based vs policy-based RL."
        ),
        "results": results,
    }
    out = OUT / "suite_metrics.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    for k, v in results.items():
        print(f"  {k:12s}  mean_eval={v['mean_eval_return']:.1f}")


if __name__ == "__main__":
    main()
