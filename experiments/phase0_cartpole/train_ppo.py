#!/usr/bin/env python3
"""Phase 0 — CartPole PPO (Nguyen homework: policy RL on a known game).

Run (from repo root, with .venv311 active):
  python experiments/phase0_cartpole/train_ppo.py
  python experiments/phase0_cartpole/plot_learning.py
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "phase0_cartpole" / "results"
OUT.mkdir(parents=True, exist_ok=True)


class EpisodeReturnCallback(BaseCallback):
    """Log finished-episode returns for the learning curve."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.returns: list[float] = []
        self.timesteps: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.returns.append(float(info["episode"]["r"]))
                self.timesteps.append(int(self.num_timesteps))
        return True


def evaluate_policy(model: PPO, n_episodes: int = 20, seed: int = 0) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        total += ep_ret
    env.close()
    return total / n_episodes


def evaluate_random(n_episodes: int = 20, seed: int = 0) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        total += ep_ret
    env.close()
    return total / n_episodes


def main() -> None:
    total_timesteps = 50_000
    seed = 42

    env = Monitor(gym.make("CartPole-v1"))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        seed=seed,
    )
    cb = EpisodeReturnCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb)

    model_path = OUT / "ppo_cartpole.zip"
    model.save(str(model_path))

    random_mean = evaluate_random(n_episodes=30, seed=seed)
    ppo_mean = evaluate_policy(model, n_episodes=30, seed=seed)

    payload = {
        "env": "CartPole-v1",
        "algorithm": "PPO (policy-based — not Q-learning)",
        "total_timesteps": total_timesteps,
        "seed": seed,
        "random_mean_return_30ep": random_mean,
        "ppo_mean_return_30ep": ppo_mean,
        "episode_returns": cb.returns,
        "episode_timesteps": cb.timesteps,
        "model_path": str(model_path.relative_to(ROOT)),
        "four_beats": [
            "predict: policy(s) -> action probs",
            "score: PPO clipped surrogate + value loss",
            "differentiate: loss.backward()",
            "update: optimizer.step() on theta and phi",
        ],
        "state_meaning": {
            "s[0]": "cart position",
            "s[1]": "cart velocity",
            "s[2]": "pole angle",
            "s[3]": "pole angular velocity",
        },
        "actions": {0: "push left", 1: "push right"},
        "reward": "+1 per timestep the pole stays upright (episode ends on failure)",
    }
    metrics_path = OUT / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {metrics_path}")
    print(f"random mean return: {random_mean:.1f}")
    print(f"PPO mean return:    {ppo_mean:.1f}")


if __name__ == "__main__":
    main()
