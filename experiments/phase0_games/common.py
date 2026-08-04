"""Shared helpers for the Phase 0 game suites (CartPole / Flappy / LunarLander)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeReturnCallback(BaseCallback):
    """Collect per-episode returns (and the cumulative timestep at episode end)
    from SB3's Monitor wrapper, so we can plot learning curves later."""

    def __init__(self):
        super().__init__()
        self.returns: list[float] = []
        self.timesteps: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.returns.append(float(ep["r"]))
                self.timesteps.append(int(self.num_timesteps))
        return True


def make_env(env_id: str) -> gym.Env:
    """Create an env; Flappy Bird needs its registration import + kwargs."""
    if env_id.startswith("FlappyBird"):
        import flappy_bird_gymnasium  # noqa: F401  (registers the env)

        return gym.make(env_id, use_lidar=False)
    return gym.make(env_id)


def evaluate_random(env_id: str, n_episodes: int = 30, seed: int = 0) -> float:
    env = make_env(env_id)
    rng = np.random.default_rng(seed)
    totals = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action = int(rng.integers(env.action_space.n))
            obs, r, term, trunc, _ = env.step(action)
            total += float(r)
            done = term or trunc
        totals.append(total)
    env.close()
    return float(np.mean(totals))


def evaluate_sb3(model, env_id: str, n_episodes: int = 30, seed: int = 0) -> float:
    env = make_env(env_id)
    totals = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            total += float(r)
            done = term or trunc
        totals.append(total)
    env.close()
    return float(np.mean(totals))
