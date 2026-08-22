"""Episode-sampling wrapper so Stable-Baselines3 sees the same task as the
from-scratch agents.

SB3 treats one environment as one long-running task, whereas the trading task is
a distribution over monthly episodes. This wrapper draws a new month on every
reset, from a generator seeded by the run seed, so the SB3 agent's seed-to-seed
spread reflects the same two sources of variation as the scratch agents: network
initialisation and the order in which months are seen.

`action_masks()` is exposed so `sb3_contrib.MaskablePPO` can apply the same
feasibility constraint the scratch agents apply. Using plain `PPO` here would
leave the policy free to put probability mass on infeasible trades and rely on
the environment to force a Hold, which is a different learning problem from the
one REINFORCE and Deep Q are solving.
"""
from __future__ import annotations

from typing import Callable, Sequence

import gymnasium as gym
import numpy as np


class MonthSampler(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        factory: Callable[[dict], gym.Env],
        episodes: Sequence[dict],
        *,
        seed: int = 0,
    ):
        super().__init__()
        if not episodes:
            raise ValueError("episodes must be non-empty")
        self.factory = factory
        self.episodes = list(episodes)
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        probe = factory(self.episodes[0])
        self.observation_space = probe.observation_space
        self.action_space = probe.action_space
        self.initial_cash = probe.initial_cash
        self._env = probe
        self._current_id = self.episodes[0].get("id")

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:  # honour SB3's reseeding so runs stay reproducible
            self._rng = np.random.default_rng(seed)
        episode = self.episodes[int(self._rng.integers(0, len(self.episodes)))]
        self._current_id = episode.get("id")
        self._env = self.factory(episode)
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    @property
    def current_episode_id(self):
        return self._current_id
