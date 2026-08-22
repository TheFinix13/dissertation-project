"""Multi-month episode sampler wrapping FullStateTradingEnv (for SB3 agents)."""
from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np

from env_full import FullStateTradingEnv


class MonthSamplerFullEnv(gym.Env):
    """Each reset() picks a price path from `price_list` (train months)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        price_list: Sequence[np.ndarray],
        *,
        initial_cash: float = 10_000.0,
        fee: float = 0.0005,
        sample_seed: int | None = None,
    ):
        super().__init__()
        if not price_list:
            raise ValueError("price_list empty")
        self.price_list = [np.asarray(p, dtype=np.float64).reshape(-1) for p in price_list]
        self.initial_cash = initial_cash
        self.fee = fee
        self._rng = np.random.default_rng(sample_seed)
        self._env: FullStateTradingEnv | None = None
        tmp = self._make(self.price_list[0])
        self.observation_space = tmp.observation_space
        self.action_space = tmp.action_space

    def _make(self, prices: np.ndarray) -> FullStateTradingEnv:
        return FullStateTradingEnv(prices, initial_cash=self.initial_cash, fee=self.fee)

    def action_masks(self) -> np.ndarray:
        assert self._env is not None
        return self._env.action_masks()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(0, len(self.price_list)))
        if options and "index" in options:
            idx = int(options["index"])
        self._env = self._make(self.price_list[idx])
        obs, info = self._env.reset(seed=seed)
        info["episode_index"] = idx
        return obs, info

    def step(self, action):
        assert self._env is not None
        return self._env.step(action)
