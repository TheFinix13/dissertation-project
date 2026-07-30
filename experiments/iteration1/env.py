"""Iteration-1 trading environment — simple one-stock MDP.

State:  s_t = [ΔP_t, C_t, n_t]
Actions: 0 Hold, 1 Buy one share, 2 Sell one share (masked)
Reward: r_t = W_{t+1} - W_t,  W_t = C_t + n_t * P_t
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class OneStockDiscreteEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        *,
        initial_cash: float = 10_000.0,
        fee: float = 0.0005,
        max_steps: int | None = None,
    ):
        super().__init__()
        prices = np.asarray(prices, dtype=np.float64).reshape(-1)
        if prices.ndim != 1 or len(prices) < 3:
            raise ValueError("prices must be a 1-D array with length >= 3")
        self.prices = prices
        self.initial_cash = float(initial_cash)
        self.fee = float(fee)
        self.max_steps = int(max_steps) if max_steps is not None else len(prices) - 1

        # [dP, cash, shares]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.t = 0
        self.cash = self.initial_cash
        self.shares = 0
        self._terminated = False

    def _price(self) -> float:
        return float(self.prices[self.t])

    def _dP(self) -> float:
        if self.t == 0:
            return 0.0
        p0, p1 = float(self.prices[self.t - 1]), float(self.prices[self.t])
        return (p1 - p0) / p0 if p0 != 0 else 0.0

    def _wealth(self) -> float:
        return self.cash + self.shares * self._price()

    def action_masks(self) -> np.ndarray:
        """True = legal. Hold always; Buy needs cash; Sell needs a share."""
        p = self._price()
        can_buy = self.cash >= p * (1.0 + self.fee)
        can_sell = self.shares >= 1
        return np.array([True, can_buy, can_sell], dtype=bool)

    def _obs(self) -> np.ndarray:
        return np.array([self._dP(), self.cash, float(self.shares)], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.cash = self.initial_cash
        self.shares = 0
        self._terminated = False
        return self._obs(), {"wealth": self._wealth(), "mask": self.action_masks()}

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("episode already done — call reset()")

        mask = self.action_masks()
        action = int(action)
        if action < 0 or action > 2:
            raise ValueError(f"invalid action {action}")
        if not mask[action]:
            # Illegal action → force Hold (safe default for smoke tests)
            action = 0

        w_before = self._wealth()
        p = self._price()

        if action == 1:  # Buy one
            self.cash -= p * (1.0 + self.fee)
            self.shares += 1
        elif action == 2:  # Sell one
            self.cash += p * (1.0 - self.fee)
            self.shares -= 1

        self.t += 1
        terminated = self.t >= self.max_steps
        truncated = False
        self._terminated = terminated

        # Mark-to-market after price advances
        w_after = self._wealth()
        reward = w_after - w_before
        info: dict[str, Any] = {
            "wealth": w_after,
            "cash": self.cash,
            "shares": self.shares,
            "mask": self.action_masks(),
            "action_executed": action,
            "sum_check_partial": reward,
        }
        return self._obs(), float(reward), terminated, truncated, info


def make_synthetic_day(n: int = 60, seed: int = 0, start: float = 100.0) -> np.ndarray:
    """Deterministic-ish random walk for smoke tests (not real market data)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.002, size=n)
    prices = start * np.cumprod(1.0 + rets)
    return prices.astype(np.float64)
