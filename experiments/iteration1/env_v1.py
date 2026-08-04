"""Iteration-2 trading environment (TradingEnv-v1) — 5-D state.

Extends Iteration 1's 3-D state [ΔP, C, n] with the two gaps identified in
the Chapter 3 Iteration-2 guidelines:

  s_t = [ ΔP_t,  PnL_t,  τ_t,  C_t/C_0,  n_t P_t/C_0 ]

  ΔP_t        — 1-step price return (market momentum)
  PnL_t       — unrealized return on the open position vs average entry
                price (0 when flat)  → profit-taking / stop-loss context
  τ_t         — episode progress t/T in [0,1]         → time awareness
  C_t/C_0     — cash as a fraction of starting capital (scaled per the
                Iteration-2 guidelines' Scaling/Range column)
  n_t P_t/C_0 — market value of the open position as a fraction of starting
                capital (Gap A: the network cannot multiply n × P itself)

Actions, fees, masking and the ΔW reward are identical to Iteration 1, so
any performance difference is attributable to the state alone.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class OneStockDiscreteEnvV1(gym.Env):
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

        # [dP, unrealized PnL, time progress, cash fraction, position-value fraction]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.t = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.entry_cost = 0.0  # total cost basis of open position
        self._terminated = False

    def _price(self) -> float:
        return float(self.prices[self.t])

    def _dP(self) -> float:
        if self.t == 0:
            return 0.0
        p0, p1 = float(self.prices[self.t - 1]), float(self.prices[self.t])
        return (p1 - p0) / p0 if p0 != 0 else 0.0

    def _pnl(self) -> float:
        """Unrealized return of the open position vs average entry price."""
        if self.shares == 0 or self.entry_cost <= 0:
            return 0.0
        avg_entry = self.entry_cost / self.shares
        return (self._price() - avg_entry) / avg_entry

    def _tau(self) -> float:
        return self.t / self.max_steps

    def _wealth(self) -> float:
        return self.cash + self.shares * self._price()

    def action_masks(self) -> np.ndarray:
        p = self._price()
        can_buy = self.cash >= p * (1.0 + self.fee)
        can_sell = self.shares >= 1
        return np.array([True, can_buy, can_sell], dtype=bool)

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                self._dP(),
                self._pnl(),
                self._tau(),
                self.cash / self.initial_cash,
                (self.shares * self._price()) / self.initial_cash,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.entry_cost = 0.0
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
            action = 0  # illegal action → forced Hold

        w_before = self._wealth()
        p = self._price()

        if action == 1:  # Buy one
            self.cash -= p * (1.0 + self.fee)
            self.shares += 1
            self.entry_cost += p
        elif action == 2:  # Sell one
            self.cash += p * (1.0 - self.fee)
            # Reduce cost basis proportionally (average-cost accounting)
            if self.shares > 0:
                self.entry_cost -= self.entry_cost / self.shares
            self.shares -= 1
            if self.shares == 0:
                self.entry_cost = 0.0

        self.t += 1
        terminated = self.t >= self.max_steps
        truncated = False
        self._terminated = terminated

        w_after = self._wealth()
        reward = w_after - w_before
        info: dict[str, Any] = {
            "wealth": w_after,
            "cash": self.cash,
            "shares": self.shares,
            "unrealized_pnl": self._pnl(),
            "mask": self.action_masks(),
            "action_executed": action,
        }
        return self._obs(), float(reward), terminated, truncated, info
