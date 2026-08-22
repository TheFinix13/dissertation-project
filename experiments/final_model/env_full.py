"""Final trading environment — complete 9-D state, divisible asset.

Implements the model agreed after Recording 52:

- The asset is divisible ("consider it's kind of Bitcoin" — Nguyen):
  Buy/Sell moves a fixed monetary slice Δ = slice_frac · C0, so the
  Discrete(3) action set is fixed and almost always feasible.
- The state is designed to "see the whole world" that is observable:

    market block   s[0] ΔP_t        1-step price return
                   s[1] mom_t       k-step return  P_t / P_{t-k} − 1
                   s[2] vol_t       std of 1-step returns over last k steps
                   s[3] magap_t     P_t / MA_k(P) − 1  (trend position)
    clock          s[4] τ_t         t / T  in [0, 1]
    balance sheet  s[5] C_t / C0    cash fraction
                   s[6] h_t·P_t/C0  position value fraction
                   s[7] PnL_t       unrealized return vs average entry
                   s[8] W_t/C0 − 1  wealth return since episode start

- Reward: r_t = W_{t+1} − W_t (mark-to-market, net of proportional fees).
- Final step force-liquidates the position so every episode ends in cash.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MIN_TRADE_VALUE = 1.0  # dollars — below this Buy/Sell is masked out


class FullStateTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        *,
        initial_cash: float = 10_000.0,
        fee: float = 0.0005,
        slice_frac: float = 0.10,
        window: int = 5,
        ma_window: int = 10,
    ):
        super().__init__()
        prices = np.asarray(prices, dtype=np.float64).reshape(-1)
        if len(prices) < 3:
            raise ValueError("prices must have length >= 3")
        self.prices = prices
        self.initial_cash = float(initial_cash)
        self.fee = float(fee)
        self.trade_slice = float(slice_frac) * self.initial_cash
        self.window = int(window)
        self.ma_window = int(ma_window)
        self.max_steps = len(prices) - 1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0 Hold, 1 Buy, 2 Sell

        self.t = 0
        self.cash = self.initial_cash
        self.units = 0.0          # fractional units of the asset
        self.entry_cost = 0.0     # total cost basis of open position
        self.n_trades = 0
        self._terminated = False

    # ---------- feature helpers ----------

    def _price(self, t: int | None = None) -> float:
        return float(self.prices[self.t if t is None else t])

    def _dP(self) -> float:
        if self.t == 0:
            return 0.0
        p0, p1 = self._price(self.t - 1), self._price(self.t)
        return (p1 - p0) / p0 if p0 != 0 else 0.0

    def _momentum(self) -> float:
        k = min(self.window, self.t)
        if k == 0:
            return 0.0
        p0, p1 = self._price(self.t - k), self._price(self.t)
        return (p1 - p0) / p0 if p0 != 0 else 0.0

    def _volatility(self) -> float:
        if self.t < 2:
            return 0.0
        lo = max(0, self.t - self.window)
        seg = self.prices[lo : self.t + 1]
        rets = np.diff(seg) / seg[:-1]
        return float(np.std(rets)) if len(rets) >= 2 else 0.0

    def _ma_gap(self) -> float:
        lo = max(0, self.t - self.ma_window + 1)
        ma = float(np.mean(self.prices[lo : self.t + 1]))
        return (self._price() - ma) / ma if ma != 0 else 0.0

    def _pnl(self) -> float:
        if self.units <= 0 or self.entry_cost <= 0:
            return 0.0
        avg_entry = self.entry_cost / self.units
        return (self._price() - avg_entry) / avg_entry

    def _wealth(self) -> float:
        return self.cash + self.units * self._price()

    # ---------- gym API ----------

    def action_masks(self) -> np.ndarray:
        can_buy = self.cash >= MIN_TRADE_VALUE
        can_sell = self.units * self._price() >= MIN_TRADE_VALUE
        return np.array([True, can_buy, can_sell], dtype=bool)

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                self._dP(),
                self._momentum(),
                self._volatility(),
                self._ma_gap(),
                self.t / self.max_steps,
                self.cash / self.initial_cash,
                (self.units * self._price()) / self.initial_cash,
                self._pnl(),
                self._wealth() / self.initial_cash - 1.0,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.cash = self.initial_cash
        self.units = 0.0
        self.entry_cost = 0.0
        self.n_trades = 0
        self._terminated = False
        return self._obs(), {"wealth": self._wealth(), "mask": self.action_masks()}

    def _buy(self, spend: float) -> None:
        """Spend `spend` dollars of cash; fee is charged inside the spend."""
        p = self._price()
        bought = spend / (p * (1.0 + self.fee))
        self.cash -= spend
        self.units += bought
        self.entry_cost += bought * p
        self.n_trades += 1

    def _sell(self, value: float) -> None:
        """Liquidate `value` dollars of position at current price."""
        p = self._price()
        sold = min(self.units, value / p)
        if self.units > 0:
            self.entry_cost -= self.entry_cost * (sold / self.units)
        self.units -= sold
        self.cash += sold * p * (1.0 - self.fee)
        if self.units * p < MIN_TRADE_VALUE:  # flush dust
            self.cash += self.units * p * (1.0 - self.fee)
            self.units = 0.0
            self.entry_cost = 0.0
        self.n_trades += 1

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("episode already done — call reset()")

        mask = self.action_masks()
        action = int(action)
        if not 0 <= action <= 2:
            raise ValueError(f"invalid action {action}")
        illegal = not mask[action]
        if illegal:
            action = 0  # forced Hold

        w_before = self._wealth()

        if action == 1:
            self._buy(min(self.cash, self.trade_slice))
        elif action == 2:
            self._sell(min(self.units * self._price(), self.trade_slice))

        self.t += 1
        terminated = self.t >= self.max_steps
        self._terminated = terminated

        if terminated and self.units > 0:
            # Mandatory end-of-episode liquidation (no overnight exposure)
            self._sell(self.units * self._price())

        w_after = self._wealth()
        reward = w_after - w_before
        info: dict[str, Any] = {
            "wealth": w_after,
            "cash": self.cash,
            "units": self.units,
            "n_trades": self.n_trades,
            "illegal": illegal,
            "mask": self.action_masks(),
            "action_executed": action,
        }
        return self._obs(), float(reward), terminated, False, info
