"""Trading environment for the final experiments.

Design decisions that differ from the earlier environment, each traceable to a
specific defect:

* **Market features are supplied, not computed.** The environment receives a
  precomputed feature matrix aligned to its price series. It cannot truncate a
  rolling window at the episode boundary because it never computes one.

* **The observed feature set is selectable.** All ladder rungs share this single
  code path, so a difference between rungs cannot be an implementation
  difference.

* **`wealth_frac` is available but excluded from every rung up to S5.** It is
  pointwise recoverable, since `cash_frac + exposure_frac = W_t / C_0`.

* **`drawdown` is a path statistic** and is therefore not recoverable from any
  set of current-step quantities.

* **Admissibility is enforced at construction.** A drawdown-penalised reward
  requires `drawdown` in the observation, otherwise the reward is not a
  function of the state and the object is not an MDP. Passing
  ``risk_lambda > 0`` without that feature raises. The MDP validity argument in
  Chapter 3 is thus checked by the code rather than asserted in prose.

* **The action model is selectable.** ``"slice"`` commits a fixed sum of money;
  ``"share"`` buys one unit, whose cost depends on the price level, so the
  economic meaning of the action drifts across the sample. Keeping both lets
  the state axis and the action axis be varied independently.

Reward:
    r_t = dW_t - lambda * C_0 * max(0, dd_{t+1} - dd_t)

`dW_t` is the mark-to-market wealth change net of fees. The penalty multiplies
the *increase* in fractional drawdown by `C_0` to put both terms in dollars, so
`lambda` is dimensionless. With ``risk_lambda = 0`` the reward is exactly
`dW_t` and the rewards telescope to `W_T - W_0`.
"""
from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from features import ACCOUNT_FEATURES, MARKET_FEATURES, order_features

MIN_TRADE_VALUE = 1.0  # dollars; below this, Buy/Sell is masked out

ACTION_HOLD, ACTION_BUY, ACTION_SELL = 0, 1, 2


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        market: np.ndarray,
        feature_names: Sequence[str],
        *,
        initial_cash: float = 10_000.0,
        fee: float = 0.0005,
        action_model: str = "slice",
        slice_frac: float = 0.10,
        risk_lambda: float = 0.0,
    ):
        super().__init__()
        prices = np.asarray(prices, dtype=np.float64).reshape(-1)
        market = np.asarray(market, dtype=np.float64)
        if len(prices) < 3:
            raise ValueError("prices must have length >= 3")
        if market.ndim != 2 or market.shape[0] != len(prices):
            raise ValueError(
                f"market must be (len(prices), n_market); got {market.shape} "
                f"for {len(prices)} prices")
        if market.shape[1] != len(MARKET_FEATURES):
            raise ValueError(
                f"market must have {len(MARKET_FEATURES)} columns in "
                f"features.MARKET_FEATURES order; got {market.shape[1]}")
        if not np.all(np.isfinite(market)):
            raise ValueError("market feature matrix contains non-finite values")
        if action_model not in ("slice", "share"):
            raise ValueError(f"action_model must be 'slice' or 'share'; got {action_model!r}")

        self.feature_names = order_features(feature_names)
        if risk_lambda > 0.0 and "drawdown" not in self.feature_names:
            raise ValueError(
                "risk_lambda > 0 requires 'drawdown' in feature_names: a "
                "drawdown-penalised reward is not measurable from a state that "
                "omits the running peak, so the process would not be an MDP")

        self._market_index = {name: i for i, name in enumerate(MARKET_FEATURES)}
        self._market_cols = [self._market_index[n] for n in self.feature_names
                             if n in self._market_index]
        self._account_names = [n for n in self.feature_names if n in ACCOUNT_FEATURES]

        self.prices = prices
        self.market = market
        self.initial_cash = float(initial_cash)
        self.fee = float(fee)
        self.action_model = action_model
        self.slice_frac = float(slice_frac)
        self.trade_slice = self.slice_frac * self.initial_cash
        self.risk_lambda = float(risk_lambda)
        self.max_steps = len(prices) - 1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_names),), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.t = 0
        self.cash = self.initial_cash
        self.units = 0.0
        self.entry_cost = 0.0
        self.peak_wealth = self.initial_cash
        self.n_trades = 0
        self.fees_paid = 0.0
        self._terminated = False

    # ------------------------------------------------------------ helpers

    def _price(self) -> float:
        return float(self.prices[self.t])

    def _wealth(self) -> float:
        return self.cash + self.units * self._price()

    def _pnl(self) -> float:
        if self.units <= 0.0 or self.entry_cost <= 0.0:
            return 0.0
        avg_entry = self.entry_cost / self.units
        return (self._price() - avg_entry) / avg_entry

    def _drawdown(self) -> float:
        if self.peak_wealth <= 0.0:
            return 0.0
        return max(0.0, (self.peak_wealth - self._wealth()) / self.peak_wealth)

    def _account_value(self, name: str) -> float:
        if name == "clock":
            return self.t / self.max_steps
        if name == "cash_frac":
            return self.cash / self.initial_cash
        if name == "exposure_frac":
            return (self.units * self._price()) / self.initial_cash
        if name == "shares_raw":
            return self.units
        if name == "pnl":
            return self._pnl()
        if name == "drawdown":
            return self._drawdown()
        if name == "wealth_frac":
            return self._wealth() / self.initial_cash - 1.0
        raise KeyError(name)

    def _obs(self) -> np.ndarray:
        market_row = self.market[self.t]
        account = {n: self._account_value(n) for n in self._account_names}
        values = [
            float(market_row[self._market_index[n]]) if n in self._market_index
            else account[n]
            for n in self.feature_names
        ]
        return np.asarray(values, dtype=np.float32)

    # --------------------------------------------------------- gym API

    def action_masks(self) -> np.ndarray:
        p = self._price()
        if self.action_model == "share":
            can_buy = self.cash >= p * (1.0 + self.fee)
            can_sell = self.units >= 1.0
        else:
            can_buy = self.cash >= MIN_TRADE_VALUE
            can_sell = self.units * p >= MIN_TRADE_VALUE
        return np.array([True, bool(can_buy), bool(can_sell)], dtype=bool)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.cash = self.initial_cash
        self.units = 0.0
        self.entry_cost = 0.0
        self.peak_wealth = self.initial_cash
        self.n_trades = 0
        self.fees_paid = 0.0
        self._terminated = False
        return self._obs(), {
            "wealth": self._wealth(),
            "drawdown": self._drawdown(),
            "mask": self.action_masks(),
        }

    def _buy(self, spend: float) -> None:
        """Spend `spend` dollars of cash. The fee is charged inside the spend."""
        p = self._price()
        gross = spend / (1.0 + self.fee)
        bought = gross / p
        self.cash -= spend
        self.units += bought
        self.entry_cost += gross
        self.fees_paid += spend - gross
        self.n_trades += 1

    def _sell(self, value: float) -> None:
        """Liquidate up to `value` dollars of position at the current price."""
        p = self._price()
        sold = min(self.units, value / p)
        if self.units > 0.0:
            self.entry_cost -= self.entry_cost * (sold / self.units)
        self.units -= sold
        proceeds = sold * p
        self.cash += proceeds * (1.0 - self.fee)
        self.fees_paid += proceeds * self.fee
        if self.units * p < MIN_TRADE_VALUE:  # flush dust so "flat" means flat
            dust = self.units * p
            self.cash += dust * (1.0 - self.fee)
            self.fees_paid += dust * self.fee
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
            action = ACTION_HOLD

        w_before = self._wealth()
        dd_before = self._drawdown()

        if action == ACTION_BUY:
            if self.action_model == "share":
                self._buy(min(self.cash, self._price() * (1.0 + self.fee)))
            else:
                self._buy(min(self.cash, self.trade_slice))
        elif action == ACTION_SELL:
            if self.action_model == "share":
                self._sell(min(self.units, 1.0) * self._price())
            else:
                self._sell(min(self.units * self._price(), self.trade_slice))

        self.t += 1
        terminated = self.t >= self.max_steps
        self._terminated = terminated

        if terminated and self.units > 0.0:
            self._sell(self.units * self._price())  # no position carried overnight

        w_after = self._wealth()
        self.peak_wealth = max(self.peak_wealth, w_after)
        dd_after = self._drawdown()

        dw = w_after - w_before
        penalty = self.risk_lambda * self.initial_cash * max(0.0, dd_after - dd_before)
        reward = dw - penalty

        info: dict[str, Any] = {
            "wealth": w_after,
            "dw": dw,                 # pure wealth change; telescopes to W_T - W_0
            "penalty": penalty,
            "drawdown": dd_after,
            "cash": self.cash,
            "units": self.units,
            "exposure": self.units * self._price(),
            "n_trades": self.n_trades,
            "fees_paid": self.fees_paid,
            "illegal": illegal,
            "mask": self.action_masks(),
            "action_executed": action,
        }
        return self._obs(), float(reward), terminated, False, info


def make_env_factory(
    feature_names: Sequence[str],
    *,
    initial_cash: float = 10_000.0,
    fee: float = 0.0005,
    action_model: str = "slice",
    slice_frac: float = 0.10,
    risk_lambda: float = 0.0,
):
    """Return `factory(episode) -> TradingEnv` for a fixed configuration.

    `episode` is a dict with 'prices' and 'market', as produced by `data.load`.
    """
    def factory(episode: dict) -> TradingEnv:
        return TradingEnv(
            episode["prices"], episode["market"], feature_names,
            initial_cash=initial_cash, fee=fee, action_model=action_model,
            slice_frac=slice_frac, risk_lambda=risk_lambda)
    return factory
