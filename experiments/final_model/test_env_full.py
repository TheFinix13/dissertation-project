"""Unit tests for FullStateTradingEnv — accounting identity, masks, liquidation.

Run:  ./venv/bin/python -m pytest experiments/final_model/test_env_full.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from env_full import FullStateTradingEnv, MIN_TRADE_VALUE


def make_prices(n: int = 25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=n)
    return 100.0 * np.cumprod(1.0 + rets)


def rollout(env, policy):
    obs, info = env.reset()
    w0 = info["wealth"]
    total_r = 0.0
    done = False
    while not done:
        a = policy(obs, env.action_masks())
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        done = term or trunc
    return w0, info["wealth"], total_r, info


def test_accounting_identity_random():
    """sum of rewards must equal final wealth minus initial wealth."""
    rng = np.random.default_rng(1)
    for seed in range(5):
        env = FullStateTradingEnv(make_prices(seed=seed))
        w0, w1, total_r, _ = rollout(
            env, lambda o, m: int(rng.choice(np.flatnonzero(m)))
        )
        assert abs((w1 - w0) - total_r) < 1e-6


def test_do_nothing_is_flat():
    env = FullStateTradingEnv(make_prices())
    w0, w1, total_r, info = rollout(env, lambda o, m: 0)
    assert w1 == pytest.approx(w0)
    assert info["n_trades"] == 0


def test_buy_charges_fee():
    prices = np.full(10, 100.0)
    env = FullStateTradingEnv(prices, fee=0.001)
    env.reset()
    _, r, *_ = env.step(1)  # buy $1000 slice at flat price
    # only cost is the fee on the traded amount: 1000 * (0.001/1.001)
    assert r == pytest.approx(-1000.0 * 0.001 / 1.001, rel=1e-6)


def test_forced_liquidation_at_end():
    env = FullStateTradingEnv(make_prices())
    _, _, _, info = rollout(env, lambda o, m: 1 if m[1] else 0)  # buy always
    assert info["units"] == 0.0
    assert info["cash"] > 0


def test_mask_blocks_sell_when_flat():
    env = FullStateTradingEnv(make_prices())
    env.reset()
    mask = env.action_masks()
    assert mask[0] and mask[1] and not mask[2]
    _, _, _, _, info = env.step(2)  # illegal sell → forced hold
    assert info["illegal"] and info["action_executed"] == 0


def test_buy_until_cash_exhausted_then_masked():
    env = FullStateTradingEnv(make_prices(n=40))
    env.reset()
    for _ in range(15):
        if not env.action_masks()[1]:
            break
        env.step(1)
    assert env.cash < MIN_TRADE_VALUE
    assert not env.action_masks()[1]


def test_obs_shape_and_clock():
    env = FullStateTradingEnv(make_prices())
    obs, _ = env.reset()
    assert obs.shape == (9,)
    assert obs[4] == 0.0  # tau at open


def test_tau_reaches_one():
    env = FullStateTradingEnv(make_prices())
    obs, _ = env.reset()
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(0)
        done = term or trunc
    assert obs[4] == pytest.approx(1.0)
