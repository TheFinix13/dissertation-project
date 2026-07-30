#!/usr/bin/env python3
"""Unit tests — accounting identity + mask legality (doer hygiene)."""
from __future__ import annotations

import numpy as np

from env import OneStockDiscreteEnv, make_synthetic_day


def test_accounting_identity():
    prices = make_synthetic_day(40, seed=1)
    env = OneStockDiscreteEnv(prices, initial_cash=5000.0, fee=0.0005)
    obs, info = env.reset()
    w0 = info["wealth"]
    total_r = 0.0
    done = False
    rng = np.random.default_rng(0)
    while not done:
        mask = env.action_masks()
        a = int(rng.choice(np.flatnonzero(mask)))
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        done = term or trunc
    assert abs((info["wealth"] - w0) - total_r) < 1e-6


def test_mask_blocks_buy_without_cash():
    prices = np.array([100.0, 101.0, 102.0, 103.0])
    env = OneStockDiscreteEnv(prices, initial_cash=50.0, fee=0.0)  # cannot buy $100 share
    env.reset()
    mask = env.action_masks()
    assert mask[0] and not mask[1] and not mask[2]


def test_mask_blocks_sell_without_shares():
    prices = np.array([100.0, 101.0, 102.0, 103.0])
    env = OneStockDiscreteEnv(prices, initial_cash=10_000.0, fee=0.0)
    env.reset()
    mask = env.action_masks()
    assert mask[0] and mask[1] and not mask[2]


def test_forced_hold_on_illegal_action():
    prices = np.array([100.0, 101.0, 102.0, 103.0])
    env = OneStockDiscreteEnv(prices, initial_cash=50.0, fee=0.0)
    env.reset()
    obs, r, term, trunc, info = env.step(1)  # illegal buy
    assert info["action_executed"] == 0
    assert info["shares"] == 0


if __name__ == "__main__":
    test_accounting_identity()
    test_mask_blocks_buy_without_cash()
    test_mask_blocks_sell_without_shares()
    test_forced_hold_on_illegal_action()
    print("ALL TESTS PASSED")
