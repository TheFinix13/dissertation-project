#!/usr/bin/env python3
"""Iteration-1 baselines on a synthetic day (smoke test for Nguyen meeting)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from env import OneStockDiscreteEnv, make_synthetic_day

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "iteration1" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def run_episode(env: OneStockDiscreteEnv, policy) -> dict:
    obs, info = env.reset()
    rewards = []
    wealth = [info["wealth"]]
    actions = []
    done = False
    while not done:
        mask = env.action_masks()
        a = int(policy(obs, mask))
        obs, r, terminated, truncated, info = env.step(a)
        rewards.append(r)
        wealth.append(info["wealth"])
        actions.append(info["action_executed"])
        done = terminated or truncated
    delta_w = wealth[-1] - wealth[0]
    return {
        "delta_w": delta_w,
        "sum_rewards": float(np.sum(rewards)),
        "n_trades": int(sum(1 for a in actions if a in (1, 2))),
        "terminal_wealth": wealth[-1],
        "wealth_path": wealth,
    }


def policy_hold(obs, mask) -> int:
    return 0


def policy_buy_and_hold(obs, mask) -> int:
    # Buy once when possible and shares==0; else Hold
    cash, shares = float(obs[1]), float(obs[2])
    if shares < 0.5 and mask[1]:
        return 1
    return 0


def policy_random(obs, mask, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


def main() -> None:
    prices = make_synthetic_day(n=60, seed=7)
    fee = 0.0005
    cash0 = 10_000.0
    rows = {}

    env = OneStockDiscreteEnv(prices, initial_cash=cash0, fee=fee)
    rows["B0_do_nothing"] = run_episode(env, policy_hold)

    env = OneStockDiscreteEnv(prices, initial_cash=cash0, fee=fee)
    rows["B1_buy_and_hold"] = run_episode(env, policy_buy_and_hold)

    rng = np.random.default_rng(0)
    env = OneStockDiscreteEnv(prices, initial_cash=cash0, fee=fee)
    rows["B2_random"] = run_episode(env, lambda o, m: policy_random(o, m, rng))

    # Honesty checks
    for name, r in rows.items():
        gap = abs(r["delta_w"] - r["sum_rewards"])
        r["accounting_gap"] = gap
        r["accounting_ok"] = gap < 1e-6
        # Drop long path from JSON summary (keep separately if needed)
        r.pop("wealth_path", None)

    assert rows["B0_do_nothing"]["delta_w"] == 0.0
    assert rows["B0_do_nothing"]["n_trades"] == 0

    out = {
        "note": "Synthetic-day smoke test only — not a market claim.",
        "fee": fee,
        "initial_cash": cash0,
        "n_bars": len(prices),
        "baselines": rows,
    }
    path = OUT / "baseline_smoke.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
