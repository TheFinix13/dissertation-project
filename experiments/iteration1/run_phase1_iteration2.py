#!/usr/bin/env python3
"""Phase 1 — Iteration 2 (5-D state) vs Iteration 1 (3-D state) on SPY months.

Trains PPO on TradingEnv-v1 with the same data split, fee, budget and seed
as run_phase1_spy_daily.py, so the ONLY difference is the state vector.

Run:
  .venv311/bin/python experiments/iteration1/run_phase1_iteration2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_v1 import OneStockDiscreteEnvV1  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "iteration1" / "data"
RES = ROOT / "experiments" / "iteration1" / "results"
CHARTS = ROOT / "reports" / "generated" / "charts"
RES.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

FEE = 0.0005
CASH0 = 10_000.0
TIMESTEPS = 80_000
SEED = 42


class MonthSamplerEnvV1(gym.Env):
    """reset() samples a month from price_list; wraps OneStockDiscreteEnvV1."""

    metadata = {"render_modes": []}

    def __init__(self, price_list: Sequence[np.ndarray], *, sample_seed: int | None = None):
        super().__init__()
        self.price_list = [np.asarray(p, dtype=np.float64).reshape(-1) for p in price_list]
        self._rng = np.random.default_rng(sample_seed)
        self._env: OneStockDiscreteEnvV1 | None = None
        tmp = OneStockDiscreteEnvV1(self.price_list[0], initial_cash=CASH0, fee=FEE)
        self.observation_space = tmp.observation_space
        self.action_space = tmp.action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        idx = int(self._rng.integers(0, len(self.price_list)))
        self._env = OneStockDiscreteEnvV1(self.price_list[idx], initial_cash=CASH0, fee=FEE)
        return self._env.reset(seed=seed)

    def step(self, action):
        assert self._env is not None
        return self._env.step(action)


def load_split():
    meta = json.loads((DATA / "spy_daily_meta.json").read_text())
    npz = np.load(DATA / "spy_daily_episodes.npz")
    train = [npz[f"train_{i}"] for i in meta["train_ids"]]
    test = [npz[f"test_{i}"] for i in meta["test_ids"]]
    return meta, train, test


def run_episode(prices: np.ndarray, policy) -> dict:
    env = OneStockDiscreteEnvV1(prices, initial_cash=CASH0, fee=FEE)
    obs, info = env.reset()
    rewards, wealth, actions = [], [info["wealth"]], []
    done = False
    while not done:
        a = int(policy(obs, env.action_masks()))
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        wealth.append(info["wealth"])
        actions.append(info["action_executed"])
        done = term or trunc
    return {
        "delta_w": wealth[-1] - wealth[0],
        "n_trades": int(sum(a in (1, 2) for a in actions)),
        "wealth_path": wealth,
        "accounting_gap": abs((wealth[-1] - wealth[0]) - float(np.sum(rewards))),
    }


def summarize(rows: list[dict]) -> dict:
    dw = np.array([r["delta_w"] for r in rows], dtype=float)
    trades = np.array([r["n_trades"] for r in rows], dtype=float)
    gaps = np.array([r["accounting_gap"] for r in rows], dtype=float)
    return {
        "n_episodes": len(rows),
        "mean_delta_w": float(dw.mean()),
        "std_delta_w": float(dw.std(ddof=1)) if len(dw) > 1 else 0.0,
        "mean_trades": float(trades.mean()),
        "max_accounting_gap": float(gaps.max()),
        "accounting_ok": bool(gaps.max() < 1e-5),
    }


def main() -> None:
    meta, train_prices, test_prices = load_split()

    env = MonthSamplerEnvV1(train_prices, sample_seed=0)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=64,
                n_epochs=8, gamma=0.99, verbose=1, seed=SEED)
    model.learn(total_timesteps=TIMESTEPS)
    model_path = RES / "ppo_spy_daily_iter2.zip"
    model.save(str(model_path))

    def ppo_policy(obs, mask):
        a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        return a if mask[a] else 0

    def bah_max(obs, mask):
        return 1 if mask[1] else 0

    rows_ppo = [run_episode(p, ppo_policy) for p in test_prices]
    rows_bah = [run_episode(p, bah_max) for p in test_prices]

    # Behavioural check: identical to buy-max on how many months?
    identical = 0
    for p in test_prices:
        e1 = OneStockDiscreteEnvV1(p, initial_cash=CASH0, fee=FEE)
        e2 = OneStockDiscreteEnvV1(p, initial_cash=CASH0, fee=FEE)
        o1, _ = e1.reset(); o2, _ = e2.reset()
        acts1, acts2, done = [], [], False
        while not done:
            a1 = ppo_policy(o1, e1.action_masks())
            a2 = bah_max(o2, e2.action_masks())
            o1, _, t1, tr1, i1 = e1.step(a1)
            o2, _, t2, tr2, i2 = e2.step(a2)
            acts1.append(i1["action_executed"]); acts2.append(i2["action_executed"])
            done = t1 or tr1
        identical += int(acts1 == acts2)

    summary = {
        "A2_ppo_iter2_5d": summarize(rows_ppo),
        "B1b_buy_max_hold": summarize(rows_bah),
    }
    wins = sum(1 for a, b in zip(rows_ppo, rows_bah) if a["delta_w"] > b["delta_w"] + 1e-9)
    payload = {
        "meta": {
            "env": "OneStockDiscreteEnvV1 (5-D scaled state: dP, PnL, tau, C/C0, nP/C0)",
            "same_as_iteration1": "data split, fee, cash, timesteps, seed",
            "n_test": meta["n_test"],
            "fee": FEE,
            "initial_cash": CASH0,
            "ppo_timesteps": TIMESTEPS,
            "seed": SEED,
            "model": str(model_path.relative_to(ROOT)),
        },
        "summary": summary,
        "behaviour": {
            "months_identical_to_buy_max": identical,
            "n_test_months": len(test_prices),
            "win_months_vs_buy_max": wins,
        },
    }
    (RES / "phase1_iteration2_results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["behaviour"], indent=2))

    # Comparison chart: Iter1 (=B1b) vs Iter2 mean ΔW
    iter1 = json.loads((RES / "phase1_spy_daily_results.json").read_text())["summary"]
    labels = ["B0\ndo-nothing", "B2\nrandom", "B1b buy-max\n(= Iter-1 PPO)", "Iter-2 PPO\n(5-D state)"]
    means = [iter1["B0_do_nothing"]["mean_delta_w"], iter1["B2_random"]["mean_delta_w"],
             iter1["B1b_buy_max_hold"]["mean_delta_w"], summary["A2_ppo_iter2_5d"]["mean_delta_w"]]
    stds = [iter1["B0_do_nothing"]["std_delta_w"], iter1["B2_random"]["std_delta_w"],
            iter1["B1b_buy_max_hold"]["std_delta_w"], summary["A2_ppo_iter2_5d"]["std_delta_w"]]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=160)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4,
           color=["#888", "#c47a2c", "#2f5f8f", "#1a7a4c"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean ΔW on test months ($)")
    ax.set_title("Phase 1 — Iteration 2 (5-D state) vs Iteration 1 on SPY test months")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "phase1_iteration2_delta_w.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / "phase1_iteration2_delta_w.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RES / 'phase1_iteration2_results.json'}")


if __name__ == "__main__":
    main()
