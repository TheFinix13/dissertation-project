#!/usr/bin/env python3
"""Supplementary runs strengthening the ablation:

1. PPO seeds 45 and 46 (five total) — solidify the bimodality claim.
2. Preliminary risk-aware reward: retrain REINFORCE + DQN (seed 42) with
   r'_t = dW_t - lambda * max(0, D_t - D_{t-1}) where D = peak wealth - W.
   Evaluated on the TRUE wealth metrics (dW, MDD) so results are comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "iteration1"))

from dqn_trading import make_dqn_policy, train_dqn_trading  # noqa: E402
from env_full import FullStateTradingEnv  # noqa: E402
from month_sampler_full import MonthSamplerFullEnv  # noqa: E402
from reinforce_trading import make_reinforce_policy, train_reinforce_trading  # noqa: E402

DATA = ROOT / "experiments" / "iteration1" / "data"
RES = HERE / "results"
FEE = 0.0005
CASH0 = 10_000.0
TIMESTEPS = 80_000
LAM = 0.25


class DrawdownPenaltyEnv(gym.Wrapper):
    """Reward shaping: r' = dW - lam * increase in drawdown from peak."""

    def __init__(self, env: FullStateTradingEnv, lam: float):
        super().__init__(env)
        self.lam = lam
        self._peak = None
        self._prev_dd = 0.0

    def action_masks(self):
        return self.env.action_masks()

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._peak = info["wealth"]
        self._prev_dd = 0.0
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        w = info["wealth"]
        self._peak = max(self._peak, w)
        dd = self._peak - w
        shaped = r - self.lam * max(0.0, dd - self._prev_dd)
        self._prev_dd = dd
        return obs, shaped, term, trunc, info


def load_split():
    meta = json.loads((DATA / "spy_daily_meta.json").read_text())
    npz = np.load(DATA / "spy_daily_episodes.npz")
    return meta, [npz[f"train_{i}"] for i in meta["train_ids"]], \
        [npz[f"test_{i}"] for i in meta["test_ids"]]


def run_episode(prices, policy):
    env = FullStateTradingEnv(prices, initial_cash=CASH0, fee=FEE)
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
    return {"delta_w": wealth[-1] - wealth[0], "n_trades": int(info["n_trades"]),
            "actions": actions,
            "gap": abs((wealth[-1] - wealth[0]) - float(np.sum(rewards)))}


def summarize(rows):
    dw = np.array([r["delta_w"] for r in rows])
    ret = dw / CASH0
    eq = np.concatenate([[CASH0], CASH0 + np.cumsum(dw)])
    mdd = float(np.max(1.0 - eq / np.maximum.accumulate(eq)))
    return {"mean_delta_w": float(dw.mean()), "std_delta_w": float(dw.std(ddof=1)),
            "mean_trades": float(np.mean([r["n_trades"] for r in rows])),
            "sharpe_monthly": float(ret.mean() / ret.std(ddof=1)),
            "max_drawdown": mdd,
            "accounting_ok": bool(max(r["gap"] for r in rows) < 1e-5)}


def main():
    from stable_baselines3 import PPO

    meta, train_prices, test_prices = load_split()
    bah_rows = [run_episode(p, lambda o, m: 1 if m[1] else 0) for p in test_prices]
    out = {}

    # ---- 1. PPO seeds 45, 46 ----
    ppo_extra = {}
    for seed in [45, 46]:
        print(f"=== PPO seed {seed} ===", flush=True)
        env = MonthSamplerFullEnv(train_prices, initial_cash=CASH0, fee=FEE, sample_seed=0)
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=64,
                    n_epochs=8, gamma=0.99, verbose=0, seed=seed)
        model.learn(total_timesteps=TIMESTEPS)

        def pol(obs, mask, m=model):
            a = int(m.predict(obs, deterministic=True)[0])
            return a if mask[a] else 0

        rows = [run_episode(p, pol) for p in test_prices]
        ident = sum(1 for a, b in zip(rows, bah_rows) if a["actions"] == b["actions"])
        ppo_extra[str(seed)] = {**summarize(rows), "months_identical_to_B1b": ident}
        print(ppo_extra[str(seed)], flush=True)
    out["ppo_extra_seeds"] = ppo_extra

    # ---- 2. risk-aware preliminary (seed 42) ----
    def make_shaped(prices):
        return DrawdownPenaltyEnv(
            FullStateTradingEnv(prices, initial_cash=CASH0, fee=FEE), LAM)

    print("=== risk-aware REINFORCE ===", flush=True)
    rf = train_reinforce_trading(make_shaped, train_prices, obs_dim=9,
                                 total_timesteps=TIMESTEPS, seed=42)
    rows_rf = [run_episode(p, make_reinforce_policy(rf.policy)) for p in test_prices]
    ident_rf = sum(1 for a, b in zip(rows_rf, bah_rows) if a["actions"] == b["actions"])

    print("=== risk-aware DQN ===", flush=True)
    dq = train_dqn_trading(make_shaped, train_prices, obs_dim=9,
                           total_timesteps=TIMESTEPS, seed=42)
    rows_dq = [run_episode(p, make_dqn_policy(dq.qnet)) for p in test_prices]
    ident_dq = sum(1 for a, b in zip(rows_dq, bah_rows) if a["actions"] == b["actions"])

    out["risk_aware"] = {
        "lambda": LAM,
        "note": "trained on shaped reward; evaluated on TRUE wealth metrics",
        "reinforce": {**summarize(rows_rf), "months_identical_to_B1b": ident_rf},
        "dqn": {**summarize(rows_dq), "months_identical_to_B1b": ident_dq},
        "B1b_reference": summarize(bah_rows),
    }
    print(json.dumps(out["risk_aware"], indent=2))

    (RES / "supplementary_results.json").write_text(json.dumps(out, indent=2))
    print("SUPPLEMENTARY_DONE")


if __name__ == "__main__":
    main()
