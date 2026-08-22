#!/usr/bin/env python3
"""One-off patch: recompute the B2 random baseline with the pilot-consistent
protocol (single RNG across all test months) and update JSON + bar chart.

The original run_full_ablation.py accidentally recreated the RNG per month.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from env_full import FullStateTradingEnv  # noqa: E402

DATA = ROOT / "experiments" / "iteration1" / "data"
RES = HERE / "results"
CHARTS = ROOT / "reports" / "generated" / "charts"
CASH0 = 10_000.0
FEE = 0.0005


def run_episode(prices, policy):
    env = FullStateTradingEnv(prices, initial_cash=CASH0, fee=FEE)
    obs, info = env.reset()
    rewards, wealth, actions = [], [info["wealth"]], []
    illegal = 0
    done = False
    while not done:
        a = int(policy(obs, env.action_masks()))
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        wealth.append(info["wealth"])
        actions.append(info["action_executed"])
        illegal += int(info["illegal"])
        done = term or trunc
    return {
        "delta_w": wealth[-1] - wealth[0],
        "n_trades": int(info["n_trades"]),
        "illegal": illegal,
        "actions": actions,
        "accounting_gap": abs((wealth[-1] - wealth[0]) - float(np.sum(rewards))),
    }


def summarize(rows):
    dw = np.array([r["delta_w"] for r in rows], dtype=float)
    trades = np.array([r["n_trades"] for r in rows], dtype=float)
    gaps = np.array([r["accounting_gap"] for r in rows], dtype=float)
    monthly_ret = dw / CASH0
    sharpe = float(monthly_ret.mean() / monthly_ret.std(ddof=1)) if monthly_ret.std(ddof=1) > 1e-12 else 0.0
    equity = np.concatenate([[CASH0], CASH0 + np.cumsum(dw)])
    peak = np.maximum.accumulate(equity)
    mdd = float(np.max(1.0 - equity / peak))
    return {
        "n_episodes": len(rows),
        "mean_delta_w": float(dw.mean()),
        "std_delta_w": float(dw.std(ddof=1)),
        "mean_trades": float(trades.mean()),
        "sharpe_monthly": sharpe,
        "max_drawdown": mdd,
        "max_accounting_gap": float(gaps.max()),
        "accounting_ok": bool(gaps.max() < 1e-5),
    }


def main():
    meta = json.loads((DATA / "spy_daily_meta.json").read_text())
    npz = np.load(DATA / "spy_daily_episodes.npz")
    test = [npz[f"test_{i}"] for i in meta["test_ids"]]

    rng = np.random.default_rng(0)
    policy = lambda o, m: int(rng.choice(np.flatnonzero(m)))  # noqa: E731
    rows = [run_episode(p, policy) for p in test]

    path = RES / "full_ablation_results.json"
    d = json.loads(path.read_text())

    s = summarize(rows)
    b1b_dw = d["monthly_delta_w"]["B1b_buy_and_hold"]
    s["win_rate_vs_B1b"] = float(
        np.mean([r["delta_w"] > b + 1e-9 for r, b in zip(rows, b1b_dw)]))
    s["months_identical_to_B1b"] = 0
    d["summary"]["B2_random"] = s
    d["monthly_delta_w"]["B2_random"] = [round(r["delta_w"], 4) for r in rows]
    d["illegal_rates"]["B2_random"] = float(np.mean([r["illegal"] for r in rows]))
    d["meta"]["b2_note"] = "B2 recomputed with single RNG across months (pilot-consistent protocol)."
    path.write_text(json.dumps(d, indent=2))

    # regenerate bar chart
    keys = ["B0_do_nothing", "B2_random", "B1b_buy_and_hold",
            "A_reinforce", "A_dqn", "A_ppo"]
    labels = ["B0\ndo-nothing", "B2\nrandom", "B1b\nbuy & hold",
              "REINFORCE\n(scratch)", "Deep Q\n(scratch)", "PPO\n(SB3)"]
    means = [d["summary"][k]["mean_delta_w"] for k in keys]
    stds = [d["summary"][k]["std_delta_w"] for k in keys]
    colors = ["#888", "#c47a2c", "#2f5f8f", "#7a4cc4", "#1a7a4c", "#c44c7a"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=160)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean ΔW on 26 test months ($)")
    ax.set_title("Algorithm ablation — complete 9-D state, divisible asset\n"
                 "(matched 80k-step budget, seed 42)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "ablation_delta_w.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / "ablation_delta_w.pdf", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(s, indent=2))
    print("patched", path)


if __name__ == "__main__":
    main()
