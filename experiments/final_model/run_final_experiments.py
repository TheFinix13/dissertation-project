#!/usr/bin/env python3
"""Final-model experiments: scratch REINFORCE vs scratch Deep Q
on the complete 9-D state, divisible-asset trading MDP.

Protocol (matches the pilot study for comparability):
  SPY daily closes 2018-2024, chronological 58/26 month split,
  C0 = $10k, fee = 5 bps, 80k env-step budget per seed, seeds {42,43,44}.

Baselines:
  B0  do-nothing        — always Hold
  B1b buy-and-hold      — Buy whenever legal until fully invested, then Hold
  B2  random (masked)   — uniform over legal actions

Run:
  ./venv/bin/python experiments/final_model/run_final_experiments.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "iteration1"))

from dqn_trading import make_dqn_policy, train_dqn_trading  # noqa: E402
from env_full import FullStateTradingEnv  # noqa: E402
from reinforce_trading import make_reinforce_policy, train_reinforce_trading  # noqa: E402

DATA = ROOT / "experiments" / "iteration1" / "data"
RES = HERE / "results"
CHARTS = ROOT / "reports" / "generated" / "charts"
RES.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

FEE = 0.0005
CASH0 = 10_000.0
TIMESTEPS = 80_000
SEEDS = [42, 43, 44]
OBS_DIM = 9


def make_env(prices: np.ndarray) -> FullStateTradingEnv:
    return FullStateTradingEnv(prices, initial_cash=CASH0, fee=FEE)


def load_split():
    meta = json.loads((DATA / "spy_daily_meta.json").read_text())
    npz = np.load(DATA / "spy_daily_episodes.npz")
    train = [npz[f"train_{i}"] for i in meta["train_ids"]]
    test = [npz[f"test_{i}"] for i in meta["test_ids"]]
    return meta, train, test


def run_episode(prices: np.ndarray, policy) -> dict:
    env = make_env(prices)
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
        "wealth_path": wealth,
        "accounting_gap": abs((wealth[-1] - wealth[0]) - float(np.sum(rewards))),
    }


def summarize(rows: list[dict]) -> dict:
    dw = np.array([r["delta_w"] for r in rows], dtype=float)
    trades = np.array([r["n_trades"] for r in rows], dtype=float)
    gaps = np.array([r["accounting_gap"] for r in rows], dtype=float)
    monthly_ret = dw / CASH0
    sharpe = float(monthly_ret.mean() / monthly_ret.std(ddof=1)) if monthly_ret.std(ddof=1) > 1e-12 else 0.0
    # max drawdown across the concatenated test-month equity curve
    equity = CASH0 + np.cumsum(dw)
    peak = np.maximum.accumulate(np.concatenate([[CASH0], equity]))
    mdd = float(np.max(1.0 - np.concatenate([[CASH0], equity]) / peak))
    return {
        "n_episodes": len(rows),
        "mean_delta_w": float(dw.mean()),
        "std_delta_w": float(dw.std(ddof=1)) if len(dw) > 1 else 0.0,
        "mean_trades": float(trades.mean()),
        "sharpe_monthly": sharpe,
        "max_drawdown": mdd,
        "max_accounting_gap": float(gaps.max()),
        "accounting_ok": bool(gaps.max() < 1e-5),
    }


def win_rate(agent_rows, ref_rows) -> float:
    wins = sum(1 for a, b in zip(agent_rows, ref_rows) if a["delta_w"] > b["delta_w"] + 1e-9)
    return wins / max(1, len(agent_rows))


# ---------- baselines ----------

def policy_hold(obs, mask):
    return 0


def policy_bah(obs, mask):
    return 1 if mask[1] else 0


def make_random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def _p(obs, mask):
        return int(rng.choice(np.flatnonzero(mask)))

    return _p


def main() -> None:
    meta, train_prices, test_prices = load_split()
    timings: dict[str, float] = {}

    baseline_methods = {
        "B0_do_nothing": policy_hold,
        "B1b_buy_and_hold": policy_bah,
        "B2_random": make_random_policy(0),
    }
    all_rows: dict[str, list[dict]] = {
        name: [run_episode(p, pol) for p in test_prices]
        for name, pol in baseline_methods.items()
    }

    per_seed: dict[str, dict[int, dict]] = {"reinforce": {}, "dqn": {}}
    for seed in SEEDS:
        print(f"=== REINFORCE (scratch, masked, 9-D) seed={seed} ===", flush=True)
        t0 = time.time()
        rf = train_reinforce_trading(
            make_env, train_prices, obs_dim=OBS_DIM,
            total_timesteps=TIMESTEPS, seed=seed,
        )
        timings[f"reinforce_s{seed}"] = time.time() - t0
        torch.save(rf.policy.state_dict(), RES / f"reinforce_full_seed{seed}.pt")
        rows = [run_episode(p, make_reinforce_policy(rf.policy)) for p in test_prices]
        per_seed["reinforce"][seed] = {
            "rows": rows,
            "train_curve": {"t": rf.episode_timesteps, "ret": rf.episode_returns},
        }
        print(f"  {timings[f'reinforce_s{seed}']:.0f}s, "
              f"mean dW={np.mean([r['delta_w'] for r in rows]):.2f}", flush=True)

        print(f"=== Deep Q (scratch, masked, 9-D) seed={seed} ===", flush=True)
        t0 = time.time()
        dq = train_dqn_trading(
            make_env, train_prices, obs_dim=OBS_DIM,
            total_timesteps=TIMESTEPS, seed=seed,
        )
        timings[f"dqn_s{seed}"] = time.time() - t0
        torch.save(dq.qnet.state_dict(), RES / f"dqn_full_seed{seed}.pt")
        rows = [run_episode(p, make_dqn_policy(dq.qnet)) for p in test_prices]
        per_seed["dqn"][seed] = {
            "rows": rows,
            "train_curve": {"t": dq.episode_timesteps, "ret": dq.episode_returns},
        }
        print(f"  {timings[f'dqn_s{seed}']:.0f}s, "
              f"mean dW={np.mean([r['delta_w'] for r in rows]):.2f}", flush=True)

    # primary seed (42) rows go into the main comparison table
    all_rows["A_reinforce"] = per_seed["reinforce"][SEEDS[0]]["rows"]
    all_rows["A_dqn"] = per_seed["dqn"][SEEDS[0]]["rows"]

    summary = {name: summarize(rows) for name, rows in all_rows.items()}
    for name in summary:
        summary[name]["win_rate_vs_B1b"] = win_rate(all_rows[name], all_rows["B1b_buy_and_hold"])

    # multi-seed spread for the learned methods
    seed_spread = {
        algo: {
            str(seed): {
                "mean_delta_w": float(np.mean([r["delta_w"] for r in d["rows"]])),
                "mean_trades": float(np.mean([r["n_trades"] for r in d["rows"]])),
            }
            for seed, d in per_seed[algo].items()
        }
        for algo in per_seed
    }

    payload = {
        "meta": {
            "purpose": (
                "Final-model experiments: complete 9-D state, divisible asset, "
                "scratch REINFORCE vs scratch Deep Q, per Recording 52 directives."
            ),
            "state": [
                "dP", "momentum_5", "volatility_5", "ma_gap_10", "tau",
                "cash/C0", "position_value/C0", "unrealized_pnl", "wealth/C0-1",
            ],
            "action_model": "Discrete(3); Buy/Sell a $1000 slice (10% of C0); divisible asset",
            "ticker": meta["ticker"],
            "bar": meta["bar"],
            "episode_def": meta["episode_def"],
            "n_train": meta["n_train"],
            "n_test": meta["n_test"],
            "fee": FEE,
            "initial_cash": CASH0,
            "timesteps_budget": TIMESTEPS,
            "seeds": SEEDS,
            "timings_s": {k: round(v, 1) for k, v in timings.items()},
        },
        "summary": summary,
        "seed_spread": seed_spread,
        "monthly_delta_w": {
            name: [round(r["delta_w"], 4) for r in rows] for name, rows in all_rows.items()
        },
    }
    out = RES / "final_model_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- chart: mean dW by method ----
    keys = ["B0_do_nothing", "B2_random", "B1b_buy_and_hold", "A_reinforce", "A_dqn"]
    labels = ["B0\ndo-nothing", "B2\nrandom", "B1b\nbuy & hold", "REINFORCE\n(scratch)", "Deep Q\n(scratch)"]
    means = [summary[k]["mean_delta_w"] for k in keys]
    stds = [summary[k]["std_delta_w"] for k in keys]
    colors = ["#888", "#c47a2c", "#2f5f8f", "#7a4cc4", "#1a7a4c"]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=160)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean ΔW on 26 test months ($)")
    ax.set_title("Final model — complete 9-D state, divisible asset\n"
                 "(scratch REINFORCE vs scratch Deep Q, seed 42, 80k steps)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "final_model_delta_w.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / "final_model_delta_w.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {CHARTS / 'final_model_delta_w.png'}")


if __name__ == "__main__":
    main()
