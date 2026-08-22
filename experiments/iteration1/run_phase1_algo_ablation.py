#!/usr/bin/env python3
"""Phase 1 — policy-family algorithm ablation on the Iteration-1 trading MDP.

Why this exists (for Dr Nguyen): "why PPO?" should be evidence, not habit.
We run three POLICY-BASED algorithms on the *identical* MDP, data split,
fee, cash, seed and timestep budget, then evaluate greedily with the same
feasibility mask on the same 26 held-out SPY months:

  - REINFORCE  (from scratch, masked Monte-Carlo policy gradient — we own it)
  - A2C        (actor-critic, unclipped — SB3)
  - PPO        (clipped policy gradient — SB3; the dissertation primary)

Value-based methods are deliberately excluded here: tabular Q cannot index a
continuous cash/price state, and DQN is off the policy-based track Nguyen
locked (it stays a Phase-0 comparator only).

Run:
  .venv311/bin/python experiments/iteration1/run_phase1_algo_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import OneStockDiscreteEnv  # noqa: E402
from month_env import MonthSamplerEnv  # noqa: E402
from reinforce_trading import make_reinforce_policy, train_reinforce_trading  # noqa: E402

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


def load_split():
    meta = json.loads((DATA / "spy_daily_meta.json").read_text())
    npz = np.load(DATA / "spy_daily_episodes.npz")
    train = [npz[f"train_{i}"] for i in meta["train_ids"]]
    test = [npz[f"test_{i}"] for i in meta["test_ids"]]
    return meta, train, test


def run_episode(prices: np.ndarray, policy) -> dict:
    env = OneStockDiscreteEnv(prices, initial_cash=CASH0, fee=FEE)
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
        "actions": actions,
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


def win_rate(agent_rows, ref_rows) -> float:
    wins = sum(1 for a, b in zip(agent_rows, ref_rows) if a["delta_w"] > b["delta_w"] + 1e-9)
    return wins / max(1, len(agent_rows))


def b1b_match_months(agent_rows, b1b_rows) -> int:
    """How many test months the agent's executed action sequence is identical
    to fully-invested buy-and-hold (the Iteration-1 collapse signature)."""
    return sum(1 for a, b in zip(agent_rows, b1b_rows) if a["actions"] == b["actions"])


# ---------- baseline policies (same as run_phase1_spy_daily.py) ----------

def policy_hold(obs, mask):
    return 0


def policy_bah_one(obs, mask):
    if float(obs[2]) < 0.5 and mask[1]:
        return 1
    return 0


def policy_bah_max(obs, mask):
    return 1 if mask[1] else 0


def make_random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def _p(obs, mask):
        return int(rng.choice(np.flatnonzero(mask)))

    return _p


def make_sb3_policy(model):
    def _p(obs, mask):
        a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        return a if mask[a] else 0

    return _p


def main() -> None:
    meta, train_prices, test_prices = load_split()
    timings: dict[str, float] = {}

    # ---- REINFORCE (scratch, masked) — matched timestep budget ----
    print("=== REINFORCE (scratch, masked) ===")
    t0 = time.time()
    rf = train_reinforce_trading(
        lambda p: OneStockDiscreteEnv(p, initial_cash=CASH0, fee=FEE),
        train_prices,
        obs_dim=3,
        total_timesteps=TIMESTEPS,
        seed=SEED,
    )
    timings["reinforce_s"] = time.time() - t0
    import torch

    torch.save(rf.policy.state_dict(), RES / "reinforce_spy_daily.pt")
    print(f"  trained {rf.total_timesteps} steps / {len(rf.episode_returns)} episodes "
          f"in {timings['reinforce_s']:.0f}s")

    # ---- A2C — same budget, SB3 defaults matched to Phase-0 choices ----
    print("=== A2C ===")
    t0 = time.time()
    a2c_env = MonthSamplerEnv(train_prices, initial_cash=CASH0, fee=FEE, sample_seed=0)
    a2c = A2C("MlpPolicy", a2c_env, learning_rate=7e-4, n_steps=5, gamma=0.99,
              verbose=0, seed=SEED)
    a2c.learn(total_timesteps=TIMESTEPS)
    a2c.save(str(RES / "a2c_spy_daily.zip"))
    timings["a2c_s"] = time.time() - t0
    print(f"  trained in {timings['a2c_s']:.0f}s")

    # ---- PPO — reuse the Iteration-1 model (identical protocol) ----
    print("=== PPO (Iteration-1 model) ===")
    ppo_path = RES / "ppo_spy_daily.zip"
    if ppo_path.exists():
        ppo = PPO.load(str(ppo_path))
        ppo_source = "loaded existing Iteration-1 model (run_phase1_spy_daily.py)"
    else:
        t0 = time.time()
        ppo_env = MonthSamplerEnv(train_prices, initial_cash=CASH0, fee=FEE, sample_seed=0)
        ppo = PPO("MlpPolicy", ppo_env, learning_rate=3e-4, n_steps=512, batch_size=64,
                  n_epochs=8, gamma=0.99, verbose=0, seed=SEED)
        ppo.learn(total_timesteps=TIMESTEPS)
        ppo.save(str(ppo_path))
        timings["ppo_s"] = time.time() - t0
        ppo_source = "retrained (no existing model found)"
    print(f"  {ppo_source}")

    # ---- Evaluate everything on the same 26 test months ----
    methods = {
        "B0_do_nothing": policy_hold,
        "B1_buy_one_hold": policy_bah_one,
        "B1b_buy_max_hold": policy_bah_max,
        "B2_random": make_random_policy(0),
        "A1_reinforce": make_reinforce_policy(rf.policy),
        "A1_a2c": make_sb3_policy(a2c),
        "A1_ppo": make_sb3_policy(ppo),
    }
    all_rows = {name: [run_episode(p, pol) for p in test_prices]
                for name, pol in methods.items()}

    summary = {name: summarize(rows) for name, rows in all_rows.items()}
    for name in summary:
        summary[name]["win_rate_vs_B1b_max"] = win_rate(all_rows[name], all_rows["B1b_buy_max_hold"])
        summary[name]["months_identical_to_B1b"] = b1b_match_months(
            all_rows[name], all_rows["B1b_buy_max_hold"])

    payload = {
        "meta": {
            "purpose": (
                "Policy-family ablation: same MDP, split, fee, cash, seed and "
                "timestep budget — only the learning algorithm changes. "
                "Justifies the PPO choice with same-MDP evidence."
            ),
            "ticker": meta["ticker"],
            "bar": meta["bar"],
            "episode_def": meta["episode_def"],
            "n_train": meta["n_train"],
            "n_test": meta["n_test"],
            "fee": FEE,
            "initial_cash": CASH0,
            "timesteps_budget": TIMESTEPS,
            "seed": SEED,
            "ppo_source": ppo_source,
            "reinforce_note": (
                "Scratch masked Monte-Carlo policy gradient "
                "(experiments/iteration1/reinforce_trading.py); trained to the "
                "same env-step budget as the SB3 agents."
            ),
            "excluded": (
                "Tabular Q (continuous state cannot be indexed) and DQN "
                "(value-based; off the policy track Nguyen locked — Phase-0 "
                "comparator only)."
            ),
            "timings_s": {k: round(v, 1) for k, v in timings.items()},
        },
        "summary": summary,
    }
    out = RES / "phase1_algo_ablation.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- Chart: mean ΔW by method ----
    keys = ["B0_do_nothing", "B2_random", "B1_buy_one_hold", "B1b_buy_max_hold",
            "A1_reinforce", "A1_a2c", "A1_ppo"]
    labels = ["B0\ndo-nothing", "B2\nrandom", "B1\nbuy-one", "B1b\nbuy-max",
              "REINFORCE\n(scratch)", "A2C", "PPO"]
    means = [summary[k]["mean_delta_w"] for k in keys]
    stds = [summary[k]["std_delta_w"] for k in keys]
    colors = ["#888", "#c47a2c", "#4a6fa5", "#2f5f8f", "#7a4cc4", "#c44c7a", "#1a7a4c"]

    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=160)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean ΔW on test months ($)")
    ax.set_title("Phase 1 — policy-family ablation on the same trading MDP\n"
                 "(REINFORCE vs A2C vs PPO, matched 80k-step budget, SPY daily months)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "phase1_algo_ablation_delta_w.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / "phase1_algo_ablation_delta_w.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {CHARTS / 'phase1_algo_ablation_delta_w.png'}")


if __name__ == "__main__":
    main()
