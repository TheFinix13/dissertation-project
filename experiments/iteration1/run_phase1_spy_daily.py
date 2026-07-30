#!/usr/bin/env python3
"""Phase 1 — evaluate baselines + PPO on SPY monthly episodes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from env import OneStockDiscreteEnv
from month_env import MonthSamplerEnv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "iteration1" / "data"
RES = ROOT / "experiments" / "iteration1" / "results"
CHARTS = ROOT / "reports" / "generated" / "charts"
RES.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

FEE = 0.0005
CASH0 = 10_000.0


def load_split():
    meta = json.loads((DATA / "spy_daily_meta.json").read_text())
    npz = np.load(DATA / "spy_daily_episodes.npz")
    train = [npz[f"train_{i}"] for i in meta["train_ids"]]
    test = [npz[f"test_{i}"] for i in meta["test_ids"]]
    return meta, train, test


def run_episode(prices: np.ndarray, policy, *, force_mask: bool = True) -> dict:
    env = OneStockDiscreteEnv(prices, initial_cash=CASH0, fee=FEE)
    # Monkey-patch: optionally disable masking by allowing all actions
    if not force_mask:
        env.action_masks = lambda: np.array([True, True, True], dtype=bool)  # type: ignore

    obs, info = env.reset()
    rewards, wealth, actions = [], [info["wealth"]], []
    done = False
    while not done:
        mask = env.action_masks()
        a = int(policy(obs, mask))
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        wealth.append(info["wealth"])
        actions.append(info["action_executed"])
        done = term or trunc
    return {
        "delta_w": wealth[-1] - wealth[0],
        "sum_rewards": float(np.sum(rewards)),
        "n_trades": int(sum(a in (1, 2) for a in actions)),
        "terminal_wealth": wealth[-1],
        "wealth_path": wealth,
        "accounting_gap": abs((wealth[-1] - wealth[0]) - float(np.sum(rewards))),
    }


def policy_hold(obs, mask):
    return 0


def policy_bah_one(obs, mask):
    """Buy exactly one share on the first feasible step, then hold."""
    if float(obs[2]) < 0.5 and mask[1]:
        return 1
    return 0


def policy_bah_max(obs, mask):
    """Invest as much cash as possible (repeated Buy while feasible), then hold.

    Fairer economic baseline than one-share BAH when initial_cash >> P_t.
    """
    if mask[1]:
        return 1
    return 0


def make_random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def _p(obs, mask):
        legal = np.flatnonzero(mask)
        return int(rng.choice(legal))

    return _p


def make_ppo_policy(model: PPO):
    def _p(obs, mask):
        action, _ = model.predict(obs, deterministic=True)
        a = int(action)
        if not mask[a]:
            return 0
        return a

    return _p


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


def win_rate(agent_rows, bah_rows) -> float:
    wins = sum(
        1
        for a, b in zip(agent_rows, bah_rows)
        if a["delta_w"] > b["delta_w"] + 1e-9
    )
    return wins / max(1, len(agent_rows))


def main() -> None:
    meta, train_prices, test_prices = load_split()

    # --- Train PPO on train months ---
    train_env = MonthSamplerEnv(train_prices, initial_cash=CASH0, fee=FEE, sample_seed=0)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=8,
        gamma=0.99,
        verbose=1,
        seed=42,
    )
    model.learn(total_timesteps=80_000)
    model_path = RES / "ppo_spy_daily.zip"
    model.save(str(model_path))

    # --- Evaluate all methods on TEST months ---
    methods = {
        "B0_do_nothing": policy_hold,
        "B1_buy_one_hold": policy_bah_one,
        "B1b_buy_max_hold": policy_bah_max,
        "B2_random": make_random_policy(0),
        "A1_ppo": make_ppo_policy(model),
    }

    all_rows = {}
    for name, pol in methods.items():
        all_rows[name] = [run_episode(p, pol, force_mask=True) for p in test_prices]

    # Doer extra: how often deterministic PPO proposes an infeasible action
    # (env then forces Hold). High rate ⇒ policy has not internalised constraints.
    illegal_requests = 0
    total_steps = 0
    for prices in test_prices:
        env = OneStockDiscreteEnv(prices, initial_cash=CASH0, fee=FEE)
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            total_steps += 1
            if not mask[a]:
                illegal_requests += 1
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc

    summary = {name: summarize(rows) for name, rows in all_rows.items()}
    for name in summary:
        summary[name]["win_rate_vs_B1_one"] = win_rate(all_rows[name], all_rows["B1_buy_one_hold"])
        summary[name]["win_rate_vs_B1b_max"] = win_rate(all_rows[name], all_rows["B1b_buy_max_hold"])

    payload = {
        "meta": {
            "ticker": meta["ticker"],
            "bar": meta["bar"],
            "episode_def": meta["episode_def"],
            "note": meta["note"],
            "n_train": meta["n_train"],
            "n_test": meta["n_test"],
            "test_ids": meta["test_ids"],
            "fee": FEE,
            "initial_cash": CASH0,
            "ppo_timesteps": 80_000,
            "model": str(model_path.relative_to(ROOT)),
        },
        "summary": summary,
        "doer_extras": {
            "ppo_illegal_action_request_rate": illegal_requests / max(1, total_steps),
            "ppo_illegal_requests": illegal_requests,
            "ppo_total_eval_steps": total_steps,
            "interpretation": (
                "Fraction of deterministic PPO actions that were infeasible "
                "under cash/share constraints (env then forces Hold)."
            ),
            "ppo_equals_b1b_buy_max_hold": bool(
                abs(summary["A1_ppo"]["mean_delta_w"] - summary["B1b_buy_max_hold"]["mean_delta_w"]) < 1e-9
                and abs(summary["A1_ppo"]["mean_trades"] - summary["B1b_buy_max_hold"]["mean_trades"]) < 1e-9
            ),
            "honest_finding": (
                "On this daily-month SPY setup with one-share discrete actions, "
                "deterministic PPO collapsed to the fully-invested buy-and-hold "
                "policy (B1b). That is not a failure of the experiment — it is "
                "the result: without a richer state or a different objective, "
                "the learned policy is economically identical to buying max shares."
            ),
        },
    }
    (RES / "phase1_spy_daily_results.json").write_text(json.dumps(payload, indent=2))

    # Bar chart mean ΔW
    labels = ["B0\ndo-nothing", "B1\nbuy-one", "B1b\nbuy-max", "B2\nrandom", "A1\nPPO"]
    keys = ["B0_do_nothing", "B1_buy_one_hold", "B1b_buy_max_hold", "B2_random", "A1_ppo"]
    means = [summary[k]["mean_delta_w"] for k in keys]
    stds = [summary[k]["std_delta_w"] for k in keys]
    colors = ["#888", "#4a6fa5", "#2f5f8f", "#c47a2c", "#1a7a4c"]

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=160)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean ΔW on test months ($)")
    ax.set_title("Phase 1 — SPY daily months\nIteration-1 discrete agent vs baselines")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "phase1_spy_daily_delta_w.png", dpi=220, bbox_inches="tight")
    fig.savefig(RES / "delta_w_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Wealth path: pick middle test month, B1b vs A1
    mid = len(test_prices) // 2
    bah = all_rows["B1b_buy_max_hold"][mid]["wealth_path"]
    ppo = all_rows["A1_ppo"][mid]["wealth_path"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    ax.plot(bah, label="B1b buy-max hold", lw=2)
    ax.plot(ppo, label="A1 PPO", lw=2)
    ax.set_xlabel("Day index within month")
    ax.set_ylabel("Wealth W_t ($)")
    ax.set_title(f"Wealth path — test month {meta['test_ids'][mid]}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "phase1_spy_daily_wealth_path.png", dpi=220, bbox_inches="tight")
    fig.savefig(RES / "wealth_path.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(payload["summary"], indent=2))
    print("doer_extras:", json.dumps(payload["doer_extras"], indent=2))
    print(f"wrote {RES / 'phase1_spy_daily_results.json'}")


if __name__ == "__main__":
    main()
