#!/usr/bin/env python3
"""Full algorithm ablation on the final 9-D divisible-asset trading MDP.

Methods (Nguyen priority order):
  1. REINFORCE (from scratch, masked)   — Monte-Carlo policy gradient
  2. Deep Q     (from scratch, masked)  — value-based, replay + target net
  3. PPO        (SB3, stretch row)      — clipped policy gradient

Plus: baselines, per-seed spread, learning curves, wealth paths and
exposure profiles for representative months, and a fee-sensitivity study.

Run:
  ./venv/bin/python experiments/final_model/run_full_ablation.py
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
from month_sampler_full import MonthSamplerFullEnv  # noqa: E402
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
FEE_GRID = [0.0, 0.0005, 0.002]  # 0, 5 bps, 20 bps


def make_env(prices: np.ndarray, fee: float = FEE) -> FullStateTradingEnv:
    return FullStateTradingEnv(prices, initial_cash=CASH0, fee=fee)


def load_split():
    meta = json.loads((DATA / "spy_daily_meta.json").read_text())
    npz = np.load(DATA / "spy_daily_episodes.npz")
    train = [npz[f"train_{i}"] for i in meta["train_ids"]]
    test = [npz[f"test_{i}"] for i in meta["test_ids"]]
    return meta, train, test


def run_episode(prices: np.ndarray, policy, fee: float = FEE) -> dict:
    env = make_env(prices, fee)
    obs, info = env.reset()
    rewards, wealth, actions, exposure = [], [info["wealth"]], [], [0.0]
    illegal = 0
    done = False
    while not done:
        a = int(policy(obs, env.action_masks()))
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        wealth.append(info["wealth"])
        actions.append(info["action_executed"])
        exposure.append(info["units"] * float(env.prices[min(env.t, len(env.prices) - 1)]) / CASH0)
        illegal += int(info["illegal"])
        done = term or trunc
    return {
        "delta_w": wealth[-1] - wealth[0],
        "n_trades": int(info["n_trades"]),
        "illegal": illegal,
        "actions": actions,
        "wealth_path": wealth,
        "exposure_path": exposure,
        "accounting_gap": abs((wealth[-1] - wealth[0]) - float(np.sum(rewards))),
    }


def summarize(rows: list[dict]) -> dict:
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


def identical_months(agent_rows, ref_rows) -> int:
    return sum(1 for a, b in zip(agent_rows, ref_rows) if a["actions"] == b["actions"])


def policy_hold(obs, mask):
    return 0


def policy_bah(obs, mask):
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


def downsample_curve(t: list, ret: list, n: int = 200) -> dict:
    if len(t) <= n:
        return {"t": t, "ret": ret}
    idx = np.linspace(0, len(t) - 1, n).astype(int)
    return {"t": [t[i] for i in idx], "ret": [ret[i] for i in idx]}


def main() -> None:
    from stable_baselines3 import PPO  # deferred: slow import

    meta, train_prices, test_prices = load_split()
    test_ids = meta["test_ids"]
    timings: dict[str, float] = {}

    # ---------- baselines ----------
    random_policy = make_random_policy(0)  # ONE rng across all test months
    all_rows: dict[str, list[dict]] = {
        "B0_do_nothing": [run_episode(p, policy_hold) for p in test_prices],
        "B1b_buy_and_hold": [run_episode(p, policy_bah) for p in test_prices],
        "B2_random": [run_episode(p, random_policy) for p in test_prices],
    }

    # ---------- learned methods, 3 seeds each ----------
    per_seed: dict[str, dict[int, dict]] = {"reinforce": {}, "dqn": {}, "ppo": {}}

    for seed in SEEDS:
        print(f"=== REINFORCE seed={seed} ===", flush=True)
        t0 = time.time()
        rf = train_reinforce_trading(make_env, train_prices, obs_dim=OBS_DIM,
                                     total_timesteps=TIMESTEPS, seed=seed)
        timings[f"reinforce_s{seed}"] = time.time() - t0
        torch.save(rf.policy.state_dict(), RES / f"reinforce_full_seed{seed}.pt")
        per_seed["reinforce"][seed] = {
            "rows": [run_episode(p, make_reinforce_policy(rf.policy)) for p in test_prices],
            "curve": downsample_curve(rf.episode_timesteps, rf.episode_returns),
        }
        print(f"  {timings[f'reinforce_s{seed}']:.0f}s", flush=True)

        print(f"=== Deep Q seed={seed} ===", flush=True)
        t0 = time.time()
        dq = train_dqn_trading(make_env, train_prices, obs_dim=OBS_DIM,
                               total_timesteps=TIMESTEPS, seed=seed)
        timings[f"dqn_s{seed}"] = time.time() - t0
        torch.save(dq.qnet.state_dict(), RES / f"dqn_full_seed{seed}.pt")
        per_seed["dqn"][seed] = {
            "rows": [run_episode(p, make_dqn_policy(dq.qnet)) for p in test_prices],
            "curve": downsample_curve(dq.episode_timesteps, dq.episode_returns),
        }
        print(f"  {timings[f'dqn_s{seed}']:.0f}s", flush=True)

        print(f"=== PPO (SB3) seed={seed} ===", flush=True)
        t0 = time.time()
        ppo_env = MonthSamplerFullEnv(train_prices, initial_cash=CASH0, fee=FEE, sample_seed=0)
        ppo = PPO("MlpPolicy", ppo_env, learning_rate=3e-4, n_steps=512, batch_size=64,
                  n_epochs=8, gamma=0.99, verbose=0, seed=seed)
        ppo.learn(total_timesteps=TIMESTEPS)
        ppo.save(str(RES / f"ppo_full_seed{seed}.zip"))
        timings[f"ppo_s{seed}"] = time.time() - t0
        per_seed["ppo"][seed] = {
            "rows": [run_episode(p, make_sb3_policy(ppo)) for p in test_prices],
            "curve": None,
        }
        print(f"  {timings[f'ppo_s{seed}']:.0f}s", flush=True)

    all_rows["A_reinforce"] = per_seed["reinforce"][SEEDS[0]]["rows"]
    all_rows["A_dqn"] = per_seed["dqn"][SEEDS[0]]["rows"]
    all_rows["A_ppo"] = per_seed["ppo"][SEEDS[0]]["rows"]

    summary = {name: summarize(rows) for name, rows in all_rows.items()}
    for name in summary:
        summary[name]["win_rate_vs_B1b"] = win_rate(all_rows[name], all_rows["B1b_buy_and_hold"])
        summary[name]["months_identical_to_B1b"] = identical_months(
            all_rows[name], all_rows["B1b_buy_and_hold"])

    seed_spread = {
        algo: {
            str(seed): {
                "mean_delta_w": float(np.mean([r["delta_w"] for r in d["rows"]])),
                "mean_trades": float(np.mean([r["n_trades"] for r in d["rows"]])),
                "months_identical_to_B1b": identical_months(d["rows"], all_rows["B1b_buy_and_hold"]),
            }
            for seed, d in per_seed[algo].items()
        }
        for algo in per_seed
    }

    # ---------- fee sensitivity (seed 42, REINFORCE + DQN + B1b) ----------
    fee_sens: dict[str, dict] = {}
    for fee in FEE_GRID:
        key = f"{fee:.4f}"
        print(f"=== fee sensitivity fee={key} ===", flush=True)
        rows_b1b = [run_episode(p, policy_bah, fee=fee) for p in test_prices]
        rf = train_reinforce_trading(lambda p: make_env(p, fee), train_prices,
                                     obs_dim=OBS_DIM, total_timesteps=TIMESTEPS, seed=42)
        rows_rf = [run_episode(p, make_reinforce_policy(rf.policy), fee=fee) for p in test_prices]
        dq = train_dqn_trading(lambda p: make_env(p, fee), train_prices,
                               obs_dim=OBS_DIM, total_timesteps=TIMESTEPS, seed=42)
        rows_dq = [run_episode(p, make_dqn_policy(dq.qnet), fee=fee) for p in test_prices]
        fee_sens[key] = {
            "B1b": summarize(rows_b1b),
            "reinforce": summarize(rows_rf),
            "dqn": summarize(rows_dq),
        }

    # ---------- representative months for path plots ----------
    dw_b1b = [r["delta_w"] for r in all_rows["B1b_buy_and_hold"]]
    idx_up = int(np.argmax(dw_b1b))
    idx_down = int(np.argmin(dw_b1b))
    paths = {
        "up_month": {"id": test_ids[idx_up], "index": idx_up},
        "down_month": {"id": test_ids[idx_down], "index": idx_down},
    }
    for label, d in paths.items():
        i = d["index"]
        d["wealth"] = {
            "B1b": all_rows["B1b_buy_and_hold"][i]["wealth_path"],
            "reinforce": all_rows["A_reinforce"][i]["wealth_path"],
            "dqn": all_rows["A_dqn"][i]["wealth_path"],
            "ppo": all_rows["A_ppo"][i]["wealth_path"],
        }
        d["exposure"] = {
            "B1b": all_rows["B1b_buy_and_hold"][i]["exposure_path"],
            "dqn": all_rows["A_dqn"][i]["exposure_path"],
        }

    payload = {
        "meta": {
            "purpose": "Full algorithm ablation on the final 9-D divisible-asset MDP.",
            "ticker": meta["ticker"], "bar": meta["bar"],
            "episode_def": meta["episode_def"],
            "n_train": meta["n_train"], "n_test": meta["n_test"],
            "fee": FEE, "initial_cash": CASH0,
            "timesteps_budget": TIMESTEPS, "seeds": SEEDS,
            "fee_grid": FEE_GRID,
            "ppo_note": ("SB3 PPO trained unmasked (env forces Hold on illegal); "
                         "evaluated deterministic with mask fallback, matching pilot protocol."),
            "timings_s": {k: round(v, 1) for k, v in timings.items()},
        },
        "summary": summary,
        "seed_spread": seed_spread,
        "fee_sensitivity": fee_sens,
        "monthly_delta_w": {name: [round(r["delta_w"], 4) for r in rows]
                            for name, rows in all_rows.items()},
        "illegal_rates": {name: float(np.mean([r["illegal"] for r in rows]))
                          for name, rows in all_rows.items()},
        "curves": {
            "reinforce": per_seed["reinforce"][SEEDS[0]]["curve"],
            "dqn": per_seed["dqn"][SEEDS[0]]["curve"],
        },
        "paths": paths,
    }
    out = RES / "full_ablation_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))

    # =================== charts ===================

    # 1. mean dW bar with PPO
    keys = ["B0_do_nothing", "B2_random", "B1b_buy_and_hold",
            "A_reinforce", "A_dqn", "A_ppo"]
    labels = ["B0\ndo-nothing", "B2\nrandom", "B1b\nbuy & hold",
              "REINFORCE\n(scratch)", "Deep Q\n(scratch)", "PPO\n(SB3)"]
    means = [summary[k]["mean_delta_w"] for k in keys]
    stds = [summary[k]["std_delta_w"] for k in keys]
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

    # 2. wealth paths, up and down months
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160, sharey=False)
    for ax, label, title in [
        (axes[0], "up_month", f"Rising month ({paths['up_month']['id']})"),
        (axes[1], "down_month", f"Falling month ({paths['down_month']['id']})"),
    ]:
        w = paths[label]["wealth"]
        for name, series, color, style in [
            ("B1b buy & hold", w["B1b"], "#2f5f8f", "-"),
            ("REINFORCE", w["reinforce"], "#7a4cc4", "--"),
            ("Deep Q", w["dqn"], "#1a7a4c", "-"),
            ("PPO", w["ppo"], "#c44c7a", ":"),
        ]:
            ax.plot(series, label=name, color=color, ls=style, lw=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Trading day in month")
        ax.set_ylabel("Wealth ($)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Wealth paths on representative test months", fontsize=11)
    fig.tight_layout()
    fig.savefig(CHARTS / "ablation_wealth_paths.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / "ablation_wealth_paths.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. exposure profile, falling month (DQN vs B1b)
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=160)
    e = paths["down_month"]["exposure"]
    ax.plot(e["B1b"], label="B1b buy & hold", color="#2f5f8f", lw=1.6)
    ax.plot(e["dqn"], label="Deep Q", color="#1a7a4c", lw=1.6)
    ax.set_xlabel("Trading day in month")
    ax.set_ylabel("Position value / $C_0$")
    ax.set_title(f"Exposure during the falling month ({paths['down_month']['id']})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "ablation_exposure_down_month.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / "ablation_exposure_down_month.pdf", bbox_inches="tight")
    plt.close(fig)

    # 4. training curves (scratch methods)
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=160)
    for algo, color in [("reinforce", "#7a4cc4"), ("dqn", "#1a7a4c")]:
        c = payload["curves"][algo]
        ret = np.array(c["ret"], dtype=float)
        if len(ret) >= 11:
            kernel = np.ones(11) / 11
            smooth = np.convolve(ret, kernel, mode="same")
        else:
            smooth = ret
        ax.plot(c["t"], smooth, label={"reinforce": "REINFORCE", "dqn": "Deep Q"}[algo],
                color=color, lw=1.5)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return ($, smoothed)")
    ax.set_title("Training curves on the trading MDP (seed 42)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "ablation_training_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / "ablation_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # 5. fee sensitivity
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=160)
    xg = np.arange(len(FEE_GRID))
    width = 0.25
    for off, (algo, color, label) in enumerate([
        ("B1b", "#2f5f8f", "B1b buy & hold"),
        ("reinforce", "#7a4cc4", "REINFORCE"),
        ("dqn", "#1a7a4c", "Deep Q"),
    ]):
        vals = [fee_sens[f"{f:.4f}"][algo]["mean_delta_w"] for f in FEE_GRID]
        ax.bar(xg + (off - 1) * width, vals, width, color=color, label=label)
    ax.set_xticks(xg)
    ax.set_xticklabels(["0 bps", "5 bps", "20 bps"])
    ax.set_xlabel("Proportional fee")
    ax.set_ylabel("Mean ΔW ($)")
    ax.set_title("Fee sensitivity (seed 42, retrained per fee)")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "ablation_fee_sensitivity.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / "ablation_fee_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    print("wrote 5 charts to", CHARTS)
    print("ABLATION_DONE")


if __name__ == "__main__":
    main()
