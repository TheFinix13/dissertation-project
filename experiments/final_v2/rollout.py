"""Episode rollout and metrics.

Two metric corrections over the earlier version.

* **Drawdown is measured inside episodes, not only across them.** The earlier
  code built a single equity curve by adding each month's profit to a constant
  starting balance and took the drawdown of that. That series says nothing about
  what happened *within* a month, which is where a trader's risk actually lives.
  Both are now reported and named for what they are.

* **The accounting identity is checked on `dw`, not on the reward.** Under a
  shaped reward the two differ by the risk penalty, so testing the identity
  against the reward would fail for a correct environment.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Policy = Callable[[np.ndarray, np.ndarray], int]


def run_episode(env, policy: Policy) -> dict:
    obs, info = env.reset()
    w0 = info["wealth"]
    dws: list[float] = []
    rewards: list[float] = []
    wealth = [w0]
    exposure = [0.0]
    actions: list[int] = []
    illegal = 0

    done = False
    while not done:
        action = int(policy(obs, env.action_masks()))
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(float(reward))
        dws.append(float(info["dw"]))
        wealth.append(float(info["wealth"]))
        exposure.append(float(info["exposure"]) / env.initial_cash)
        actions.append(int(info["action_executed"]))
        illegal += int(info["illegal"])
        done = term or trunc

    w = np.asarray(wealth, dtype=float)
    peak = np.maximum.accumulate(w)
    intra_dd = float(np.max(1.0 - w / peak)) if np.all(peak > 0) else 0.0

    return {
        "delta_w": float(w[-1] - w[0]),
        "sum_dw": float(np.sum(dws)),
        "sum_reward": float(np.sum(rewards)),
        "n_trades": int(info["n_trades"]),
        "fees_paid": float(info["fees_paid"]),
        "illegal": illegal,
        "intra_dd": intra_dd,
        "mean_exposure": float(np.mean(exposure)),
        "actions": actions,
        "wealth_path": [float(x) for x in w],
        "exposure_path": exposure,
        # identity: the wealth changes must telescope to the episode's profit
        "accounting_gap": abs(float(w[-1] - w[0]) - float(np.sum(dws))),
    }


def evaluate(factory, episodes: Sequence[dict], policy: Policy) -> list[dict]:
    rows = []
    for ep in episodes:
        row = run_episode(factory(ep), policy)
        row["episode_id"] = ep.get("id")
        rows.append(row)
    return rows


def summarize(rows: Sequence[dict], initial_cash: float = 10_000.0) -> dict:
    dw = np.array([r["delta_w"] for r in rows], dtype=float)
    trades = np.array([r["n_trades"] for r in rows], dtype=float)
    gaps = np.array([r["accounting_gap"] for r in rows], dtype=float)
    intra = np.array([r["intra_dd"] for r in rows], dtype=float)
    expo = np.array([r["mean_exposure"] for r in rows], dtype=float)
    fees = np.array([r["fees_paid"] for r in rows], dtype=float)

    monthly_ret = dw / initial_cash
    sd = monthly_ret.std(ddof=1) if len(monthly_ret) > 1 else 0.0
    sharpe = float(monthly_ret.mean() / sd) if sd > 1e-12 else 0.0

    # Aggregate curve: monthly profits accumulated on a constant stake. This is
    # a summary of month-to-month consistency, not a compounded equity curve.
    equity = np.concatenate([[initial_cash], initial_cash + np.cumsum(dw)])
    peak = np.maximum.accumulate(equity)
    mdd_monthly = float(np.max(1.0 - equity / peak))

    return {
        "n_episodes": len(rows),
        "mean_delta_w": float(dw.mean()),
        "std_delta_w": float(dw.std(ddof=1)) if len(dw) > 1 else 0.0,
        "mean_trades": float(trades.mean()),
        "mean_fees_paid": float(fees.mean()),
        "mean_exposure": float(expo.mean()),
        "sharpe_monthly": sharpe,
        "mdd_monthly_pnl": mdd_monthly,
        "mdd_intra_mean": float(intra.mean()),
        "mdd_intra_max": float(intra.max()),
        "max_accounting_gap": float(gaps.max()),
        "accounting_ok": bool(gaps.max() < 1e-6),
        "illegal_rate": float(np.mean([r["illegal"] for r in rows])),
    }


def win_rate(rows: Sequence[dict], reference: Sequence[dict]) -> float:
    wins = sum(1 for a, b in zip(rows, reference)
               if a["delta_w"] > b["delta_w"] + 1e-9)
    return wins / max(1, len(rows))


def identical_months(rows: Sequence[dict], reference: Sequence[dict]) -> int:
    return sum(1 for a, b in zip(rows, reference) if a["actions"] == b["actions"])
