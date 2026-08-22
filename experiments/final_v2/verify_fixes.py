#!/usr/bin/env python3
"""Quantify the two substantive defects found in the earlier setup.

This script exists so the corrections reported in the dissertation are backed by
measured magnitudes rather than by argument alone. It trains nothing.

Defect 1 — rolling features truncated at the episode boundary.
  The earlier environment computed momentum, volatility and the trend gap inside
  each episode with a shortened window, so early days of every month received
  degenerate values. Reported here: how many days per episode were affected and
  how far the values were from the correct continuous ones.

Defect 2 — the buy-and-hold baseline was a dollar-cost-average.
  Buying one tenth of capital whenever a Buy was legal needs ten steps to reach
  full investment. Reported here: the resulting understatement of buy-and-hold
  on the real test months.

Run:
  ./venv/bin/python experiments/final_v2/verify_fixes.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data as datamod
from baselines import policy_buy_when_legal, policy_hold
from env import TradingEnv
from features import MA_WINDOW, MARKET_FEATURES, MOM_WINDOW, VOL_WINDOW, resolve_rung
from rollout import evaluate, run_episode, summarize, win_rate

HERE = Path(__file__).resolve().parent
CASH0 = 10_000.0
OLD_MOM_WINDOW = 5
OLD_VOL_WINDOW = 5
OLD_MA_WINDOW = 10


def old_style_features(prices: np.ndarray) -> np.ndarray:
    """Reproduce the earlier in-episode computation, truncated at t=0."""
    out = np.zeros((len(prices), 3))
    for t in range(len(prices)):
        k = min(OLD_MOM_WINDOW, t)
        out[t, 0] = 0.0 if k == 0 else prices[t] / prices[t - k] - 1.0

        if t < 2:
            out[t, 1] = 0.0
        else:
            seg = prices[max(0, t - OLD_VOL_WINDOW): t + 1]
            rets = np.diff(seg) / seg[:-1]
            out[t, 1] = float(np.std(rets)) if len(rets) >= 2 else 0.0

        lo = max(0, t - OLD_MA_WINDOW + 1)
        ma = float(np.mean(prices[lo: t + 1]))
        out[t, 2] = prices[t] / ma - 1.0 if ma != 0 else 0.0
    return out


def continuous_equivalent(prices_all: np.ndarray, offset: int, n: int) -> np.ndarray:
    """The same three definitions, computed with full preceding history."""
    out = np.zeros((n, 3))
    for j in range(n):
        t = offset + j
        out[j, 0] = prices_all[t] / prices_all[t - OLD_MOM_WINDOW] - 1.0
        seg = prices_all[t - OLD_VOL_WINDOW: t + 1]
        rets = np.diff(seg) / seg[:-1]
        out[j, 1] = float(np.std(rets))
        ma = float(np.mean(prices_all[t - OLD_MA_WINDOW + 1: t + 1]))
        out[j, 2] = prices_all[t] / ma - 1.0
    return out


def defect_1(parts) -> dict:
    all_eps = parts["train"] + parts["val"] + parts["test"]
    prices_all = np.concatenate([e["prices"] for e in all_eps])
    offsets, pos = [], 0
    for e in all_eps:
        offsets.append(pos)
        pos += len(e["prices"])

    rows = []
    for e, off in zip(all_eps, offsets):
        n = len(e["prices"])
        if off < OLD_MA_WINDOW:  # first episode has no preceding history at all
            continue
        old = old_style_features(e["prices"])
        new = continuous_equivalent(prices_all, off, n)
        err = np.abs(old - new)
        affected = int(np.sum(np.any(err > 1e-12, axis=1)))
        rows.append({
            "id": e["id"],
            "n_bars": n,
            "days_affected": affected,
            "frac_affected": affected / n,
            "day0_mom_old": float(old[0, 0]),
            "day0_mom_true": float(new[0, 0]),
            "day0_vol_old": float(old[0, 1]),
            "day0_vol_true": float(new[0, 1]),
            "max_abs_err_mom": float(err[:, 0].max()),
            "max_abs_err_vol": float(err[:, 1].max()),
            "max_abs_err_gap": float(err[:, 2].max()),
        })

    frac = np.array([r["frac_affected"] for r in rows])
    return {
        "n_episodes_checked": len(rows),
        "mean_frac_of_episode_affected": float(frac.mean()),
        "max_frac_of_episode_affected": float(frac.max()),
        "mean_days_affected": float(np.mean([r["days_affected"] for r in rows])),
        "day0_momentum_was_always_zero": all(r["day0_mom_old"] == 0.0 for r in rows),
        "day0_volatility_was_always_zero": all(r["day0_vol_old"] == 0.0 for r in rows),
        "mean_true_day0_momentum_abs": float(
            np.mean([abs(r["day0_mom_true"]) for r in rows])),
        "worst_abs_error": {
            "momentum": float(max(r["max_abs_err_mom"] for r in rows)),
            "volatility": float(max(r["max_abs_err_vol"] for r in rows)),
            "trend_gap": float(max(r["max_abs_err_gap"] for r in rows)),
        },
        "note": (
            f"windows compared at the earlier settings (mom {OLD_MOM_WINDOW}, "
            f"vol {OLD_VOL_WINDOW}, MA {OLD_MA_WINDOW}) so the difference "
            f"isolates truncation rather than the new window lengths "
            f"(now mom {MOM_WINDOW}, vol {VOL_WINDOW}, MA {MA_WINDOW})"),
    }


def defect_2(parts) -> dict:
    test = parts["test"]
    names = resolve_rung("S3")

    def factory(slice_frac):
        def f(ep):
            return TradingEnv(ep["prices"], ep["market"], names,
                              initial_cash=CASH0, fee=0.0005, slice_frac=slice_frac)
        return f

    slice_rows = evaluate(factory(0.10), test, policy_buy_when_legal)
    true_rows = evaluate(factory(1.0), test, policy_buy_when_legal)
    hold_rows = evaluate(factory(0.10), test, policy_hold)

    s_slice = summarize(slice_rows, CASH0)
    s_true = summarize(true_rows, CASH0)

    return {
        "B1a_slice_bah": s_slice,
        "B1b_true_bah": s_true,
        "B0_do_nothing": summarize(hold_rows, CASH0),
        "understatement_mean_delta_w": s_true["mean_delta_w"] - s_slice["mean_delta_w"],
        "understatement_pct": (
            100.0 * (s_true["mean_delta_w"] - s_slice["mean_delta_w"])
            / abs(s_true["mean_delta_w"]) if s_true["mean_delta_w"] != 0 else 0.0),
        "months_true_beats_slice": int(sum(
            1 for a, b in zip(true_rows, slice_rows) if a["delta_w"] > b["delta_w"])),
        "n_test_months": len(test),
        "slice_win_rate_vs_true": win_rate(slice_rows, true_rows),
    }


def integrity_checks(parts) -> dict:
    names = resolve_rung("S4")
    gaps, dds = [], []
    for split in ("train", "val", "test"):
        for ep in parts[split]:
            env = TradingEnv(ep["prices"], ep["market"], names,
                             initial_cash=CASH0, fee=0.0005, risk_lambda=0.0)
            row = run_episode(env, policy_buy_when_legal)
            gaps.append(row["accounting_gap"])
            dds.append(row["intra_dd"])
    return {
        "episodes_checked": len(gaps),
        "max_accounting_gap": float(max(gaps)),
        "accounting_ok": bool(max(gaps) < 1e-9),
        "mean_intra_episode_drawdown_buy_and_hold": float(np.mean(dds)),
        "max_intra_episode_drawdown_buy_and_hold": float(np.max(dds)),
    }


def main() -> None:
    meta, parts = datamod.load()
    payload = {
        "data": {
            "counts": meta["counts"],
            "train_span": [meta["ids"]["train"][0], meta["ids"]["train"][-1]],
            "val_span": [meta["ids"]["val"][0], meta["ids"]["val"][-1]],
            "test_span": [meta["ids"]["test"][0], meta["ids"]["test"][-1]],
            "market_features": list(MARKET_FEATURES),
        },
        "defect_1_episode_boundary_truncation": defect_1(parts),
        "defect_2_baseline_was_dollar_cost_average": defect_2(parts),
        "integrity": integrity_checks(parts),
    }
    out = HERE / "results" / "verify_fixes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
