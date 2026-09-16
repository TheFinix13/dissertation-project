#!/usr/bin/env python3
"""Tuned stop-loss baseline: the simple rule the viva asked about.

STUDY: post-viva engineering — NOT in the submitted dissertation. Writes
results/stop_loss.json.

Why this exists (16 Sep 2026, from the viva of 14 Sep). Prof Nikitopoulos
argued that if a simple stop-loss can cut crash losses, a learned agent
that does the same adds no value. The dissertation never measured that
rule, so the objection could not be answered with a number. This script
produces the number.

The rule: fully invested at the first bar, like true buy-and-hold, then
sell everything when the position falls a threshold below its entry
(fixed stop) or below its running peak since entry (trailing stop), and
stay in cash until the month's forced liquidation.

Protocol, mirroring the main study's validation discipline:

1. Sweep variant (fixed/trailing) x threshold on the 300 simulated
   VALIDATION episodes only.
2. Select the variant with the highest mean wealth change there.
3. Report that one selection, once, on the 600 held-out simulated test
   episodes (overall and per market condition) and on the 180 real
   transfer months (overall, per window, and the 2008 crisis band).

Comparators come from `results/sim_results.json` (DQN across 6 seeds)
and from re-evaluated buy-and-hold on the same episode sets.

Run:
  ./venv/bin/python experiments/final_v2/run_stop_loss.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import real_transfer
import sim_data
from baselines import StopLossPolicy, policy_buy_when_legal
from features import resolve_rung
from harness import INITIAL_CASH, Config, provenance, write_results
from rollout import evaluate, summarize

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "stop_loss.json"
SIM_RESULTS = HERE / "results" / "sim_results.json"

#: The sweep grid. Thresholds are fractions of the entry (fixed) or of the
#: running peak since entry (trailing).
THRESHOLDS = (0.02, 0.05, 0.08, 0.10, 0.15)
VARIANTS = ("fixed", "trailing")

#: The 2008 crisis band used in Chapter 5's worked example.
CRISIS_FIRST, CRISIS_LAST = "2008-09", "2009-03"


def eval_rule(factory, episodes, feature_names, *, threshold, trailing):
    """Evaluate one stop-loss configuration; fresh policy per call."""
    policy = StopLossPolicy(feature_names, threshold=threshold,
                            trailing=trailing)
    return evaluate(factory, episodes, policy)


def per_regime(rows) -> dict:
    out = {}
    for regime in sim_data.REGIMES:
        sub = [r for r in rows if sim_data.regime_of(r["episode_id"]) == regime]
        if sub:
            out[regime] = summarize(sub, INITIAL_CASH)
    return out


def per_window(rows) -> dict:
    out = {}
    for window in real_transfer.WINDOWS:
        sub = [r for r in rows
               if real_transfer.window_of(r["episode_id"]) == window]
        if sub:
            out[window] = summarize(sub, INITIAL_CASH)
    return out


def crisis_band(rows) -> dict:
    """Sum of monthly wealth changes across the 2008-09 .. 2009-03 band."""
    band = [r for r in rows
            if CRISIS_FIRST <= r["episode_id"] <= CRISIS_LAST]
    oct08 = [r for r in rows if r["episode_id"] == "2008-10"]
    return {
        "months": [r["episode_id"] for r in band],
        "sum_delta_w": float(sum(r["delta_w"] for r in band)),
        "mean_exposure": float(np.mean([r["mean_exposure"] for r in band]))
        if band else 0.0,
        "oct_2008_delta_w": float(oct08[0]["delta_w"]) if oct08 else None,
        "oct_2008_exposure": float(oct08[0]["mean_exposure"]) if oct08 else None,
    }


def dqn_reference() -> dict | None:
    """Pull the DQN transfer comparison out of the main-study results."""
    if not SIM_RESULTS.exists():
        return None
    d = json.loads(SIM_RESULTS.read_text())
    dqn = d["results"]["dqn"]
    # Crisis band per seed, from the persisted per-month transfer rows.
    band_sums = []
    for seed_row in dqn["per_seed"]:
        months = seed_row["real_per_month"]
        band_sums.append(sum(m["delta_w"] for m in months
                             if CRISIS_FIRST <= m["id"] <= CRISIS_LAST))
    return {
        "sim_test_delta_w": dqn["across_seeds"]["mean_delta_w"],
        "transfer_delta_w": dqn["across_seeds"]["real_transfer_delta_w"],
        "transfer_sharpe": [s["real_transfer"]["sharpe_monthly"]
                            for s in dqn["per_seed"]],
        "crisis_band_sum_delta_w": {
            "mean": float(np.mean(band_sums)),
            "min": float(np.min(band_sums)),
            "max": float(np.max(band_sums)),
        },
        "source": str(SIM_RESULTS.name),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sim_meta, sim_parts = sim_data.load()
    transfer_meta, transfer_parts = real_transfer.load()
    transfer_eps = real_transfer.all_episodes(transfer_parts)

    # Same state and fee as the studies; the rule needs clock/pnl/exposure,
    # all inside S3. The slice override makes it an investor benchmark with
    # buy-and-hold's powers, not a policy from the agents' action set.
    cfg = Config("STOPLOSS_S3", resolve_rung("S3"), rung="S3")
    factory = cfg.factory(slice_frac=1.0, risk_lambda=0.0)

    print(f"sweep: {len(VARIANTS) * len(THRESHOLDS)} configurations on "
          f"{len(sim_parts['val'])} simulated validation episodes")

    # ---- Stage 1: tune on the simulated validation set only
    sweep = []
    for variant in VARIANTS:
        for thr in THRESHOLDS:
            rows = eval_rule(factory, sim_parts["val"], cfg.feature_names,
                             threshold=thr, trailing=(variant == "trailing"))
            s = summarize(rows, INITIAL_CASH)
            sweep.append({"variant": variant, "threshold": thr,
                          "val_summary": s})
            print(f"  {variant:8s} thr={thr:.2f}: val dW={s['mean_delta_w']:+8.2f} "
                  f"expo={s['mean_exposure']:.2f} ddI={s['mdd_intra_mean']:.3f}")

    chosen = max(sweep, key=lambda r: r["val_summary"]["mean_delta_w"])
    print(f"\nselected on validation: {chosen['variant']} stop at "
          f"{chosen['threshold']:.0%} (val dW={chosen['val_summary']['mean_delta_w']:+.2f})")

    # ---- Stage 2: report the selection once on the held-out sets
    trailing = chosen["variant"] == "trailing"
    thr = chosen["threshold"]

    sim_rows = eval_rule(factory, sim_parts["test"], cfg.feature_names,
                         threshold=thr, trailing=trailing)
    real_rows = eval_rule(factory, transfer_eps, cfg.feature_names,
                          threshold=thr, trailing=trailing)

    # Buy-and-hold on the same sets, same environment powers.
    bah_sim = evaluate(factory, sim_parts["test"], policy_buy_when_legal)
    bah_real = evaluate(factory, transfer_eps, policy_buy_when_legal)

    report = {
        "sim_test": summarize(sim_rows, INITIAL_CASH),
        "sim_test_per_regime": per_regime(sim_rows),
        "transfer": summarize(real_rows, INITIAL_CASH),
        "transfer_per_window": per_window(real_rows),
        "transfer_crisis_band": crisis_band(real_rows),
        "transfer_per_month": [
            {"id": r["episode_id"], "delta_w": r["delta_w"],
             "n_trades": r["n_trades"], "mean_exposure": r["mean_exposure"],
             "intra_dd": r["intra_dd"]} for r in real_rows],
    }
    bah_report = {
        "sim_test": summarize(bah_sim, INITIAL_CASH),
        "transfer": summarize(bah_real, INITIAL_CASH),
        "transfer_crisis_band": crisis_band(bah_real),
    }

    print(f"\nheld-out results for the {chosen['variant']} {thr:.0%} stop:")
    print(f"  sim test  dW={report['sim_test']['mean_delta_w']:+8.2f} "
          f"(BAH {bah_report['sim_test']['mean_delta_w']:+8.2f})")
    print(f"  transfer  dW={report['transfer']['mean_delta_w']:+8.2f} "
          f"sharpe={report['transfer']['sharpe_monthly']:.3f} "
          f"(BAH {bah_report['transfer']['mean_delta_w']:+8.2f} / "
          f"{bah_report['transfer']['sharpe_monthly']:.3f})")
    cb, cb_b = report["transfer_crisis_band"], bah_report["transfer_crisis_band"]
    print(f"  crisis band {CRISIS_FIRST}..{CRISIS_LAST}: "
          f"stop {cb['sum_delta_w']:+8.2f} vs BAH {cb_b['sum_delta_w']:+8.2f}; "
          f"Oct 2008 {cb['oct_2008_delta_w']:+8.2f} vs {cb_b['oct_2008_delta_w']:+8.2f}")

    dqn = dqn_reference()
    if dqn:
        print(f"  DQN (6-seed mean, sim_results.json): transfer "
              f"dW={dqn['transfer_delta_w']['mean']:+.2f}, crisis band "
              f"{dqn['crisis_band_sum_delta_w']['mean']:+.2f}")

    payload = {
        "study": ("post-viva stop-loss baseline: does a simple rule match "
                  "the learned agent's crash protection?"),
        "not_in_dissertation": True,
        "protocol": {
            "rule": ("fully invested at first bar; sell all when the position "
                     "falls `threshold` below entry (fixed) or below its peak "
                     "since entry (trailing); stay in cash to liquidation"),
            "tuning": "selected on the 300 simulated validation episodes only",
            "selection_rule": "highest mean wealth change on validation",
            "grid": {"variants": list(VARIANTS),
                     "thresholds": list(THRESHOLDS)},
            "environment": ("S3 features, fee 5 bps, slice_frac=1.0 "
                            "(buy-and-hold powers)"),
        },
        "config": cfg.as_dict(),
        "sweep_validation": sweep,
        "chosen": {"variant": chosen["variant"], "threshold": thr},
        "stop_loss": report,
        "buy_and_hold": bah_report,
        "dqn_reference": dqn,
        "provenance": provenance(),
    }
    write_results(RESULTS.with_name(args.out) if args.out else RESULTS, payload)


if __name__ == "__main__":
    main()
