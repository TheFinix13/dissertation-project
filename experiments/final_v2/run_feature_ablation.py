#!/usr/bin/env python3
"""Feature ablation: does Deep Q-learning actually USE the volatility signals?

STUDY: post-viva engineering — NOT in the submitted dissertation. Writes
results/feature_ablation.json.

Why this exists (14 Sep 2026, the evening after the viva). In the viva I
claimed the October 2008 zero-exposure behaviour came from the volatility
features matching the falling conditions learned in training. Dr Nguyen
asked how I knew. I did not — it was a hypothesis presented as a finding.

This script settles it by ablation. Deep Q-learning is retrained under the
identical protocol to `run_sim.py` (same 3,000 simulated episodes, same
seeds, same 300,000 steps, same reward, same slice, same fee) with only
the state changed:

  S3_full     the dissertation's 9 features
  S3_novol    the same minus vol_k and atr_norm      <- the test
  S3_account  account features only, no market data  <- floor control

If the volatility claim is right, removing those two features should
degrade the crash behaviour: exposure through the 2008 window should rise
towards buy-and-hold's, and the down-regime loss should worsen. If nothing
changes, the claim was wrong and some other feature carries the signal.

The floor control matters because "no market features at all" bounds how
much of the behaviour is attributable to market information of ANY kind
rather than to the clock and the account state.

Run (from experiments/final_v2):
  ../../venv/bin/python run_feature_ablation.py
  ../../venv/bin/python run_feature_ablation.py --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import real_transfer
import sim_data
from features import resolve_rung
from harness import (INITIAL_CASH, SEEDS, Config, aggregate, provenance,
                     run_baselines, write_results)
from run_sim import across_seed_extras, per_regime, per_window, train_and_eval_seed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "feature_ablation.json"

#: The two volatility signals under test. vol_k is close-to-close standard
#: deviation over 5 days; atr_norm is the 14-day average true range over
#: price. They are the only two features that measure variability.
VOL_FEATURES = ("vol_k", "atr_norm")

#: Months to inspect for the crash behaviour. 2008-10 is the month quoted in
#: the deck; the wider band is reported so the finding does not rest on one
#: month.
CRASH_MONTH = "2008-10"
CRASH_BAND = ("2008-09", "2008-10", "2008-11", "2008-12")


def build_cells() -> dict[str, Config]:
    full = resolve_rung("S3")
    novol = tuple(f for f in full if f not in VOL_FEATURES)
    account = tuple(f for f in full
                    if f in ("clock", "cash_frac", "exposure_frac", "pnl"))
    return {
        "S3_full": Config("S3_full", full, rung="S3"),
        "S3_novol": Config("S3_novol", novol, rung="S3-novol"),
        "S3_account": Config("S3_account", account, rung="S3-account"),
    }


def exposure_in(per_month: list[dict], ids) -> float:
    """Mean exposure over the named months, or nan if none are present."""
    vals = [m["mean_exposure"] for m in per_month if m["id"] in ids]
    return float(np.mean(vals)) if vals else float("nan")


def delta_in(per_month: list[dict], ids) -> float:
    vals = [m["delta_w"] for m in per_month if m["id"] in ids]
    return float(np.sum(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.timesteps, args.seeds = 3_000, [42]

    sim_meta, sim_parts = sim_data.load()
    transfer_meta, transfer_parts = real_transfer.load()
    transfer_eps = real_transfer.all_episodes(transfer_parts)
    cells = build_cells()

    print(f"sim train={len(sim_parts['train'])} test={len(sim_parts['test'])}  "
          f"transfer={len(transfer_eps)} months")
    for name, cfg in cells.items():
        print(f"  {name:11s} obs_dim={len(cfg.feature_names)}  "
              f"{', '.join(cfg.feature_names)}")

    # Baselines are state-independent, so one config is enough to score them.
    ref = cells["S3_full"]
    sim_base = run_baselines(ref, sim_parts["test"])
    real_base = run_baselines(ref, transfer_eps)
    sim_ref_rows = sim_base["B1b_true_bah"]["_rows"]
    real_ref_rows = real_base["B1b_true_bah"]["_rows"]
    bah_month = {m["id"]: m for m in real_base["B1b_true_bah"]["per_month"]}
    print(f"\nbuy-and-hold {CRASH_MONTH}: "
          f"dW {bah_month[CRASH_MONTH]['delta_w']:+.2f}, "
          f"exposure {bah_month[CRASH_MONTH]['mean_exposure']:.3f}")

    results: dict = {}
    for name, cfg in cells.items():
        print(f"\n=== {name} ({len(cfg.feature_names)} features) ===")
        per_seed = []
        for seed in args.seeds:
            t0 = time.time()
            row = train_and_eval_seed(
                "dqn", cfg, sim_parts["train"], sim_parts["test"],
                transfer_eps, sim_ref_rows, real_ref_rows,
                seed=seed, timesteps=args.timesteps)
            per_seed.append(row)
            pm = row["real_per_month"]
            print(f"  seed {seed}: sim dW={row['summary']['mean_delta_w']:+8.2f}  "
                  f"down={row['per_regime']['down']['mean_delta_w']:+8.2f}  "
                  f"real dW={row['real_transfer']['mean_delta_w']:+7.2f}  "
                  f"expo(2008-10)={exposure_in(pm, {CRASH_MONTH}):.3f}  "
                  f"expo(band)={exposure_in(pm, set(CRASH_BAND)):.3f}  "
                  f"cond={'state' if row['summary']['state_dependent'] else 'MASK-ONLY'}"
                  f"  ({time.time() - t0:.0f}s)")

        entry = {"config": cfg.as_dict(), "per_seed": per_seed,
                 "across_seeds": aggregate(per_seed)}
        across_seed_extras(entry)
        entry["crash"] = {
            "month": CRASH_MONTH,
            "band": list(CRASH_BAND),
            "exposure_month": float(np.mean(
                [exposure_in(s["real_per_month"], {CRASH_MONTH})
                 for s in per_seed])),
            "exposure_band": float(np.mean(
                [exposure_in(s["real_per_month"], set(CRASH_BAND))
                 for s in per_seed])),
            "delta_band": float(np.mean(
                [delta_in(s["real_per_month"], set(CRASH_BAND))
                 for s in per_seed])),
        }
        results[name] = entry

    payload = {
        "study": "feature ablation: are the volatility signals load-bearing?",
        "question": "Dr Nguyen, viva 14 Sep 2026: 'how do you know that?'",
        "ablated": list(VOL_FEATURES),
        "algo": "dqn",
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "network": "pyramid MLP, hidden (64, 32)",
        "baselines_sim": {k: {"summary": v["summary"]}
                          for k, v in sim_base.items()},
        "baselines_real": {k: {"summary": v["summary"]}
                           for k, v in real_base.items()},
        "bah_crash": {
            "month": CRASH_MONTH,
            "delta_w": bah_month[CRASH_MONTH]["delta_w"],
            "exposure": bah_month[CRASH_MONTH]["mean_exposure"],
            "band_delta_w": float(np.sum(
                [bah_month[m]["delta_w"] for m in CRASH_BAND])),
            "band_exposure": float(np.mean(
                [bah_month[m]["mean_exposure"] for m in CRASH_BAND])),
        },
        "provenance": provenance(),
        "results": results,
    }
    write_results(RESULTS, payload)

    print("\n=== ABLATION SUMMARY (6-seed means) ===")
    hdr = (f"{'cell':<11} {'sim dW':>9} {'spread':>7} {'down':>9} "
           f"{'real dW':>8} {'expo 08-10':>10} {'expo band':>9} {'state':>6}")
    print(hdr)
    for name, e in results.items():
        a, c = e["across_seeds"], e["crash"]
        print(f"{name:<11} {a['mean_delta_w']['mean']:+9.2f} "
              f"{a['mean_delta_w']['spread']:7.2f} "
              f"{a['regime_down_delta_w']['mean']:+9.2f} "
              f"{a['real_transfer_delta_w']['mean']:+8.2f} "
              f"{c['exposure_month']:10.3f} {c['exposure_band']:9.3f} "
              f"{a['n_state_dependent']}/{a['n_seeds']:>4}")
    b = payload["bah_crash"]
    print(f"{'buy&hold':<11} {'—':>9} {'—':>7} "
          f"{real_base['B1b_true_bah']['summary']['mean_delta_w']:>9} "
          f"{'—':>8} {b['exposure']:10.3f} {b['band_exposure']:9.3f}")
    print("\nFEATURE_ABLATION_DONE")


if __name__ == "__main__":
    main()
