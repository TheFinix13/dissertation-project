#!/usr/bin/env python3
"""Simulated-data study: train on generated episodes, test under control.

This is the redesign of the final study after the real-data version was found
to be data-starved (60 training episodes against a ~16k-parameter network).
Three changes, each fixing one identified defect:

1. **Training data is simulated** (`sim_data.py`): thousands of episodes,
   balanced across up, down and flat regimes, drawn from a generator
   calibrated on the real training years. Fixes the episode count and the
   class imbalance at once.

2. **Testing is controlled.** The simulated test set is a fresh draw from the
   same generator, reported overall and per regime, so the study can state
   what each agent does when the market rises, falls, or drifts sideways -
   not merely what it did in one historical period.

3. **Real data becomes the transfer test.** Each trained policy is also
   evaluated, unchanged, on the real SPY test months (2024-25). The gap
   between simulated and real performance measures the domain gap rather
   than being confounded with the learning question.

The state (S3), reward (wealth change), action model (quarter-slice), fee and
baselines are all unchanged from the real-data study, so the two studies
differ in exactly one thing: the data.

Run:
  ./venv/bin/python experiments/final_v2/run_sim.py
  ./venv/bin/python experiments/final_v2/run_sim.py --smoke
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import data as real_data
import sim_data
from features import resolve_rung
from harness import (INITIAL_CASH, SEEDS, Config, aggregate, provenance,
                     run_baselines, train_one, write_results)
from rollout import evaluate, mask_determined, summarize, win_rate

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "sim_results.json"

ALGOS = ("reinforce", "dqn", "ppo")


def per_regime(rows: list[dict], initial_cash: float) -> dict:
    """Split evaluation rows by the regime encoded in the episode id."""
    out = {}
    for regime in sim_data.REGIMES:
        sub = [r for r in rows if sim_data.regime_of(r["episode_id"]) == regime]
        if sub:
            out[regime] = summarize(sub, initial_cash)
    return out


def trim(summary: dict) -> dict:
    """Drop bulky per-row fields before serialising."""
    return {k: v for k, v in summary.items() if k not in ("detail",)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.timesteps, args.seeds = 3_000, [42]

    sim_meta, sim_parts = sim_data.load()
    real_meta, real_parts = real_data.load()
    print(f"sim data: train={len(sim_parts['train'])} val={len(sim_parts['val'])} "
          f"test={len(sim_parts['test'])} episodes")
    print(f"real transfer set: {len(real_parts['test'])} months "
          f"({real_meta['ids']['test'][0]}..{real_meta['ids']['test'][-1]})")

    cfg = Config("SIM_S3_wealth", resolve_rung("S3"), rung="S3")

    # Baselines on both evaluation sets, per regime on the simulated one.
    sim_base = run_baselines(cfg, sim_parts["test"])
    real_base = run_baselines(cfg, real_parts["test"])
    sim_ref_rows = sim_base["B1b_true_bah"]["_rows"]
    real_ref_rows = real_base["B1b_true_bah"]["_rows"]
    print("\nbaselines (simulated test):")
    for name, b in sim_base.items():
        s = b["summary"]
        print(f"  {name:16s} dW={s['mean_delta_w']:+8.2f}  expo={s['mean_exposure']:.2f}")
        b["per_regime"] = per_regime(b["_rows"], INITIAL_CASH)

    results: dict = {}
    for algo in args.algos:
        print(f"\n{algo}: {args.timesteps:,} steps x seeds {args.seeds}")
        per_seed = []
        for seed in args.seeds:
            t0 = time.time()
            policy, trace = train_one(algo, cfg, sim_parts["train"],
                                      seed=seed, total_timesteps=args.timesteps)
            eval_factory = cfg.factory(risk_lambda=0.0)

            sim_rows = evaluate(eval_factory, sim_parts["test"], policy)
            sim_summary = summarize(sim_rows, INITIAL_CASH)
            sim_summary["win_rate_vs_ref"] = win_rate(sim_rows, sim_ref_rows)
            conditioning = mask_determined(eval_factory, sim_parts["test"], policy)
            sim_summary["state_dependent"] = conditioning["state_dependent"]

            real_rows = evaluate(eval_factory, real_parts["test"], policy)
            real_summary = summarize(real_rows, INITIAL_CASH)
            real_summary["win_rate_vs_ref"] = win_rate(real_rows, real_ref_rows)

            per_seed.append({
                "seed": seed,
                "summary": sim_summary,           # aggregate() reads this
                "per_regime": per_regime(sim_rows, INITIAL_CASH),
                "real_transfer": real_summary,
                "conditioning": trim(conditioning),
                "train": {k: v for k, v in trace.items()
                          if k in ("wall_seconds", "note")},
                "real_per_month": [
                    {"id": r["episode_id"], "delta_w": r["delta_w"],
                     "n_trades": r["n_trades"]} for r in real_rows],
            })
            reg = per_seed[-1]["per_regime"]
            reg_str = "  ".join(
                f"{k}={reg[k]['mean_delta_w']:+7.2f}" for k in sim_data.REGIMES
                if k in reg)
            print(f"  seed {seed}: sim dW={sim_summary['mean_delta_w']:+8.2f} "
                  f"[{reg_str}]  real dW={real_summary['mean_delta_w']:+8.2f}  "
                  f"cond={'state' if conditioning['state_dependent'] else 'MASK-ONLY'}  "
                  f"({time.time() - t0:.0f}s)")

        entry = {"per_seed": per_seed, "across_seeds": aggregate(per_seed)}
        # Across-seed per-regime and transfer means.
        for regime in sim_data.REGIMES:
            vals = [s["per_regime"][regime]["mean_delta_w"] for s in per_seed
                    if regime in s["per_regime"]]
            entry["across_seeds"][f"regime_{regime}_delta_w"] = {
                "mean": float(np.mean(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals))}
        transfer = [s["real_transfer"]["mean_delta_w"] for s in per_seed]
        entry["across_seeds"]["real_transfer_delta_w"] = {
            "mean": float(np.mean(transfer)), "min": float(np.min(transfer)),
            "max": float(np.max(transfer))}
        results[algo] = entry

    payload = {
        "study": "simulated-data study: controlled regimes, real-data transfer",
        "config": cfg.as_dict(),
        "network": "pyramid MLP, hidden (64, 32)",
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "sim_meta": {k: sim_meta[k] for k in
                     ("generator", "episode_def", "counts_per_regime",
                      "calibration", "counts")},
        "real_test_months": real_meta["ids"]["test"],
        "baselines_sim": sim_base,
        "baselines_real": real_base,
        "provenance": provenance(),
        "results": results,
    }
    write_results(RESULTS.with_name(args.out) if args.out else RESULTS, payload)


if __name__ == "__main__":
    main()
