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
   evaluated, unchanged, on every real SPY month outside the generator's
   calibration window (`real_transfer.py`): 2006-2017 and 2023-2025, 180
   months including the 2008 crisis. The gap between simulated and real
   performance measures the domain gap rather than being confounded with
   the learning question, and the early window supplies the real declines
   that the 2024-25 test years lacked.

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

import real_transfer
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


def per_window(rows: list[dict], initial_cash: float) -> dict:
    """Split real-transfer evaluation rows by transfer window (early/late)."""
    out = {}
    for window in real_transfer.WINDOWS:
        sub = [r for r in rows
               if real_transfer.window_of(r["episode_id"]) == window]
        if sub:
            out[window] = summarize(sub, initial_cash)
    return out


def trim(summary: dict) -> dict:
    """Drop bulky per-row fields before serialising."""
    return {k: v for k, v in summary.items() if k not in ("detail",)}


def train_and_eval_seed(algo: str, cfg: Config, train_eps: list[dict],
                        sim_test: list[dict], transfer_eps: list[dict],
                        sim_ref_rows, real_ref_rows, *,
                        seed: int, timesteps: int) -> dict:
    """Train one seed on simulated data, evaluate on sim test and real transfer.

    Shared by `run_sim.py` (the main comparison) and `run_sim_suite.py`
    (feature and reward cells), so every cell reports the same quantities.
    Evaluation always scores the plain wealth change (`risk_lambda=0`).
    """
    policy, trace = train_one(algo, cfg, train_eps,
                              seed=seed, total_timesteps=timesteps)
    eval_factory = cfg.factory(risk_lambda=0.0)

    sim_rows = evaluate(eval_factory, sim_test, policy)
    sim_summary = summarize(sim_rows, INITIAL_CASH)
    sim_summary["win_rate_vs_ref"] = win_rate(sim_rows, sim_ref_rows)
    conditioning = mask_determined(eval_factory, sim_test, policy)
    sim_summary["state_dependent"] = conditioning["state_dependent"]

    real_rows = evaluate(eval_factory, transfer_eps, policy)
    real_summary = summarize(real_rows, INITIAL_CASH)
    real_summary["win_rate_vs_ref"] = win_rate(real_rows, real_ref_rows)

    return {
        "seed": seed,
        "summary": sim_summary,           # aggregate() reads this
        "per_regime": per_regime(sim_rows, INITIAL_CASH),
        "real_transfer": real_summary,
        "real_transfer_windows": per_window(real_rows, INITIAL_CASH),
        "conditioning": trim(conditioning),
        "train": {k: v for k, v in trace.items()
                  if k in ("wall_seconds", "note")},
        "real_per_month": [
            {"id": r["episode_id"], "delta_w": r["delta_w"],
             "n_trades": r["n_trades"], "mean_exposure": r["mean_exposure"],
             "intra_dd": r["intra_dd"]} for r in real_rows],
    }


def across_seed_extras(entry: dict) -> None:
    """Attach per-regime and transfer across-seed means to an algo entry."""
    per_seed = entry["per_seed"]
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
    for window in real_transfer.WINDOWS:
        vals = [s["real_transfer_windows"][window]["mean_delta_w"]
                for s in per_seed if window in s["real_transfer_windows"]]
        entry["across_seeds"][f"transfer_{window}_delta_w"] = {
            "mean": float(np.mean(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals))}


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
    transfer_meta, transfer_parts = real_transfer.load()
    transfer_eps = real_transfer.all_episodes(transfer_parts)
    print(f"sim data: train={len(sim_parts['train'])} val={len(sim_parts['val'])} "
          f"test={len(sim_parts['test'])} episodes")
    print(f"real transfer set: {len(transfer_eps)} months "
          f"(early {transfer_meta['ids']['early'][0]}..{transfer_meta['ids']['early'][-1]}, "
          f"late {transfer_meta['ids']['late'][0]}..{transfer_meta['ids']['late'][-1]})")

    cfg = Config("SIM_S3_wealth", resolve_rung("S3"), rung="S3")

    # Baselines on both evaluation sets: per regime on the simulated one,
    # per window on the real one.
    sim_base = run_baselines(cfg, sim_parts["test"])
    real_base = run_baselines(cfg, transfer_eps)
    sim_ref_rows = sim_base["B1b_true_bah"]["_rows"]
    real_ref_rows = real_base["B1b_true_bah"]["_rows"]
    print("\nbaselines (simulated test | real transfer):")
    for name in sim_base:
        s, r = sim_base[name]["summary"], real_base[name]["summary"]
        print(f"  {name:16s} sim dW={s['mean_delta_w']:+8.2f}  "
              f"real dW={r['mean_delta_w']:+8.2f}")
        sim_base[name]["per_regime"] = per_regime(sim_base[name]["_rows"],
                                                  INITIAL_CASH)
        real_base[name]["per_window"] = per_window(real_base[name]["_rows"],
                                                   INITIAL_CASH)

    results: dict = {}
    for algo in args.algos:
        print(f"\n{algo}: {args.timesteps:,} steps x seeds {args.seeds}")
        per_seed = []
        for seed in args.seeds:
            t0 = time.time()
            row = train_and_eval_seed(
                algo, cfg, sim_parts["train"], sim_parts["test"], transfer_eps,
                sim_ref_rows, real_ref_rows,
                seed=seed, timesteps=args.timesteps)
            per_seed.append(row)
            reg = row["per_regime"]
            reg_str = "  ".join(
                f"{k}={reg[k]['mean_delta_w']:+7.2f}" for k in sim_data.REGIMES
                if k in reg)
            win = row["real_transfer_windows"]
            print(f"  seed {seed}: sim dW={row['summary']['mean_delta_w']:+8.2f} "
                  f"[{reg_str}]  real dW={row['real_transfer']['mean_delta_w']:+8.2f} "
                  f"(early {win['early']['mean_delta_w']:+7.2f}, "
                  f"late {win['late']['mean_delta_w']:+7.2f})  "
                  f"cond={'state' if row['summary']['state_dependent'] else 'MASK-ONLY'}  "
                  f"({time.time() - t0:.0f}s)")

        entry = {"per_seed": per_seed, "across_seeds": aggregate(per_seed)}
        across_seed_extras(entry)
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
        "transfer_meta": {k: transfer_meta[k] for k in
                          ("windows", "counts", "excluded_calibration_window")},
        "transfer_months": {w: transfer_meta["ids"][w]
                            for w in real_transfer.WINDOWS},
        "baselines_sim": sim_base,
        "baselines_real": real_base,
        "provenance": provenance(),
        "results": results,
    }
    write_results(RESULTS.with_name(args.out) if args.out else RESULTS, payload)


if __name__ == "__main__":
    main()
