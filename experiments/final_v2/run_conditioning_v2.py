#!/usr/bin/env python3
"""Conditioning diagnostic v2: all three algorithms at both trade slices.

The original conditioning study (`run_conditioning.py`) predates the slice
calibration: it ran at the 0.10 C0 slice, covered only REINFORCE and DQN, and
crossed them with a reward-scaling axis that the dissertation does not use.
This rerun reports the diagnostic the chapter actually needs:

  * the frozen nine-feature state and the wealth-change reward, exactly as
    defined in Chapter 3 (no reward-scale axis);
  * both action models: the original 0.10 C0 slice and the calibrated
    0.25 C0 slice adopted in Section 5.3;
  * all three learners: REINFORCE, DQN and MaskablePPO;
  * six freshly trained seeds per cell; nothing is loaded from disk.

Baselines are re-evaluated inside each slice because the always-buy ceiling
and the slice schedule depend on the slice. The SHA-256 of the episode file
is recorded so the run is provably on the canonical dataset documented in
Chapter 4 (re-downloading would back-adjust prices and silently desynchronise
this run from every other result in the chapter).

Run:
  ./venv/bin/python experiments/final_v2/run_conditioning_v2.py
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from features import resolve_rung
from harness import (Config, evaluate_agent, load_data, provenance,
                     run_baselines, write_results)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "conditioning_v2_results.json"

SEEDS = (42, 43, 44, 45, 46, 47)
SLICES = (0.10, 0.25)
ALGOS = ("reinforce", "dqn", "ppo")


def dataset_fingerprint() -> dict:
    out = {}
    for name in ("spy_episodes.npz", "spy_meta.json"):
        p = HERE / "data" / name
        out[name] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=80_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    ap.add_argument("--slices", type=float, nargs="+", default=list(SLICES))
    args = ap.parse_args()

    meta, parts = load_data()
    train_eps, val_eps = parts["train"], parts["val"]

    payload = {
        "study": ("policy conditioning at the frozen nine-feature state, "
                  "all algorithms, both trade slices"),
        "eval_split": "val",
        "reward": "wealth change (Chapter 3), no scaling axis",
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "slices": args.slices,
        "provenance": provenance(),
        "dataset": dataset_fingerprint(),
        "cells": {},
        "baselines": {},
    }

    for slice_frac in args.slices:
        tag = f"slice{slice_frac:g}"
        cfg = Config(f"S3_{tag}", resolve_rung("S3"), rung="S3",
                     slice_frac=slice_frac)
        base = run_baselines(cfg, val_eps)
        reference_rows = base["B1b_true_bah"]["_rows"]
        payload["baselines"][tag] = base

        print(f"\n=== slice {slice_frac:g} C0 | validation "
              f"({len(val_eps)} months), {args.timesteps:,} steps, "
              f"seeds {args.seeds} ===")
        for name, b in base.items():
            s = b["summary"]
            print(f"  {name:16s} dW={s['mean_delta_w']:+8.2f}  "
                  f"expo={s['mean_exposure']:.3f}")

        for algo in args.algos:
            key = f"{algo}__{tag}"
            print(f"\n  {algo} @ {slice_frac:g} C0")
            res = evaluate_agent(algo, cfg, train_eps, val_eps,
                                 reference_rows, seeds=args.seeds,
                                 total_timesteps=args.timesteps,
                                 reward_scale=1.0)
            agg = res["across_seeds"]
            print(f"      mean={agg['mean_delta_w']['mean']:+.2f}  "
                  f"spread={agg['mean_delta_w']['spread']:.2f}  "
                  f"state-dependent {agg['n_state_dependent']}"
                  f"/{agg['n_seeds']}")
            res["config"] = cfg.as_dict()
            payload["cells"][key] = res

    print("\n  summary")
    width = max(len(k) for k in payload["cells"])
    print(f"      {'cell':<{width}}  {'mean dW':>9}  {'spread':>8}  "
          f"{'expo':>6}  state-dep")
    for key, res in payload["cells"].items():
        agg = res["across_seeds"]
        print(f"      {key:<{width}}  {agg['mean_delta_w']['mean']:+9.2f}  "
              f"{agg['mean_delta_w']['spread']:8.2f}  "
              f"{agg['mean_exposure']['mean']:6.3f}  "
              f"{agg['n_state_dependent']}/{agg['n_seeds']}")

    write_results(RESULTS, payload)
    print("CONDITIONING_V2_DONE")


if __name__ == "__main__":
    main()
