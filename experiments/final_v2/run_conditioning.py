#!/usr/bin/env python3
"""Diagnostic: why is the seed-to-seed spread as large as the signal?

The first ladder run produced a spread across seeds that exceeded every
difference between adjacent rungs, so the stopping rule could not discriminate
between representations. That is a statement about the measurement, not about
the states, and it has to be understood before the ladder result means anything.

Two hypotheses, both testable on the validation split.

**H1: the learners collapse onto policies that ignore the observation.** Several
runs returned exactly the wealth change of a reference strategy, which is what a
policy that consults only the feasibility mask would produce. If that is what is
happening, no state can look better than any other, because the state is not
being read. `mask_determined` tests this directly.

**H2: Deep Q-learning is badly conditioned in dollar units.** Q estimates a sum
of dollar rewards across a month, so targets reach the low hundreds while the
network is initialised to output values near zero. A squared-error loss on such
targets produces very large early gradients. Dividing the reward by the initial
capital rescales targets to the order of 0.02 without changing the MDP or any
reported metric.

The experiment crosses the two learners with the two reward scales at the frozen
state, on validation, with a wider seed set than the main study, because the
question here is about the variance of the estimator rather than about the
policies themselves.

Run:
  ./venv/bin/python experiments/final_v2/run_conditioning.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from features import resolve_rung
from harness import (Config, evaluate_agent, load_data, provenance,
                     run_baselines, write_results)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "conditioning_results.json"

SEEDS = (42, 43, 44, 45, 46, 47)
SCALES = {"dollars": 1.0, "fraction_of_capital": 1.0 / 10_000.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=80_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--algos", nargs="+", default=["reinforce", "dqn"])
    args = ap.parse_args()

    meta, parts = load_data()
    train_eps, val_eps = parts["train"], parts["val"]

    cfg = Config("S3", resolve_rung("S3"), rung="S3")
    base = run_baselines(cfg, val_eps)
    reference_rows = base["B1b_true_bah"]["_rows"]

    print(f"\nconditioning diagnostic on validation ({len(val_eps)} months), "
          f"{args.timesteps:,} steps, seeds {args.seeds}")
    for name, b in base.items():
        s = b["summary"]
        print(f"  {name:16s} dW={s['mean_delta_w']:+8.2f}  "
              f"expo={s['mean_exposure']:.2f}")

    payload = {
        "study": "seed variance and policy conditioning at the frozen state",
        "hypotheses": {
            "H1": "learners collapse onto mask-determined policies",
            "H2": "Deep Q-learning is badly conditioned in dollar units",
        },
        "eval_split": "val",
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "scales": SCALES,
        "provenance": provenance(),
        "baselines": base,
        "cells": {},
    }

    for algo in args.algos:
        for scale_name, scale in SCALES.items():
            key = f"{algo}__{scale_name}"
            print(f"\n  {algo}, reward in {scale_name}")
            res = evaluate_agent(algo, cfg, train_eps, val_eps, reference_rows,
                                 seeds=args.seeds,
                                 total_timesteps=args.timesteps,
                                 reward_scale=scale)
            agg = res["across_seeds"]
            dw = [s["summary"]["mean_delta_w"] for s in res["per_seed"]]
            print(f"      spread={agg['mean_delta_w']['spread']:.2f}  "
                  f"mean={agg['mean_delta_w']['mean']:+.2f}  "
                  f"state-dependent in {agg['n_state_dependent']}"
                  f"/{agg['n_seeds']} seeds")
            print(f"      per-seed dW: "
                  f"{', '.join(f'{v:+.1f}' for v in sorted(dw))}")
            payload["cells"][key] = res

    print("\n  summary")
    rows = []
    for key, res in payload["cells"].items():
        agg = res["across_seeds"]
        rows.append((key, agg["mean_delta_w"]["mean"],
                     agg["mean_delta_w"]["spread"],
                     agg["n_state_dependent"], agg["n_seeds"]))
    width = max(len(r[0]) for r in rows)
    print(f"      {'cell':<{width}}  {'mean dW':>9}  {'spread':>8}  state-dep")
    for key, mean, spread, nsd, n in rows:
        print(f"      {key:<{width}}  {mean:+9.2f}  {spread:8.2f}  {nsd}/{n}")

    # The verdict is recorded rather than left to the reader, because the main
    # study's configuration is chosen from it.
    verdicts = {}
    for algo in args.algos:
        d = payload["cells"].get(f"{algo}__dollars", {}).get("across_seeds")
        f = payload["cells"].get(f"{algo}__fraction_of_capital", {}).get("across_seeds")
        if not d or not f:
            continue
        verdicts[algo] = {
            "spread_dollars": d["mean_delta_w"]["spread"],
            "spread_fraction": f["mean_delta_w"]["spread"],
            "state_dependent_dollars": f"{d['n_state_dependent']}/{d['n_seeds']}",
            "state_dependent_fraction": f"{f['n_state_dependent']}/{f['n_seeds']}",
            "scaling_reduces_spread": bool(
                f["mean_delta_w"]["spread"] < d["mean_delta_w"]["spread"]),
            "scaling_improves_conditioning": bool(
                f["n_state_dependent"] > d["n_state_dependent"]),
        }
    payload["verdict"] = verdicts
    print("\n  verdict")
    for algo, v in verdicts.items():
        print(f"      {algo}: spread {v['spread_dollars']:.1f} -> "
              f"{v['spread_fraction']:.1f}; state-dependent "
              f"{v['state_dependent_dollars']} -> {v['state_dependent_fraction']}")

    write_results(RESULTS, payload)


if __name__ == "__main__":
    main()
