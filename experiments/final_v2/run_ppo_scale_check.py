#!/usr/bin/env python3
"""Does the PPO collapse in the main study come from the reward's dollar scale?

STUDY: post-viva, Gate 3 diagnostic — writes results/ppo_scale_check.json.
Not in the dissertation.

A known-answer probe (notes/post-viva/2026-09-29-gates-2-4-audit.md) showed
PPO with the SB3 defaults failing a two-regime toy world that Deep Q-learning
solves. Dividing the reward by the initial capital, or removing SB3's joint
gradient clipping, fixed it. This script asks the same question on the main
study's simulated data. It trains PPO with raw-dollar and capital-scaled
rewards and scores both on the simulated validation split only. The test
split and the real months are not touched.

Run:
  ../../venv/bin/python run_ppo_scale_check.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import sim_data
from features import resolve_rung
from harness import (INITIAL_CASH, Config, provenance, run_baselines, train_one,
                     write_results)
from rollout import evaluate, mask_determined, summarize
from run_sim import per_regime

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "ppo_scale_check.json"

SCALES = {"dollars": 1.0, "fraction_of_capital": 1.0 / INITIAL_CASH}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--algo", default="ppo",
                    help="dqn or reinforce gives the reference on the same months")
    ap.add_argument("--scales", nargs="+", default=list(SCALES), choices=list(SCALES))
    args = ap.parse_args()
    out = OUT if args.algo == "ppo" else OUT.with_name(f"scale_check_{args.algo}.json")

    _, parts = sim_data.load()
    cfg = Config("SIM_S3_wealth", resolve_rung("S3"), rung="S3")
    base = run_baselines(cfg, parts["val"])
    eval_factory = cfg.factory(risk_lambda=0.0)

    cells: dict = {}
    for name in args.scales:
        scale = SCALES[name]
        rows_out = []
        for seed in args.seeds:
            t0 = time.time()
            policy, _ = train_one(args.algo, cfg, parts["train"], seed=seed,
                                  total_timesteps=args.timesteps,
                                  reward_scale=scale)
            rows = evaluate(eval_factory, parts["val"], policy)
            s = summarize(rows, INITIAL_CASH)
            cond = mask_determined(eval_factory, parts["val"], policy)
            reg = per_regime(rows, INITIAL_CASH)
            rows_out.append({
                "seed": seed,
                "mean_delta_w": s["mean_delta_w"],
                "mean_exposure": s["mean_exposure"],
                "mean_trades": s["mean_trades"],
                "state_dependent": cond["state_dependent"],
                "per_regime": {g: v["mean_delta_w"] for g, v in reg.items()},
                "wall_seconds": round(time.time() - t0, 1),
            })
            print(f"{name:20s} seed {seed}: dW={s['mean_delta_w']:+8.2f} "
                  f"expo={s['mean_exposure']:.2f} "
                  f"cond={'state' if cond['state_dependent'] else 'MASK-ONLY'} "
                  f"({rows_out[-1]['wall_seconds']}s)", flush=True)
        cells[name] = {"reward_scale": scale, "per_seed": rows_out}

    write_results(out, {
        "study": f"{args.algo} reward-scale check on the simulated validation split",
        "algo": args.algo,
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "eval_split": "sim val (100 per regime)",
        "baselines_val": {k: v["summary"]["mean_delta_w"] for k, v in base.items()},
        "cells": cells,
        "provenance": provenance(),
    })


if __name__ == "__main__":
    main()
