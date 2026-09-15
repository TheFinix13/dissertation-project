#!/usr/bin/env python3
"""The pilot's full experiment suite, rerun on the simulated data.

STUDY: main (simulated redo) — writes results/sim_{slice,lambda,cells,fees}.json.
In the dissertation.

`run_sim.py` answers the algorithm question under the fixed setup (pyramid
network, 3,000 balanced simulated episodes). The pilot answered four further
questions - trade sizing, state features, the risk-aware reward, transaction
costs - and those answers rested on the data-starved study. This script
re-establishes each one under the corrected setup, mirroring the pilot's
scripts stage for stage so every design rule carries over unchanged:

  slice    `calibrate_slice.py` on the simulated validation split: the
           exposure-ceiling arithmetic, no training involved.
  lambda   `run_final.py` stage 1 on the simulated validation split: the
           risk-penalty sweep with the same floors (retain 50% of the
           unpenalised return and 50% of its exposure), DQN as selector.
  cells    `run_final.py` stage 2 for the state and reward axes: S4 x wealth
           and S4 x risk-aware on the simulated test split (per regime),
           each policy also dropped onto the extended real transfer set.
  fees     `run_fees.py` on the simulated validation split: the same fee
           grid, plus the behavioural-response check.

Sweeps (lambda, fees) rank configurations rather than report final numbers,
so they run at half the main training budget, mirroring the pilot's use of a
reduced budget for its validation sweep.

Run:
  ./venv/bin/python experiments/final_v2/run_sim_suite.py --stage all
  ./venv/bin/python experiments/final_v2/run_sim_suite.py --stage lambda
  ./venv/bin/python experiments/final_v2/run_sim_suite.py --stage all --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import real_transfer
import sim_data
from baselines import policy_buy_when_legal
from env import make_env_factory
from features import resolve_rung
from harness import (FEE, INITIAL_CASH, SEEDS, Config, aggregate,
                     evaluate_agent, provenance, run_baselines, write_results)
from rollout import evaluate, summarize
from run_sim import across_seed_extras, per_regime, per_window, train_and_eval_seed

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

ALGOS = ("reinforce", "dqn", "ppo")

# ------------------------------ pilot constants, unchanged ------------------

SLICE_GRID = (0.05, 0.10, 0.25, 0.50)
EXPOSURE_TARGET = 0.90

LAMBDA_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
LAMBDA_SELECTOR = "dqn"
RETURN_FLOOR_FRAC = 0.50
EXPOSURE_FLOOR_FRAC = 0.50

FEE_GRID_BPS = (0.0, 5.0, 10.0, 25.0, 50.0)


# ------------------------------------------------------------------- slice

def stage_slice(sim_parts, args) -> None:
    """Exposure-ceiling arithmetic on the simulated validation split."""
    val = sim_parts["val"]
    mean_len = sum(len(e["prices"]) for e in val) / len(val)

    bah = summarize(
        evaluate(make_env_factory(resolve_rung("S3"), slice_frac=1.0),
                 val, policy_buy_when_legal), INITIAL_CASH)
    threshold = EXPOSURE_TARGET * bah["mean_exposure"]

    rows = []
    print(f"\nslice ceilings on simulated validation ({len(val)} episodes, "
          f"mean length {mean_len:.1f} bars)")
    print(f"  buy-and-hold exposure {bah['mean_exposure']:.3f}; "
          f"ceiling must reach {threshold:.3f}")
    for frac in SLICE_GRID:
        factory = make_env_factory(resolve_rung("S3"), slice_frac=frac)
        s = summarize(evaluate(factory, val, policy_buy_when_legal),
                      INITIAL_CASH)
        row = {"slice_frac": frac, "steps_to_full": round(1.0 / frac),
               "mean_exposure": s["mean_exposure"],
               "mean_delta_w": s["mean_delta_w"],
               "reaches_target": bool(s["mean_exposure"] >= threshold)}
        rows.append(row)
        print(f"  slice {frac:4.2f}: exposure {s['mean_exposure']:.3f}  "
              f"{'yes' if row['reaches_target'] else 'no'}")

    reaching = [r["slice_frac"] for r in rows if r["reaches_target"]]
    chosen = min(reaching) if reaching else max(SLICE_GRID)
    print(f"  selected slice = {chosen:.2f} C_0")

    write_results(RESULTS_DIR / "sim_slice.json", {
        "study": "trade-slice ceiling on the simulated validation split",
        "rule": (f"smallest slice whose attainable exposure ceiling reaches "
                 f"{EXPOSURE_TARGET:.0%} of buy-and-hold's exposure"),
        "split_used": "sim val",
        "mean_episode_length": mean_len,
        "buy_and_hold": {"mean_exposure": bah["mean_exposure"],
                         "mean_delta_w": bah["mean_delta_w"]},
        "exposure_threshold": threshold,
        "grid": rows,
        "chosen_slice_frac": chosen,
        "provenance": provenance(),
    })


# ------------------------------------------------------------------ lambda

def stage_lambda(sim_parts, args) -> dict:
    """Risk-penalty selection on the simulated validation split (pilot rule)."""
    train_eps, val_eps = sim_parts["train"], sim_parts["val"]
    features = resolve_rung("S4")

    ref_cfg = Config("SIM_S4_lambda_ref", features, rung="S4")
    reference_rows = run_baselines(ref_cfg, val_eps)["B1b_true_bah"]["_rows"]

    trials = {}
    print(f"\nlambda selection on simulated validation ({len(val_eps)} episodes, "
          f"{LAMBDA_SELECTOR}, {args.sweep_timesteps:,} steps)")
    for lam in (0.0, *args.lambda_grid):
        cfg = Config(f"SIM_S4_lam{lam}", features, rung="S4", risk_lambda=lam)
        print(f"  lambda = {lam}")
        res = evaluate_agent(LAMBDA_SELECTOR, cfg, train_eps, val_eps,
                             reference_rows, seeds=args.seeds,
                             total_timesteps=args.sweep_timesteps)
        trials[lam] = res["across_seeds"]

    unpenalised = trials[0.0]
    base_dw = unpenalised["mean_delta_w"]["mean"]
    base_expo = unpenalised["mean_exposure"]["mean"]
    dw_floor = RETURN_FLOOR_FRAC * base_dw
    expo_floor = EXPOSURE_FLOOR_FRAC * base_expo
    print(f"  unpenalised: dW={base_dw:+.2f}, exposure={base_expo:.3f}")
    print(f"  floors: dW >= {dw_floor:+.2f} and exposure >= {expo_floor:.3f}")

    admissible = []
    for lam in args.lambda_grid:
        t = trials[lam]
        ok = (t["mean_delta_w"]["mean"] >= dw_floor
              and t["mean_exposure"]["mean"] >= expo_floor)
        if ok:
            admissible.append(lam)
        print(f"    lambda={lam:<5g} dW={t['mean_delta_w']['mean']:+8.2f}  "
              f"expo={t['mean_exposure']['mean']:.3f}  "
              f"ddI={t['mdd_intra_mean']['mean']:.4f}  "
              f"{'pass' if ok else 'fail'}")

    chosen = max(admissible) if admissible else min(args.lambda_grid)
    print(f"  selected lambda = {chosen}"
          + ("" if admissible else " (fallback: smallest on the grid)"))

    payload = {
        "study": "risk-penalty selection on the simulated validation split",
        "rule": (f"largest lambda retaining >= {RETURN_FLOOR_FRAC:.0%} of the "
                 f"unpenalised mean wealth change and >= "
                 f"{EXPOSURE_FLOOR_FRAC:.0%} of its mean exposure"),
        "selector_algo": LAMBDA_SELECTOR,
        "grid": list(args.lambda_grid),
        "sweep_timesteps": args.sweep_timesteps,
        "seeds": list(args.seeds),
        "unpenalised": {"mean_delta_w": base_dw, "mean_exposure": base_expo,
                        "mdd_intra_mean": unpenalised["mdd_intra_mean"]["mean"]},
        "return_floor": dw_floor,
        "exposure_floor": expo_floor,
        "cleared_floors": admissible,
        "chosen": chosen,
        "fell_back": not admissible,
        "trials": {str(k): v for k, v in trials.items()},
        "provenance": provenance(),
    }
    write_results(RESULTS_DIR / "sim_lambda.json", payload)
    return payload


# ------------------------------------------------------------------- cells

def stage_cells(sim_parts, transfer_eps, args) -> None:
    """State and reward axes on the simulated test split + real transfer."""
    lam = args.lambda_fixed
    if lam is None:
        lam_file = RESULTS_DIR / "sim_lambda.json"
        if not lam_file.exists():
            raise SystemExit("run --stage lambda first, or pass --lambda-fixed")
        lam = float(json.loads(lam_file.read_text())["chosen"])

    s4 = resolve_rung("S4")
    cells = [
        Config("SIM_S4_wealth", s4, rung="S4"),
        Config("SIM_S4_risk", s4, rung="S4", risk_lambda=lam),
    ]

    out = {}
    for cfg in cells:
        print(f"\ncell {cfg.name} (lambda={cfg.risk_lambda}, "
              f"{args.timesteps:,} steps, seeds {list(args.seeds)})")
        sim_base = run_baselines(cfg, sim_parts["test"])
        real_base = run_baselines(cfg, transfer_eps)
        sim_ref = sim_base["B1b_true_bah"]["_rows"]
        real_ref = real_base["B1b_true_bah"]["_rows"]
        for name in sim_base:
            sim_base[name]["per_regime"] = per_regime(
                sim_base[name]["_rows"], INITIAL_CASH)
            real_base[name]["per_window"] = per_window(
                real_base[name]["_rows"], INITIAL_CASH)

        entry = {"config": cfg.as_dict(), "baselines_sim": sim_base,
                 "baselines_real": real_base, "algos": {}}
        for algo in args.algos:
            print(f"  {algo}")
            per_seed = []
            for seed in args.seeds:
                t0 = time.time()
                row = train_and_eval_seed(
                    algo, cfg, sim_parts["train"], sim_parts["test"],
                    transfer_eps, sim_ref, real_ref,
                    seed=seed, timesteps=args.timesteps)
                per_seed.append(row)
                print(f"    seed {seed}: sim dW="
                      f"{row['summary']['mean_delta_w']:+8.2f}  real dW="
                      f"{row['real_transfer']['mean_delta_w']:+8.2f}  "
                      f"cond={'state' if row['summary']['state_dependent'] else 'MASK-ONLY'}  "
                      f"({time.time() - t0:.0f}s)")
            algo_entry = {"per_seed": per_seed,
                          "across_seeds": aggregate(per_seed)}
            across_seed_extras(algo_entry)
            entry["algos"][algo] = algo_entry
        out[cfg.name] = entry

    write_results(RESULTS_DIR / "sim_cells.json", {
        "study": "state and reward axes on the simulated data",
        "cells": {
            "SIM_S4_wealth": "adds drawdown to the state, reward unchanged",
            "SIM_S4_risk": ("adds drawdown to the state and to the reward, "
                            f"lambda = {lam} (selected on sim validation)"),
        },
        "lambda": lam,
        "timesteps": args.timesteps,
        "seeds": list(args.seeds),
        "provenance": provenance(),
        "results": out,
    })


# -------------------------------------------------------------------- fees

def stage_fees(sim_parts, args) -> None:
    """Fee sensitivity on the simulated validation split (pilot grid)."""
    train_eps, val_eps = sim_parts["train"], sim_parts["val"]
    features = resolve_rung("S3")

    print(f"\nfee sensitivity on simulated validation ({len(val_eps)} episodes, "
          f"{args.sweep_timesteps:,} steps, seeds {list(args.seeds)})")
    payload = {
        "study": "fee sensitivity on the simulated data at the frozen state",
        "eval_split": "sim val",
        "sweep_timesteps": args.sweep_timesteps,
        "seeds": list(args.seeds),
        "fees_bps": list(args.fees_bps),
        "provenance": provenance(),
        "levels": {},
    }

    for bps in args.fees_bps:
        fee = bps / 10_000.0
        cfg = Config(f"SIM_fee{bps:g}bps", features, rung="S3", fee=fee)
        print(f"\n  fee = {bps:g} bps")
        base = run_baselines(cfg, val_eps)
        reference_rows = base["B1b_true_bah"]["_rows"]
        print(f"      B1b_true_bah     dW="
              f"{base['B1b_true_bah']['summary']['mean_delta_w']:+8.2f}")
        entry = {"config": cfg.as_dict(), "baselines": base, "algos": {}}
        for algo in args.algos:
            print(f"    {algo}")
            entry["algos"][algo] = evaluate_agent(
                algo, cfg, train_eps, val_eps, reference_rows,
                seeds=args.seeds, total_timesteps=args.sweep_timesteps)
        payload["levels"][f"{bps:g}"] = entry

    print("\n  behavioural response to cost")
    response = {}
    for algo in args.algos:
        trades = [payload["levels"][f"{b:g}"]["algos"][algo]["across_seeds"]
                  ["mean_trades"]["mean"] for b in args.fees_bps]
        dws = [payload["levels"][f"{b:g}"]["algos"][algo]["across_seeds"]
               ["mean_delta_w"]["mean"] for b in args.fees_bps]
        spreads = [payload["levels"][f"{b:g}"]["algos"][algo]["across_seeds"]
                   ["mean_trades"]["spread"] for b in args.fees_bps]
        change = trades[0] - trades[-1]
        noise = max(spreads)
        response[algo] = {
            "fees_bps": list(args.fees_bps),
            "mean_trades": trades,
            "mean_delta_w": dws,
            "trades_change_zero_to_max_fee": change,
            "max_seed_spread_on_trades": noise,
            "responds_beyond_noise": bool(abs(change) > noise),
        }
        print(f"      {algo:10s} trades "
              f"{' -> '.join(f'{t:.1f}' for t in trades)}  "
              f"(change {change:+.1f}, noise {noise:.1f}, "
              f"{'responds' if abs(change) > noise else 'FLAT'})")
    payload["behavioural_response"] = response

    write_results(RESULTS_DIR / "sim_fees.json", payload)


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("slice", "lambda", "cells", "fees", "all"),
                    default="all")
    ap.add_argument("--timesteps", type=int, default=300_000,
                    help="budget for the test-reported cells")
    ap.add_argument("--sweep-timesteps", type=int, default=150_000,
                    help="budget for validation sweeps, which only rank")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    ap.add_argument("--lambda-grid", type=float, nargs="+",
                    default=list(LAMBDA_GRID))
    ap.add_argument("--lambda-fixed", type=float, default=None)
    ap.add_argument("--fees-bps", type=float, nargs="+",
                    default=list(FEE_GRID_BPS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.timesteps, args.sweep_timesteps = 3_000, 2_000
        args.seeds = [42]
        args.lambda_grid = [0.1]
        args.fees_bps = [0.0, 50.0]

    sim_meta, sim_parts = sim_data.load()
    transfer_meta, transfer_parts = real_transfer.load()
    transfer_eps = real_transfer.all_episodes(transfer_parts)
    print(f"sim data: train={len(sim_parts['train'])} "
          f"val={len(sim_parts['val'])} test={len(sim_parts['test'])}; "
          f"real transfer {len(transfer_eps)} months")

    stages = (("slice", "lambda", "cells", "fees") if args.stage == "all"
              else (args.stage,))
    for stage in stages:
        t0 = time.time()
        if stage == "slice":
            stage_slice(sim_parts, args)
        elif stage == "lambda":
            stage_lambda(sim_parts, args)
        elif stage == "cells":
            stage_cells(sim_parts, transfer_eps, args)
        elif stage == "fees":
            stage_fees(sim_parts, args)
        print(f"\nstage {stage} done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
