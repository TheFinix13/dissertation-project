#!/usr/bin/env python3
"""Final experiments: three algorithms, three admissible cells, one action axis.

Structure
---------
Stage 1 (validation) selects the one free hyper-parameter of the risk-aware
reward, the penalty weight lambda. The criterion is declared before the sweep:
take the **largest** lambda whose mean wealth change on validation stays within
one across-seed spread of the unpenalised agent. Selecting by wealth change
alone would always return lambda = 0 and make the risk experiment vacuous;
selecting by drawdown alone would return the largest lambda on offer, since a
policy that never trades has no drawdown. The rule as stated asks for the
strongest risk penalty that does not cost return, which is the trade-off the
experiment is about.

Stage 2 (test, evaluated once) reports:

  * three admissible state-reward cells,
        S3 x wealth change   -- the baseline
        S4 x wealth change   -- does risk history help on its own?
        S4 x risk-aware      -- does penalising risk change behaviour?
    The fourth pairing, S3 x risk-aware, is inadmissible: the reward depends on
    peak wealth, which S3 does not carry. It is excluded by construction, and
    `assert_inadmissible_cell_rejected` records that the environment refuses to
    build it rather than leaving the claim to prose.

  * an action-representation comparison at the frozen state, fixed monetary
    slice against whole-share, so that state, reward and action are each varied
    one at a time.

All learned results use seeds {42, 43, 44}. Reference strategies are recomputed
inside each configuration, because a baseline's numbers depend on the fee and
the action model.

Run:
  ./venv/bin/python experiments/final_v2/run_final.py
  ./venv/bin/python experiments/final_v2/run_final.py --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env import TradingEnv
from features import resolve_rung
from harness import (SEEDS, Config, evaluate_agent, load_data, provenance,
                     run_baselines, write_results)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "final_results.json"

ALGOS = ("reinforce", "dqn", "ppo")

#: Candidate penalty weights. The grid is centred low on purpose. Because the
#: penalty is `lambda * C_0 * d(drawdown)`, a one-percent deepening of drawdown
#: costs `lambda * 100` dollars, while a typical day's wealth change at realistic
#: exposure is a few tens of dollars. Values of order one therefore let the
#: penalty dominate the return term outright, and the agent's best reply is to
#: stop trading. The grid spans two orders of magnitude so the selection rule can
#: locate the point where the penalty starts to bind rather than assuming it.
LAMBDA_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)

#: The algorithm used to select lambda. One learner is enough for a
#: hyper-parameter choice, and using the from-scratch value-based method keeps
#: the selection independent of the third-party implementation.
LAMBDA_SELECTOR = "dqn"


def assert_inadmissible_cell_rejected() -> dict:
    """Confirm the environment refuses the S3 x risk-aware pairing.

    This is the code counterpart of the admissibility argument in Chapter 3. If
    this check ever stopped raising, the experimental design would silently
    include a cell that is not an MDP.
    """
    try:
        TradingEnv(np.array([100.0, 101.0, 102.0]),
                   np.zeros((3, 7)), resolve_rung("S3"), risk_lambda=1.0)
    except ValueError as exc:
        return {"rejected": True, "message": str(exc)}
    return {"rejected": False,
            "message": "environment accepted an inadmissible configuration"}


def select_lambda(parts, *, seeds, timesteps, grid, log=print) -> dict:
    """Stage 1: choose lambda on validation under the pre-declared rule."""
    train_eps, val_eps = parts["train"], parts["val"]
    features = resolve_rung("S4")

    ref_cfg = Config("S4_lambda_ref", features, rung="S4")
    reference_rows = run_baselines(ref_cfg, val_eps)["B1b_true_bah"]["_rows"]

    trials = {}
    log(f"\nstage 1: selecting lambda on validation "
        f"({len(val_eps)} months, {LAMBDA_SELECTOR})")
    for lam in (0.0, *grid):
        cfg = Config(f"S4_lam{lam}", features, rung="S4", risk_lambda=lam)
        log(f"  lambda = {lam}")
        res = evaluate_agent(LAMBDA_SELECTOR, cfg, train_eps, val_eps,
                             reference_rows, seeds=seeds,
                             total_timesteps=timesteps, log=log)
        trials[lam] = res["across_seeds"]

    unpenalised = trials[0.0]
    tolerance = unpenalised["mean_delta_w"]["spread"]
    floor = unpenalised["mean_delta_w"]["mean"] - tolerance

    admissible = [lam for lam in grid
                  if trials[lam]["mean_delta_w"]["mean"] >= floor]
    chosen = max(admissible) if admissible else min(grid)

    log(f"\n  unpenalised mean dW = {unpenalised['mean_delta_w']['mean']:+.2f}, "
        f"seed spread = {tolerance:.2f}")
    log(f"  return floor = {floor:+.2f}; lambda values clearing it: "
        f"{admissible or 'none'}")
    log(f"  selected lambda = {chosen}"
        + ("" if admissible else " (fallback: smallest on the grid)"))

    return {
        "rule": ("largest lambda whose validation mean wealth change stays "
                 "within one across-seed spread of the unpenalised agent"),
        "declared_before_running": True,
        "selector_algo": LAMBDA_SELECTOR,
        "grid": list(grid),
        "unpenalised_mean_delta_w": unpenalised["mean_delta_w"]["mean"],
        "seed_spread": tolerance,
        "return_floor": floor,
        "cleared_floor": admissible,
        "chosen": chosen,
        "fell_back": not admissible,
        "trials": {str(k): v for k, v in trials.items()},
    }


def run_cells(parts, cells, *, algos, seeds, timesteps, log=print) -> dict:
    """Stage 2: evaluate each configuration on the held-out test months."""
    train_eps, test_eps = parts["train"], parts["test"]
    out = {}
    for cfg in cells:
        log(f"\n  {cfg.name}  ({len(cfg.feature_names)}-D, "
            f"reward={cfg.as_dict()['reward']}, actions={cfg.action_model})")
        base = run_baselines(cfg, test_eps)
        reference_rows = base["B1b_true_bah"]["_rows"]
        for name, b in base.items():
            s = b["summary"]
            log(f"      {name:16s} dW={s['mean_delta_w']:+8.2f}  "
                f"trades={s['mean_trades']:5.1f}  expo={s['mean_exposure']:.2f}  "
                f"ddI={s['mdd_intra_mean']:.3f}")
        entry = {"config": cfg.as_dict(), "baselines": base, "algos": {}}
        for algo in algos:
            log(f"    {algo}")
            entry["algos"][algo] = evaluate_agent(
                algo, cfg, train_eps, test_eps, reference_rows,
                seeds=seeds, total_timesteps=timesteps, log=log)
        out[cfg.name] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=60_000)
    ap.add_argument("--lambda-timesteps", type=int, default=40_000,
                    help="budget for the validation sweep, which only ranks")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    ap.add_argument("--lambda-grid", type=float, nargs="+", default=list(LAMBDA_GRID))
    ap.add_argument("--lambda-fixed", type=float, default=None,
                    help="skip stage 1 and use this value")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.timesteps, args.lambda_timesteps = 2_000, 1_000
        args.seeds, args.algos = [42], ["reinforce", "dqn", "ppo"]
        args.lambda_grid = [1.0]

    meta, parts = load_data()

    admissibility = assert_inadmissible_cell_rejected()
    print(f"\nadmissibility guard: S3 x risk-aware rejected = "
          f"{admissibility['rejected']}")

    if args.lambda_fixed is not None:
        lam_sel = {"chosen": args.lambda_fixed, "rule": "supplied on the command line"}
    else:
        lam_sel = select_lambda(parts, seeds=args.seeds,
                               timesteps=args.lambda_timesteps,
                               grid=args.lambda_grid)
    lam = float(lam_sel["chosen"])

    s3, s4 = resolve_rung("S3"), resolve_rung("S4")
    cells = [
        Config("S3_wealth", s3, rung="S3"),
        Config("S4_wealth", s4, rung="S4"),
        Config("S4_risk", s4, rung="S4", risk_lambda=lam),
        Config("S3_wealth_share", s3, rung="S3", action_model="share"),
    ]

    print(f"\nstage 2: test split ({len(parts['test'])} months), "
          f"{args.timesteps:,} steps, seeds {args.seeds}, lambda = {lam}")
    results = run_cells(parts, cells, algos=args.algos, seeds=args.seeds,
                        timesteps=args.timesteps)

    payload = {
        "study": "final model: state, reward and action axes",
        "eval_split": "test",
        "test_months": [e["id"] for e in parts["test"]],
        "train_months": [e["id"] for e in parts["train"]],
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "admissibility_guard": admissibility,
        "lambda_selection": lam_sel,
        "cells": {
            "S3_wealth": "baseline: nine-feature state, wealth-change reward",
            "S4_wealth": "adds drawdown to the state, reward unchanged",
            "S4_risk": "adds drawdown to the state and to the reward",
            "S3_wealth_share": "baseline state and reward, whole-share actions",
            "excluded": ("S3 x risk-aware: reward depends on peak wealth, "
                         "which the state omits"),
        },
        "provenance": provenance(),
        "data_meta": {k: meta[k] for k in ("ticker", "range", "split_rule",
                                          "warmup_bars", "counts")},
        "results": results,
    }
    write_results(RESULTS, payload)


if __name__ == "__main__":
    main()
