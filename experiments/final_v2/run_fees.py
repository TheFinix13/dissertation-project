#!/usr/bin/env python3
"""Fee sensitivity: how much does the assumed transaction cost matter?

The headline experiments assume 5 basis points per trade, which is defensible
for a retail investor trading one of the world's most liquid instruments but is
still an assumption, and a favourable one. A trading result that only survives
at an optimistic cost is not a trading result, so the fee is swept across the
range a real participant might face:

    0 bps    frictionless, the theoretical ceiling
    5 bps    the headline assumption
    10 bps   a widely quoted retail commission
    25 bps   a costly retail arrangement, or thin liquidity
    50 bps   deliberately punitive, to locate the point where trading stops
             paying at all

Two things are measured, and the second matters more. The obvious one is how
much profit each fee level removes. The more informative one is whether the
agent's *behaviour* responds: an agent that has learned that trading is costly
should trade less as the cost rises. An agent whose trade count is flat across
a tenfold increase in cost has not represented the cost at all, whatever its
returns look like.

This runs on the validation split. It is a robustness check on an assumption,
not a result about held-out performance, and it should not consume test months.

Run:
  ./venv/bin/python experiments/final_v2/run_fees.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from features import resolve_rung
from harness import (SEEDS, Config, evaluate_agent, load_data, provenance,
                     run_baselines, write_results)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "fee_results.json"

FEE_GRID_BPS = (0.0, 5.0, 10.0, 25.0, 50.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=80_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--algos", nargs="+", default=["reinforce", "dqn"])
    ap.add_argument("--fees-bps", type=float, nargs="+", default=list(FEE_GRID_BPS))
    args = ap.parse_args()

    meta, parts = load_data()
    train_eps, val_eps = parts["train"], parts["val"]
    features = resolve_rung("S3")

    print(f"\nfee sensitivity on validation ({len(val_eps)} months), "
          f"{args.timesteps:,} steps, seeds {args.seeds}")

    payload = {
        "study": "fee sensitivity at the frozen state",
        "eval_split": "val",
        "rationale": ("the 5 bps headline is an assumption; a result that only "
                      "survives at an optimistic cost is not a result"),
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "fees_bps": args.fees_bps,
        "provenance": provenance(),
        "levels": {},
    }

    for bps in args.fees_bps:
        fee = bps / 10_000.0
        cfg = Config(f"fee{bps:g}bps", features, rung="S3", fee=fee)
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
                seeds=args.seeds, total_timesteps=args.timesteps)
        payload["levels"][f"{bps:g}"] = entry

    # Does behaviour respond to cost, or only profit? A flat trade count across a
    # tenfold cost increase means the agent never represented the cost.
    print("\n  behavioural response to cost")
    response = {}
    for algo in args.algos:
        trades = [payload["levels"][f"{b:g}"]["algos"][algo]["across_seeds"]
                  ["mean_trades"]["mean"] for b in args.fees_bps]
        dws = [payload["levels"][f"{b:g}"]["algos"][algo]["across_seeds"]
               ["mean_delta_w"]["mean"] for b in args.fees_bps]
        spreads = [payload["levels"][f"{b:g}"]["algos"][algo]["across_seeds"]
                   ["mean_trades"]["spread"] for b in args.fees_bps]
        # Compare the change in trade count with the seed noise on it, so
        # "responds" means "beyond what seeds alone would produce".
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
              f"(change {change:+.1f}, seed noise {noise:.1f}, "
              f"{'responds' if abs(change) > noise else 'FLAT'})")
        print(f"      {'':10s} dW     "
              f"{' -> '.join(f'{d:+.0f}' for d in dws)}")
    payload["behavioural_response"] = response

    write_results(RESULTS, payload)


if __name__ == "__main__":
    main()
