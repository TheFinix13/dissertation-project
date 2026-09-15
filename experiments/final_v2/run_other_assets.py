#!/usr/bin/env python3
"""Evaluate the simulated-study policies on OTHER assets — demo gate.

STUDY: post-viva engineering — NOT in the submitted dissertation. Writes
results/other_assets.json (gates the demo page).

Purpose (13 Sep 2026): the viva QR demo shows the SPY transfer test
animated. Before any other asset appears on that page, its numbers are
produced here and inspected offline. An asset ships only if the story
is defensible; anything that embarrasses is cut before it is seen.

Protocol: identical to the SPY transfer test. Policies are retrained
with the same seeds on the same 3,000 simulated episodes (training is
deterministic per seed, so these are the dissertation's policies), then
run unchanged on the new asset's real months outside the 2018-2022
calibration window. The agents tune nothing on any of these months —
for the new tickers they have never seen the asset at all.

Run (from experiments/final_v2):
  ../../venv/bin/python run_other_assets.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import data as real_data
import real_transfer
import sim_data
from features import resolve_rung
from harness import INITIAL_CASH, SEEDS, Config, run_baselines, train_one
from rollout import evaluate

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "other_assets.json"

TICKERS = ("AAPL", "QQQ")
ALGOS = ("dqn", "reinforce", "ppo")
TIMESTEPS = 300_000


def build_transfer(ticker: str) -> list[dict]:
    """Same windows as the SPY transfer set, for a different ticker."""
    df = real_data.fetch(ticker, start=real_transfer.WARMUP_START,
                         end=real_data.NOMINAL_END)
    eps = real_data.build_episodes(
        df, nominal_start=real_transfer.NOMINAL_START)
    keep = [e for e in eps if real_transfer.window_of(e["id"]) is not None]
    if len(keep) < 150:
        raise RuntimeError(f"{ticker}: only {len(keep)} transfer months")
    return keep


def main() -> None:
    sim_meta, sim_parts = sim_data.load()
    cfg = Config("SIM_S3_wealth", resolve_rung("S3"), rung="S3")
    eval_factory = cfg.factory(risk_lambda=0.0)

    assets: dict[str, list[dict]] = {}
    out: dict = {"protocol": "SPY transfer protocol, other tickers",
                 "timesteps": TIMESTEPS, "seeds": list(SEEDS),
                 "assets": {}}
    for t in TICKERS:
        eps = build_transfer(t)
        assets[t] = eps
        base = run_baselines(cfg, eps)
        out["assets"][t] = {
            "n_months": len(eps),
            "ids": [e["id"] for e in eps],
            "bah_per_month": [
                {"id": r["episode_id"], "delta_w": r["delta_w"]}
                for r in base["B1b_true_bah"]["_rows"]],
            "bah_mean": float(np.mean(
                [r["delta_w"] for r in base["B1b_true_bah"]["_rows"]])),
            "algos": {a: [] for a in ALGOS},
        }
        print(f"{t}: {len(eps)} months, BH mean "
              f"{out['assets'][t]['bah_mean']:+.2f} $/month", flush=True)

    for algo in ALGOS:
        for seed in SEEDS:
            t0 = time.time()
            policy, _ = train_one(algo, cfg, sim_parts["train"],
                                  seed=seed, total_timesteps=TIMESTEPS)
            line = [f"{algo} seed {seed}:"]
            for t in TICKERS:
                rows = evaluate(eval_factory, assets[t], policy)
                out["assets"][t]["algos"][algo].append([
                    {"id": r["episode_id"], "delta_w": r["delta_w"],
                     "mean_exposure": r["mean_exposure"]} for r in rows])
                line.append(
                    f"{t} {np.mean([r['delta_w'] for r in rows]):+7.2f}")
            line.append(f"({time.time() - t0:.0f}s)")
            print("  ".join(line), flush=True)

    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}")

    # gate summary
    print("\n=== GATE SUMMARY ($/month, mean of seeds) ===")
    for t in TICKERS:
        a = out["assets"][t]
        print(f"{t}: BH {a['bah_mean']:+.2f}", end="")
        for algo in ALGOS:
            means = [np.mean([m["delta_w"] for m in seed_rows])
                     for seed_rows in a["algos"][algo]]
            print(f"  {algo} {np.mean(means):+.2f} "
                  f"[{np.min(means):+.2f}..{np.max(means):+.2f}]", end="")
        print()


if __name__ == "__main__":
    main()
