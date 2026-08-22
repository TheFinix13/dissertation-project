#!/usr/bin/env python3
"""Calibrate the trade slice against the exposure ceiling it implies.

The problem this exposes
------------------------
A Buy commits a fixed fraction of initial capital, so becoming fully invested
takes `1 / slice_frac` steps. A monthly episode is only about 21 bars long, so
the slice does not merely control how finely the agent can adjust its position:
it imposes a hard ceiling on the *average* exposure attainable within an episode,
and therefore on the return attainable, no matter how skilful the policy is.

With the slice originally used, a tenth of capital, the ceiling is a mean
exposure near 0.69 against buy-and-hold's 0.91. An agent under that setting
cannot beat buy-and-hold in a rising market, and reporting that it failed to
would be reporting arithmetic as though it were a result about learning.

The rule
--------
Choose the smallest slice whose attainable exposure ceiling reaches 90% of
buy-and-hold's exposure, so the benchmark is reachable in principle and any
shortfall is attributable to the policy rather than to the action set. Smallest,
because a finer slice gives the agent more control over position size, which is
worth keeping wherever it is free.

The criterion is stated on exposure rather than on return because the exposure
ceiling increases monotonically with the slice, whereas the return ceiling does
not: a coarser slice buys at a different set of prices, so it can earn slightly
less while holding slightly more. Selecting on a non-monotone quantity would
make the choice depend on the particular path the validation year happened to
take.

This needs no training and no test data. The ceiling is a property of the
environment and the episode lengths, measured by running the fastest legal
accumulation policy, so the calibration is arithmetic rather than tuning.

Run:
  ./venv/bin/python experiments/final_v2/calibrate_slice.py
"""
from __future__ import annotations

import json
from pathlib import Path

import data as data_mod
from baselines import policy_buy_when_legal
from env import make_env_factory
from features import resolve_rung
from rollout import evaluate, summarize

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "slice_calibration.json"

GRID = (0.05, 0.10, 0.25, 0.50)
INITIAL_CASH = 10_000.0

#: Fraction of buy-and-hold's exposure the ceiling must reach.
EXPOSURE_TARGET = 0.90


def ceiling_for(episodes, slice_frac: float) -> dict:
    """Best attainable exposure and return at this granularity.

    `policy_buy_when_legal` accumulates as fast as the action model permits, so
    its mean exposure is the ceiling for any policy using the same slice.
    """
    factory = make_env_factory(resolve_rung("S3"), slice_frac=slice_frac)
    s = summarize(evaluate(factory, episodes, policy_buy_when_legal), INITIAL_CASH)
    return {"slice_frac": slice_frac,
            "steps_to_full": round(1.0 / slice_frac),
            "mean_exposure": s["mean_exposure"],
            "mean_delta_w": s["mean_delta_w"],
            "mdd_intra_mean": s["mdd_intra_mean"]}


def main() -> None:
    meta, parts = data_mod.load()
    val = parts["val"]
    mean_len = sum(len(e["prices"]) for e in val) / len(val)

    bah = summarize(
        evaluate(make_env_factory(resolve_rung("S3"), slice_frac=1.0),
                 val, policy_buy_when_legal), INITIAL_CASH)

    rows = [ceiling_for(val, f) for f in GRID]

    print(f"calibration on the validation split ({len(val)} months, "
          f"mean length {mean_len:.1f} bars)\n")
    print(f"  buy-and-hold: exposure {bah['mean_exposure']:.3f}, "
          f"dW {bah['mean_delta_w']:+.2f}\n")
    threshold = EXPOSURE_TARGET * bah["mean_exposure"]
    print(f"  exposure the ceiling must reach: {threshold:.3f} "
          f"({EXPOSURE_TARGET:.0%} of buy-and-hold)\n")
    print(f"  {'slice':>7} {'steps':>6} {'ceiling expo':>13} "
          f"{'ceiling dW':>11} {'sufficient':>11}")
    for r in rows:
        ok = r["mean_exposure"] >= threshold
        r["reaches_target"] = bool(ok)
        print(f"  {r['slice_frac']:7.2f} {r['steps_to_full']:6d} "
              f"{r['mean_exposure']:13.3f} {r['mean_delta_w']:+11.2f} "
              f"{'yes' if ok else 'no':>11}")

    reaching = [r for r in rows if r["reaches_target"]]
    chosen = min(r["slice_frac"] for r in reaching) if reaching else max(GRID)

    print(f"\n  selected slice = {chosen:.2f} C_0 "
          f"(= ${chosen * INITIAL_CASH:,.0f})")
    if not reaching:
        print("  warning: no granularity on the grid reaches the target")

    payload = {
        "study": "trade-slice calibration against the exposure ceiling",
        "rule": (f"smallest slice whose attainable exposure ceiling reaches "
                 f"{EXPOSURE_TARGET:.0%} of buy-and-hold's exposure"),
        "exposure_target_fraction": EXPOSURE_TARGET,
        "exposure_threshold": threshold,
        "requires_training": False,
        "requires_test_data": False,
        "split_used": "val",
        "mean_episode_length": mean_len,
        "buy_and_hold": {"mean_exposure": bah["mean_exposure"],
                         "mean_delta_w": bah["mean_delta_w"]},
        "grid": rows,
        "chosen_slice_frac": chosen,
        "chosen_slice_dollars": chosen * INITIAL_CASH,
        "note": ("the ceiling is measured by the fastest legal accumulation "
                 "policy, so it bounds every policy at that granularity"),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
