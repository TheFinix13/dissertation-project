#!/usr/bin/env python3
"""Saturation study: does each rung of the state ladder change what is learned?

Chapter 3 argues the ladder from information requirements. This script tests it,
and it does so on the **validation** year, not the test years. That is
deliberate. Choosing where the ladder stops is a design decision, and a design
decision taken on the test set would make the final numbers a report on months
that had already influenced the design. The test years are spent once, by
`run_final.py`, on the configuration this script selects.

The stopping rule was fixed before the first run (Chapter 3, evaluation
protocol): the ladder terminates at the first rung where the next rung moves
mean wealth change, mean intra-episode drawdown and trades per month by less
than the spread observed across seeds {42, 43, 44}. The rule is applied here in
code, so the reported stopping point is not a judgement made after seeing the
table.

Only the state varies. The reward is the wealth change throughout, and the
action model is the fixed monetary slice throughout, so any difference between
adjacent rows is attributable to the information added between them. The
risk-aware reward is a separate axis and belongs to `run_final.py`.

Run:
  ./venv/bin/python experiments/final_v2/run_ladder.py
  ./venv/bin/python experiments/final_v2/run_ladder.py --smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

from features import LADDER, LADDER_LIMITATION, resolve_rung
from harness import (SEEDS, Config, aggregate, evaluate_agent, load_data,
                     provenance, run_baselines, write_results)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "ladder_results.json"

#: In ladder order. S4 appears with the wealth-change reward only: this asks
#: whether risk history helps by itself, separately from whether a risk-aware
#: reward helps.
RUNGS = ("S0", "S1", "S2", "S3a", "S3", "S4", "S5")

ALGOS = ("reinforce", "dqn")

#: Metrics the stopping rule reads, and the direction that counts as an
#: improvement. Only magnitudes matter for saturation, but the sign is recorded
#: so the table can be read without consulting the code.
RULE_METRICS = {
    "mean_delta_w": "higher is better",
    "mdd_intra_mean": "lower is better",
    "mean_trades": "behavioural, no preferred direction",
}


def apply_stopping_rule(by_rung: dict[str, dict]) -> dict:
    """Find the first rung whose successor changes nothing beyond seed noise.

    A rung is declared saturating when, for every metric in the rule, the change
    to the next rung is smaller than the larger of the two rungs' across-seed
    spreads. Comparing against the spread rather than a fixed threshold means
    the bar adapts to how noisy the training actually was.
    """
    order = [r for r in RUNGS if r in by_rung]
    comparisons = []
    stop_at = None

    for a, b in zip(order, order[1:]):
        metrics = {}
        saturated = True
        for metric in RULE_METRICS:
            va = by_rung[a][metric]
            vb = by_rung[b][metric]
            change = vb["mean"] - va["mean"]
            noise = max(va["spread"], vb["spread"])
            beyond = abs(change) > noise
            metrics[metric] = {
                "from": va["mean"], "to": vb["mean"], "change": change,
                "seed_spread": noise, "beyond_noise": bool(beyond),
            }
            if beyond:
                saturated = False
        comparisons.append({"from": a, "to": b, "saturated": saturated,
                            "metrics": metrics})
        if saturated and stop_at is None:
            stop_at = a

    return {
        "rule": ("terminate at the first rung whose successor moves every rule "
                 "metric by less than the across-seed spread"),
        "rule_metrics": RULE_METRICS,
        "declared_before_running": True,
        "comparisons": comparisons,
        "stops_at": stop_at,
        "note": ("None means no adjacent pair saturated, so the ladder did not "
                 "reach a stopping point within the rungs tested"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=60_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--rungs", nargs="+", default=list(RUNGS))
    ap.add_argument("--algos", nargs="+", default=list(ALGOS))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budget and one seed, to check the wiring only")
    args = ap.parse_args()

    if args.smoke:
        args.timesteps, args.seeds = 2_000, [42]
        args.rungs, args.algos = ["S0", "S3"], ["reinforce", "dqn"]

    meta, parts = load_data()
    train_eps = parts["train"]
    eval_eps = parts[args.split]
    print(f"ladder study: evaluating on the {args.split} split "
          f"({len(eval_eps)} months), {args.timesteps:,} steps, "
          f"seeds {args.seeds}\n")

    payload = {
        "study": "state ladder saturation",
        "eval_split": args.split,
        "eval_months": [e["id"] for e in eval_eps],
        "train_months": [e["id"] for e in train_eps],
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "reward": "wealth change throughout; the reward axis is not varied here",
        "action_model": "fixed monetary slice throughout",
        "provenance": provenance(),
        "data_meta": {k: meta[k] for k in ("ticker", "range", "split_rule",
                                           "warmup_bars", "counts")},
        "rungs": {},
    }

    # Baselines depend only on the environment, not on the observation, so one
    # evaluation covers every rung. S3's configuration is used as the carrier.
    ref_cfg = Config("baselines", resolve_rung("S3"), rung="S3")
    print("  reference strategies")
    base = run_baselines(ref_cfg, eval_eps)
    reference_rows = base["B1b_true_bah"]["_rows"]
    for name, b in base.items():
        s = b["summary"]
        print(f"      {name:16s} dW={s['mean_delta_w']:+8.2f}  "
              f"trades={s['mean_trades']:5.1f}  expo={s['mean_exposure']:.2f}  "
              f"ddI={s['mdd_intra_mean']:.3f}")
    payload["baselines"] = base

    for rung in args.rungs:
        cfg = Config(rung, resolve_rung(rung), rung=rung)
        print(f"\n  {rung}  ({len(cfg.feature_names)}-D: "
              f"{', '.join(cfg.feature_names)})")
        print(f"      limitation: {LADDER_LIMITATION[rung]}")
        entry = {"config": cfg.as_dict(),
                 "limitation": LADDER_LIMITATION[rung],
                 "algos": {}}
        for algo in args.algos:
            print(f"    {algo}")
            entry["algos"][algo] = evaluate_agent(
                algo, cfg, train_eps, eval_eps, reference_rows,
                seeds=args.seeds, total_timesteps=args.timesteps)
        payload["rungs"][rung] = entry

    # The stopping rule reads one algorithm at a time, because saturation is a
    # property of the representation under a given learner. It is also applied
    # to the two learners pooled, which is the summary Chapter 5 quotes.
    payload["saturation"] = {}
    for algo in args.algos:
        by_rung = {r: payload["rungs"][r]["algos"][algo]["across_seeds"]
                   for r in args.rungs}
        payload["saturation"][algo] = apply_stopping_rule(by_rung)

    pooled = {}
    for r in args.rungs:
        seeds_all = [s for algo in args.algos
                     for s in payload["rungs"][r]["algos"][algo]["per_seed"]]
        pooled[r] = aggregate(seeds_all)
    payload["saturation"]["pooled"] = apply_stopping_rule(pooled)

    print("\n  saturation")
    for algo, res in payload["saturation"].items():
        print(f"      {algo:10s} stops at {res['stops_at']}")

    write_results(RESULTS, payload)


if __name__ == "__main__":
    main()
