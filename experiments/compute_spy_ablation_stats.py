"""Paired statistics for the SPY ablation pilot.

Runs paired-by-seed Wilcoxon signed-rank + paired-bootstrap CI for the
six per-arm cells of the SPY ablation pilot, so the table in
Section 5.5 carries inference instead of just point estimates.

Run:
    venv/bin/python experiments/compute_spy_ablation_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
PER_CELL = ROOT / "results" / "per_cell"
OUT = ROOT.parent / "reports" / "generated" / "stats"


PATTERNS = {
    "A_baseline": "baseline__SPY__seed*__phase2.json",
    "B_aleatoric": "probabilistic__SPY__seed*__phase2_aleatoric.json",
    "C_epistemic": "probabilistic__SPY__seed*__phase2_epistemic.json",
    "D_state_only": "ablation_state_only__SPY__seed*__ablation_spy_phase2.json",
    "E_guard_only": "ablation_guard_only__SPY__seed*__ablation_spy_phase2.json",
    "F_scaling_only": "ablation_scaling_only__SPY__seed*__ablation_spy_phase2.json",
}

METRICS = ("final_portfolio_value", "sharpe_ratio", "max_drawdown")

BOOTSTRAP = 10_000
RNG = np.random.default_rng(20260616)


def load(pattern: str) -> dict[int, dict]:
    out = {}
    for path in sorted(PER_CELL.glob(pattern)):
        data = json.loads(path.read_text())
        out[int(data["seed"])] = data
    return out


def paired_test(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 5:
        return {"n": int(len(x)), "p": float("nan"), "median_diff": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    res = stats.wilcoxon(x, y, alternative="two-sided")
    diffs = x - y
    idx = RNG.integers(0, len(diffs), size=(BOOTSTRAP, len(diffs)))
    boot = np.median(diffs[idx], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n": int(len(x)),
        "p": float(res.pvalue),
        "median_diff": float(np.median(diffs)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def main() -> None:
    arms = {label: load(pattern) for label, pattern in PATTERNS.items()}
    common_seeds = sorted(set.intersection(*[set(d.keys()) for d in arms.values()]))

    out = {"common_seeds": common_seeds}
    contrasts = (
        ("B_aleatoric", "A_baseline"),
        ("C_epistemic", "A_baseline"),
        ("D_state_only", "A_baseline"),
        ("E_guard_only", "A_baseline"),
        ("F_scaling_only", "A_baseline"),
        ("B_aleatoric", "D_state_only"),
        ("B_aleatoric", "E_guard_only"),
        ("B_aleatoric", "F_scaling_only"),
    )

    for metric in METRICS:
        out[metric] = {}
        for left, right in contrasts:
            x = np.array([arms[left][s][metric] for s in common_seeds])
            y = np.array([arms[right][s][metric] for s in common_seeds])
            out[metric][f"{left}_vs_{right}"] = paired_test(x, y)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spy_ablation_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for metric in METRICS:
        print(f"\n=== {metric} ===")
        for k, v in out[metric].items():
            print(f"  {k:36s} n={v['n']:2d}  diff={v['median_diff']:+.4g}  "
                  f"CI=[{v['ci_lo']:+.4g}, {v['ci_hi']:+.4g}]  p={v['p']:.3g}")


if __name__ == "__main__":
    main()
