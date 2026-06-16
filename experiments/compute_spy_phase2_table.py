"""Compute the SPY headline case-study table from Phase-2 per-cell JSONs.

Replaces the legacy 10k-step SPY case-study numbers in Chapter 5 with
the Phase-2-protocol-compliant 50k-step numbers committed at
experiments/results/per_cell/{baseline|probabilistic}__SPY__*.json.

Also computes the same statistics for the three ablation arms
(state_only, guard_only, scaling_only) if Phase-2 budget per-cell JSONs
exist for them at experiments/results/per_cell/ablation_*__SPY__*.json.

Run:
    venv/bin/python experiments/compute_spy_phase2_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PER_CELL = ROOT / "results" / "per_cell"
OUT = ROOT.parent / "reports" / "generated" / "stats"

METRICS = (
    "final_portfolio_value",
    "sharpe_ratio",
    "max_drawdown",
    "capital_preservation_rate_95pct_hwm",
    "var_95_violation_rate",
)


def load_arm(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(PER_CELL.glob(pattern)):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            pass
    return rows


def summarise(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"arm": label, "n_seeds": 0}
    out = {"arm": label, "n_seeds": len(rows)}
    for m in METRICS:
        vals = [r[m] for r in rows if m in r]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[f"{m}_median"] = float(np.median(arr))
        out[f"{m}_q25"] = float(np.quantile(arr, 0.25))
        out[f"{m}_q75"] = float(np.quantile(arr, 0.75))
        out[f"{m}_mean"] = float(np.mean(arr))
    return out


def main() -> None:
    arms = {
        "A_baseline_phase2": "baseline__SPY__seed*__phase2.json",
        "B_aleatoric_phase2": "probabilistic__SPY__seed*__phase2_aleatoric.json",
        "C_epistemic_phase2": "probabilistic__SPY__seed*__phase2_epistemic.json",
        "D_state_only_phase2": "ablation_state_only__SPY__seed*__ablation_spy_phase2.json",
        "E_guard_only_phase2": "ablation_guard_only__SPY__seed*__ablation_spy_phase2.json",
        "F_scaling_only_phase2": "ablation_scaling_only__SPY__seed*__ablation_spy_phase2.json",
    }
    summaries = {label: summarise(load_arm(pattern), label) for label, pattern in arms.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spy_phase2_case_study.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    md = ["# SPY case study at Phase-2 budget (50k timesteps, 10 seeds)", ""]
    md.append(r"| Arm | n | Final \$ (med) | Final \$ IQR | Sharpe (med) | MDD (med) | Preserv. (med) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, s in summaries.items():
        n = s.get("n_seeds", 0)
        if not n:
            md.append(f"| {label} | 0 | -- | -- | -- | -- | -- |")
            continue
        md.append(
            f"| {label} | {n} | "
            f"\\${s['final_portfolio_value_median']:,.0f} | "
            f"\\${s['final_portfolio_value_q25']:,.0f} -- \\${s['final_portfolio_value_q75']:,.0f} | "
            f"{s['sharpe_ratio_median']:+.4f} | "
            f"{s['max_drawdown_median']:.4f} | "
            f"{s['capital_preservation_rate_95pct_hwm_median']:.4f} |"
        )
    (OUT / "spy_phase2_case_study.md").write_text("\n".join(md), encoding="utf-8")

    for label, s in summaries.items():
        n = s.get("n_seeds", 0)
        if not n:
            print(f"{label}: no per-cell data yet.")
            continue
        print(
            f"{label}: n={n}, "
            f"final med=${s['final_portfolio_value_median']:,.0f}, "
            f"Sharpe med={s['sharpe_ratio_median']:+.4f}, "
            f"MDD med={s['max_drawdown_median']:.4f}"
        )


if __name__ == "__main__":
    main()
