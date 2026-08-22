#!/usr/bin/env python3
"""Generate the per-month results appendix from the results JSON.

The appendix is generated rather than transcribed so that it cannot drift from
the numbers it reports. Regenerating it after any re-run is a one-line command,
and a mismatch between the table and the results file becomes impossible rather
than merely unlikely.

Run:
  ./venv/bin/python experiments/final_v2/make_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE.parents[1] / "latex" / "final_dissertation" / "appendix_permonth.tex"

CELL = "S3_wealth"
SEED = 42


def per_month(entry: dict, algo: str, seed: int) -> dict[str, float]:
    for s in entry["algos"][algo]["per_seed"]:
        if s["seed"] == seed:
            return {m["id"]: m["delta_w"] for m in s["per_month"]}
    raise KeyError(f"seed {seed} not found for {algo}")


def baseline_month(entry: dict, name: str) -> dict[str, float]:
    return {m["id"]: m["delta_w"] for m in entry["baselines"][name]["per_month"]}


def main() -> None:
    final = json.loads((RESULTS / "final_results.json").read_text())
    risk = json.loads((RESULTS / "final_results_risk.json").read_text())
    entry = final["results"][CELL]

    cols = {
        "B1b buy\\&hold": baseline_month(entry, "B1b_true_bah"),
        "B1a slice sched.": baseline_month(entry, "B1a_slice_bah"),
        "REINFORCE": per_month(entry, "reinforce", SEED),
        "Deep Q": per_month(entry, "dqn", SEED),
        "PPO": per_month(entry, "ppo", SEED),
        "Deep Q (risk)": per_month(risk["results"]["S4_risk"], "dqn", SEED),
        "Random": baseline_month(entry, "B2_random"),
    }
    months = sorted(cols["B1b buy\\&hold"])

    lines = [
        r"\chapter{Per-Month Test Results}",
        r"\label{app:permonth}",
        "",
        rf"Wealth change $\Delta W$ in dollars for every held-out test month, at",
        rf"seed {SEED}, with $C_0 = \$10{{,}}000$ and a 5-basis-point fee. The",
        r"learned columns use the nine-feature state and the wealth-change reward",
        r"except the final one, which uses the risk-aware reward at",
        r"$\lambda = 0.10$ (Section~\ref{sec:res_risk}). B0 do-nothing is",
        r"identically zero and is omitted.",
        "",
        r"Two patterns are visible by inspection and are discussed in",
        r"Section~\ref{sec:res_final}. Months where REINFORCE equals the slice",
        r"schedule to the cent are months in which it executed an identical",
        r"action sequence, which is what a mask-determined policy does. And every",
        r"column beats buy-and-hold in the same months, namely the months in which",
        r"the market fell, because holding less is what they all have in common.",
        "",
        r"This table is generated from the committed results files by",
        r"\texttt{make\_tables.py} rather than transcribed.",
        "",
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Per-month $\Delta W$ (\$) on the {len(months)} held-out test months.}}",
        r"\label{tab:permonth}",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        r"\textbf{Month} & "
        + " & ".join(rf"\textbf{{{c}}}" for c in cols) + r" \\",
        r"\midrule",
    ]

    for m in months:
        cells = [f"{cols[c].get(m, float('nan')):.2f}" for c in cols]
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")

    lines += [r"\midrule", r"\textbf{Mean} & "
              + " & ".join(
                  rf"\textbf{{{sum(cols[c][m] for m in months) / len(months):.2f}}}"
                  for c in cols) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}  ({len(months)} months, {len(cols)} columns)")


if __name__ == "__main__":
    main()
