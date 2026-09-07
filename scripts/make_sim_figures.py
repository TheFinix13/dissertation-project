#!/usr/bin/env python3
"""Generate the simulated-data study figure for Chapter 5.

Reads experiments/final_v2/results/sim_results.json; nothing typed by hand.

Output (PDF, into latex/dissertation/figs5/):
    fig5_sim.pdf   two panels: per-regime mean wealth change per algorithm
                   vs buy-and-hold, and per-seed sim-test vs real-transfer
                   wealth change with state-dependence marked.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "final_v2" / "results"
OUT = ROOT / "latex" / "dissertation" / "figs5"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "figure.dpi": 150,
})

C = {"reinforce": "#0072B2", "dqn": "#D55E00", "ppo": "#009E73"}
LABEL = {"reinforce": "REINFORCE", "dqn": "Deep Q", "ppo": "PPO"}
C_BAH = "#333333"

j = json.load(open(RES / "sim_results.json"))
regimes = ("up", "down", "flat")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.9),
                               gridspec_kw={"width_ratios": [1.15, 1.0]})

# Left: per-regime means, algorithms + buy-and-hold.
x = np.arange(len(regimes))
width = 0.2
bah = j["baselines_sim"]["B1a_slice_bah"]["per_regime"]
series = [("dqn", C["dqn"]), ("reinforce", C["reinforce"]), ("ppo", C["ppo"])]
for i, (algo, color) in enumerate(series):
    vals = [j["results"][algo]["across_seeds"][f"regime_{r}_delta_w"]["mean"]
            for r in regimes]
    ax1.bar(x + (i - 1.5) * width, vals, width, color=color, label=LABEL[algo])
ax1.bar(x + 1.5 * width, [bah[r]["mean_delta_w"] for r in regimes], width,
        color=C_BAH, alpha=0.55, label="always-buy")
ax1.axhline(0, color="black", lw=0.7)
ax1.set_xticks(x, [r.capitalize() for r in regimes])
ax1.set_ylabel("mean wealth change per episode ($)")
ax1.set_xlabel("simulated test regime")
ax1.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")

# Right: per-seed sim vs real transfer.
for algo, color in series:
    for s in j["results"][algo]["per_seed"]:
        filled = s["summary"]["state_dependent"]
        ax2.scatter(s["summary"]["mean_delta_w"],
                    s["real_transfer"]["mean_delta_w"],
                    s=26, facecolor=color if filled else "white",
                    edgecolor=color, linewidth=1.1, zorder=3)
ax2.axhline(0, color="black", lw=0.7)
ax2.axvline(0, color="black", lw=0.7)
ax2.axhline(j["baselines_real"]["B1a_slice_bah"]["summary"]["mean_delta_w"],
            color=C_BAH, lw=1.0, ls="--")
ax2.annotate("real buy-and-hold", xy=(-95, 183), fontsize=7, color=C_BAH)
ax2.set_xlabel("simulated test ($/episode)")
ax2.set_ylabel("real transfer ($/month)")
handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec=c, label=LABEL[a])
           for a, c in [("dqn", C["dqn"]), ("reinforce", C["reinforce"]),
                        ("ppo", C["ppo"])]]
handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#666",
                          label="mask-only seed"))
ax2.legend(handles=handles, fontsize=7, frameon=False, loc="center left")

fig.tight_layout()
fig.savefig(OUT / "fig5_sim.pdf")
print(f"wrote {OUT / 'fig5_sim.pdf'}")
