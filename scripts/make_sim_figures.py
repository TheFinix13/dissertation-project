#!/usr/bin/env python3
"""Generate the simulated-data study figures for Chapter 5.

Reads experiments/final_v2/results/*.json; nothing typed by hand.

Output (PDF, into latex/dissertation/figs5/):
    fig5_sim.pdf       two panels: per-regime mean wealth change per
                       algorithm vs always-buy, and per-seed sim-test vs
                       real-transfer wealth change with state-dependence.
    fig5_transfer.pdf  cumulative wealth change of the Deep Q agents vs
                       buy-and-hold across the 180 real transfer months,
                       with the excluded calibration window marked.
    fig5_lambda.pdf    risk-penalty sweep on the simulated validation
                       split (mirrors the pilot's lambda figure).
    fig5_fees.pdf      fee sweep on the simulated validation split
                       (mirrors the pilot's fee figure).

Figures whose results file is missing are skipped with a notice.
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
SERIES = [("dqn", C["dqn"]), ("reinforce", C["reinforce"]), ("ppo", C["ppo"])]


def load(name: str) -> dict | None:
    path = RES / name
    if not path.exists():
        print(f"skip: {name} not found")
        return None
    return json.load(open(path))


# ---------------------------------------------------------------- fig5_sim

def fig_sim(j: dict) -> None:
    regimes = ("up", "down", "flat")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.9),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Left: per-regime means, algorithms + always-buy.
    x = np.arange(len(regimes))
    width = 0.2
    bah = j["baselines_sim"]["B1a_slice_bah"]["per_regime"]
    for i, (algo, color) in enumerate(SERIES):
        vals = [j["results"][algo]["across_seeds"][f"regime_{r}_delta_w"]["mean"]
                for r in regimes]
        ax1.bar(x + (i - 1.5) * width, vals, width, color=color,
                label=LABEL[algo])
    ax1.bar(x + 1.5 * width, [bah[r]["mean_delta_w"] for r in regimes], width,
            color=C_BAH, alpha=0.55, label="always-buy")
    ax1.axhline(0, color="black", lw=0.7)
    ax1.set_xticks(x, [r.capitalize() for r in regimes])
    ax1.set_ylabel("mean wealth change per episode ($)")
    ax1.set_xlabel("simulated test condition")
    ax1.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")

    # Right: per-seed sim vs real transfer.
    for algo, color in SERIES:
        for s in j["results"][algo]["per_seed"]:
            filled = s["summary"]["state_dependent"]
            ax2.scatter(s["summary"]["mean_delta_w"],
                        s["real_transfer"]["mean_delta_w"],
                        s=26, facecolor=color if filled else "white",
                        edgecolor=color, linewidth=1.1, zorder=3)
    real_bah = j["baselines_real"]["B1b_true_bah"]["summary"]["mean_delta_w"]
    ax2.axhline(0, color="black", lw=0.7)
    ax2.axvline(0, color="black", lw=0.7)
    ax2.axhline(real_bah, color=C_BAH, lw=1.0, ls="--")
    ax2.annotate("real buy-and-hold", xy=(0.02, real_bah), fontsize=7,
                 color=C_BAH, xycoords=("axes fraction", "data"),
                 xytext=(0, 3), textcoords="offset points")
    ax2.set_xlabel("simulated test ($/episode)")
    ax2.set_ylabel("real transfer ($/month)")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec=c,
                          label=LABEL[a]) for a, c in SERIES]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white",
                              mec="#666", label="mask-only seed"))
    ax2.legend(handles=handles, fontsize=7, frameon=False, loc="center left")

    fig.tight_layout()
    fig.savefig(OUT / "fig5_sim.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig5_sim.pdf'}")


# ----------------------------------------------------------- fig5_transfer

def fig_transfer(j: dict) -> None:
    """Cumulative wealth change over the 180 real transfer months."""
    months = j["transfer_months"]["early"] + j["transfer_months"]["late"]
    n_early = len(j["transfer_months"]["early"])
    idx = {m: k for k, m in enumerate(months)}

    bah_pm = {r["id"]: r["delta_w"]
              for r in j["baselines_real"]["B1b_true_bah"]["per_month"]}
    bah = np.array([bah_pm[m] for m in months])

    # DQN mean across seeds, plus min/max band.
    per_seed = []
    for s in j["results"]["dqn"]["per_seed"]:
        pm = {r["id"]: r["delta_w"] for r in s["real_per_month"]}
        per_seed.append(np.array([pm[m] for m in months]))
    dqn = np.vstack(per_seed)

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    xs = np.arange(len(months))
    ax.plot(xs, np.cumsum(bah), color=C_BAH, lw=1.4, label="buy-and-hold")
    cum = np.cumsum(dqn, axis=1)
    ax.fill_between(xs, cum.min(axis=0), cum.max(axis=0),
                    color=C["dqn"], alpha=0.18, lw=0)
    ax.plot(xs, cum.mean(axis=0), color=C["dqn"], lw=1.4,
            label="Deep Q (6-seed mean)")
    ax.axhline(0, color="black", lw=0.6)

    # Calibration-window break between the two windows.
    ax.axvline(n_early - 0.5, color="#888888", lw=0.9, ls=":")
    ax.annotate("2018–2022\nexcluded\n(calibration)", xy=(n_early - 0.5, 0.98),
                xycoords=("data", "axes fraction"), ha="center", va="top",
                fontsize=6.5, color="#555555")

    # Year ticks every 24 months.
    ticks = [k for k, m in enumerate(months) if m.endswith("-01")][::2]
    ax.set_xticks(ticks, [months[k][:4] for k in ticks])
    ax.set_ylabel("cumulative wealth change ($)")
    ax.set_xlabel("real transfer month")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "fig5_transfer.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig5_transfer.pdf'}")


# ------------------------------------------------------------- fig5_lambda

def fig_lambda(j: dict) -> None:
    keys = sorted(j["trials"], key=float)
    lams = [float(k) for k in keys]
    dw = [j["trials"][k]["mean_delta_w"]["mean"] for k in keys]
    expo = [j["trials"][k]["mean_exposure"]["mean"] for k in keys]
    dd = [j["trials"][k]["mdd_intra_mean"]["mean"] for k in keys]
    base_expo = j["unpenalised"]["mean_exposure"]
    base_dd = j["unpenalised"]["mdd_intra_mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax1.plot(lams, dw, "o-", color=C["dqn"], lw=1.3, ms=4)
    ax1.axhline(j["return_floor"], color="#888", lw=1.0, ls="--")
    ax1.annotate(f"return floor (+${j['return_floor']:.2f})",
                 xy=(0.35, j["return_floor"]), fontsize=7, color="#555",
                 xytext=(0, 4), textcoords="offset points")
    ax1.axvline(j["chosen"], color="#888", lw=0.8, ls=":")
    ax1.set_xlabel("risk-penalty weight $\\lambda$")
    ax1.set_ylabel("mean wealth change ($/episode)")

    ax2.plot(lams, [e / base_expo if base_expo else 0 for e in expo], "s-",
             color="#0072B2", lw=1.3, ms=4, label="exposure / unpenalised")
    ax2.plot(lams, [d / base_dd if base_dd else 0 for d in dd], "^-",
             color="#D55E00", lw=1.3, ms=4, label="drawdown / unpenalised")
    ax2.axhline(1.0, color="black", lw=0.6)
    ax2.axvline(j["chosen"], color="#888", lw=0.8, ls=":")
    ax2.set_xlabel("risk-penalty weight $\\lambda$")
    ax2.set_ylabel("relative to unpenalised")
    ax2.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_lambda.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig5_lambda.pdf'}")


# --------------------------------------------------------------- fig5_fees

def fig_fees(j: dict) -> None:
    fees = [float(b) for b in j["fees_bps"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))

    bah = [j["levels"][f"{b:g}"]["baselines"]["B1b_true_bah"]["summary"]
           ["mean_delta_w"] for b in fees]
    ax1.plot(fees, bah, "o-", color=C_BAH, lw=1.3, ms=4, label="buy-and-hold")
    for algo, color in SERIES:
        dw = [j["levels"][f"{b:g}"]["algos"][algo]["across_seeds"]
              ["mean_delta_w"]["mean"] for b in fees]
        ax1.plot(fees, dw, "o-", color=color, lw=1.3, ms=4, label=LABEL[algo])
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_xlabel("fee (basis points)")
    ax1.set_ylabel("mean wealth change ($/episode)")
    ax1.legend(fontsize=7, frameon=False)

    for algo, color in SERIES:
        tr = [j["levels"][f"{b:g}"]["algos"][algo]["across_seeds"]
              ["mean_trades"]["mean"] for b in fees]
        ax2.plot(fees, tr, "o-", color=color, lw=1.3, ms=4, label=LABEL[algo])
    ax2.set_xlabel("fee (basis points)")
    ax2.set_ylabel("trades per episode")
    ax2.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_fees.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig5_fees.pdf'}")


if __name__ == "__main__":
    sim = load("sim_results.json")
    if sim:
        fig_sim(sim)
        fig_transfer(sim)
    lam = load("sim_lambda.json")
    if lam:
        fig_lambda(lam)
    fees = load("sim_fees.json")
    if fees:
        fig_fees(fees)
