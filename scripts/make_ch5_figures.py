#!/usr/bin/env python3
"""Generate the Chapter 5 figures for the supervisor preview.

Every number is read from the result JSONs under experiments/final_v2/results;
nothing is typed in by hand, so the figures cannot drift from the tables.

Outputs (PDF, into latex/dissertation/figs5/):
    fig5_slice.pdf      exposure ceiling per trade slice vs buy-and-hold
    fig5_seeds.pdf      per-seed conditioning outcomes (REINFORCE vs Deep Q)
    fig5_state.pdf      9-feature vs 10-feature state on the test months
    fig5_cumwealth.pdf  cumulative monthly wealth changes on the test months
    fig5_effic.pdf      wealth change vs exposure and vs drawdown (2 panels)
    fig5_months.pdf     buy-and-hold vs Deep Q monthly wealth changes, all 24 months
    fig5_mscatter.pdf   Deep Q monthly wealth change against buy-and-hold's
    fig5_lambda.pdf     risk-penalty sweep: wealth change and exposure vs lambda
    fig5_fees.pdf       fee sensitivity: wealth change and trading activity
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

C_REI = "#0072B2"   # blue
C_DQN = "#D55E00"   # vermillion
C_PPO = "#009E73"   # green
C_BAH = "#333333"   # near-black
C_REF = "#999999"   # grey


def load(name: str):
    return json.load(open(RES / name))


# ------------------------------------------------------------- fig5_slice
def fig_slice() -> None:
    j = load("slice_calibration.json")
    grid = j["grid"]
    labels = [f"{g['slice_frac']:.2f}" r"$C_0$" for g in grid]
    expo = [g["mean_exposure"] for g in grid]
    bah = j["buy_and_hold"]["mean_exposure"]
    target = j["exposure_threshold"]

    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    bars = ax.bar(labels, expo, width=0.55, color=["#b8cbe0", "#b8cbe0", C_REI, "#b8cbe0"])
    ax.axhline(bah, color=C_BAH, lw=1.2, ls="--")
    ax.axhline(target, color=C_DQN, lw=1.0, ls=":")
    ax.annotate(f"buy-and-hold ({bah:.3f})", xy=(0.02, bah), xytext=(0.02, bah + 0.03),
                textcoords="data", fontsize=8, color=C_BAH)
    ax.annotate(f"90% target ({target:.3f})", xy=(2.55, target), xytext=(2.55, target - 0.09),
                fontsize=8, color=C_DQN)
    for b, g in zip(bars, grid):
        ax.annotate(f"{g['mean_exposure']:.3f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylabel("attainable mean exposure")
    ax.set_xlabel("trade slice (fraction of initial capital)")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_slice.pdf")
    plt.close(fig)


# -------------------------------------------------------------- fig5_test
def _test_rows():
    """Rows for the held-out comparison. S4_risk comes from the lambda=0.10 rerun."""
    fin = load("final_results.json")
    risk = load("final_results_risk.json")
    cells = [
        ("S3_wealth", "9-feature state", fin),
        ("S4_wealth", "10-feature state", fin),
        ("S4_risk", "10-feature + risk reward", risk),
        ("S3_wealth_share", "whole-share action", fin),
        ("S3_wealth_slice10", r"0.10$C_0$ action", fin),
    ]
    rows = []
    for cell, label, src in cells:
        block = src["results"][cell]
        entry = {"label": label, "methods": {}}
        for m in ("reinforce", "dqn", "ppo"):
            a = block["algos"][m]["across_seeds"]
            seeds = [s["summary"]["mean_delta_w"] for s in block["algos"][m]["per_seed"]]
            entry["methods"][m] = {
                "mean": a["mean_delta_w"]["mean"],
                "min": min(seeds),
                "max": max(seeds),
                "expo": a["mean_exposure"]["mean"],
                "dd": a["mdd_intra_mean"]["mean"],
            }
        rows.append(entry)
    bah = fin["results"]["S3_wealth"]["baselines"]["B1b_true_bah"]["summary"]
    return rows, bah


def fig_seeds() -> None:
    """Per-seed conditioning outcomes on validation, both slices, all algos."""
    j = load("conditioning_v2_results.json")
    algos = [("reinforce", C_REI, "REINFORCE"),
             ("dqn", C_DQN, "Deep Q"),
             ("ppo", C_PPO, "PPO")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    for ax, tag, title in [(axes[0], "slice0.1", "original $0.10\\,C_0$ slice"),
                           (axes[1], "slice0.25", "calibrated $0.25\\,C_0$ slice")]:
        ceiling = j["baselines"][tag]["B1a_slice_bah"]["summary"]["mean_delta_w"]
        for x, (algo, color, label) in enumerate(algos):
            seeds = j["cells"][f"{algo}__{tag}"]["per_seed"]
            for k, s in enumerate(seeds):
                dw = s["summary"]["mean_delta_w"]
                dep = s["conditioning"]["state_dependent"]
                jitter = (k - 2.5) * 0.055
                if dep:
                    ax.scatter(x + jitter, dw, s=38, color=color, zorder=3)
                else:
                    ax.scatter(x + jitter, dw, s=38, facecolors="none",
                               edgecolors=color, linewidths=1.3, zorder=3)
        ax.axhline(ceiling, color=C_BAH, ls="--", lw=1.0)
        ax.annotate(f"always-buy ceiling  +${ceiling:.2f}", xy=(-0.42, ceiling),
                    xytext=(-0.42, ceiling + 5), fontsize=7.5, color=C_BAH)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["REINFORCE", "Deep Q", "PPO"], fontsize=8)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean wealth change per month ($)")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="", color="grey", label="state-dependent seed"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", color="grey",
               label="mask-only seed"),
    ]
    axes[0].legend(handles=handles, fontsize=7.5, frameon=False,
                   loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_seeds.pdf")
    plt.close(fig)


def fig_state() -> None:
    """9-feature vs 10-feature state under the wealth reward, test months.

    Three panels (return, exposure, drawdown) so the reader can see that all
    three quantities move together, not just the return.
    """
    fin = load("final_results.json")
    names = [("reinforce", "REINFORCE", C_REI), ("dqn", "Deep Q", C_DQN), ("ppo", "PPO", C_PPO)]
    metrics = [("mean_delta_w", "mean wealth change ($)", "mean_delta_w"),
               ("mean_exposure", "mean exposure", "mean_exposure"),
               ("mdd_intra_mean", "mean intra-episode drawdown", "mdd_intra_mean")]
    bah = fin["results"]["S3_wealth"]["baselines"]["B1b_true_bah"]["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.6))
    w = 0.34
    for ax, (key, ylab, _) in zip(axes, metrics):
        for i, (m, label, color) in enumerate(names):
            for dx, cell, alpha in ((-w / 2, "S3_wealth", 1.0), (w / 2, "S4_wealth", 0.45)):
                blk = fin["results"][cell]["algos"][m]
                mean = blk["across_seeds"][key]["mean"]
                seeds = [s["summary"][key] for s in blk["per_seed"]]
                ax.bar(i + dx, mean, width=w, color=color, alpha=alpha)
                ax.plot([i + dx, i + dx], [min(seeds), max(seeds)], color="black", lw=0.8)
        ax.axhline(bah[key], color=C_BAH, ls="--", lw=0.9)
        ax.set_xticks(range(3))
        ax.set_xticklabels([n[1] for n in names], fontsize=7)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=7)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color="grey", alpha=1.0, label="9-feature state"),
               Patch(color="grey", alpha=0.45, label="10-feature state"),
               Line2D([], [], color=C_BAH, ls="--", lw=0.9, label="buy-and-hold")]
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig5_state.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cumwealth() -> None:
    """Cumulative sum of mean monthly wealth changes across the 24 test months."""
    j = load("final_results.json")
    block = j["results"]["S3_wealth"]
    months = [m["id"] for m in block["baselines"]["B1b_true_bah"]["per_month"]]

    def cum(vals):
        out, t = [], 0.0
        for v in vals:
            t += v
            out.append(t)
        return out

    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for key, label, color, ls in [("B1b_true_bah", "buy-and-hold", C_BAH, "--"),
                                  ("B1a_slice_bah", "always-buy (0.25 $C_0$)", C_REF, ":")]:
        vals = [m["delta_w"] for m in block["baselines"][key]["per_month"]]
        ax.plot(range(24), cum(vals), color=color, ls=ls, lw=1.3, label=label)
    for m, label, color in [("reinforce", "REINFORCE", C_REI),
                            ("dqn", "Deep Q", C_DQN), ("ppo", "PPO", C_PPO)]:
        per_seed = block["algos"][m]["per_seed"]
        means = [sum(s["per_month"][k]["delta_w"] for s in per_seed) / len(per_seed)
                 for k in range(24)]
        ax.plot(range(24), cum(means), color=color, lw=1.4, label=label)
    ticks = list(range(0, 24, 3))
    ax.set_xticks(ticks)
    ax.set_xticklabels([months[t] for t in ticks], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("cumulative monthly wealth changes ($)")
    # C0 = $10,000, so $100 of monthly wealth change is 1% of initial capital.
    ax2 = ax.secondary_yaxis("right", functions=(lambda v: v / 100.0,
                                                 lambda v: v * 100.0))
    ax2.set_ylabel("normalised (% of $C_0$)")
    ax2.tick_params(labelsize=7)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_cumwealth.pdf")
    plt.close(fig)


def fig_effic() -> None:
    """Wealth change against exposure (left) and against drawdown (right)."""
    rows, bah = _test_rows()
    fin = load("final_results.json")
    base = fin["results"]["S3_wealth"]["baselines"]
    markers = {"reinforce": "o", "dqn": "s", "ppo": "^"}
    colors = {"reinforce": C_REI, "dqn": C_DQN, "ppo": C_PPO}
    labels = {"reinforce": "REINFORCE", "dqn": "Deep Q", "ppo": "PPO"}

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    for ax, xkey, bx, xlab in [
            (axes[0], "expo", bah["mean_exposure"], "mean exposure"),
            (axes[1], "dd", bah["mdd_intra_mean"], "mean intra-episode drawdown")]:
        slope = bah["mean_delta_w"] / bx
        ax.plot([0, bx * 1.12], [0, slope * bx * 1.12], color=C_REF, lw=1.0, ls="--",
                zorder=1, label="proportional to buy-and-hold")
        seen = set()
        for entry in rows:
            for m, v in entry["methods"].items():
                ax.scatter(v[xkey], v["mean"], marker=markers[m], color=colors[m],
                           s=26, zorder=3, label=labels[m] if m not in seen else None)
                seen.add(m)
        s = base["B1b_true_bah"]["summary"]
        xv = s["mean_exposure"] if xkey == "expo" else s["mdd_intra_mean"]
        ax.scatter(xv, s["mean_delta_w"], marker="*", s=95, color=C_BAH, zorder=4,
                   label="buy-and-hold" if ax is axes[0] else None)
        ax.set_xlabel(xlab)
    axes[0].set_ylabel("mean wealth change ($)")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_effic.pdf")
    plt.close(fig)


def _monthly_bah_dqn():
    """Per-month wealth changes: buy-and-hold and the Deep Q seed mean."""
    j = load("final_results.json")
    block = j["results"]["S3_wealth"]
    months = [m["id"] for m in block["baselines"]["B1b_true_bah"]["per_month"]]
    bah = [m["delta_w"] for m in block["baselines"]["B1b_true_bah"]["per_month"]]
    per = block["algos"]["dqn"]["per_seed"]
    dqn = [sum(s["per_month"][k]["delta_w"] for s in per) / len(per)
           for k in range(len(months))]
    expo_ratio = (block["algos"]["dqn"]["across_seeds"]["mean_exposure"]["mean"]
                  / block["baselines"]["B1b_true_bah"]["summary"]["mean_exposure"])
    return months, bah, dqn, expo_ratio


def fig_months() -> None:
    """Monthly wealth changes for buy-and-hold and Deep Q across the test set."""
    months, bah, dqn, _ = _monthly_bah_dqn()
    x = list(range(len(months)))
    w = 0.4

    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    i_mar = months.index("2025-03")
    ax.axvspan(i_mar - 0.55, i_mar + 0.55, color=C_DQN, alpha=0.10)
    ax.bar([v - w / 2 for v in x], bah, width=w, color=C_BAH, alpha=0.85,
           label="buy-and-hold")
    ax.bar([v + w / 2 for v in x], dqn, width=w, color=C_DQN, alpha=0.85,
           label="Deep Q (seed mean)")
    ax.axhline(0, color="black", lw=0.7)
    ax.annotate("March 2025\n$-$398.41 vs $-$184.04",
                xy=(i_mar, min(bah)), xytext=(i_mar - 6.2, min(bah) + 30),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.7))
    ticks = list(range(0, len(months), 3))
    ax.set_xticks(ticks)
    ax.set_xticklabels([months[t] for t in ticks], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("wealth change ($)")
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_months.pdf")
    plt.close(fig)


def fig_mscatter() -> None:
    """Deep Q monthly wealth change against buy-and-hold's, with the
    exposure-proportional line."""
    months, bah, dqn, expo_ratio = _monthly_bah_dqn()

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    lo, hi = min(bah) * 1.08, max(bah) * 1.08
    ax.plot([lo, hi], [expo_ratio * lo, expo_ratio * hi], color=C_REF, lw=1.0,
            ls="--", zorder=1, label=f"proportional to exposure ({expo_ratio:.2f}x)")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.scatter(bah, dqn, s=26, color=C_DQN, zorder=3, label="one test month")
    for mid, dx, dy in [("2025-03", 6, -4), ("2024-04", 6, -4), ("2025-05", -8, 4)]:
        k = months.index(mid)
        ax.annotate(mid, xy=(bah[k], dqn[k]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5,
                    ha="left" if dx > 0 else "right")
    ax.set_xlabel("buy-and-hold wealth change in the month ($)")
    ax.set_ylabel("Deep Q wealth change in the month ($)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_mscatter.pdf")
    plt.close(fig)


# ------------------------------------------------------------ fig5_lambda
def fig_lambda() -> None:
    """Two panels: return vs lambda (with the selection floor), and exposure
    plus drawdown relative to the unpenalised agent, which makes the cliff
    between lambda = 0.10 and 0.25 directly readable."""
    j = load("final_results.json")
    trials = j["lambda_selection"]["trials"]
    lams = [l for l in ["0.0", "0.05", "0.1", "0.25", "0.5", "1.0", "2.0"] if l in trials]
    x = list(range(len(lams)))
    dw = [trials[l]["mean_delta_w"]["mean"] for l in lams]
    ex = [trials[l]["mean_exposure"]["mean"] for l in lams]
    dd = [trials[l]["mdd_intra_mean"]["mean"] for l in lams]
    ex_rel = [100.0 * v / ex[0] for v in ex]
    dd_rel = [100.0 * v / dd[0] for v in dd]
    floor = 0.5 * dw[0]

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.7))
    a.plot(x, dw, "o-", color=C_DQN, lw=1.4)
    a.axhline(floor, color=C_BAH, ls=":", lw=1.0)
    a.annotate(f"selection floor  +${floor:.2f}", xy=(len(x) - 0.2, floor),
               xytext=(len(x) - 0.2, floor + 5), fontsize=7.5, ha="right", color=C_BAH)
    a.annotate(r"chosen $\lambda=0.10$", xy=(2, dw[2]), xytext=(2.5, dw[2] + 35),
               fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
    a.set_xticks(x)
    a.set_xticklabels(lams, fontsize=7)
    a.set_xlabel(r"penalty weight $\lambda$")
    a.set_ylabel("mean wealth change ($)")

    b.plot(x, ex_rel, "s-", color=C_REI, lw=1.4, label="exposure")
    b.plot(x, dd_rel, "^--", color=C_PPO, lw=1.4, label="drawdown")
    b.axvspan(2.5, 3.5, color=C_DQN, alpha=0.08)
    b.annotate("cliff", xy=(3.0, 55), fontsize=8, ha="center", color=C_DQN)
    b.set_xticks(x)
    b.set_xticklabels(lams, fontsize=7)
    b.set_xlabel(r"penalty weight $\lambda$")
    b.set_ylabel(r"relative to unpenalised (%)")
    b.legend(fontsize=8, frameon=False)
    fig.suptitle("Deep Q-learning on the validation months", fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_lambda.pdf")
    plt.close(fig)


# -------------------------------------------------------------- fig5_fees
def fig_fees() -> None:
    j = load("fee_results.json")
    fees = [0, 5, 10, 25, 50]
    bah = [j["levels"][str(f)]["baselines"]["B1b_true_bah"]["summary"]["mean_delta_w"] for f in fees]
    rei = [j["levels"][str(f)]["algos"]["reinforce"]["across_seeds"]["mean_delta_w"]["mean"] for f in fees]
    dqn = [j["levels"][str(f)]["algos"]["dqn"]["across_seeds"]["mean_delta_w"]["mean"] for f in fees]
    ppo = [j["levels"][str(f)]["algos"]["ppo"]["across_seeds"]["mean_delta_w"]["mean"] for f in fees]
    rei_t = j["behavioural_response"]["reinforce"]["mean_trades"]
    dqn_t = j["behavioural_response"]["dqn"]["mean_trades"]
    ppo_t = j["behavioural_response"]["ppo"]["mean_trades"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.8))
    a.plot(fees, bah, "*-", color=C_BAH, lw=1.2, label="buy-and-hold")
    a.plot(fees, rei, "o-", color=C_REI, lw=1.2, label="REINFORCE")
    a.plot(fees, dqn, "s-", color=C_DQN, lw=1.2, label="Deep Q")
    a.plot(fees, ppo, "^-", color=C_PPO, lw=1.2, label="PPO")
    a.set_xlabel("transaction fee (basis points)")
    a.set_ylabel("mean wealth change ($)")
    a2 = a.secondary_yaxis("right", functions=(lambda v: v / 100.0,
                                               lambda v: v * 100.0))
    a2.set_ylabel("normalised (% of $C_0$)", fontsize=8)
    a2.tick_params(labelsize=7)
    a.legend(fontsize=7.5, frameon=False)

    b.plot(fees, rei_t, "o-", color=C_REI, lw=1.2, label="REINFORCE")
    b.plot(fees, dqn_t, "s-", color=C_DQN, lw=1.2, label="Deep Q")
    b.plot(fees, ppo_t, "^-", color=C_PPO, lw=1.2, label="PPO")
    b.set_xlabel("transaction fee (basis points)")
    b.set_ylabel("mean trades per month")
    b.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_fees.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for stale in ("fig5_test.pdf", "fig5_expo.pdf"):
        (OUT / stale).unlink(missing_ok=True)
    fig_slice()
    fig_seeds()
    fig_state()
    fig_cumwealth()
    fig_effic()
    fig_months()
    fig_mscatter()
    fig_lambda()
    fig_fees()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("fig5_*.pdf"))))
