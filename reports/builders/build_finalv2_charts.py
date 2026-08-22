#!/usr/bin/env python3
"""Chapter 5 charts for the final_v2 experiment suite.

Reads the JSON records written by experiments/final_v2/run_*.py and emits
PNG + PDF pairs under reports/generated/charts/.

Run:
  venv/bin/python reports/builders/build_finalv2_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "final_v2" / "results"
CHARTS = ROOT / "reports" / "generated" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

C_REINFORCE = "#4a6fa5"
C_DQN = "#c4952c"
C_PPO = "#1a7a4c"
C_BAH = "#aa3333"
C_SLICE = "#8a5aa8"
C_GREY = "#8a8a8a"

ALGO_STYLE = {
    "reinforce": ("REINFORCE", C_REINFORCE),
    "dqn": ("Deep Q-learning", C_DQN),
    "ppo": ("PPO", C_PPO),
}


def save(fig, name: str) -> None:
    fig.savefig(CHARTS / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {CHARTS / (name + '.png')}")


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


# ----------------------------------------------------------------- figure 1
def chart_return_vs_exposure(final: dict, risk: dict) -> None:
    """Return against what was risked to earn it.

    In both panels the dashed ray joins the origin to buy-and-hold. A strategy
    on that ray earns exactly in proportion to what it commits, which is what
    holding a smaller fraction of buy-and-hold achieves with no learning at all.
    Only a point above the ray represents skill.
    """
    cell = final["results"]["S3_wealth"]
    base = cell["baselines"]
    b1b = base["B1b_true_bah"]["summary"]
    rd = risk["results"]["S4_risk"]["algos"]["dqn"]["across_seeds"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), dpi=160)

    panels = (
        (axes[0], "mean_exposure", "mdd", 1.0,
         "Mean exposure (fraction of wealth in the asset)",
         "Return against participation"),
        (axes[1], "mdd_intra_mean", None, 0.035,
         "Mean intra-episode drawdown",
         "Return against drawdown incurred"),
    )

    for ax, key, _unused, xmax, xlabel, title in panels:
        bx = b1b["mean_exposure"] if key == "mean_exposure" else b1b[key]
        xs = np.linspace(0, xmax, 50)
        ax.plot(xs, xs * b1b["mean_delta_w"] / bx, color=C_BAH, lw=1.3,
                ls="--", zorder=1, label="proportional to buy-and-hold")

        for bkey, label, colour, marker in (
            ("B1b_true_bah", "B1b buy-and-hold", C_BAH, "*"),
            ("B1a_slice_bah", "B1a slice schedule", C_SLICE, "D"),
            ("B2_random", "B2 random legal", C_GREY, "s"),
        ):
            s = base[bkey]["summary"]
            ax.scatter([s[key]], [s["mean_delta_w"]], s=110, color=colour,
                       marker=marker, zorder=3, edgecolor="white",
                       linewidth=0.8, label=label)

        for algo, (label, colour) in ALGO_STYLE.items():
            a = cell["algos"][algo]["across_seeds"]
            ax.errorbar(a[key]["mean"], a["mean_delta_w"]["mean"],
                        yerr=a["mean_delta_w"]["spread"] / 2.0,
                        xerr=a[key]["spread"] / 2.0,
                        color=colour, lw=1.1, capsize=3, zorder=2)
            ax.scatter([a[key]["mean"]], [a["mean_delta_w"]["mean"]], s=95,
                       color=colour, marker="o", zorder=3, edgecolor="white",
                       linewidth=0.8, label=label)

        ax.scatter([rd[key]["mean"]], [rd["mean_delta_w"]["mean"]], s=95,
                   facecolor="white", edgecolor=C_DQN, linewidth=1.8,
                   marker="o", zorder=3, label="Deep Q, risk-aware reward")

        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 245)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[1].set_xticks([0.0, 0.01, 0.02, 0.03])
    axes[0].set_ylabel("Mean monthly wealth change ($)")
    for ax in axes:
        ax.text(0.03, 0.94, "learned points sit on the ray to within the seed\n"
                            "range: less return for less risk, in proportion",
                transform=ax.transAxes, fontsize=10.5, color="#333333",
                va="top", ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Held-out months: no learned agent earns more per unit of risk "
                 "than buy-and-hold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "finalv2_return_vs_exposure")


# ----------------------------------------------------------------- figure 2
def chart_ladder(ladder: dict) -> None:
    rungs = list(ladder["rungs"].keys())
    dims = [ladder["rungs"][r]["config"]["obs_dim"] for r in rungs]
    labels = [f"$S_{{{r[1:]}}}$\n({d}D)" for r, d in zip(rungs, dims)]

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=160)
    x = np.arange(len(rungs))
    w = 0.38

    for off, algo in ((-w / 2, "reinforce"), (w / 2, "dqn")):
        label, colour = ALGO_STYLE[algo]
        means, spreads, statedep = [], [], []
        for r in rungs:
            a = ladder["rungs"][r]["algos"][algo]["across_seeds"]
            means.append(a["mean_delta_w"]["mean"])
            spreads.append(a["mean_delta_w"]["spread"])
            statedep.append(a["n_state_dependent"])
        ax.bar(x + off, means, w, color=colour, label=label, zorder=2)
        ax.errorbar(x + off, means, yerr=np.array(spreads) / 2.0, fmt="none",
                    ecolor="#333333", lw=1.1, capsize=3, zorder=3)
        for xi, nd in zip(x + off, statedep):
            ax.text(xi, -7, f"{nd}/6", ha="center", va="top",
                    fontsize=8, color=colour)

    b1b = ladder["baselines"]["B1b_true_bah"]["summary"]["mean_delta_w"]
    ax.axhline(b1b, color=C_BAH, lw=1.4, ls="--", zorder=1)
    ax.text(-0.45, b1b + 3, f"buy-and-hold ${b1b:.0f}",
            color=C_BAH, fontsize=11.5, ha="left", va="bottom")
    ax.axhline(0, color="black", lw=0.8)

    ax.set_ylim(-26, 232)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean monthly wealth change ($)")
    ax.set_xlabel("State ladder rung (whiskers span the six seeds; "
                  "figures above the axis count state-dependent seeds)")
    ax.set_title("The ladder on validation: no rung separates from its neighbour")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", ncol=2, fontsize=11.5, framealpha=0.95)
    fig.tight_layout()
    save(fig, "finalv2_ladder")


# ----------------------------------------------------------------- figure 3
def chart_wealth_paths(final: dict) -> None:
    cell = final["results"]["S3_wealth"]
    months = final["test_months"]
    x = np.arange(len(months) + 1)

    def cum(per_month):
        return np.concatenate([[0.0], np.cumsum([m["delta_w"] for m in per_month])])

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=160)

    for key, label, colour, ls, lw in (
        ("B1b_true_bah", "B1b buy-and-hold", C_BAH, "-", 4.0),
        ("B1a_slice_bah", "B1a slice schedule", C_SLICE, "-", 2.2),
        ("B2_random", "B2 random legal", C_GREY, ":", 1.8),
    ):
        ax.plot(x, cum(cell["baselines"][key]["per_month"]), lw=lw,
                color=colour, ls=ls, label=label, alpha=0.9)

    styles = {"reinforce": ((0, (5, 2)), 1.9),
              "dqn": ("-", 2.2),
              "ppo": ((0, (1, 1.6)), 2.2)}
    for algo, (label, colour) in ALGO_STYLE.items():
        runs = cell["algos"][algo]["per_seed"]
        paths = np.vstack([cum(r["per_month"]) for r in runs])
        ax.fill_between(x, paths.min(axis=0), paths.max(axis=0),
                        color=colour, alpha=0.13, lw=0)
        order = np.argsort(paths[:, -1])
        median = paths[order[len(order) // 2]]
        ls, lw = styles[algo]
        ax.plot(x, median, lw=lw, ls=ls, color=colour,
                label=f"{label} (median seed)")

    ax.axhline(0, color="black", lw=0.8)
    ax.annotate("REINFORCE, PPO and the slice schedule\n"
                "coincide because they are the same policy:\n"
                "buy whenever a Buy is legal",
                xy=(16.0, 3220), xytext=(2.2, 3450),
                fontsize=11.5, color="#333333", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#cccccc", alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#777777", lw=1.0))
    ticks = x[::3]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["start"] + [months[i - 1] for i in ticks[1:]],
                       rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Cumulative wealth change over 24 months ($)")
    ax.set_title("Held-out months: cumulative profit, shading spans all six seeds")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              fontsize=11.5, frameon=False)
    fig.tight_layout()
    save(fig, "finalv2_wealth_paths")


# ----------------------------------------------------------------- figure 4
def chart_lambda(final: dict) -> None:
    trials = final["lambda_selection"]["trials"]
    lams = sorted(float(k) for k in trials)
    ret = [trials[f"{l}"]["mean_delta_w"]["mean"] for l in lams]
    expo = [trials[f"{l}"]["mean_exposure"]["mean"] for l in lams]
    dd = [trials[f"{l}"]["mdd_intra_mean"]["mean"] for l in lams]

    base_ret, base_expo = ret[0], expo[0]
    xt = np.arange(len(lams))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1), dpi=160)

    ax1.plot(xt, ret, "-o", color=C_DQN, lw=2.0, label="mean $\\Delta W$")
    ax1.axhline(0.5 * base_ret, color="#555555", lw=1.2, ls="--",
                label="return floor (half unpenalised)")
    ax1b = ax1.twinx()
    ax1b.plot(xt, expo, "-s", color=C_REINFORCE, lw=1.8, label="mean exposure")
    ax1b.axhline(0.5 * base_expo, color=C_REINFORCE, lw=1.0, ls=":",
                 label="exposure floor")
    ax1b.set_ylabel("Mean exposure", color=C_REINFORCE)
    ax1b.tick_params(axis="y", colors=C_REINFORCE)
    ax1b.set_ylim(0, 0.55)
    ax1.set_ylabel("Mean monthly wealth change ($)", color=C_DQN)
    ax1.tick_params(axis="y", colors=C_DQN)
    ax1.set_xticks(xt)
    ax1.set_xticklabels([("0\n(none)" if l == 0 else f"{l:g}") for l in lams])
    ax1.set_xlabel("Penalty weight $\\lambda$")
    ax1.set_title("Return and participation collapse together")
    ax1.grid(True, alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10,
               framealpha=0.95)

    sel = 0.10
    ax1.axvline(lams.index(sel), color="#333333", lw=1.0, alpha=0.6)
    ax1.text(lams.index(sel) - 0.14, max(ret) * 0.42, "selected",
             fontsize=10.5, rotation=90, va="center", ha="right")

    ax2.plot([0, base_expo * 1.08], [0, dd[0] * 1.08], color=C_BAH, lw=1.3,
             ls="--", zorder=1, label="proportional to the unpenalised agent")
    ax2.plot(expo, dd, "-o", color=C_DQN, lw=2.0, zorder=2,
             label="penalty sweep")
    offsets = {0.0: (-46, 4), 0.05: (8, -6), 0.1: (-44, 6), 0.25: (4, 10)}
    for l, e, d in zip(lams, expo, dd):
        if l not in offsets:
            continue
        ax2.annotate(f"$\\lambda$={l:g}", (e, d), textcoords="offset points",
                     xytext=offsets[l], fontsize=10.5, color="#444444")
    ax2.annotate("$\\lambda \\geq 0.5$: the agent\nnever opens a position",
                 xy=(0.006, 0.0004), xytext=(0.235, 0.0042),
                 fontsize=10.5, color="#444444",
                 arrowprops=dict(arrowstyle="->", color="#777777", lw=1.0))
    ax2.set_xlim(-0.02, 0.58)
    ax2.set_xlabel("Mean exposure")
    ax2.set_ylabel("Mean intra-episode drawdown")
    ax2.set_title("Drawdown falls only because exposure falls")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    save(fig, "finalv2_lambda_frontier")


# ----------------------------------------------------------------- figure 5
def chart_fees(fees: dict) -> None:
    bps = fees["fees_bps"]
    xt = np.arange(len(bps))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1), dpi=160)

    b1b = [fees["levels"][f"{int(b)}"]["baselines"]["B1b_true_bah"]
           ["summary"]["mean_delta_w"] for b in bps]
    ax1.plot(xt, b1b, "-*", color=C_BAH, lw=2.0, ms=11, label="B1b buy-and-hold")

    for algo in ("reinforce", "dqn"):
        label, colour = ALGO_STYLE[algo]
        means, spreads = [], []
        for b in bps:
            a = fees["levels"][f"{int(b)}"]["algos"][algo]["across_seeds"]
            means.append(a["mean_delta_w"]["mean"])
            spreads.append(a["mean_delta_w"]["spread"])
        means = np.array(means)
        spreads = np.array(spreads)
        ax1.fill_between(xt, means - spreads / 2, means + spreads / 2,
                         color=colour, alpha=0.14, lw=0)
        ax1.plot(xt, means, "-o", color=colour, lw=2.0, label=label)

    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(xt)
    ax1.set_xticklabels([f"{b:g}" for b in bps])
    ax1.set_xlabel("Transaction fee (basis points)")
    ax1.set_ylabel("Mean monthly wealth change ($)")
    ax1.set_title("Profit: every strategy earns less")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=11.5)

    for algo in ("reinforce", "dqn"):
        label, colour = ALGO_STYLE[algo]
        beh = fees["behavioural_response"][algo]
        ax2.plot(xt, beh["mean_trades"], "-o", color=colour, lw=2.0,
                 label=f"{label}"
                       f"{' (responds)' if beh['responds_beyond_noise'] else ' (no trend)'}")
    ax2.set_xticks(xt)
    ax2.set_xticklabels([f"{b:g}" for b in bps])
    ax2.set_xlabel("Transaction fee (basis points)")
    ax2.set_ylabel("Trades per month")
    ax2.set_title("Behaviour: only Deep Q responds")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=11.5)

    fig.suptitle("Fee sensitivity on validation, six seeds per point",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "finalv2_fees")


def main() -> None:
    final = load("final_results.json")
    risk = load("final_results_risk.json")
    ladder = load("ladder_results.json")
    fees = load("fee_results.json")

    chart_return_vs_exposure(final, risk)
    chart_ladder(ladder)
    chart_wealth_paths(final)
    chart_lambda(final)
    chart_fees(fees)


if __name__ == "__main__":
    main()
