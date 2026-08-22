#!/usr/bin/env python3
"""Render the linear Algorithm-1 ASCII block as a Word-ready PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "generated" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

LINES = [
    "ALGORITHM 1: High-Frequency Baseline Discrete Trading Agent",
    "            (policy-gradient update; PPO is one option)",
    "=" * 72,
    "Input  : Historical prices {P_t}, fee c, epochs E, days-per-epoch K",
    "Output : Optimised policy parameters θ*",
    "",
    " 1.  Initialise policy θ and value φ randomly",
    " 2.  FOR epoch = 1 to E DO",
    " 3.      FOR each trading day (episode) k = 1 to K DO",
    " 4.          Initialise state  s_0 = [ΔP_0, h_0 = 0]ᵀ",
    " 5.          Initialise buffer B = {}",
    " 6.          FOR minute t = 0 to T−1  (T = 390) DO",
    " 7.              Compute π_θ(a_t | s_t)",
    " 8.              Sample a_t ~ π_θ   where a_t ∈ {0:Hold, 1:Buy, 2:Sell}",
    " 9.              IF a_t = 1 THEN h_{t+1} = 1     (Buy / Long)",
    "10.              IF a_t = 2 THEN h_{t+1} = 0     (Sell / Cash)",
    "11.              IF a_t = 0 THEN h_{t+1} = h_t   (Hold)",
    "12.              Observe  ΔP_{t+1} = (P_{t+1} − P_t) / P_t",
    "13.              Reward   r_t = (h_t · ΔP_{t+1}) − c · |h_{t+1} − h_t|",
    "14.              Next state s_{t+1} = [ΔP_{t+1}, h_{t+1}]ᵀ",
    "15.              Store (s_t, a_t, r_t, s_{t+1}) in B",
    "16.          END FOR",
    "17.      END FOR",
    "18.      Update θ by policy gradient / PPO loss on B",
    "19.      Update value network φ",
    "20.  END FOR",
    "21.  RETURN θ*",
    "=" * 72,
    "Iteration 1 only — 3 actions, binary holding, one stock, one day = one trajectory",
]


def main() -> None:
    bg, card, text, muted, accent = (
        "#ffffff",
        "#f7f8fa",
        "#1a1a1a",
        "#555555",
        "#1a5f7a",
    )
    n = len(LINES)
    fig_h = max(8.5, 0.28 * n + 1.2)
    fig, ax = plt.subplots(figsize=(9.0, fig_h), dpi=200)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 2)
    ax.axis("off")

    box = FancyBboxPatch(
        (0.25, 0.4),
        9.5,
        n + 1.2,
        boxstyle="round,pad=0.04,rounding_size=0.2",
        linewidth=1.1,
        edgecolor="#cfd6dd",
        facecolor=card,
    )
    ax.add_patch(box)

    y = n + 0.9
    for i, line in enumerate(LINES):
        color = accent if i < 2 or line.startswith("=") else text
        weight = "bold" if i < 2 or line.startswith("ALGORITHM") else "normal"
        if line.startswith("Iteration 1"):
            color, weight = muted, "normal"
        ax.text(
            0.55,
            y - i * 1.0,
            line,
            ha="left",
            va="top",
            fontsize=8.2,
            color=color,
            fontweight=weight,
            fontfamily="monospace",
        )

    stem = OUT / "algorithm1_baseline_discrete_ascii"
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight", facecolor=bg)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"wrote {stem}.png")
    print(f"wrote {stem}.pdf")


if __name__ == "__main__":
    main()
