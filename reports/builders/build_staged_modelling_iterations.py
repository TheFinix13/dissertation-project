#!/usr/bin/env python3
"""Render the staged-modelling iteration roadmap as a Word-ready PNG/PDF."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[1] / "generated" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ITERATIONS = [
    (
        "Iteration 1: Simple Baseline",
        [
            "Discrete Actions {Hold, Buy, Sell}",
            r"Binary Holdings  $h_t \in \{0, 1\}$",
            "Simple Step-Return Reward",
        ],
    ),
    (
        "Iteration 2: Multi-Sizing & Shorting",
        [
            r"Expanded Discrete Actions  $\{-2,-1,0,+1,+2\}$",
            r"Short Selling Support  ($h_t \in \{-1,0,1\}$)",
            "Volatility-Penalized Reward",
        ],
    ),
    (
        "Iteration 3: Continuous Action Sizing",
        [
            r"Continuous Action Space  $a_t \in [-1.0,+1.0]$",
            "Dynamic Portfolio Capital Allocation",
            "Feature Vector State (Technical Indicators)",
        ],
    ),
    (
        "Iteration 4: Multi-Asset Portfolio Allocation",
        [
            r"Vector Actions  $w_t = [w_1, w_2, \ldots, w_N]^\top$",
            "Asset Covariance Matrix in State Space",
            "Drawdown Penalties & Risk Constraints",
        ],
    ),
]


def render(*, dark: bool) -> None:
    if dark:
        bg, card, text, muted, accent, edge = (
            "#1e1e1e",
            "#2d2d2d",
            "#e8e8e8",
            "#a0a0a0",
            "#7ec8e3",
            "#3a3a3a",
        )
        suffix = "_dark"
    else:
        # Word / print friendly
        bg, card, text, muted, accent, edge = (
            "#ffffff",
            "#f4f6f8",
            "#1a1a1a",
            "#555555",
            "#1a5f7a",
            "#cfd6dd",
        )
        suffix = ""

    fig_h = 9.2
    fig, ax = plt.subplots(figsize=(7.2, fig_h), dpi=200)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")

    # Outer rounded frame (Gemini-style container)
    outer = FancyBboxPatch(
        (0.35, 0.4),
        9.3,
        19.2,
        boxstyle="round,pad=0.05,rounding_size=0.35",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=card,
    )
    ax.add_patch(outer)

    ax.text(
        5.0,
        19.05,
        "Staged modelling progression",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=text,
        fontfamily="monospace",
    )
    ax.text(
        5.0,
        18.45,
        "simple MDP  →  richer actions / state / risk",
        ha="center",
        va="top",
        fontsize=8.5,
        color=muted,
        fontfamily="monospace",
    )

    # Block geometry (top → bottom)
    block_top = 17.6
    block_h = 3.15
    gap = 0.85  # space for connector between blocks
    left, width = 1.0, 8.0

    for i, (title, bullets) in enumerate(ITERATIONS):
        y_top = block_top - i * (block_h + gap)
        y_bot = y_top - block_h

        box = FancyBboxPatch(
            (left, y_bot),
            width,
            block_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.0,
            edgecolor=accent if i == 0 else edge,
            facecolor=bg if not dark else "#252525",
        )
        ax.add_patch(box)

        ax.text(
            left + 0.25,
            y_top - 0.35,
            f"[ {title} ]",
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=accent if i == 0 else text,
            fontfamily="monospace",
        )

        for j, bullet in enumerate(bullets):
            ax.text(
                left + 0.45,
                y_top - 0.95 - j * 0.55,
                f"•  {bullet}",
                ha="left",
                va="top",
                fontsize=9,
                color=text,
                fontfamily="monospace",
            )

        # Connector to next block
        if i < len(ITERATIONS) - 1:
            cx = 5.0
            y1 = y_bot - 0.08
            y2 = y_bot - gap + 0.28
            ax.plot([cx, cx], [y1, y2], color=muted, lw=1.4, solid_capstyle="round")
            ax.annotate(
                "",
                xy=(cx, y_bot - gap + 0.12),
                xytext=(cx, y2),
                arrowprops=dict(arrowstyle="-|>", color=muted, lw=1.4),
            )

    ax.text(
        5.0,
        0.75,
        "Iteration 1 = Friday baseline   ·   later iterations earned one at a time",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=muted,
        fontfamily="monospace",
        style="italic",
    )

    fig.tight_layout(pad=0.3)
    stem = OUT_DIR / f"staged_modelling_iterations{suffix}"
    fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight", facecolor=bg)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"wrote {stem}.png")
    print(f"wrote {stem}.pdf")


if __name__ == "__main__":
    render(dark=False)
    render(dark=True)
