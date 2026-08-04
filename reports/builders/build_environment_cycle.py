#!/usr/bin/env python3
"""Render Environment Cycle (Policy ↔ TradingEnv) as Word-ready PNG/PDF.

Matches the Chapter 4 ASCII sketch: Policy Network π_θ(a|s) exchanges
Action a_t with TradingEnv (Gymnasium), which returns State s_{t+1} and
Reward R_t, with Market Data Stream nested inside the env.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "generated" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

FS_TITLE = 16
FS_BOX = 13
FS_SUB = 11
FS_NEST = 10.5
FS_ARROW = 11
LW = 2.2
ARR = 15
BOX_LW = 2.4


def draw(path_png: Path, path_pdf: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Outer dashed frame
    ax.add_patch(
        Rectangle(
            (0.35, 0.45), 9.3, 4.0,
            fill=False, linestyle=(0, (6, 4)), linewidth=1.6,
            edgecolor="#666666",
        )
    )
    ax.text(
        5.0, 4.65, "Environment Cycle",
        ha="center", va="center",
        fontsize=FS_TITLE, fontweight="bold", color="#1a1a1a",
        fontfamily="sans-serif",
    )

    # Policy box (left)
    ax.add_patch(
        FancyBboxPatch(
            (0.7, 1.35), 3.2, 2.5,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=BOX_LW, edgecolor="#1f4e79", facecolor="#e8f1fb",
        )
    )
    ax.text(
        2.3, 2.95, "Policy Network",
        ha="center", va="center",
        fontsize=FS_BOX, fontweight="bold", color="#1f4e79",
        fontfamily="sans-serif",
    )
    ax.text(
        2.3, 2.25, r"$\pi_\theta(a \mid s)$",
        ha="center", va="center",
        fontsize=FS_SUB + 2, color="#243447",
        fontfamily="sans-serif",
    )

    # TradingEnv box (right)
    ax.add_patch(
        FancyBboxPatch(
            (6.1, 1.35), 3.2, 2.5,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=BOX_LW, edgecolor="#1a6b3c", facecolor="#e8f6ee",
        )
    )
    ax.text(
        7.7, 3.25, "TradingEnv",
        ha="center", va="center",
        fontsize=FS_BOX, fontweight="bold", color="#1a6b3c",
        fontfamily="sans-serif",
    )
    ax.text(
        7.7, 2.85, "(Gymnasium)",
        ha="center", va="center",
        fontsize=FS_SUB, color="#2d5a3f",
        fontfamily="sans-serif",
    )

    # Nested Market Data Stream
    ax.add_patch(
        FancyBboxPatch(
            (6.45, 1.55), 2.5, 0.85,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6, edgecolor="#5a7a8a", facecolor="#f4f7f9",
        )
    )
    ax.text(
        7.7, 1.97, "Market Data Stream",
        ha="center", va="center",
        fontsize=FS_NEST, color="#334455",
        fontfamily="sans-serif", fontweight="medium",
    )

    # Top arrow: Action a_t (policy → env)
    ax.add_patch(
        FancyArrowPatch(
            (3.95, 3.2), (6.05, 3.2),
            arrowstyle="->", mutation_scale=ARR, lw=LW, color="#1f4e79",
            connectionstyle="arc3,rad=0",
        )
    )
    ax.text(
        5.0, 3.55, r"Action $a_t$",
        ha="center", va="center",
        fontsize=FS_ARROW, fontweight="bold", color="#1f4e79",
        fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#c5d4e8", linewidth=0.9),
    )

    # Bottom arrow: State + Reward (env → policy)
    ax.add_patch(
        FancyArrowPatch(
            (6.05, 2.0), (3.95, 2.0),
            arrowstyle="->", mutation_scale=ARR, lw=LW, color="#1a6b3c",
            connectionstyle="arc3,rad=0",
        )
    )
    ax.text(
        5.0, 1.45, r"State $s_{t+1}$, Reward $R_t$",
        ha="center", va="center",
        fontsize=FS_ARROW, fontweight="bold", color="#1a6b3c",
        fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#c5e0d0", linewidth=0.9),
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(path_png, dpi=220, bbox_inches="tight", facecolor="white")
    if path_pdf is not None:
        fig.savefig(path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path_png}")
    if path_pdf is not None:
        print(f"wrote {path_pdf}")


def main() -> None:
    draw(
        OUT / "environment_cycle.png",
        OUT / "environment_cycle.pdf",
    )
    # Also drop a copy under latex/tikz for easy Word paste next to other figs
    latex_png = Path(__file__).resolve().parents[2] / "latex" / "tikz" / "environment_cycle_word.png"
    latex_png.parent.mkdir(parents=True, exist_ok=True)
    draw(latex_png, None)


if __name__ == "__main__":
    main()
