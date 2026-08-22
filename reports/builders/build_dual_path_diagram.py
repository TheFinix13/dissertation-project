#!/usr/bin/env python3
"""Render the dual-path architecture diagram for Word / slides insertion.

Each node combines plain-language labels (image 2) with formulas (image 1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "generated" / "charts"


def _box(
    ax,
    xy,
    w,
    h,
    title,
    subtitle=None,
    fc="#ffffff",
    ec="#333333",
    title_size=9,
    sub_size=8.5,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            weight="bold",
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=sub_size,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            weight="bold",
        )
    return patch


def _arrow(ax, start, end, text=None, rad=0.0, fontsize=8, text_offset=0.03):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="#333333",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + text_offset, text, ha="center", va="bottom", fontsize=fontsize)


def build_diagram() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bw = 0.44

    # --- Forecaster chain (centre top) ---
    _box(ax, (0.28, 0.90), bw, 0.06, "Probabilistic LSTM Forecaster", fc="#dbeafe")
    _box(
        ax,
        (0.28, 0.81),
        bw,
        0.07,
        "Predictive variance",
        r"$\hat{\sigma}_t^2$",
        fc="#eff6ff",
    )
    _box(
        ax,
        (0.28, 0.71),
        bw,
        0.08,
        "Daily uncertainty score",
        r"Window normalization: $u_t \in [0,1]$",
        fc="#eff6ff",
    )

    _arrow(ax, (0.5, 0.90), (0.5, 0.88))
    _arrow(ax, (0.5, 0.81), (0.5, 0.79))
    _arrow(ax, (0.5, 0.71), (0.5, 0.66))

    # --- Path 1 (left) ---
    _box(
        ax,
        (0.04, 0.52),
        0.38,
        0.09,
        "Path 1: State context",
        r"Visible in agent's state: $s_t = (\ldots,\, u_t)$",
        fc="#ffedd5",
    )
    _box(
        ax,
        (0.06, 0.40),
        0.34,
        0.07,
        "Trading agent (PPO)",
        fc="#fef9c3",
    )

    _arrow(ax, (0.42, 0.71), (0.23, 0.61), rad=0.18)
    _arrow(ax, (0.23, 0.52), (0.23, 0.47))

    # --- Path 2 (right) ---
    _box(
        ax,
        (0.58, 0.52),
        0.38,
        0.09,
        "Path 2: Execution rules",
        r"Modifies realised trade size $v_t$",
        fc="#ede9fe",
    )
    _box(
        ax,
        (0.60, 0.39),
        0.34,
        0.08,
        "Soft dial: shrinks trade size",
        r"$\max(1 - u_t,\, s_{\min})$",
        fc="#fee2e2",
    )
    _box(
        ax,
        (0.60, 0.27),
        0.34,
        0.08,
        "Hard guard: blocks new buys",
        r"$\mathbf{1}(a_t \leq 0 \text{ or } u_t < \tau)$",
        fc="#fee2e2",
    )

    _arrow(ax, (0.58, 0.71), (0.77, 0.61), rad=-0.18)
    _arrow(ax, (0.77, 0.52), (0.77, 0.47))
    _arrow(ax, (0.77, 0.39), (0.77, 0.35))

    # PPO → execution
    _arrow(
        ax,
        (0.40, 0.435),
        (0.60, 0.31),
        text="Proposes trade $a_t$",
        rad=-0.10,
        fontsize=8.5,
    )

    # --- Output ---
    _box(
        ax,
        (0.28, 0.12),
        bw,
        0.08,
        "Realised market trade",
        r"$v_t$",
        fc="#dcfce7",
    )
    _arrow(ax, (0.77, 0.27), (0.62, 0.20), rad=0.14)
    _arrow(ax, (0.23, 0.40), (0.38, 0.20), rad=-0.14)

    ax.text(
        0.5,
        0.03,
        "Path 1 lets PPO learn from uncertainty; Path 2 enforces caution at execution.\n"
        r"Soft dial shrinks every trade; hard guard blocks new buys when $u_t \geq \tau$; "
        "sells are never blocked.",
        ha="center",
        va="center",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )

    fig.tight_layout()
    png = OUT_DIR / "dual_path_architecture.png"
    pdf = OUT_DIR / "dual_path_architecture.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    build_diagram()
