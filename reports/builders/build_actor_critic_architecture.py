#!/usr/bin/env python3
"""Render Actor–Critic / PPO architectural flow as Word-ready PNG/PDF.

Default (light) is high-contrast for pasting onto a white Word page.
Dark variant is optional for screen/slides.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "generated" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

FS_TITLE = 17
FS_SUB = 11.5
FS_BOX = 13.5
FS_BOX_SUB = 11
FS_LABEL = 11
LW = 2.4
ARR = 16
BOX_LW = 2.6


def box(ax, x, y, w, h, title, subtitle=None, *, edge, face, title_c, sub_c):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=BOX_LW, edgecolor=edge, facecolor=face,
        )
    )
    if subtitle:
        ax.text(x, y + 0.30, title, ha="center", va="center",
                fontsize=FS_BOX, fontweight="bold", color=title_c, fontfamily="sans-serif")
        ax.text(x, y - 0.34, subtitle, ha="center", va="center",
                fontsize=FS_BOX_SUB, color=sub_c, fontfamily="monospace", fontweight="medium")
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=FS_BOX, fontweight="bold", color=title_c, fontfamily="sans-serif")


def v_arrow(ax, x, y1, y2, color, label_face, *, label=None, side="right", dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            (x, y1), (x, y2),
            arrowstyle="->", mutation_scale=ARR, lw=LW, color=color,
            linestyle=(0, (5, 3)) if dashed else "solid",
            shrinkA=1, shrinkB=1,
        )
    )
    if label:
        gap = 0.20
        ax.text(
            x + gap if side == "right" else x - gap,
            (y1 + y2) / 2,
            label,
            ha="left" if side == "right" else "right",
            va="center",
            fontsize=FS_LABEL,
            color=color,
            fontfamily="monospace",
            fontweight="bold",
            linespacing=1.3,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=label_face,
                edgecolor="#bbbbbb" if label_face != "#1a1a1a" else "#555555",
                linewidth=0.8,
                alpha=1.0,
            ),
        )


def diverge_from_source(ax, x_mid, y_src, x_left, x_right, y_dst, color):
    y_rail = (y_src + y_dst) / 2
    ax.plot([x_mid, x_mid], [y_src, y_rail], color=color, lw=LW, solid_capstyle="round")
    ax.plot([x_left, x_right], [y_rail, y_rail], color=color, lw=LW, solid_capstyle="round")
    for x in (x_left, x_right):
        ax.add_patch(
            FancyArrowPatch(
                (x, y_rail), (x, y_dst),
                arrowstyle="->", mutation_scale=ARR, lw=LW, color=color,
                shrinkA=0, shrinkB=2,
            )
        )


def render(*, dark: bool) -> None:
    if dark:
        # Screen / slide dark theme
        bg, face, edge, title_c, sub_c, accent, line, label_face = (
            "#121212", "#1f1f1f", "#e0e0e0", "#ffffff", "#e8e8e8",
            "#8fd3f0", "#e0e0e0", "#1a1a1a",
        )
        suffix = "_dark"
    else:
        # High-contrast Word / print theme (stays readable on white page)
        bg, face, edge, title_c, sub_c, accent, line, label_face = (
            "#ffffff", "#d9e6f2", "#0b2c4a", "#000000", "#111111",
            "#0b2c4a", "#1a1a1a", "#ffffff",
        )
        suffix = ""

    fig, ax = plt.subplots(figsize=(14.0, 15.5), dpi=260)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 32)
    ax.axis("off")

    ax.text(14, 31.2, "Actor–Critic architectural flow  (Iteration 1)",
            ha="center", va="top", fontsize=FS_TITLE, fontweight="bold",
            color=title_c, fontfamily="sans-serif")
    ax.text(14, 30.2,
            "How the MDP, policy π_θ and value estimator V_φ interact during training",
            ha="center", va="top", fontsize=FS_SUB, color=sub_c, fontfamily="sans-serif")

    xl, xr, xm = 7.0, 21.0, 14.0

    box(ax, xm, 28.2, 12.5, 1.7,
        "ENVIRONMENT", "(High-Frequency Stock Market)",
        edge=edge, face=face, title_c=title_c, sub_c=sub_c)

    v_arrow(ax, xm, 27.3, 25.85, line, label_face,
            label="State Vector\n$s_t = [\\Delta P_t,\\ h_t]^{\\top}$", side="right")

    box(ax, xm, 24.9, 12.5, 1.4,
        "AGENT STATE", None,
        edge=edge, face=face, title_c=title_c, sub_c=sub_c)

    diverge_from_source(ax, xm, 24.15, xl, xr, 22.35, line)

    box(ax, xl, 21.2, 8.4, 1.8,
        "POLICY NETWORK", r"($\pi_\theta$)",
        edge=edge, face=face, title_c=title_c, sub_c=sub_c)
    box(ax, xr, 21.2, 8.4, 1.8,
        "CRITIC NETWORK", r"(Value $V_\phi$)",
        edge=edge, face=face, title_c=title_c, sub_c=sub_c)

    v_arrow(ax, xl, 20.25, 18.55, line, label_face,
            label="Probability Vector\n$\\pi_\\theta(a_t\\mid s_t)$", side="right")
    v_arrow(ax, xr, 20.25, 18.55, line, label_face,
            label="Estimated Baseline\nValue $V_\\phi(s_t)$", side="right")

    box(ax, xl, 17.4, 8.4, 1.9,
        "ACTION SELECTION", r"$a_t \in \{\mathrm{Hold},\ \mathrm{Buy},\ \mathrm{Sell}\}$",
        edge=edge, face=face, title_c=title_c, sub_c=sub_c)
    box(ax, xr, 17.4, 9.2, 1.9,
        "ADVANTAGE CALCULATOR",
        r"$A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$",
        edge=edge, face=face, title_c=title_c, sub_c=sub_c)

    v_arrow(ax, xl, 16.4, 14.55, line, label_face,
            label="Executes Action $a_t$", side="right", dashed=True)

    box(ax, xl, 13.15, 9.0, 2.3,
        "REWARD MECHANISM",
        r"$r_t = (h_t\cdot\Delta P_{t+1}) - c\cdot|h_{t+1}-h_t|$",
        edge=edge, face=face, title_c=title_c, sub_c=sub_c)

    y_rail = 10.35
    ppo_top = 9.55

    ax.plot([xl, xl], [11.95, y_rail], color=line, lw=LW, solid_capstyle="round")
    ax.plot([xl, xm], [y_rail, y_rail], color=line, lw=LW, solid_capstyle="round")
    ax.text(
        xl + 0.28, (11.95 + y_rail) / 2,
        "Returns Reward $r_t$\n& Next State $s_{t+1}$",
        ha="left", va="center", fontsize=FS_LABEL, color=line,
        fontfamily="monospace", fontweight="bold", linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=label_face,
                  edgecolor="#bbbbbb", linewidth=0.8, alpha=1.0),
    )

    ax.plot([xr, xr], [16.4, y_rail], color=line, lw=LW, linestyle=(0, (5, 3)),
            solid_capstyle="round")
    ax.plot([xr, xm], [y_rail, y_rail], color=line, lw=LW, solid_capstyle="round")

    ax.add_patch(
        FancyArrowPatch(
            (xm, y_rail), (xm, ppo_top),
            arrowstyle="->", mutation_scale=ARR, lw=LW, color=line,
            shrinkA=0, shrinkB=2,
        )
    )

    box(ax, xm, 8.2, 12.0, 2.2,
        "PPO LOSS  &  GRADIENT UPDATE",
        r"$\theta_{e+1} \leftarrow \theta_e + \alpha\,\nabla_\theta L$",
        edge=edge, face=face, title_c=title_c, sub_c=sub_c)

    ax.text(xm, 6.0,
            "Left path = decide & act   ·   Right path = score how good that was   ·   Bottom = learn",
            ha="center", va="center", fontsize=10.5, color=sub_c, fontfamily="sans-serif",
            style="italic")
    ax.text(xm, 5.1,
            "Matches Algorithm 1 (Iteration 1): discrete actions, binary holding, one stock",
            ha="center", va="center", fontsize=10, color=sub_c, fontfamily="monospace")

    stem = OUT / f"actor_critic_architecture{suffix}"
    fig.tight_layout(pad=0.35)
    # High DPI so Word doesn't soft-blur when scaled
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor=bg)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"wrote {stem}.png")
    print(f"wrote {stem}.pdf")


if __name__ == "__main__":
    render(dark=False)
    render(dark=True)
