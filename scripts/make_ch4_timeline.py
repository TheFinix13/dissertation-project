#!/usr/bin/env python3
"""Generate Figure 4.1: how the two studies use the real SPY months.

One timeline (2006-2026), two rows. The first study trains, validates,
and tests inside 2018-2025. The main study reads only 2018-2022 (to
calibrate the generator) and keeps every other real month as its
transfer test; its training and validation data are simulated.

Output: latex/dissertation/figs5/fig4_timeline.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "dissertation" / "figs5"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "figure.dpi": 150,
})

C_TRAIN = "#aec7e8"   # training / calibration months
C_VAL = "#ffdd8f"     # validation months
C_TEST = "#b5e2b5"    # held-out test / transfer months
C_UNUSED = "#f0f0f0"  # not used

BAR_H = 0.52


def block(ax, y, x0, x1, color, label, sub="", hatch=None):
    ax.add_patch(Rectangle((x0, y - BAR_H / 2), x1 - x0, BAR_H,
                           facecolor=color, edgecolor="#555", lw=0.6,
                           hatch=hatch))
    xm = (x0 + x1) / 2
    ax.text(xm, y + 0.09, label, ha="center", va="center",
            fontsize=7.6, fontweight="bold")
    if sub:
        ax.text(xm, y - 0.13, sub, ha="center", va="center", fontsize=6.8)


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 1.25))

    y = 0.0

    # --- main study ----------------------------------------------------
    block(ax, y, 2006, 2018, C_TEST, "transfer test: 144 months",
          "incl. the 2008 crisis")
    block(ax, y, 2018, 2023, C_TRAIN, "generator calibration",
          "no other use")
    block(ax, y, 2023, 2026, C_TEST, "transfer test", "36 months")

    for yr in (2006, 2018, 2023, 2026):
        ax.axvline(yr, color="#999", lw=0.4, ls=":", zorder=0)
        ax.text(yr, -0.62, str(yr), ha="center", fontsize=7.4, color="#444")

    ax.set_xlim(2005.5, 2026.5)
    ax.set_ylim(-0.85, 0.45)
    ax.axis("off")
    fig.tight_layout()
    out = OUT / "fig4_timeline.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
