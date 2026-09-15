"""Render every diagram used in the viva PPTX.

Design brief: examiners are ML researchers, not RL specialists, and know
no finance. Every slide gets one big picture that carries the idea; the
text on the slide is a single caption line. This script produces those
pictures.

Output: PNGs in latex/viva/figs/ at 220 dpi (crisp on projectors).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "viva" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# --- shared style --------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelweight": "bold",
    "figure.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

C = {
    "dqn": "#e07a1f",
    "reinforce": "#20807d",
    "ppo": "#6f42c1",
    "bh": "#3a3a3a",
    "ab": "#9a9a9a",
    "green": "#2a9d3d",
    "red": "#c0392b",
    "blue": "#1f5aa8",
    "bg": "#f7f4ee",
    "grid": "#dcd6c8",
}


def save(name: str) -> Path:
    p = OUT / name
    plt.savefig(p)
    plt.close()
    return p


# --- slide 3: clean RL loop diagram --------------------------------------
def slide3_rl_loop():
    """Clean agent-environment loop, self-contained, no external callouts."""
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # two boxes, narrower, further apart so arrows are long
    def box(x, y, w, h, colour, head, body):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=colour, edgecolor="none"))
        ax.text(x + w / 2, y + h - 0.42, head, ha="center", va="top",
                fontsize=17, fontweight="bold", color="white")
        ax.text(x + w / 2, y + h * 0.42, body, ha="center", va="center",
                fontsize=12, color="white", linespacing=1.5)

    box(0.5, 1.9, 3.7, 2.7, C["blue"], "AGENT",
        "A small neural\nnetwork that outputs\nBuy, Hold, or Sell.")
    box(7.8, 1.9, 3.7, 2.7, "#4a6b3a", "ENVIRONMENT",
        "One month of daily\nprices. Executes the\naction, applies fees.")

    # top arrow with wide gap for labels
    ax.annotate("", xy=(7.75, 4.30), xytext=(4.25, 4.30),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.6",
                                color=C["ppo"], lw=3, mutation_scale=20))
    ax.text(6.0, 4.85, "ACTION",
            ha="center", fontsize=14, fontweight="bold", color=C["ppo"])
    ax.text(6.0, 4.05, "Buy · Hold · Sell",
            ha="center", fontsize=11, color="#555", style="italic")

    # bottom arrow going the other way
    ax.annotate("", xy=(4.25, 2.20), xytext=(7.75, 2.20),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.6",
                                color=C["dqn"], lw=3, mutation_scale=20))
    ax.text(6.0, 2.75, "STATE  +  REWARD",
            ha="center", fontsize=14, fontweight="bold", color=C["dqn"])
    ax.text(6.0, 2.42, "market signal  +  wealth change",
            ha="center", fontsize=10.5, color="#555", style="italic")

    # bottom takeaway strip — short enough to fit
    ax.add_patch(mpatches.FancyBboxPatch(
        (2.0, 0.15), 8.0, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#111", edgecolor="none"))
    ax.text(6, 0.60,
            "The agent is never shown the right answer — only the reward.",
            ha="center", va="center", fontsize=13, color="white", fontweight="bold")

    # small header above
    ax.text(6, 5.55, "One month = about 21 of these loops.",
            ha="center", fontsize=12, color="#666", style="italic")

    save("slide3_rl_loop.png")


# --- slide 3 (v2): the dissertation loop, redrawn with human wording -----
def slide3_env_loop():
    """Same layout AND serif font as the Chapter 3 figure (environment top,
    agent bottom, curved arrows, numbered side notes), plain-language text,
    with the reward formula and the dissertation's closing line."""
    SERIF = "STIXGeneral"  # closest bundled match to the dissertation's Times
    with plt.rc_context({"mathtext.fontset": "stix"}):
        fig, ax = plt.subplots(figsize=(13, 6.0))
        ax.set_xlim(0, 13)
        ax.set_ylim(-0.75, 5.6)
        ax.axis("off")

        def box(cx, cy, w, h, face, edge, head, body):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                boxstyle="round,pad=0.03,rounding_size=0.10",
                facecolor=face, edgecolor=edge, linewidth=1.4))
            # clear gap between the bold header and the body lines
            ax.text(cx, cy + h / 2 - 0.16, head, ha="center", va="top",
                    fontsize=15.5, fontweight="bold", color="#1a1a1a",
                    family=SERIF)
            ax.text(cx, cy - 0.32, body, ha="center", va="center",
                    fontsize=11.5, color="#1a1a1a", linespacing=1.55,
                    family=SERIF)

        # environment on top, agent below — same as the dissertation figure
        box(6.5, 4.55, 5.6, 1.5, "#e4e4f7", "#8888bb",
            "Trading Environment",
            "One month of daily market prices, plus the\ncash balance, holdings and trading fees")
        box(6.5, 1.45, 6.0, 1.5, "#fbf7d8", "#b8a94a",
            "Trading Agent",
            "A small neural network (policy \u03c0 or Q-network)\nLooks at the current state, then picks: Hold, Buy or Sell")

        # curved arrows: left = state (env -> agent), right = action (agent -> env)
        ax.annotate("", xy=(4.2, 1.85), xytext=(3.6, 4.30),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.5",
                                    color="#333", lw=2.2,
                                    connectionstyle="arc3,rad=0.35"))
        ax.annotate("", xy=(9.4, 4.30), xytext=(8.8, 1.85),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.5",
                                    color="#333", lw=2.2,
                                    connectionstyle="arc3,rad=0.35"))

        # numbered side notes, human wording, serif
        ax.text(1.55, 3.0,
                "\u2460 The state\nWhat the agent can see right now:\nmarket signals, its own portfolio,\ntime left in the month, and which\nactions are currently allowed",
                ha="center", va="center", fontsize=11.5, color="#1a1a1a",
                linespacing=1.55, family=SERIF)
        ax.text(11.35, 3.0,
                "\u2461 The action\nHold, Buy or Sell.\nThe trade happens, the fee is\npaid, and the market moves\none day forward",
                ha="center", va="center", fontsize=11.5, color="#1a1a1a",
                linespacing=1.55, family=SERIF)

        # reward block — centred under the agent box, like the dissertation:
        # formula first, then the human line, then the PDF's closing line as is
        ax.text(6.5, 0.28,
                "\u2462 Reward:  $r_t = W_{t+1} - W_t$",
                ha="center", va="center", fontsize=12.5, color="#1a1a1a",
                family=SERIF)
        ax.text(6.5, -0.36,
                "How much the portfolio's wealth changed after fees;\n"
                "the model learns from this outcome, updates $\\theta$ or "
                "$\\phi$ and the process repeats",
                ha="center", va="center", fontsize=11.5, color="#1a1a1a",
                linespacing=1.5, family=SERIF)

        save("slide3_env_loop.png")


# --- slide 2: the everyday decision --------------------------------------
def slide2_decision():
    """A stock chart with three annotated moments: buy, hold, sell."""
    rng = np.random.default_rng(3)
    n = 90
    ret = rng.normal(0.001, 0.012, n)
    ret[25:32] -= 0.008  # dip
    ret[55:70] += 0.006  # rally
    price = 100 * np.exp(np.cumsum(ret))
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(x, price, color=C["blue"], lw=2.2)
    ax.fill_between(x, price.min() - 2, price, color=C["blue"], alpha=0.08)

    # three annotated decision points
    for xi, label, colour in [
        (18, "BUY", C["green"]),
        (45, "HOLD", "#666666"),
        (74, "SELL", C["red"]),
    ]:
        y = price[xi]
        ax.scatter([xi], [y], s=180, color=colour, zorder=5, edgecolor="white", lw=2)
        ax.annotate(
            label,
            xy=(xi, y),
            xytext=(xi, y + 5.5),
            ha="center",
            fontsize=17,
            fontweight="bold",
            color=colour,
            arrowprops=dict(arrowstyle="-", color=colour, lw=1.5),
        )

    ax.set_xlabel("Trading days")
    ax.set_ylabel("Share price ($)")
    ax.set_xticks([])
    ax.set_ylim(price.min() - 3, price.max() + 9)
    save("slide2_decision.png")


# --- slide 4: three algorithms as boxes ----------------------------------
def slide4_algorithms():
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.4, C["reinforce"], "REINFORCE",
         "Learns the POLICY directly.",
         "Plays a full month, then looks\nat the outcome. Actions that led\nto profit become more likely;\nactions that led to loss\nbecome less likely.",
         "Simple and transparent, but noisy —\none lucky month can teach bad habits.\nClassic 1992 baseline, built from scratch."),
        (4.2, C["dqn"], "Deep Q-learning",
         "Learns action VALUES.",
         "For each situation, estimates\nhow much Buy, Hold and Sell\nare each worth in dollars —\nthen picks the most\nvaluable one.",
         "Keeps a memory of past days and\nre-learns from them over and over.\nBuilt from scratch."),
        (8.0, C["ppo"], "PPO",
         "Learns the POLICY, safely.",
         "Same idea as REINFORCE, but\nevery update is capped — the\npolicy can only change by a\nsmall, controlled amount\nat a time.",
         "The cap stops one bad batch from\nwrecking what has been learned.\nModern default, standard library."),
    ]

    for x, colour, name, kind, how, note in boxes:
        # header bar
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 4.4), 3.6, 1.15, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=colour, edgecolor="none"))
        ax.text(x + 1.8, 4.98, name, ha="center", va="center",
                fontsize=17, fontweight="bold", color="white")
        # body
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.2), 3.6, 4.15, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="white", edgecolor=colour, linewidth=2))
        ax.text(x + 1.8, 3.95, kind, ha="center", va="top",
                fontsize=12.5, fontweight="bold", color=colour)
        ax.text(x + 1.8, 3.40, how, ha="center", va="top",
                fontsize=11, color="#333333", linespacing=1.45)
        ax.text(x + 1.8, 1.28, note, ha="center", va="top",
                fontsize=9.5, style="italic", color="#666666",
                linespacing=1.4)

    save("slide4_algorithms.png")


# --- slide 5: rules of the experiment (two aligned timelines) ------------
def slide5_rules():
    """Two stacked timelines sharing a 2006-2025 year axis.

    Top panel: first study (real data) — TRAIN 2018-22, VAL 2023, TEST 2024-25.
    Bottom panel: main study (simulator + transfer) — CALIBRATE 2018-22,
    with a simulator side panel and TRANSFER TEST bands covering every real
    month outside the calibrate window (2006-17 and 2023-25).
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12.5, 5.4), sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.55},
    )

    year_lo, year_hi = 2006.0, 2026.0  # bands span the full axis, no stubs
    palette = {
        "unused":    "#dedbd2",
        "train":     C["blue"],
        "val":       "#e0a020",
        "test":      "#2a9d3d",
        "calibrate": C["blue"],
        "transfer":  "#4a4a4a",
    }

    def draw_panel(ax, title, bands, calibrate_dashed=False,
                   side_panel_text=None):
        for start, end, kind, label in bands:
            colour = palette[kind]
            rect = mpatches.FancyBboxPatch(
                (start, 0.35), end - start, 0.55,
                boxstyle="round,pad=0.0,rounding_size=0.03",
                facecolor=colour, edgecolor="none",
                linewidth=0,
            )
            ax.add_patch(rect)
            if kind == "calibrate" and calibrate_dashed:
                # dashed border to signal "not training data itself"
                outline = mpatches.Rectangle(
                    (start, 0.35), end - start, 0.55,
                    facecolor="none", edgecolor="white",
                    linewidth=1.8, linestyle=(0, (4, 3)),
                )
                ax.add_patch(outline)
            if label:
                ax.text(
                    (start + end) / 2, 0.625, label,
                    ha="center", va="center",
                    fontsize=11 if (end - start) > 1.8 else 9.5,
                    fontweight="bold", color="white",
                )
        # panel title above the timeline
        ax.text(year_lo, 1.28, title, ha="left", va="center",
                fontsize=13.5, fontweight="bold", color="#1f335a")
        # optional side panel (simulator description)
        if side_panel_text:
            # place above the CALIBRATE band, well right of the panel title
            panel_left = 2013.5
            panel_w = 4.2
            box = mpatches.FancyBboxPatch(
                (panel_left, 1.05), panel_w, 0.55,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor="white", edgecolor=C["dqn"], linewidth=1.6,
            )
            ax.add_patch(box)
            ax.text(panel_left + panel_w / 2, 1.32, side_panel_text,
                    ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=C["dqn"])
            # arrow from the panel down to the middle of the CALIBRATE band
            ax.annotate(
                "", xy=(2020.5, 0.92), xytext=(panel_left + panel_w * 0.7, 1.05),
                arrowprops=dict(arrowstyle="->", color=C["dqn"], lw=1.4),
            )
        ax.set_ylim(0.0, 1.75)
        ax.set_xlim(year_lo, year_hi)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0)

    # Top — first study: its own proportional bar, no year axis.
    # Just TRAIN / VAL / TEST in time order, sized by month counts.
    ax_top.set_xlim(0, 96)
    ax_top.set_ylim(0.0, 1.75)
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    top_bands = [
        (0, 60, palette["train"], "TRAIN · 60 months", "earliest data"),
        (60, 72, palette["val"], "VAL · 12 months", ""),
        (72, 96, palette["test"], "TEST · 24 months", "most recent data"),
    ]
    for start, end, colour, label, under in top_bands:
        ax_top.add_patch(mpatches.FancyBboxPatch(
            (start, 0.35), end - start, 0.55,
            boxstyle="round,pad=0.0,rounding_size=0.03",
            facecolor=colour, edgecolor="none"))
        ax_top.text((start + end) / 2, 0.625, label,
                    ha="center", va="center",
                    fontsize=11 if (end - start) > 14 else 9.5,
                    fontweight="bold", color="white")
        if under:
            ax_top.text((start + end) / 2, 0.16, under,
                        ha="center", va="center", fontsize=9.5,
                        color="#777", style="italic")
    # arrow showing time order
    ax_top.annotate("", xy=(96, 1.06), xytext=(0, 1.06),
                    arrowprops=dict(arrowstyle="->", color="#999", lw=1.2))
    ax_top.text(48, 1.18, "time \u2192  (strictly chronological, no shuffling)",
                ha="center", va="bottom", fontsize=10, color="#777",
                style="italic")
    ax_top.text(0, 1.50, "First study — real SPY market data",
                ha="left", va="center", fontsize=13.5,
                fontweight="bold", color="#1f335a")

    # Bottom — main study (simulator + transfer test)
    draw_panel(
        ax_bot,
        "Main study — simulator + transfer test",
        [
            (2006.0, 2018.0, "transfer",  "TRANSFER TEST · 144 mo  (+36 mo after 2022 = 180)"),
            (2018.0, 2023.0, "calibrate", "CALIBRATE\n60 mo"),
            (2023.0, 2026.0, "transfer",  "TRANSFER TEST\n36 mo"),
        ],
        calibrate_dashed=True,
        side_panel_text="TRAIN on simulator\n(3,000 balanced episodes)",
    )

    # Shared year axis on the bottom panel only
    year_ticks = [2006, 2010, 2014, 2018, 2022, 2025]
    ax_bot.set_xticks(year_ticks)
    ax_bot.set_xticklabels([str(y) for y in year_ticks], fontsize=11)
    ax_bot.tick_params(axis="x", pad=4)

    # Small legend for the transfer / calibrate colours
    legend_handles = [
        mpatches.Patch(facecolor=palette["train"], label="TRAIN"),
        mpatches.Patch(facecolor=palette["val"], label="VAL"),
        mpatches.Patch(facecolor=palette["test"], label="TEST"),
        mpatches.Patch(facecolor=palette["calibrate"], label="CALIBRATE (simulator fit)"),
        mpatches.Patch(facecolor=palette["transfer"], label="TRANSFER TEST (real months, never trained on)"),
    ]
    fig.legend(
        handles=legend_handles, ncol=3,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        frameon=False, fontsize=10.5,
    )

    fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.14)
    save("slide5_rules.png")


# --- slide 6: first result bar chart -------------------------------------
def slide6_first_result():
    labels = ["Buy-and-hold\n(benchmark)", "Always-buy\n(no learning)",
              "REINFORCE", "PPO", "Deep Q-learning"]
    vals = [180.25, 179.75, 171.51, 127.35, 74.68]
    colours = [C["bh"], C["ab"], C["reinforce"], C["ppo"], C["dqn"]]
    state_dep = ["", "", "0 / 6", "0 / 6", "6 / 6"]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.20)
    bars = ax.bar(labels, vals, color=colours, edgecolor="white",
                  linewidth=1.5, width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 7,
                f"${v:.2f}", ha="center", fontsize=13.5, fontweight="bold")

    # state-dependence annotation as a separate row below the x-axis labels
    for i, sd in enumerate(state_dep):
        if sd:
            # "6 / 6" → green (fully state-dependent); "0 / 6" → red
            colour = C["green"] if sd.startswith("6") else C["red"]
            ax.annotate(sd, xy=(i, 0), xytext=(0, -62),
                        textcoords="offset points",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color=colour)
        else:
            ax.annotate("—", xy=(i, 0), xytext=(0, -62),
                        textcoords="offset points",
                        ha="center", va="center", fontsize=13, color="#999")
    fig.text(0.02, 0.10, "policy depends\non state?", fontsize=10.5,
             color="#555", fontweight="bold", ha="left", va="center")

    ax.axhline(180.25, color=C["bh"], lw=1, ls="--", alpha=0.35)
    ax.set_ylabel("$ earned per month  (test 2024–2025)", fontsize=12)
    ax.set_ylim(0, 220)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=8, labelsize=10.5)
    save("slide6_first_result.png")


# --- slide 7: the behavioural test ---------------------------------------
def slide7_state_test():
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # two "moments" — same mask, different states
    def moment(x, market_up, action, colour, label):
        # market box
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 4.0), 3.8, 1.5, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=C["bg"], edgecolor="#888", linewidth=1.2))
        ax.text(x + 1.9, 5.15, label, ha="center", fontsize=13,
                fontweight="bold", color="#333")
        arrow = "↑ RISING" if market_up else "↓ FALLING"
        acolour = C["green"] if market_up else C["red"]
        ax.text(x + 1.9, 4.5, arrow, ha="center", fontsize=15,
                fontweight="bold", color=acolour)
        # mask
        ax.text(x + 1.9, 3.5, "same actions available:\nBuy · Hold · Sell",
                ha="center", fontsize=11, color="#666")
        # agent decision
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + 0.6, 1.4), 2.6, 1.4, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=colour, edgecolor="none"))
        ax.text(x + 1.9, 2.1, action, ha="center", va="center",
                fontsize=17, fontweight="bold", color="white")
        ax.annotate("", xy=(x + 1.9, 2.85), xytext=(x + 1.9, 3.4),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.5))

    moment(0.6, True, "BUY", C["green"], "Moment A")
    moment(7.6, False, "SELL", C["red"], "Moment B")

    # verdict
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.6, 0.05), 10.8, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#111", edgecolor="none"))
    ax.text(6, 0.52,
            "Same mask, different action  ⇒  the agent is reading its state.",
            ha="center", va="center", fontsize=15, color="white", fontweight="bold")

    save("slide7_state_test.png")


# --- slides 7/8: the diagnosis panels, one per study ----------------------
def _diagnosis_panel(ax, episodes, params, title, verdict, ok):
    colour = C["green"] if ok else C["red"]
    ax.bar([0], [episodes], width=0.55, color=C["blue"])
    ax.bar([1], [params], width=0.55, color=colour)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["training episodes", "model parameters"], fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(20, 60000)
    for xi, v in zip([0, 1], [episodes, params]):
        ax.text(xi, v * 1.25, f"{v:,}", ha="center",
                fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=14, pad=8)
    ax.text(0.5, -0.24, verdict, transform=ax.transAxes,
            ha="center", fontsize=12.5, fontweight="bold", color=colour)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def slide7_diagnosis_first():
    """First study only: 60 episodes vs 18,000 parameters, with the
    MLP architecture spelled out so the 18,000 is not magic."""
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    _diagnosis_panel(ax, 60, 18000, "First study (real data)",
                     "Far too few examples to train 18,000 weights.\n"
                     "The model memorises instead of learning.", False)
    # where the 18,000 comes from
    ax.text(0.5, 1.13,
            "multi-layer perceptron: 9 inputs \u2192 128 \u2192 128 \u2192 "
            "3 actions \u2248 18,000 trainable weights",
            transform=ax.transAxes, ha="center", fontsize=10.5,
            color="#555", style="italic")
    plt.subplots_adjust(bottom=0.18, top=0.82)
    save("slide7_diagnosis_first.png")


def slide8_diagnosis_main():
    """Main study only: 3,000 episodes vs 2,800 parameters."""
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    _diagnosis_panel(ax, 3000, 2800, "Main study (simulator)",
                     "3,000 examples comfortably cover 2,800 weights.\n"
                     "The model has to learn a general rule.", True)
    ax.text(0.5, 1.13,
            "MLP cut to 9 inputs \u2192 64 \u2192 32 \u2192 3 actions "
            "\u2248 2,800 trainable weights",
            transform=ax.transAxes, ha="center", fontsize=10.5,
            color="#555", style="italic")
    plt.subplots_adjust(bottom=0.18, top=0.82)
    save("slide8_diagnosis_main.png")


# --- slide 9: the simulator ----------------------------------------------
def slide9_simulator():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=False)
    rng = np.random.default_rng(11)
    specs = [
        ("UP",  0.00244, 0.00967, C["green"], "rising"),
        ("DOWN", -0.00358, 0.02000, C["red"], "falling"),
        ("FLAT", 0.00012, 0.01095, "#888888", "sideways"),
    ]
    for ax, (name, mu, sig, colour, sub) in zip(axes, specs):
        for _ in range(6):
            r = rng.normal(mu, sig, 21)
            p = 100 * np.exp(np.cumsum(r))
            ax.plot(p, color=colour, alpha=0.35, lw=1.4)
        ax.set_title(f"{name}\n{sub} market", fontsize=14, fontweight="bold",
                     color=colour, pad=6)
        ax.set_xlabel("day", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # no suptitle — the on-slide text carries this line so the images
    # stay clear of any text row
    save("slide9_simulator.png")


# --- slide 10: the flip --------------------------------------------------
def slide10_flip():
    """Slide 9 images: per-condition results (left) and the overall
    balanced-test result (right), as two separate PNGs."""
    slide10_conditions()
    slide10_overall()


def slide10_conditions():
    """Per-condition results (Table 5.3) — standalone image, goes on
    the LEFT of slide 9."""
    cond_series = [
        ("Deep Q", C["dqn"], [473.72, -103.35, 10.45]),
        ("REINFORCE", C["reinforce"], [228.58, -248.32, 2.60]),
        ("PPO", C["ppo"], [67.69, -99.04, -2.14]),
        ("Always-buy", C["ab"], [477.22, -663.91, 10.58]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.13)
    x = np.arange(3)
    n = len(cond_series)
    bw = 0.19
    for k, (name, colour, vals) in enumerate(cond_series):
        offs = (k - (n - 1) / 2) * bw
        ax.bar(x + offs, vals, width=bw * 0.92, color=colour,
               edgecolor="white", linewidth=0.8, label=name)
    # label only the two bars that carry the argument, up and down
    ax.text(x[0] + (0 - 1.5) * bw, 473.72 + 18, "+$473.72",
            ha="center", fontsize=9, fontweight="bold", color=C["dqn"])
    ax.text(x[0] + (3 - 1.5) * bw, 477.22 + 18, "+$477.22",
            ha="center", fontsize=9, fontweight="bold", color="#777")
    ax.text(x[1] + (0 - 1.5) * bw, -103.35 - 22, "\u2212$103.35",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C["dqn"])
    ax.text(x[1] + (3 - 1.5) * bw, -663.91 - 22, "\u2212$663.91",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#777")
    # the two takeaway annotations (kept clear of bars and labels)
    ax.text(0.40, 0.97,
            "rising markets: Deep Q keeps 99%\nof what always-buy earns",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, color=C["green"], fontweight="bold",
            linespacing=1.35)
    ax.text(0.02, 0.04,
            "falling markets: Deep Q loses 84% less\n"
            "— it steps aside instead of holding on",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, color=C["red"], fontweight="bold",
            linespacing=1.35)
    ax.axhline(0, color="#555", lw=1.1)
    # title wording set by Fiyin (12 Sep)
    ax.set_title("600 test episodes categorised by market condition",
                 fontsize=12, color="#444", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["UP\nrising", "DOWN\nfalling", "FLAT\nsideways"],
                       fontsize=10)
    ax.set_ylim(-760, 590)
    ax.set_yticks([-600, -400, -200, 0, 200, 400])
    ax.set_ylabel("$ earned per episode", fontsize=11)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.legend(loc="center left", frameon=False, fontsize=9.5,
              bbox_to_anchor=(0.66, 0.72))
    save("slide10_conditions.png")


def slide10_overall():
    """Overall balanced-test result — standalone image, goes on the
    RIGHT of slide 9."""
    import matplotlib.transforms as mtransforms
    labels = ["Buy-and-\nhold", "Always-\nbuy", "REIN-\nFORCE",
              "PPO", "Deep Q-\nlearning"]
    colours = [C["bh"], C["ab"], C["reinforce"], C["ppo"], C["dqn"]]
    sim = [-60.29, -58.70, -5.71, -11.16, 126.94]
    state_dep = ["—", "—", "1 / 6", "0 / 6", "6 / 6"]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.subplots_adjust(left=0.155, right=0.98, top=0.86, bottom=0.24)
    bars = ax.bar(labels, sim, color=colours, edgecolor="white",
                  linewidth=1.2, width=0.62)
    for b, v in zip(bars, sim):
        cx = b.get_x() + b.get_width() / 2
        if v >= 0:
            ax.text(cx, v + 5, f"+${v:.2f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=C["green"])
        else:
            ax.text(cx, v - 5, f"\u2212${abs(v):.2f}", ha="center",
                    va="top", fontsize=10.5, fontweight="bold",
                    color=C["red"])
    ax.axhline(0, color="#555", lw=1.1)
    ax.text(-0.42, 6, "doing nothing = $0", fontsize=9.5,
            color="#555", ha="left", va="bottom", style="italic")
    # title wording set by Fiyin (12 Sep) — no second line
    ax.set_title("All 600 episodes overall",
                 fontsize=12, color="#444", pad=8)
    ax.set_ylabel("$ earned per episode", fontsize=11)
    ax.set_ylim(-95, 160)
    ax.set_yticks([-50, 0, 50, 100, 150])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6, labelsize=9.5)

    # decisions-changed row below the x-axis labels
    trans = mtransforms.blended_transform_factory(ax.transData,
                                                  ax.transAxes)
    for i, sd in enumerate(state_dep):
        if sd == "—":
            ax.text(i, -0.235, "—", transform=trans, ha="center",
                    va="center", fontsize=11.5, color="#999")
        else:
            c = C["green"] if sd.startswith("6") else (
                "#c07020" if sd.startswith("1") else C["red"])
            ax.text(i, -0.235, sd, transform=trans, ha="center",
                    va="center", fontsize=11.5, fontweight="bold",
                    color=c)
    fig.text(0.005, 0.075, "decisions changed\nwith the state?",
             fontsize=8, color="#555", fontweight="bold",
             ha="left", va="center")
    save("slide10_overall.png")


# --- slide 11: the 2008 stress test --------------------------------------
def slide11_stress():
    """Slide 10 images: cumulative lines (left) and transfer means
    (right), as two separate PNGs."""
    slide11_lines()
    slide11_bars()


def slide11_lines():
    """Cumulative wealth over the 180 transfer months, ALL methods,
    drawn from the actual experiment results (sim_results.json) so it
    matches the dissertation figure — buy-and-hold genuinely finishes
    higher; the agents' win is the 2008 behaviour and risk-adjusted
    return."""
    import json
    res = json.loads(
        (ROOT / "experiments/final_v2/results/sim_results.json").read_text())

    def algo_cum(name):
        seeds = res["results"][name]["per_seed"]
        monthly = np.mean(
            [[m["delta_w"] for m in s["real_per_month"]] for s in seeds],
            axis=0)
        return np.cumsum(monthly)

    bh_monthly = [m["delta_w"]
                  for m in res["baselines_real"]["B1b_true_bah"]["per_month"]]
    bh_cum = np.cumsum(bh_monthly)
    dqn_cum = algo_cum("dqn")
    rf_cum = algo_cum("reinforce")
    ppo_cum = algo_cum("ppo")
    ids = [m["id"] for m in
           res["baselines_real"]["B1b_true_bah"]["per_month"]]

    real_x = list(range(1, 181))

    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.10)
    ax.plot(real_x, bh_cum, color=C["bh"], lw=2.3, label="Buy-and-hold")
    ax.plot(real_x, dqn_cum, color=C["dqn"], lw=2.3,
            label="Deep Q-learning (our agent)")
    ax.plot(real_x, rf_cum, color=C["reinforce"], lw=1.7, alpha=0.9,
            label="REINFORCE")
    ax.plot(real_x, ppo_cum, color=C["ppo"], lw=1.7, alpha=0.9,
            label="PPO")
    ax.axhline(0, color="#aaa", lw=0.6)

    # dashed separator: months 1-144 are 2006-2017, months 145-180 are
    # 2023-2025 (2018-2022 is the excluded calibration window)
    ax.axvline(144.5, color="#999", lw=1.0, ls="--")
    ax.text(0.795, 0.98, "2018–2022 excluded\n(calibration) ",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#777")

    # 2008 highlight band (Sep 2008 - May 2009)
    i_sep08 = ids.index("2008-09") + 1
    i_may09 = ids.index("2009-05") + 1
    ax.axvspan(i_sep08, i_may09, color=C["red"], alpha=0.12)
    ax.annotate("2008 crisis: our agent held\nZERO shares through Oct 2008",
                xy=((i_sep08 + i_may09) / 2, min(bh_cum) * 0.9),
                xytext=(58, min(bh_cum) * 0.75),
                fontsize=10.5, fontweight="bold", color=C["red"],
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.5,
                                connectionstyle="arc3,rad=0.15"))

    # x-axis: year labels
    year_ticks = [1, 25, 49, 73, 97, 121, 145, 169]
    year_labels = ["2006", "2008", "2010", "2012", "2014", "2016", "2023", "2025"]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, fontsize=9.5)
    ax.set_ylabel("cumulative wealth change ($)")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_title("All methods, month by month — real experiment data\n"
                 "(2008 crash highlighted)",
                 fontsize=11.5, color="#444", pad=8)
    save("slide11_lines.png")


def slide11_bars():
    """Mean $ per month over the 180 transfer months, ALL methods
    (Table 5.3 'Real' column) — standalone image."""
    labels = ["Buy-and-\nhold", "Always-\nbuy", "REIN-\nFORCE",
              "PPO", "Deep Q-\nlearning"]
    vals = [80.24, 76.48, 31.66, 8.32, 60.28]
    colours = [C["bh"], C["ab"], C["reinforce"], C["ppo"], C["dqn"]]

    fig, ax = plt.subplots(figsize=(5.4, 5.3))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.87, bottom=0.22)
    bars = ax.bar(labels, vals, color=colours, edgecolor="white",
                  linewidth=1.2, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"+${v:.2f}",
                ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color="#333")
    ax.set_title("All methods on the same 180 months\n"
                 "every strategy profits — the market mostly rose",
                 fontsize=11.5, color="#444", pad=8)
    ax.set_ylabel("$ earned per month", fontsize=11)
    ax.set_ylim(0, 100)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6, labelsize=9)
    ax.text(0.5, -0.20,
            "buy-and-hold earns more on average,\n"
            "but rides the full 2008 crash",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9.5, color="#555", style="italic", linespacing=1.4)
    save("slide11_bars.png")


# --- slide 12: takeaways -------------------------------------------------
def slide12_takeaways():
    """Three findings cards — a flatter aspect so the closing slide can
    also carry a Contribution / Limitation / Future-work row underneath.
    """
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    cards = [
        (0.3, C["blue"], "1", "The data set the limit,\nnot the algorithm.",
         "60 real months could not train\n18,000 parameters. 3,000\nbalanced simulated ones could."),
        (4.15, C["dqn"], "2", "The winner flipped\nwhen we fixed the data.",
         "Deep Q-learning went from worst\n(real data) to best (+$127/ep,\n6/6 seeds, balanced test)."),
        (8.0, C["green"], "3", "The learned caution\ntransferred to 2008.",
         "On 180 unseen real months our\nagent held ZERO shares through\nOct 2008. Sharpe 0.224 vs 0.198."),
    ]

    for x, colour, num, head, body in cards:
        # card
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.15), 3.6, 3.7, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="white", edgecolor=colour, linewidth=2.5))
        # top strip for the badge and heading
        badge_y = 3.35
        circ = mpatches.Circle((x + 0.5, badge_y), 0.28,
                               facecolor=colour, edgecolor="none")
        ax.add_patch(circ)
        ax.text(x + 0.5, badge_y, num, ha="center", va="center",
                fontsize=15, fontweight="bold", color="white")
        # heading text below the badge
        ax.text(x + 1.95, 2.72, head, ha="center", va="center",
                fontsize=13, fontweight="bold", color=colour,
                linespacing=1.30)
        # body
        ax.text(x + 1.8, 1.15, body, ha="center", va="center",
                fontsize=11.2, color="#333", linespacing=1.55)

    save("slide12_takeaways.png")


# --- convert existing dissertation figures we reuse ---------------------
def convert_pdfs():
    """PDF → PNG for slides that reuse dissertation figures."""
    import pymupdf
    reuse = [
        (ROOT / "latex/dissertation/figs5/fig5_transfer.pdf", "fig5_transfer.png"),
        (ROOT / "latex/dissertation/figs5/fig5_sim.pdf", "fig5_sim.png"),
    ]
    for pdf, out in reuse:
        if not pdf.exists():
            continue
        doc = pymupdf.open(pdf)
        pix = doc[0].get_pixmap(dpi=250)
        pix.save(OUT / out)


def main():
    slide2_decision()
    slide3_rl_loop()
    slide3_env_loop()
    slide4_algorithms()
    slide5_rules()
    slide6_first_result()
    slide7_state_test()
    slide7_diagnosis_first()
    slide8_diagnosis_main()
    slide9_simulator()
    slide10_flip()
    slide11_stress()
    slide12_takeaways()
    convert_pdfs()
    print("Wrote figures to", OUT)
    for p in sorted(OUT.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    main()
