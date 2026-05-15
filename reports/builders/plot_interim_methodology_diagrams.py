"""Generate methodology diagrams for the interim review.

Run standalone:
    venv/bin/python reports/builders/plot_interim_methodology_diagrams.py

Outputs in reports/generated/charts/:
    interim_fig_data_splits.png
    interim_fig_rl_loop.png
    interim_fig_training_pipeline.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _box(ax, xy, w, h, text, fc="#E8F4FC", ec="#2166AC", fontsize=9):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.35, shrinkA=3, shrinkB=3, mutation_scale=11),
    )


def _arrow_poly(ax, points):
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if i < len(points) - 2:
            ax.plot([x1, x2], [y1, y2], color="#222", lw=1.35, solid_capstyle="round")
        else:
            _arrow(ax, x1, y1, x2, y2)


def render_data_splits(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("Figure A \u2014 Data splits (no future leakage)", fontsize=11, fontweight="bold", pad=10)

    _box(ax, (0.3, 1.35), 4.0, 1.05, "Train\n2009 \u2013 2018\nForecaster learns patterns", fc="#D5E8F4")
    _box(ax, (4.65, 1.35), 2.15, 1.05, "Validation\n2019 \u2013 2021\nTune thresholds", fc="#FEE9C7")
    _box(ax, (7.05, 1.35), 2.55, 1.05, "Test\n2022 \u2013 2025\nFinal reported metrics", fc="#DFF0D8")

    _arrow(ax, 4.3, 1.88, 4.62, 1.88)
    _arrow(ax, 6.82, 1.88, 7.02, 1.88)

    ax.text(5, 0.4,
        "\u2460 Strictly chronological (no random shuffle).\n"
        "\u2461 Train: history for forecaster + RL simulation.\n"
        "\u2462 Validation: tune uncertainty threshold \u2014 no final scores.\n"
        "\u2463 Test: headline Sharpe, drawdown, preservation (first time numbers count).",
        ha="center", fontsize=8, color="#333", linespacing=1.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_rl_loop(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Figure B \u2014 RL training loop (one step)", fontsize=11, fontweight="bold", pad=10)

    cx = 5.0
    _box(ax, (3.25, 5.05), 3.5, 0.95,
         "Environment (StockEnv)\nprices + portfolio + uncertainty", fc="#E8F4FC")
    _box(ax, (3.25, 3.25), 3.5, 0.95,
         "PPO Policy (neural network)\nmaps state \u2192 action a \u2208 [\u22121, 1]", fc="#FFF2CC")
    _box(ax, (2.35, 1.05), 5.3, 1.12,
         "Reward and transition\nr = 100 \u00b7 ln(V_next / V_now)\nadvance one day \u2192 new state", fc="#E7F4E4", fontsize=8)

    _arrow(ax, cx, 5.05, cx, 4.25)
    ax.text(5.4, 4.62, "\u2460 observation\n(state s_t)", fontsize=7.5, color="#333")

    _arrow(ax, cx, 3.25, cx, 2.22)
    ax.text(5.4, 2.75, "\u2461 action a_t;\ntrade executed", fontsize=7.5, color="#333")

    _arrow_poly(ax, [(3.25, 1.6), (0.85, 1.6), (0.85, 5.53), (3.25, 5.53)])
    ax.text(0.15, 3.45, "\u2462 PPO updates\nfrom rollouts;\nrepeat", fontsize=7.5, color="#333")

    ax.text(5, 6.4, "Read: top \u2192 middle \u2192 bottom \u2192 left return = one training cycle.",
            ha="center", fontsize=8, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_training_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Figure C \u2014 End-to-end training pipeline", fontsize=11, fontweight="bold", pad=10)

    y_row = 5.35
    h = 0.95
    _box(ax, (0.35, y_row), 2.45, h, "\u2460 Prices\nYahoo daily closes", fc="#F0F0F0")
    _box(ax, (3.05, y_row), 2.95, h, "\u2461 LSTM forecaster\nGaussian NLL loss\n\u2192 uncertainty u_t", fc="#E3F2FD")
    _box(ax, (6.35, y_row), 3.3, h, "\u2462 Inject u_t into StockEnv\n(baseline: u_t = 0)", fc="#E8DAEF")

    cy = y_row + h / 2
    _arrow(ax, 2.8, cy, 3.03, cy)
    _arrow(ax, 6.0, cy, 6.33, cy)

    _box(ax, (1.25, 3.05), 7.5, 1.35,
         "\u2463 Train PPO (same settings both arms)\nlr 3e-4 \u00b7 n_steps 512 \u00b7 batch 64 \u00b7 epochs 5\n10,000 steps \u00d7 seeds {7, 19, 42}",
         fc="#FFF9E6", fontsize=8.5)

    _arrow(ax, 5.0, y_row, 5.0, 4.42)

    _box(ax, (1.6, 1.05), 6.8, 1.15,
         "\u2464 Evaluate on test window\nReplay policy \u2192 equity curve \u2192 Sharpe, MDD, preservation",
         fc="#DFF0D8", fontsize=8.5)

    _arrow(ax, 5.0, 3.05, 5.0, 2.22)

    ax.text(5, 0.38,
            "Baseline skips steps \u2461\u2013\u2462: same PPO, no uncertainty channel.",
            ha="center", fontsize=8, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_all(root: Path | None = None) -> dict[str, Path]:
    root = root or _repo_root()
    charts = root / "reports" / "generated" / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    paths = {
        "splits": charts / "interim_fig_data_splits.png",
        "rl_loop": charts / "interim_fig_rl_loop.png",
        "pipeline": charts / "interim_fig_training_pipeline.png",
    }
    render_data_splits(paths["splits"])
    render_rl_loop(paths["rl_loop"])
    render_training_pipeline(paths["pipeline"])
    return paths


def main() -> None:
    out = render_all()
    for k, p in out.items():
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
