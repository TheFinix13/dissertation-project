"""Reproducible dissertation chart builder (Phase-2).

Regenerates every Phase-2 dissertation chart deterministically from the
committed result CSVs in ``experiments/results/``. NO network access is
required (no yfinance): all data comes from the canonical Phase-2 outputs.

How to run (from the repository root)::

    MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig venv/bin/python reports/builders/plot_phase2_charts.py

Data sources (all committed):
  * ``summary_*_phase2.csv`` -- canonical aggregated metrics with columns
    agent,ticker,fold_id,metric,n,mean,std,median,q25,q75,min,max. Used for the
    cross-grid distribution charts (drawdown / outcome / Sharpe), the SPY
    final-value bar chart, and the aleatoric-vs-epistemic scatter.
  * ``walk_forward_*_phase2_wf.csv`` -- per-cell walk-forward metrics. Used for
    the by-fold chart.
  * ``wf_curves/*.csv`` -- real-dated daily equity curves (date,portfolio_value,
    agent,ticker,fold_id,seed). Available as a fully-offline fallback for the
    representative equity curve.
  * ``spy_repr_curve_phase2.csv`` -- a single representative SPY run over the
    2022--2025 test window with aligned baseline equity, probabilistic equity
    and the aleatoric/epistemic uncertainty scores. This is the only file that
    carries the per-day uncertainty series, so it drives the uncertainty signal
    chart and the equity/uncertainty overlay. It     is a derived artefact produced
    by training one representative seed; regenerate it with the committed script
    ``reports/builders/gen_spy_repr_curve.py`` (which needs the venv and network).
    Reading it here is deterministic and offline.

Every chart written by this script is reproducible from the committed result
files plus this script. The two legacy market-data charts
(``dataset_spy_close.png``, ``spy_intraday_realtime_proxy.png``) are NOT built
here because they require live Yahoo Finance downloads; they remain the product
of the legacy ``plot_dissertation_visuals.py`` builder and are unreferenced by
the dissertation body.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants / conventions
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "experiments" / "results"
WF_CURVES = RESULTS / "wf_curves"
CHARTS = ROOT / "reports" / "generated" / "charts"

INITIAL_CAPITAL = 1_000_000.0
TAU_QUANTILE = 0.80  # uncertainty_quantile_stop from dissertation_protocol.json
TEST_FOLD = "test_legacy"  # the 2022--2025 main-grid window

# The canonical complete walk-forward run cited by the dissertation tables
# (320 cells = 8 tickers x 4 folds x 10 seeds; 87.2% guard win-rate). A later
# partial re-run (...085445Z...) exists on disk but is incomplete, so the file
# is pinned here rather than glob-selected to keep the chart consistent with the
# text.
WF_FILE = "walk_forward_20260530T073000Z_phase2_wf.csv"

# Human-readable agent labels and a stable colour per agent.
AGENT_LABEL = {
    "buy_and_hold": "Buy & Hold",
    "ppo_standalone": "Baseline PPO",
    "ppo_with_uncertainty_guard": "Aleatoric guard",
    "ppo_with_uncertainty_guard_epistemic": "Epistemic guard",
    "stop_loss_5pct": "Stop-loss 5%",
    "stop_loss_10pct": "Stop-loss 10%",
    "all_cash": "All cash",
}
AGENT_COLOR = {
    "buy_and_hold": "#444444",
    "ppo_standalone": "#1f77b4",
    "ppo_with_uncertainty_guard": "#d62728",
    "ppo_with_uncertainty_guard_epistemic": "#ff7f0e",
    "stop_loss_5pct": "#2ca02c",
    "stop_loss_10pct": "#8c564b",
    "all_cash": "#999999",
}


def _latest(pattern: str, folder: Path = RESULTS) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matches {pattern} in {folder}")
    return files[-1]


def _grid_values(summary: pd.DataFrame, agent: str, metric: str,
                 fold: str = TEST_FOLD) -> pd.Series:
    """Per-ticker representative value (median across seeds) across the grid."""
    sub = summary[(summary.agent == agent)
                  & (summary.metric == metric)
                  & (summary.fold_id == fold)]
    return sub.set_index("ticker")["median"].dropna()


def _style_axis(ax) -> None:
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ----------------------------------------------------------------------------
# (a) drawdown distribution -- the thesis-critical corrected chart
# ----------------------------------------------------------------------------
def plot_drawdown_distribution(summary: pd.DataFrame) -> dict:
    agents = [
        "buy_and_hold", "stop_loss_5pct", "stop_loss_10pct",
        "ppo_standalone", "ppo_with_uncertainty_guard",
        "ppo_with_uncertainty_guard_epistemic",
    ]
    data, labels, colors, stats = [], [], [], {}
    for a in agents:
        vals = _grid_values(summary, a, "max_drawdown") * 100.0
        if len(vals) == 0:
            continue
        data.append(vals.values)
        labels.append(f"{AGENT_LABEL[a]}\n(n={len(vals)})")
        colors.append(AGENT_COLOR[a])
        stats[a] = dict(median=float(vals.median()), mean=float(vals.mean()),
                        worst=float(vals.max()), n=int(len(vals)))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=False,
                    medianprops=dict(color="black", linewidth=1.6),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    # jittered points
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        x = rng.normal(i + 1, 0.05, size=len(vals))
        ax.scatter(x, vals, s=10, color=colors[i], alpha=0.5, zorder=3,
                   edgecolors="none")
    bh_median = stats["buy_and_hold"]["median"]
    ax.axhline(bh_median, color="#444444", linestyle="--", linewidth=1.2,
               label=f"Buy & Hold median ({bh_median:.0f}%)")
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Maximum drawdown (%)  –  lower is safer")
    ax.set_title("Worst peak-to-trough loss across the test grid", fontsize=13,
                 fontweight="bold", pad=34)
    ax.text(0.5, 1.012,
            "Buy & Hold is the fair drawdown benchmark; the Baseline PPO's tiny "
            "drawdown comes from barely investing.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            style="italic", color="#555555")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(CHARTS / "drawdown_distribution.png", dpi=160)
    fig.savefig(CHARTS / "drawdown_distribution.pdf")
    plt.close(fig)
    return stats


# ----------------------------------------------------------------------------
# (b) outcome distribution -- final value / % return across the grid
# ----------------------------------------------------------------------------
def plot_outcome_distribution(summary: pd.DataFrame) -> None:
    agents = ["buy_and_hold", "ppo_standalone", "ppo_with_uncertainty_guard",
              "ppo_with_uncertainty_guard_epistemic"]
    data, labels, colors, maxpct = [], [], [], 0.0
    for a in agents:
        fpv = _grid_values(summary, a, "final_portfolio_value")
        ret = (fpv / INITIAL_CAPITAL - 1.0) * 100.0
        data.append(ret.values)
        labels.append(f"{AGENT_LABEL[a]}\n(n={len(ret)})")
        colors.append(AGENT_COLOR[a])
        maxpct = max(maxpct, float(ret.max()))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.6),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        x = rng.normal(i + 1, 0.05, size=len(vals))
        ax.scatter(x, vals, s=10, color=colors[i], alpha=0.5, zorder=3,
                   edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-30, 165)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total return over the test window (%)")
    ax.set_title("Spread of investment outcomes across the test grid",
                 fontsize=13, fontweight="bold", pad=34)
    ax.text(0.5, 1.012,
            f"Each dot is one stock. Axis clipped at +165%; the best guard "
            f"outliers reach about +{maxpct:.0f}%.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            style="italic", color="#555555")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(CHARTS / "outcome_distribution.png", dpi=160)
    fig.savefig(CHARTS / "outcome_distribution.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# (c) Sharpe distribution
# ----------------------------------------------------------------------------
def plot_sharpe_distribution(summary: pd.DataFrame) -> None:
    agents = ["buy_and_hold", "ppo_standalone", "ppo_with_uncertainty_guard",
              "ppo_with_uncertainty_guard_epistemic"]
    data, labels, colors = [], [], []
    for a in agents:
        vals = _grid_values(summary, a, "sharpe_ratio")
        data.append(vals.values)
        labels.append(f"{AGENT_LABEL[a]}\n(n={len(vals)})")
        colors.append(AGENT_COLOR[a])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.6),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        x = rng.normal(i + 1, 0.05, size=len(vals))
        ax.scatter(x, vals, s=10, color=colors[i], alpha=0.5, zorder=3,
                   edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Sharpe ratio (return per unit of risk)")
    ax.set_title("Risk-adjusted return across the test grid", fontsize=13,
                 fontweight="bold", pad=34)
    ax.text(0.5, 1.012,
            "The baseline clusters near zero; both uncertainty-guard agents sit "
            "clearly positive.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            style="italic", color="#555555")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(CHARTS / "sharpe_distribution.png", dpi=160)
    fig.savefig(CHARTS / "sharpe_distribution.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# (d) equity / uncertainty overlay -- the mechanism in action
# ----------------------------------------------------------------------------
def plot_equity_uncertainty_overlay(repr_curve: pd.DataFrame) -> None:
    df = repr_curve.assign(date=pd.to_datetime(repr_curve["date"]))
    df = df.sort_values("date").reset_index(drop=True)
    u = df["uncertainty_aleatoric"].values
    tau = float(np.quantile(u, TAU_QUANTILE))
    blocked = u >= tau

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.8), sharex=True,
        gridspec_kw=dict(height_ratios=[2.1, 1.0], hspace=0.12))

    ax1.plot(df["date"], df["baseline_value"], color=AGENT_COLOR["ppo_standalone"],
             linewidth=1.6, label="Baseline PPO (no uncertainty)")
    ax1.plot(df["date"], df["prob_value"],
             color=AGENT_COLOR["ppo_with_uncertainty_guard"], linewidth=1.9,
             label="Probabilistic PPO (uncertainty guard)")
    ax1.axhline(INITIAL_CAPITAL, color="black", linewidth=0.7, alpha=0.5)
    ax1.set_ylabel("Portfolio value (USD)")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v/1e6:.2f}M"))
    ax1.set_title("How the uncertainty guard works on a representative SPY run",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    _style_axis(ax1)

    ax2.plot(df["date"], u, color=AGENT_COLOR["ppo_with_uncertainty_guard"],
             linewidth=1.3)
    ax2.axhline(tau, color="black", linestyle="--", linewidth=1.1,
                label=f"Threshold $\\tau$ (80th pct = {tau:.2f})")
    # shade contiguous blocked regions
    dates = df["date"].values
    start = None
    first_label = True
    for i in range(len(blocked)):
        if blocked[i] and start is None:
            start = dates[i]
        is_end = blocked[i] and (i == len(blocked) - 1 or not blocked[i + 1])
        if is_end and start is not None:
            ax2.axvspan(start, dates[i], color="#d62728", alpha=0.18,
                        label="New buys blocked" if first_label else None)
            first_label = False
            start = None
    ax2.set_ylabel("Uncertainty $u_t$")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=8.5, ncol=2)
    _style_axis(ax2)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

    fig.savefig(CHARTS / "equity_uncertainty_overlay.png", dpi=160,
                bbox_inches="tight")
    fig.savefig(CHARTS / "equity_uncertainty_overlay.pdf",
                bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# (e) aleatoric vs epistemic scatter -- they are essentially equivalent
# ----------------------------------------------------------------------------
def plot_aleatoric_vs_epistemic(summary: pd.DataFrame) -> dict:
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    specs = [
        ("final_portfolio_value", "Final value ($M)", 1e6, axes[0]),
        ("sharpe_ratio", "Sharpe ratio", 1.0, axes[1]),
    ]
    for metric, label, scale, ax in specs:
        a = _grid_values(summary, "ppo_with_uncertainty_guard", metric)
        e = _grid_values(summary, "ppo_with_uncertainty_guard_epistemic", metric)
        common = sorted(set(a.index) & set(e.index))
        ax_x = (a.loc[common] / scale).values
        ax_y = (e.loc[common] / scale).values
        ax.scatter(ax_x, ax_y, s=22, color="#6a51a3", alpha=0.7,
                   edgecolors="white", linewidths=0.4)
        lo = float(min(ax_x.min(), ax_y.min()))
        hi = float(max(ax_x.max(), ax_y.max()))
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black",
                linestyle="--", linewidth=1.0, label="equal performance (y=x)")
        corr = float(np.corrcoef(ax_x, ax_y)[0, 1])
        out[metric] = dict(n=len(common), corr=corr)
        ax.set_xlabel(f"Aleatoric  –  {label}")
        ax.set_ylabel(f"Epistemic  –  {label}")
        ax.set_title(f"{label.split(' (')[0]}  (r = {corr:.2f}, n = {len(common)})",
                     fontsize=11)
        ax.legend(loc="upper left", fontsize=8.5)
        _style_axis(ax)
        ax.grid(True, alpha=0.2)
    fig.suptitle("Aleatoric vs epistemic uncertainty: near-identical per-stock "
                 "outcomes", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(CHARTS / "aleatoric_vs_epistemic_scatter.png", dpi=160)
    fig.savefig(CHARTS / "aleatoric_vs_epistemic_scatter.pdf")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# (f) walk-forward by fold
# ----------------------------------------------------------------------------
def plot_walk_forward_by_fold(wf: pd.DataFrame) -> None:
    folds = ["wf_2018_2019", "wf_2020_2021", "wf_2022_2023", "wf_2024_2025"]
    fold_label = {"wf_2018_2019": "2018-19", "wf_2020_2021": "2020-21",
                  "wf_2022_2023": "2022-23", "wf_2024_2025": "2024-25"}
    agents = [("ppo_standalone", "Baseline PPO"),
              ("ppo_with_uncertainty_guard", "Probabilistic PPO")]

    med_fpv = {a: [] for a, _ in agents}
    winrate = {a: [] for a, _ in agents}
    for f in folds:
        for a, _ in agents:
            sub = wf[(wf.agent == a) & (wf.fold_id == f)]
            med_fpv[a].append(sub["final_portfolio_value"].median() / 1e6)
            winrate[a].append(
                (sub["final_portfolio_value"] > INITIAL_CAPITAL).mean() * 100.0)

    x = np.arange(len(folds))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0))

    for j, (a, lab) in enumerate(agents):
        ax1.bar(x + (j - 0.5) * w, med_fpv[a], w, label=lab,
                color=AGENT_COLOR[a], alpha=0.85)
    ax1.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax1.set_xticks(x)
    ax1.set_xticklabels([fold_label[f] for f in folds])
    ax1.set_ylabel("Median final value ($M, from $1M start)")
    ax1.set_title("Median outcome by walk-forward fold", fontsize=11)
    ax1.legend(fontsize=9)
    _style_axis(ax1)

    for j, (a, lab) in enumerate(agents):
        ax2.bar(x + (j - 0.5) * w, winrate[a], w, label=lab,
                color=AGENT_COLOR[a], alpha=0.85)
    ax2.axhline(50, color="black", linewidth=0.8, linestyle=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels([fold_label[f] for f in folds])
    ax2.set_ylabel("Win-rate (% of cells finishing above $1M)")
    ax2.set_title("Win-rate by walk-forward fold", fontsize=11)
    ax2.legend(fontsize=9)
    _style_axis(ax2)

    fig.suptitle("Out-of-sample walk-forward: the guard leads on median outcome "
                 "in every period", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(CHARTS / "walk_forward_by_fold.png", dpi=160)
    fig.savefig(CHARTS / "walk_forward_by_fold.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# (g) equity curve comparison + final value bar (SPY case study)
# ----------------------------------------------------------------------------
def plot_equity_curve_comparison(repr_curve: pd.DataFrame,
                                 summary: pd.DataFrame) -> None:
    df = repr_curve.assign(date=pd.to_datetime(repr_curve["date"]))
    df = df.sort_values("date").reset_index(drop=True)
    bh_final = float(
        _grid_values(summary, "buy_and_hold", "final_portfolio_value")
        .reindex(["SPY"]).iloc[0])

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(df["date"], df["baseline_value"], color=AGENT_COLOR["ppo_standalone"],
            linewidth=1.7, label="Baseline PPO (no uncertainty signal)")
    ax.plot(df["date"], df["prob_value"],
            color=AGENT_COLOR["ppo_with_uncertainty_guard"], linewidth=2.0,
            label="Probabilistic PPO (uncertainty guard)")
    ax.axhline(bh_final, color="#444444", linestyle="--", linewidth=1.3,
               label=f"Buy & Hold final value (${bh_final/1e6:.2f}M)")
    ax.axhline(INITIAL_CAPITAL, color="black", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("Portfolio value (USD)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v/1e6:.2f}M"))
    ax.set_title("Representative SPY equity curves, 2022–2025 test window",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    _style_axis(ax)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(CHARTS / "equity_curve_comparison.png", dpi=160)
    fig.savefig(CHARTS / "equity_curve_comparison.pdf")
    plt.close(fig)


def plot_final_value_comparison(summary: pd.DataFrame) -> None:
    agents = ["buy_and_hold", "stop_loss_5pct", "ppo_standalone",
              "ppo_with_uncertainty_guard", "ppo_with_uncertainty_guard_epistemic"]
    vals, labels, colors = [], [], []
    for a in agents:
        fpv = _grid_values(summary, a, "final_portfolio_value").reindex(["SPY"])
        if fpv.isna().all():
            continue
        vals.append(float(fpv.iloc[0]) / 1e6)
        labels.append(AGENT_LABEL[a])
        colors.append(AGENT_COLOR[a])

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.axhline(INITIAL_CAPITAL / 1e6, color="black", linewidth=0.9,
               linestyle=":", label="$1.0M starting capital")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"${v:.2f}M",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Final portfolio value ($M)")
    ax.set_title("Terminal wealth on the SPY 2022–2025 test window",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    _style_axis(ax)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS / "final_value_comparison.png", dpi=160)
    fig.savefig(CHARTS / "final_value_comparison.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# (h) uncertainty signal -- illustrative single run
# ----------------------------------------------------------------------------
def plot_uncertainty_signal(repr_curve: pd.DataFrame) -> None:
    df = repr_curve.assign(date=pd.to_datetime(repr_curve["date"]))
    df = df.sort_values("date").reset_index(drop=True)
    u = df["uncertainty_aleatoric"].values
    tau = float(np.quantile(u, TAU_QUANTILE))

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(df["date"], u, color="#ff7f0e", linewidth=1.4)
    ax.fill_between(df["date"], u, tau, where=(u >= tau), color="#d62728",
                    alpha=0.25, interpolate=True)
    ax.axhline(tau, color="black", linestyle="--", linewidth=1.1,
               label=f"Threshold $\\tau$ (80th pct = {tau:.2f})")
    ax.set_ylabel("Uncertainty score $u_t$")
    ax.set_xlabel("Date")
    ax.set_title("Illustrative aleatoric uncertainty signal (single SPY run)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    _style_axis(ax)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(CHARTS / "uncertainty_signal.png", dpi=160)
    fig.savefig(CHARTS / "uncertainty_signal.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(_latest("summary_*_phase2.csv"))
    wf = pd.read_csv(RESULTS / WF_FILE)
    repr_curve = pd.read_csv(RESULTS / "spy_repr_curve_phase2.csv")

    dd = plot_drawdown_distribution(summary)
    plot_outcome_distribution(summary)
    plot_sharpe_distribution(summary)
    plot_equity_uncertainty_overlay(repr_curve)
    ae = plot_aleatoric_vs_epistemic(summary)
    plot_walk_forward_by_fold(wf)
    plot_equity_curve_comparison(repr_curve, summary)
    plot_final_value_comparison(summary)
    plot_uncertainty_signal(repr_curve)

    print("Wrote charts to", CHARTS)
    print("\n=== Per-agent maximum drawdown across the test grid (median %) ===")
    print(f"{'agent':40s} {'n':>4} {'median':>8} {'mean':>8} {'worst':>8}")
    for a, s in dd.items():
        print(f"{a:40s} {s['n']:4d} {s['median']:8.1f} {s['mean']:8.1f} "
              f"{s['worst']:8.1f}")
    print("\n=== Aleatoric vs epistemic equivalence ===")
    for m, s in ae.items():
        print(f"{m:30s} n={s['n']:3d}  correlation r={s['corr']:.3f}")


if __name__ == "__main__":
    main()
