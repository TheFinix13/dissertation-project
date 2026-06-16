"""Statistical-significance pass for the dissertation's Phase-2 results.

Consumes the committed per-cell JSON files under
`experiments/results/per_cell/` and produces the inference numbers that
Chapter 5 lacks:

  1. Paired Wilcoxon signed-rank between Arms (paired on ticker; the
     across-seed median is used as the cell statistic so the test is
     paired across tickers, not artificially inflated by within-ticker
     correlation across seeds).
  2. Paired bootstrap 95% confidence intervals for the per-ticker
     median difference (B - A) and (B - C), with 10,000 resamples.
  3. Hodges-Lehmann shift estimate as a robust effect size.

The script is intentionally compute-cheap (runs in seconds on a
laptop) and emits both a machine-readable JSON and a human-readable
Markdown summary into `reports/generated/stats/`.

Run:
    venv/bin/python experiments/stats_significance.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
PER_CELL = ROOT / "results" / "per_cell"
OUT_DIR = (ROOT.parent / "reports" / "generated" / "stats")

METRICS = (
    "final_portfolio_value",
    "sharpe_ratio",
    "max_drawdown",
    "capital_preservation_rate_95pct_hwm",
)

# Per-cell tag suffixes that identify the canonical Phase-2 main grid.
PHASE2_TAGS = ("phase2", "phase2_aleatoric", "phase2_epistemic")
WF_TAG = "phase2_wf"

BOOTSTRAP_RESAMPLES = 10_000
RNG_SEED = 20260616


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _arm_from_filename(name: str) -> str:
    """Return one of {A, B, C, WF_A, WF_B} from a per-cell filename, or '' to skip."""
    if name.startswith("baseline__"):
        return "A"
    if name.startswith("probabilistic__"):
        if "_aleatoric" in name:
            return "B"
        if "_epistemic" in name:
            return "C"
        # legacy default (aleatoric)
        return "B"
    if name.startswith("wf_baseline__"):
        return "WF_A"
    if name.startswith("wf_probabilistic__"):
        return "WF_B"
    return ""


def _tag_from_filename(name: str) -> str:
    """Trailing tag, e.g. 'phase2' / 'phase2_aleatoric' / 'phase2_wf'."""
    base = name.rstrip(".json")
    # The schema is e.g. probabilistic__SPY__seed7__phase2_aleatoric.json
    parts = base.split("__")
    return parts[-1] if parts else ""


def load_per_cell() -> pd.DataFrame:
    rows: List[dict] = []
    if not PER_CELL.exists():
        raise FileNotFoundError(f"Expected {PER_CELL} to exist with committed Phase-2 results.")
    for path in sorted(PER_CELL.glob("*.json")):
        arm = _arm_from_filename(path.name)
        if not arm:
            continue
        tag = _tag_from_filename(path.name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        row = {
            "arm": arm,
            "tag": tag,
            "ticker": payload.get("ticker"),
            "seed": payload.get("seed"),
            "fold_id": payload.get("fold_id", "test_legacy"),
        }
        for metric in METRICS:
            if metric in payload:
                row[metric] = payload[metric]
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No per-cell JSON files matched the expected schema.")
    return df


# ---------------------------------------------------------------------------
# Cell statistics
# ---------------------------------------------------------------------------

def main_grid_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, ticker) cell statistic = median across seeds.

    Pairs across tickers; collapsing seeds first prevents the within-ticker
    seed correlation from inflating the apparent sample size.
    """
    mask = df["tag"].isin(PHASE2_TAGS) & df["arm"].isin(("A", "B", "C"))
    sub = df.loc[mask].copy()
    grouped = (
        sub.groupby(["arm", "ticker"], dropna=False)[list(METRICS)]
        .median()
        .reset_index()
    )
    return grouped


def wf_cells(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["tag"] == WF_TAG) & df["arm"].isin(("WF_A", "WF_B"))
    sub = df.loc[mask].copy()
    grouped = (
        sub.groupby(["arm", "ticker", "fold_id"], dropna=False)[list(METRICS)]
        .median()
        .reset_index()
    )
    return grouped


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Two-sided Wilcoxon signed-rank on the differences (x - y)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return {"n_pairs": int(len(x)), "statistic": float("nan"), "p_value": float("nan")}
    res = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
    return {"n_pairs": int(len(x)), "statistic": float(res.statistic), "p_value": float(res.pvalue)}


def hodges_lehmann(x: np.ndarray, y: np.ndarray, max_pairs: int = 200_000) -> float:
    """Hodges-Lehmann median of pairwise differences (a robust effect size)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    diffs = (x - y)[mask]
    n = len(diffs)
    if n == 0:
        return float("nan")
    # Median of all pairwise (d_i + d_j) / 2 over i <= j.
    if n * (n + 1) // 2 > max_pairs:
        rng = np.random.default_rng(RNG_SEED)
        idx_i = rng.integers(0, n, max_pairs)
        idx_j = rng.integers(0, n, max_pairs)
        sample = (diffs[idx_i] + diffs[idx_j]) / 2.0
        return float(np.median(sample))
    pair_means = (diffs[:, None] + diffs[None, :]) / 2.0
    iu = np.triu_indices_from(pair_means)
    return float(np.median(pair_means[iu]))


def paired_bootstrap_median_diff(
    x: np.ndarray, y: np.ndarray, *, resamples: int = BOOTSTRAP_RESAMPLES
) -> Dict[str, float]:
    """Paired bootstrap of the median of (x - y).

    Returns the point estimate and the percentile 95% CI.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    diffs = (x - y)[mask]
    n = len(diffs)
    if n < 5:
        return {
            "n_pairs": int(n),
            "median_diff": float("nan"),
            "ci_lower_95": float("nan"),
            "ci_upper_95": float("nan"),
        }
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n, size=(resamples, n))
    samples = diffs[idx]
    boot_medians = np.median(samples, axis=1)
    lo, hi = np.quantile(boot_medians, [0.025, 0.975])
    return {
        "n_pairs": int(n),
        "median_diff": float(np.median(diffs)),
        "ci_lower_95": float(lo),
        "ci_upper_95": float(hi),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_main_grid_tests(grid: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    """For each metric, run B vs A, C vs A, B vs C paired tests."""
    arms = {
        a: grid[grid["arm"] == a].set_index("ticker")[list(METRICS)]
        for a in ("A", "B", "C")
    }
    contrasts = (("B", "A"), ("C", "A"), ("B", "C"))
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric in METRICS:
        out[metric] = {}
        for left, right in contrasts:
            common = arms[left][[metric]].join(arms[right][[metric]], how="inner", lsuffix="_L", rsuffix="_R")
            x = common[f"{metric}_L"].to_numpy()
            y = common[f"{metric}_R"].to_numpy()
            payload = {
                **paired_wilcoxon(x, y),
                **paired_bootstrap_median_diff(x, y),
                "hodges_lehmann": hodges_lehmann(x, y),
            }
            out[metric][f"{left}_vs_{right}"] = payload
    return out


def run_wf_tests(wf: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Walk-forward: pair on (ticker, fold)."""
    a = wf[wf["arm"] == "WF_A"].set_index(["ticker", "fold_id"])[list(METRICS)]
    b = wf[wf["arm"] == "WF_B"].set_index(["ticker", "fold_id"])[list(METRICS)]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric in METRICS:
        common = a[[metric]].join(b[[metric]], how="inner", lsuffix="_A", rsuffix="_B")
        x = common[f"{metric}_B"].to_numpy()
        y = common[f"{metric}_A"].to_numpy()
        out[metric] = {
            "WFB_vs_WFA": {
                **paired_wilcoxon(x, y),
                **paired_bootstrap_median_diff(x, y),
                "hodges_lehmann": hodges_lehmann(x, y),
            }
        }
    return out


def write_outputs(
    main_grid: pd.DataFrame,
    wf: pd.DataFrame,
    main_tests: Dict,
    wf_tests: Dict,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_grid.to_csv(OUT_DIR / "main_grid_per_ticker_medians.csv", index=False)
    wf.to_csv(OUT_DIR / "wf_per_ticker_fold_medians.csv", index=False)
    bundle = {
        "phase2_main_grid": main_tests,
        "walk_forward": wf_tests,
        "metadata": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "rng_seed": RNG_SEED,
            "n_main_grid_tickers": int(main_grid["ticker"].nunique()),
            "n_wf_pairs": int(len(wf) // 2),
        },
    }
    (OUT_DIR / "phase2_statistical_tests.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )

    md = _format_markdown(bundle, main_grid, wf)
    (OUT_DIR / "phase2_statistical_tests.md").write_text(md, encoding="utf-8")


def _fmt(v: float, fmt: str = ".4g") -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "NA"
    return format(v, fmt)


def _format_markdown(bundle: Dict, main_grid: pd.DataFrame, wf: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Phase-2 statistical-significance tests")
    lines.append("")
    lines.append(
        f"Generated offline from committed per-cell JSONs. "
        f"Paired Wilcoxon + paired bootstrap (n={bundle['metadata']['bootstrap_resamples']:,} resamples). "
        f"Cell statistic = median across seeds per ticker."
    )
    lines.append("")
    lines.append(f"- Phase-2 main grid: {bundle['metadata']['n_main_grid_tickers']} tickers per arm.")
    lines.append(f"- Walk-forward: {bundle['metadata']['n_wf_pairs']} (ticker, fold) pairs per arm.")
    lines.append("")
    lines.append("## Phase-2 main grid (2022-2025 test window)")
    lines.append("")
    for metric, contrasts in bundle["phase2_main_grid"].items():
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| Contrast | n pairs | Median diff (L-R) | 95% bootstrap CI | Hodges-Lehmann | Wilcoxon p |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for contrast_name, stats_dict in contrasts.items():
            ci = f"[{_fmt(stats_dict['ci_lower_95'])}, {_fmt(stats_dict['ci_upper_95'])}]"
            lines.append(
                f"| {contrast_name} | {stats_dict['n_pairs']} | "
                f"{_fmt(stats_dict['median_diff'])} | {ci} | "
                f"{_fmt(stats_dict['hodges_lehmann'])} | {_fmt(stats_dict['p_value'])} |"
            )
        lines.append("")
    lines.append("## Walk-forward (out-of-time, 2018-2025)")
    lines.append("")
    for metric, contrasts in bundle["walk_forward"].items():
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| Contrast | n pairs | Median diff (L-R) | 95% bootstrap CI | Hodges-Lehmann | Wilcoxon p |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for contrast_name, stats_dict in contrasts.items():
            ci = f"[{_fmt(stats_dict['ci_lower_95'])}, {_fmt(stats_dict['ci_upper_95'])}]"
            lines.append(
                f"| {contrast_name} | {stats_dict['n_pairs']} | "
                f"{_fmt(stats_dict['median_diff'])} | {ci} | "
                f"{_fmt(stats_dict['hodges_lehmann'])} | {_fmt(stats_dict['p_value'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    global BOOTSTRAP_RESAMPLES  # noqa: PLW0603
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    BOOTSTRAP_RESAMPLES = args.resamples

    df = load_per_cell()
    main_grid = main_grid_cells(df)
    wf = wf_cells(df)
    main_tests = run_main_grid_tests(main_grid)
    wf_tests = run_wf_tests(wf)
    write_outputs(main_grid, wf, main_tests, wf_tests)

    print(f"Wrote outputs to {OUT_DIR}/")
    print(f"  - phase2_statistical_tests.json")
    print(f"  - phase2_statistical_tests.md")
    print(f"  - main_grid_per_ticker_medians.csv")
    print(f"  - wf_per_ticker_fold_medians.csv")


if __name__ == "__main__":
    main()
