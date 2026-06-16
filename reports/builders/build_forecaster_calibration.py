"""Build the forecaster calibration figure for Chapter 5.

Trains the dissertation's LSTM forecaster on the training window
(2009-01-01 to 2018-12-31) and reports its predictive-interval
calibration on the validation window (2019-01-01 to 2021-12-31).

Produces three artefacts in reports/generated/charts/ + a metrics JSON:
  - forecaster_calibration.pdf  (reliability diagram, vector)
  - forecaster_calibration.png  (same, raster fallback)
  - forecaster_calibration_metrics.json (NLL, RMSE, coverages,
    Expected Calibration Error)

Determinism:
  - torch seed = 20260616
  - numpy seed = 20260616
  - yfinance is cached on disk for repeatability between runs.

Run:
    venv/bin/python reports/builders/build_forecaster_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "runners"))

from run_probabilistic_agent import ProbabilisticLSTM, build_sequences, gaussian_nll

CHARTS_DIR = ROOT / "reports" / "generated" / "charts"
STATS_DIR = ROOT / "reports" / "generated" / "stats"
CACHE_DIR = ROOT / "experiments" / "cache"

TRAIN_START, TRAIN_END = "2009-01-01", "2018-12-31"
VALID_START, VALID_END = "2019-01-01", "2021-12-31"

SEQ_LEN = 20
EPOCHS = 200            # more than the runners' 20 because this is one-shot, not 70x10 cells
LR = 1e-3
SEED = 20260616


def _fetch(ticker: str, start: str, end: str) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if "Adj Close" in df.columns:
        prices = df["Adj Close"].to_numpy().squeeze().astype("float32")
    else:
        prices = df["Close"].to_numpy().squeeze().astype("float32")
    np.save(cache_path, prices)
    return prices


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_prices = _fetch("SPY", TRAIN_START, TRAIN_END)
    valid_prices = _fetch("SPY", VALID_START, VALID_END)

    train_returns = np.diff(np.log(np.maximum(train_prices, 1e-8))).astype("float32")
    valid_returns = np.diff(np.log(np.maximum(valid_prices, 1e-8))).astype("float32")

    x_train, y_train = build_sequences(train_returns, seq_len=SEQ_LEN)
    x_valid, y_valid = build_sequences(valid_returns, seq_len=SEQ_LEN)
    xt = torch.tensor(x_train)
    yt = torch.tensor(y_train)
    xv = torch.tensor(x_valid)
    yv = torch.tensor(y_valid)

    model = ProbabilisticLSTM(dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Mini-batch SGD with a fixed shuffle (deterministic given SEED).
    n = len(xt)
    batch = 256
    perm = torch.randperm(n)
    xt, yt = xt[perm], yt[perm]

    model.train()
    train_losses = []
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch):
            xb = xt[i : i + batch]
            yb = yt[i : i + batch]
            opt.zero_grad()
            mean, log_var = model(xb)
            loss = gaussian_nll(yb, mean, log_var)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

    model.eval()
    with torch.no_grad():
        mean_v, log_var_v = model(xv)
        mean_v = mean_v.squeeze(-1).numpy()
        std_v = np.exp(0.5 * log_var_v.squeeze(-1).numpy())
        y_v_flat = yv.squeeze(-1).numpy()
        valid_nll = float(gaussian_nll(yv, torch.tensor(mean_v[:, None]), torch.tensor(np.log(std_v ** 2)[:, None])).item())

    rmse = float(np.sqrt(np.mean((y_v_flat - mean_v) ** 2)))
    mae = float(np.mean(np.abs(y_v_flat - mean_v)))

    # Empirical coverage at nominal levels.
    nominal_levels = np.linspace(0.05, 0.95, 19)  # 5% to 95% in 5pp steps
    empirical_coverages = []
    for c in nominal_levels:
        z = stats.norm.ppf(0.5 + c / 2.0)
        inside = np.abs(y_v_flat - mean_v) <= z * std_v
        empirical_coverages.append(float(inside.mean()))
    empirical_coverages = np.asarray(empirical_coverages)

    # Expected Calibration Error: mean | nominal - empirical |.
    ece = float(np.mean(np.abs(nominal_levels - empirical_coverages)))

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure: reliability diagram + interval-width scatter ---------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    ax_left.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax_left.plot(
        nominal_levels, empirical_coverages,
        marker="o", linewidth=2.0, color="#1f77b4",
        label="LSTM forecaster (SPY validation)",
    )
    ax_left.set_xlabel("Nominal coverage")
    ax_left.set_ylabel("Empirical coverage")
    ax_left.set_xlim(0.0, 1.0)
    ax_left.set_ylim(0.0, 1.0)
    ax_left.grid(True, alpha=0.3)
    ax_left.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax_left.legend(loc="lower right", fontsize=9)

    # Right panel: residuals vs predicted std.
    abs_resid = np.abs(y_v_flat - mean_v)
    ax_right.scatter(std_v, abs_resid, alpha=0.4, s=12, color="#1f77b4")
    diag_lim = max(float(std_v.max()), float(abs_resid.max()))
    ax_right.plot([0, diag_lim], [0, diag_lim], "k--", linewidth=1, label="$|y - \\hat\\mu| = \\hat\\sigma$")
    ax_right.set_xlabel("Predicted std $\\hat\\sigma_t$")
    ax_right.set_ylabel("Absolute residual $|y_t - \\hat\\mu_t|$")
    ax_right.set_title("Residual scale vs predicted spread")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"SPY LSTM forecaster: validation 2019-2021 (NLL={valid_nll:.4f}, RMSE={rmse:.5f})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(CHARTS_DIR / "forecaster_calibration.pdf", dpi=200)
    fig.savefig(CHARTS_DIR / "forecaster_calibration.png", dpi=200)
    plt.close(fig)

    # ---- Metrics JSON ---------------------------------------------------
    metrics = {
        "seq_len": SEQ_LEN,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "train_window": [TRAIN_START, TRAIN_END],
        "valid_window": [VALID_START, VALID_END],
        "n_train_sequences": int(len(x_train)),
        "n_valid_sequences": int(len(x_valid)),
        "final_train_nll": train_losses[-1],
        "valid_nll": valid_nll,
        "valid_rmse": rmse,
        "valid_mae": mae,
        "expected_calibration_error": ece,
        "nominal_levels": nominal_levels.tolist(),
        "empirical_coverages": empirical_coverages.tolist(),
        "coverage_50_nominal_50": float(empirical_coverages[np.argmin(np.abs(nominal_levels - 0.5))]),
        "coverage_80_nominal_80": float(empirical_coverages[np.argmin(np.abs(nominal_levels - 0.8))]),
        "coverage_95_nominal_95": float(empirical_coverages[np.argmin(np.abs(nominal_levels - 0.95))]),
    }
    (STATS_DIR / "forecaster_calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Wrote {CHARTS_DIR / 'forecaster_calibration.pdf'}")
    print(f"Wrote {STATS_DIR / 'forecaster_calibration_metrics.json'}")
    print(f"  valid NLL = {valid_nll:.4f}")
    print(f"  valid RMSE = {rmse:.5f}")
    print(f"  ECE = {ece:.3f}")
    print(f"  Empirical coverage @ nominal 80% = {metrics['coverage_80_nominal_80']:.3f}")
    print(f"  Empirical coverage @ nominal 95% = {metrics['coverage_95_nominal_95']:.3f}")


if __name__ == "__main__":
    main()
