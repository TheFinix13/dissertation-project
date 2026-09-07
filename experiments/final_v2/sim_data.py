#!/usr/bin/env python3
"""Simulated market data: regime-switching episodes calibrated on real SPY.

Why this module exists
----------------------
The real-data study trains on 60 monthly episodes. The policy network has
thousands of parameters, so the agents memorise the training months instead of
learning a policy (Chapter 5 documents the always-buy collapse). The root cause
is data volume, not architecture. The remedy used in the literature is to train
on simulated data drawn from a generator calibrated on the real series, and to
keep the real series for a secondary domain-transfer test.

Design
------
* **Three regimes: up, down, flat.** One episode is one regime throughout, so
  test results can be attributed to a market condition. This is the controlled
  experiment the real data cannot provide: the 2024-25 test years only rise.

* **Calibration uses the real training months only** (SPY 2018-2022). Each
  real month is labelled by its simple return (up > +2%, down < -2%, flat
  otherwise). Daily log-returns are pooled per label and give the per-regime
  drift and volatility. The intraday-range scale is fitted so the generated
  mean ATR/close matches the real per-regime mean. Validation and test months
  of the real series are never read here.

* **Balanced splits from the same distribution.** Train, validation and test
  are independent draws from the same generator with equal regime counts.
  Nothing distinguishes the splits except the random seed, so the train/test
  mismatch of the chronological real-data split cannot occur by construction.

* **Episodes are format-compatible with `data.py`.** Each episode is
  {id, prices, market} with the market features computed by the same
  `features.compute_market_features` code path on a warm-up lead that is
  generated and then discarded, exactly as the real pipeline does.

Generator
---------
Daily log-returns are i.i.d. Normal(mu_r, sigma_r) within a regime. High and
Low extend the close-to-close move by half-normal wicks whose scale is fitted
to the real per-regime ATR/close. Volume is lognormal; only relative volume
enters the feature set, so its absolute scale is irrelevant. This is the
simplest generator that supports the controlled test; richer generators
(HMM + OU, GANs) exist and are discussed in Chapter 2, but their realism is
not needed to answer a question about learning dynamics, and their opacity
would cost the control that motivates simulation in the first place.

Run:
  ./venv/bin/python experiments/final_v2/sim_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data as real_data
from features import MARKET_FEATURES, WARMUP_BARS, compute_market_features

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"

REGIMES = ("up", "down", "flat")

#: Monthly simple-return thresholds for labelling the real calibration months.
UP_THRESHOLD = 0.02
DOWN_THRESHOLD = -0.02

#: Bars per episode, matching a real trading month.
EPISODE_BARS = 21

#: Episodes per regime in each split.
COUNTS = {"train": 1000, "val": 100, "test": 200}

#: Base seeds so the three splits are independent draws.
SPLIT_SEEDS = {"train": 10_000, "val": 20_000, "test": 30_000}

#: Starting price for every generated series. The feature set is scale-free
#: (returns and ratios), and the slice action commits dollars, not shares, so
#: the level itself carries no information.
P0 = 500.0

HALF_NORMAL_MEAN = np.sqrt(2.0 / np.pi)  # E|N(0,1)|


# ------------------------------------------------------------- calibration

def label_month(prices: np.ndarray) -> str:
    r = float(prices[-1] / prices[0] - 1.0)
    if r > UP_THRESHOLD:
        return "up"
    if r < DOWN_THRESHOLD:
        return "down"
    return "flat"


def calibrate() -> dict:
    """Per-regime daily drift, volatility and range scale from real SPY train months."""
    meta, parts = real_data.load()
    atr_col = list(MARKET_FEATURES).index("atr_norm")

    pooled: dict[str, list[np.ndarray]] = {r: [] for r in REGIMES}
    pooled_atr: dict[str, list[np.ndarray]] = {r: [] for r in REGIMES}
    counts = {r: 0 for r in REGIMES}
    for ep in parts["train"]:
        regime = label_month(ep["prices"])
        counts[regime] += 1
        pooled[regime].append(np.diff(np.log(ep["prices"])))
        pooled_atr[regime].append(ep["market"][:, atr_col])

    params: dict[str, dict] = {}
    for regime in REGIMES:
        if not pooled[regime]:
            raise RuntimeError(f"no real training months labelled {regime!r}")
        rets = np.concatenate(pooled[regime])
        mu = float(rets.mean())
        sigma = float(rets.std(ddof=1))
        target_atr = float(np.concatenate(pooled_atr[regime]).mean())
        # E[TR/C] ~= E|ret| + E[upper wick] + E[lower wick]
        #          = sigma*sqrt(2/pi) + 2 * wick_scale * sqrt(2/pi)
        wick_scale = max(
            (target_atr - sigma * HALF_NORMAL_MEAN) / (2.0 * HALF_NORMAL_MEAN),
            5e-4)
        params[regime] = {
            "n_months": counts[regime],
            "n_daily_returns": int(len(rets)),
            "mu_daily_log": mu,
            "sigma_daily_log": sigma,
            "target_atr_norm": target_atr,
            "wick_scale": float(wick_scale),
        }
    return {
        "source": "real SPY training months only",
        "train_months": [meta["ids"]["train"][0], meta["ids"]["train"][-1]],
        "labelling": {"up": f"> {UP_THRESHOLD:+.0%}", "down": f"< {DOWN_THRESHOLD:+.0%}",
                      "flat": "otherwise"},
        "regimes": params,
    }


# -------------------------------------------------------------- generation

def generate_episode(rng: np.random.Generator, regime_params: dict) -> dict:
    """One synthetic episode: warm-up lead + EPISODE_BARS bars, one regime."""
    n = WARMUP_BARS + EPISODE_BARS
    mu = regime_params["mu_daily_log"]
    sigma = regime_params["sigma_daily_log"]
    wick = regime_params["wick_scale"]

    log_rets = rng.normal(mu, sigma, size=n - 1)
    close = P0 * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)]))

    prev_close = np.concatenate([[close[0]], close[:-1]])
    hi_base = np.maximum(prev_close, close)
    lo_base = np.minimum(prev_close, close)
    high = hi_base * (1.0 + np.abs(rng.normal(0.0, wick, size=n)))
    low = lo_base * (1.0 - np.abs(rng.normal(0.0, wick, size=n)))

    volume = 1e8 * np.exp(rng.normal(0.0, 0.35, size=n))

    df = pd.DataFrame({
        "Open": prev_close, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=pd.RangeIndex(n))
    feats = compute_market_features(df)

    prices = close[WARMUP_BARS:]
    market = feats.iloc[WARMUP_BARS:].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(market)):
        raise RuntimeError("non-finite features in generated episode")
    return {"prices": prices, "market": market}


def generate_split(name: str, calibration: dict) -> list[dict]:
    episodes: list[dict] = []
    for r_i, regime in enumerate(REGIMES):
        rng = np.random.default_rng(SPLIT_SEEDS[name] + r_i)
        params = calibration["regimes"][regime]
        for i in range(COUNTS[name]):
            ep = generate_episode(rng, params)
            ep["id"] = f"{regime}_{i:04d}"
            episodes.append(ep)
    return episodes


def regime_of(episode_id: str) -> str:
    return episode_id.split("_", 1)[0]


# ------------------------------------------------------------------- io

def save(parts: dict[str, list[dict]], calibration: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, part in parts.items():
        for e in part:
            arrays[f"{name}__{e['id']}__prices"] = e["prices"]
            arrays[f"{name}__{e['id']}__market"] = e["market"]
    np.savez_compressed(OUT / "sim_episodes.npz", **arrays)

    meta = {
        "generator": "regime-switching i.i.d. lognormal daily returns",
        "episode_def": f"{EPISODE_BARS} daily bars, one regime throughout",
        "warmup_bars": WARMUP_BARS,
        "p0": P0,
        "regimes": list(REGIMES),
        "counts_per_regime": COUNTS,
        "split_seeds": SPLIT_SEEDS,
        "calibration": calibration,
        "market_feature_order": list(MARKET_FEATURES),
        "counts": {k: len(v) for k, v in parts.items()},
        "ids": {k: [e["id"] for e in v] for k, v in parts.items()},
    }
    (OUT / "sim_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load() -> tuple[dict, dict[str, list[dict]]]:
    """Load cached simulated episodes; same shape as `data.load()`."""
    meta = json.loads((OUT / "sim_meta.json").read_text())
    npz = np.load(OUT / "sim_episodes.npz")
    parts: dict[str, list[dict]] = {}
    for name, ids in meta["ids"].items():
        parts[name] = [
            {"id": i,
             "prices": npz[f"{name}__{i}__prices"],
             "market": npz[f"{name}__{i}__market"]}
            for i in ids
        ]
    return meta, parts


def main() -> None:
    calibration = calibrate()
    print(json.dumps(calibration, indent=2))
    parts = {name: generate_split(name, calibration) for name in COUNTS}
    meta = save(parts, calibration)

    # Calibration check: generated mean ATR/close per regime vs target.
    atr_col = list(MARKET_FEATURES).index("atr_norm")
    for regime in REGIMES:
        got = np.concatenate([
            e["market"][:, atr_col] for e in parts["train"]
            if regime_of(e["id"]) == regime]).mean()
        want = calibration["regimes"][regime]["target_atr_norm"]
        print(f"atr_norm {regime:5s}: generated {got:.5f}  target {want:.5f}")

    print(json.dumps({"counts": meta["counts"]}, indent=2))
    print(f"wrote {OUT / 'sim_episodes.npz'}")
    print(f"wrote {OUT / 'sim_meta.json'}")


if __name__ == "__main__":
    main()
