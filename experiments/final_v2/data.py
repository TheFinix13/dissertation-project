#!/usr/bin/env python3
"""Data pipeline for the final experiments: OHLCV, continuous features, 3-way split.

Three deliberate differences from the earlier pipeline.

1. **OHLCV is retained.** The earlier pipeline kept `Close` and discarded the
   rest, which makes range-based risk (the difference between a calm flat day
   and a violently whipsawing flat day) unobservable in principle rather than
   merely absent.

2. **Features are computed on the continuous series before episodes are cut.**
   A warm-up lead is fetched before the nominal start date and then discarded,
   so the very first episode already has a fully-defined 20-day trend and a
   14-day average true range.

3. **The split is three-way and chronological.** Any hyper-parameter choice is
   made on validation; the test years are touched once. A two-way split cannot
   support the claim that no tuning saw the test set.

    train  2018-01 .. 2022-12   (60 months)
    val    2023-01 .. 2023-12   (12 months)
    test   2024-01 .. 2025-12   (24 months)

Run:
  ./venv/bin/python experiments/final_v2/data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import CANONICAL_ORDER, MARKET_FEATURES, WARMUP_BARS, compute_market_features

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"

TICKER = "SPY"
WARMUP_START = "2017-09-01"   # discarded; exists only to warm the rolling windows
NOMINAL_START = "2018-01-01"
NOMINAL_END = "2026-01-01"    # exclusive, so the series ends 2025-12-31

TRAIN_END = "2022-12"
VAL_END = "2023-12"

MIN_BARS = 15


def fetch(ticker: str = TICKER) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=WARMUP_START, end=NOMINAL_END,
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("download returned no rows")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"missing OHLCV columns: {sorted(missing)}")
    df = df[sorted(required)].dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def build_episodes(df: pd.DataFrame) -> list[dict]:
    """Compute features on the continuous series, then cut monthly episodes."""
    feats = compute_market_features(df)

    # Drop the warm-up lead only after features exist, so the first retained
    # row already carries a complete history.
    if len(df) <= WARMUP_BARS:
        raise RuntimeError("series shorter than the feature warm-up")
    keep = df.index >= pd.Timestamp(NOMINAL_START)
    if keep.sum() == 0:
        raise RuntimeError("no rows on or after the nominal start")
    first_kept = int(np.flatnonzero(keep)[0])
    if first_kept < WARMUP_BARS:
        raise RuntimeError(
            f"warm-up lead too short: only {first_kept} bars before "
            f"{NOMINAL_START}, need {WARMUP_BARS}")

    df = df.loc[keep]
    feats = feats.loc[keep]
    if feats.isna().any().any():
        bad = feats.columns[feats.isna().any()].tolist()
        raise RuntimeError(f"NaNs survived the warm-up in {bad}")

    ym = df.index.to_period("M").astype(str)
    episodes: list[dict] = []
    for month in pd.unique(ym):
        sel = ym == month
        if int(sel.sum()) < MIN_BARS:
            continue
        episodes.append({
            "id": month,
            "n_bars": int(sel.sum()),
            "start": str(df.index[sel][0].date()),
            "end": str(df.index[sel][-1].date()),
            "prices": df.loc[sel, "Close"].to_numpy(dtype=np.float64),
            "market": feats.loc[sel].to_numpy(dtype=np.float64),
        })
    return episodes


def split(episodes: list[dict]) -> dict[str, list[dict]]:
    train = [e for e in episodes if e["id"] <= TRAIN_END]
    val = [e for e in episodes if TRAIN_END < e["id"] <= VAL_END]
    test = [e for e in episodes if e["id"] > VAL_END]
    for name, part in (("train", train), ("val", val), ("test", test)):
        if not part:
            raise RuntimeError(f"{name} split is empty")
    return {"train": train, "val": val, "test": test}


def save(parts: dict[str, list[dict]]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, part in parts.items():
        for e in part:
            arrays[f"{name}__{e['id']}__prices"] = e["prices"]
            arrays[f"{name}__{e['id']}__market"] = e["market"]
    np.savez_compressed(OUT / "spy_episodes.npz", **arrays)

    meta = {
        "ticker": TICKER,
        "bar": "daily, auto-adjusted OHLCV",
        "episode_def": "one calendar month of daily bars",
        "feature_computation": (
            "market features computed on the continuous series, then sliced "
            "per month; a warm-up lead before the nominal start is discarded"),
        "warmup_bars": WARMUP_BARS,
        "warmup_start": WARMUP_START,
        "range": [NOMINAL_START, NOMINAL_END],
        "split_rule": {"train_end": TRAIN_END, "val_end": VAL_END,
                       "note": "chronological; validation carries all tuning"},
        "market_feature_order": list(MARKET_FEATURES),
        "canonical_feature_order": list(CANONICAL_ORDER),
        "counts": {k: len(v) for k, v in parts.items()},
        "ids": {k: [e["id"] for e in v] for k, v in parts.items()},
        "ranges": {
            k: [{"id": e["id"], "start": e["start"], "end": e["end"],
                 "n_bars": e["n_bars"]} for e in v]
            for k, v in parts.items()
        },
    }
    (OUT / "spy_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load() -> tuple[dict, dict[str, list[dict]]]:
    """Load cached episodes. Each episode is {id, prices, market}."""
    meta = json.loads((OUT / "spy_meta.json").read_text())
    npz = np.load(OUT / "spy_episodes.npz")
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
    df = fetch()
    episodes = build_episodes(df)
    parts = split(episodes)
    meta = save(parts)
    print(json.dumps({
        "counts": meta["counts"],
        "train": [meta["ids"]["train"][0], meta["ids"]["train"][-1]],
        "val": [meta["ids"]["val"][0], meta["ids"]["val"][-1]],
        "test": [meta["ids"]["test"][0], meta["ids"]["test"][-1]],
        "warmup_bars": meta["warmup_bars"],
        "market_features": meta["market_feature_order"],
    }, indent=2))
    print(f"wrote {OUT / 'spy_episodes.npz'}")
    print(f"wrote {OUT / 'spy_meta.json'}")


if __name__ == "__main__":
    main()
