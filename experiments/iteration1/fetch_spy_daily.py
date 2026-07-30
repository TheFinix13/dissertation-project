#!/usr/bin/env python3
"""Download SPY daily bars and split into chronological train/test day episodes.

Episode definition (honest):
  One episode = one calendar month of daily bars (~18–23 steps).
  This is NOT the 390-minute US session; we state that explicitly so Nguyen
  is not misled. Minute-bar upgrade is a later drop-in (same env API).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "iteration1" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def fetch_spy(start: str = "2018-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    df = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    if "Close" not in df.columns:
        raise RuntimeError(f"unexpected columns: {df.columns}")
    out = df[["Close"]].dropna().copy()
    out.index = pd.to_datetime(out.index)
    out["year_month"] = out.index.to_period("M").astype(str)
    return out


def months_to_episodes(df: pd.DataFrame, min_bars: int = 15) -> list[dict]:
    episodes = []
    for ym, g in df.groupby("year_month"):
        prices = g["Close"].to_numpy(dtype=np.float64)
        if len(prices) < min_bars:
            continue
        episodes.append(
            {
                "id": ym,
                "n_bars": int(len(prices)),
                "start": str(g.index[0].date()),
                "end": str(g.index[-1].date()),
                "prices": prices,
            }
        )
    return episodes


def main() -> None:
    df = fetch_spy()
    episodes = months_to_episodes(df)
    if len(episodes) < 10:
        raise RuntimeError(f"too few episodes: {len(episodes)}")

    # Chronological split — never shuffle months
    n = len(episodes)
    n_train = max(5, int(0.7 * n))
    train, test = episodes[:n_train], episodes[n_train:]

    # Persist prices as npz + metadata json (lists can't hold arrays in json)
    meta = {
        "ticker": "SPY",
        "bar": "daily",
        "episode_def": "one calendar month of daily closes",
        "note": "T ≈ 18–23, not 390. Minute-bar episodes are a later upgrade.",
        "fee_default": 0.0005,
        "n_train": len(train),
        "n_test": len(test),
        "train_ids": [e["id"] for e in train],
        "test_ids": [e["id"] for e in test],
        "train_ranges": [{"id": e["id"], "start": e["start"], "end": e["end"], "n_bars": e["n_bars"]} for e in train],
        "test_ranges": [{"id": e["id"], "start": e["start"], "end": e["end"], "n_bars": e["n_bars"]} for e in test],
    }
    (OUT / "spy_daily_meta.json").write_text(json.dumps(meta, indent=2))

    np.savez_compressed(
        OUT / "spy_daily_episodes.npz",
        **{f"train_{e['id']}": e["prices"] for e in train},
        **{f"test_{e['id']}": e["prices"] for e in test},
    )
    print(json.dumps({k: meta[k] for k in ("n_train", "n_test", "train_ids", "test_ids", "episode_def", "note")}, indent=2))
    print(f"wrote {OUT / 'spy_daily_meta.json'}")
    print(f"wrote {OUT / 'spy_daily_episodes.npz'}")


if __name__ == "__main__":
    main()
