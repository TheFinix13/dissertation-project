#!/usr/bin/env python3
"""Extended real-data transfer set: every SPY month outside the calibration window.

STUDY: main (simulated redo) — builds data/spy_transfer.npz: 180 real
months (2006-17 + 2023-25), including the 2008 crisis. In the dissertation.

Why this module exists
----------------------
The pilot split the real series 60/12/24 (train/val/test) because the agents
*trained* on real months, so most of the history had to be spent on training
and tuning. The simulated-data study removes that constraint: real data enters
it only as (a) the source of the generator's calibration moments (the pooled
2018-2022 training months) and (b) a transfer test for policies trained
entirely in simulation. Every real month outside the calibration window is
therefore available for evaluation, and a transfer test on 24 rising months
(2024-25) was the weakest form that test could take.

This module builds the extended set:

    early  2006-01 .. 2017-12   144 months. Fully out of sample - these months
                                predate the calibration window and were never
                                downloaded by the pilot pipeline. They include
                                the 2008 financial crisis, the 2011 correction
                                and the 2015-16 drawdowns: the real declines
                                the simulated down regime stands in for.
    late   2023-01 .. 2025-12   36 months. After the calibration window; the
                                pilot used 2023 for validation and 2024-25 for
                                test, but the simulated study tunes nothing on
                                real data, so all 36 are usable.

The calibration window 2018-01 .. 2022-12 is excluded: the generator's
per-regime moments were fitted on those months, so evaluating on them would
blur the transfer claim. The pilot's own cached splits (`spy_episodes.npz`)
are not touched; this module writes its own cache.

Run:
  ./venv/bin/python experiments/final_v2/real_transfer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data as real_data

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"

#: Warm-up lead mirrors the pilot's convention: fetched, used to warm the
#: rolling feature windows, then discarded.
WARMUP_START = "2005-09-01"
NOMINAL_START = "2006-01-01"

#: The generator's calibration months (the pilot's training split). Excluded.
CALIB_FIRST = "2018-01"
CALIB_LAST = "2022-12"

WINDOWS = ("early", "late")


def window_of(month_id: str) -> str | None:
    """Which transfer window a month id (YYYY-MM) belongs to, if any."""
    if month_id < CALIB_FIRST:
        return "early"
    if month_id > CALIB_LAST:
        return "late"
    return None


def build() -> dict[str, list[dict]]:
    df = real_data.fetch(start=WARMUP_START, end=real_data.NOMINAL_END)
    episodes = real_data.build_episodes(df, nominal_start=NOMINAL_START)
    parts: dict[str, list[dict]] = {w: [] for w in WINDOWS}
    for e in episodes:
        w = window_of(e["id"])
        if w is not None:
            parts[w].append(e)
    for w in WINDOWS:
        if not parts[w]:
            raise RuntimeError(f"transfer window {w!r} is empty")
    return parts


def save(parts: dict[str, list[dict]]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, part in parts.items():
        for e in part:
            arrays[f"{name}__{e['id']}__prices"] = e["prices"]
            arrays[f"{name}__{e['id']}__market"] = e["market"]
    np.savez_compressed(OUT / "spy_transfer.npz", **arrays)

    meta = {
        "ticker": real_data.TICKER,
        "bar": "daily, auto-adjusted OHLCV",
        "episode_def": "one calendar month of daily bars",
        "purpose": ("transfer evaluation only; no agent trains or tunes on "
                    "any real month in the simulated-data study"),
        "warmup_start": WARMUP_START,
        "nominal_start": NOMINAL_START,
        "excluded_calibration_window": [CALIB_FIRST, CALIB_LAST],
        "windows": {
            "early": "2006-01 .. 2017-12, predates the calibration window",
            "late": "2023-01 .. 2025-12, after the calibration window",
        },
        "counts": {k: len(v) for k, v in parts.items()},
        "ids": {k: [e["id"] for e in v] for k, v in parts.items()},
    }
    (OUT / "spy_transfer_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load() -> tuple[dict, dict[str, list[dict]]]:
    """Load cached transfer episodes; same episode shape as `data.load()`."""
    meta = json.loads((OUT / "spy_transfer_meta.json").read_text())
    npz = np.load(OUT / "spy_transfer.npz")
    parts: dict[str, list[dict]] = {}
    for name, ids in meta["ids"].items():
        parts[name] = [
            {"id": i,
             "prices": npz[f"{name}__{i}__prices"],
             "market": npz[f"{name}__{i}__market"]}
            for i in ids
        ]
    return meta, parts


def all_episodes(parts: dict[str, list[dict]]) -> list[dict]:
    """Both windows as one chronologically ordered list."""
    merged = parts["early"] + parts["late"]
    return sorted(merged, key=lambda e: e["id"])


def main() -> None:
    parts = build()
    meta = save(parts)
    print(json.dumps({
        "counts": meta["counts"],
        "early": [meta["ids"]["early"][0], meta["ids"]["early"][-1]],
        "late": [meta["ids"]["late"][0], meta["ids"]["late"][-1]],
        "excluded": meta["excluded_calibration_window"],
    }, indent=2))
    print(f"wrote {OUT / 'spy_transfer.npz'}")
    print(f"wrote {OUT / 'spy_transfer_meta.json'}")


if __name__ == "__main__":
    main()
