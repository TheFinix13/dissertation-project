#!/usr/bin/env python3
"""Freeze the post-viva evaluation data before any post-viva run.

STUDY: post-viva, Experiment 1 set-up — writes data/sim_post_viva.npz,
data/spy_holdout_1993_2005.npz, data/spy_holdout_1993_2005_raw.csv and
data/post_viva_data_meta.json. Not in the dissertation.

Two things are frozen, as `notes/post-viva/2026-09-29-experiment-designs.md`
requires.

* Fresh simulated splits from the dissertation's generator, with new seeds:
  a tuning set (100 months per regime, base seed 40,000) and a test set
  (200 per regime, base seed 50,000). The generator parameters are read
  from `data/sim_meta.json`, so the months come from exactly the same
  distribution as the dissertation's.
* The clean real hold-out, SPY from its first trading day (29 January 1993)
  to December 2005. The raw download is saved as well as the episodes,
  because Yahoo Finance re-adjusts past prices on every download.

The script prints counts and file hashes only. It never prints or computes
a return on the hold-out or the test set, so running it does not open them.

Run once:
  ../../venv/bin/python freeze_post_viva_data.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data as real_data
import sim_data

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
META = OUT / "post_viva_data_meta.json"

SPLITS = {"tune": (100, 40_000), "test": (200, 50_000)}
HOLDOUT_FETCH_START = "1993-01-29"
HOLDOUT_END = "2006-01-01"
HOLDOUT_STARTS = ("1993-03-01", "1993-04-01")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_sim(calibration: dict) -> dict[str, list[dict]]:
    parts: dict[str, list[dict]] = {}
    for name, (per_regime, base_seed) in SPLITS.items():
        eps = []
        for r_i, regime in enumerate(sim_data.REGIMES):
            rng = np.random.default_rng(base_seed + r_i)
            for i in range(per_regime):
                ep = sim_data.generate_episode(rng, calibration["regimes"][regime])
                ep["id"] = f"{regime}_{i:04d}"
                eps.append(ep)
        parts[name] = eps
    return parts


def holdout() -> tuple[pd.DataFrame, list[dict], str]:
    raw = real_data.fetch(start=HOLDOUT_FETCH_START, end=HOLDOUT_END)
    for start in HOLDOUT_STARTS:
        try:
            return raw, real_data.build_episodes(raw, nominal_start=start), start
        except RuntimeError as err:
            if "warm-up lead too short" not in str(err):
                raise
    raise RuntimeError("no hold-out start has a full warm-up")


def save_npz(path: Path, parts: dict[str, list[dict]]) -> None:
    arrays = {}
    for name, eps in parts.items():
        for e in eps:
            arrays[f"{name}__{e['id']}__prices"] = e["prices"]
            arrays[f"{name}__{e['id']}__market"] = e["market"]
    np.savez_compressed(path, **arrays)


def main() -> None:
    if META.exists():
        raise SystemExit(f"{META.name} exists; the data is already frozen")
    calibration = json.loads((OUT / "sim_meta.json").read_text())["calibration"]

    sim_parts = fresh_sim(calibration)
    sim_path = OUT / "sim_post_viva.npz"
    save_npz(sim_path, sim_parts)

    raw, episodes, start = holdout()
    raw_path = OUT / "spy_holdout_1993_2005_raw.csv"
    raw.to_csv(raw_path)
    hold_path = OUT / "spy_holdout_1993_2005.npz"
    save_npz(hold_path, {"holdout": episodes})

    meta = {
        "frozen_on": pd.Timestamp.now().isoformat(timespec="seconds"),
        "sim": {
            "generator": "dissertation generator, parameters from data/sim_meta.json",
            "splits": {k: {"per_regime": v[0], "base_seed": v[1]}
                       for k, v in SPLITS.items()},
            "ids": {k: [e["id"] for e in v] for k, v in sim_parts.items()},
            "file": sim_path.name, "sha256": sha256(sim_path)},
        "holdout": {
            "ticker": real_data.TICKER,
            "fetched": [HOLDOUT_FETCH_START, HOLDOUT_END],
            "first_month": start[:7],
            "n_months": len(episodes),
            "ids": [e["id"] for e in episodes],
            "file": hold_path.name, "sha256": sha256(hold_path),
            "raw_file": raw_path.name, "raw_sha256": sha256(raw_path),
            "raw_rows": int(len(raw))},
    }
    META.write_text(json.dumps(meta, indent=2))
    print(json.dumps({
        "sim_counts": {k: len(v) for k, v in sim_parts.items()},
        "sim_sha256": meta["sim"]["sha256"],
        "holdout_months": meta["holdout"]["n_months"],
        "holdout_range": [meta["holdout"]["ids"][0], meta["holdout"]["ids"][-1]],
        "holdout_sha256": meta["holdout"]["sha256"],
        "raw_sha256": meta["holdout"]["raw_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
