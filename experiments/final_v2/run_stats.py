#!/usr/bin/env python3
"""Statistical tests on the saved main-study results. No training.

STUDY: post-viva engineering — NOT in the submitted dissertation. Writes
results/stats.json from results/sim_results.json, results/stop_loss.json
and results/other_assets.json.

Why this exists (28 Sep 2026). The final report feedback asked for
"statistical rigour around the seed variation". The dissertation reported
six-seed means and spreads without confidence intervals or tests, so a
reader could not tell which gaps were larger than chance.

Real months (SPY, AAPL, QQQ; 180 months each). Every algorithm has a
6-seed x 180-month matrix of monthly wealth change. Two things are random:
which seeds were drawn and which months were observed. A two-level
bootstrap resamples seeds with replacement, then months in circular blocks
of six so that runs of volatile months stay together. Blocks wrap inside
each transfer window, never across the 2018-2022 gap. Buy-and-hold uses the
same month draw in every replicate, so every comparison is paired. Months
are also resampled one at a time as a sensitivity check.

Sharpe is the dissertation's definition: mean over standard deviation of
monthly return, computed per seed and then averaged across seeds (0.224 for
DQN on SPY). A seed that never trades has Sharpe 0.

Exposure-matched buy-and-hold (SPY only; the other-asset runs did not save
buy-and-hold exposure). Holding a fraction k of the buy-and-hold position
scales every monthly profit, and the proportional fee, by k. So for each
seed, k = seed mean exposure / buy-and-hold mean exposure, and the matched
comparator is k times buy-and-hold. Sharpe is unchanged by this scaling, so
the Sharpe test above is already exposure-matched. The matched mean and the
matched worst-months figure ask whether an agent did better than simply
holding less stock.

Simulated test (SPY only; 600 episodes). Only per-seed means were saved,
so the seed is the unit of analysis (n = 6) and intervals are Student t.
Buy-and-hold is deterministic on the fixed test set and enters as a
constant.

Run (from experiments/final_v2):
  ../../venv/bin/python run_stats.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

import real_transfer
from harness import INITIAL_CASH, provenance, write_results

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "stats.json"

ALGOS = ("dqn", "reinforce", "ppo")
CRISIS_FIRST, CRISIS_LAST = "2008-09", "2009-03"
CVAR_Q = 0.10


def block_indices(ids: list[str], block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap inside each transfer window.

    The real months are two separate windows (2006-17 and 2023-25). Blocks
    wrap within a window, never from December 2017 into January 2023.
    """
    n = len(ids)
    if block <= 1:
        return rng.integers(0, n, n)
    out = []
    for window in sorted({real_transfer.window_of(m) or "" for m in ids}):
        pos = np.array([i for i, m in enumerate(ids)
                        if (real_transfer.window_of(m) or "") == window])
        w = len(pos)
        starts = rng.integers(0, w, -(-w // block))
        local = ((starts[:, None] + np.arange(block)[None, :]) % w).ravel()[:w]
        out.append(pos[local])
    return np.concatenate(out)


def sharpe_rows(x: np.ndarray) -> np.ndarray:
    """Per-row Sharpe of monthly returns; rows with zero spread score 0."""
    r = x / INITIAL_CASH
    sd = r.std(axis=-1, ddof=1)
    mu = r.mean(axis=-1)
    return np.where(sd > 1e-12, mu / np.where(sd > 1e-12, sd, 1.0), 0.0)


def cvar_rows(x: np.ndarray, q: float = CVAR_Q) -> np.ndarray:
    """Mean of the worst q fraction of months, per row (dollars)."""
    k = max(1, int(round(q * x.shape[-1])))
    return np.sort(x, axis=-1)[..., :k].mean(axis=-1)


def point_stats(X: np.ndarray, bah: np.ndarray) -> dict:
    return {
        "mean_dw": float(X.mean()),
        "sharpe": float(sharpe_rows(X).mean()),
        "cvar10": float(cvar_rows(X).mean()),
        "bah_mean_dw": float(bah.mean()),
        "bah_sharpe": float(sharpe_rows(bah[None, :])[0]),
        "bah_cvar10": float(cvar_rows(bah[None, :])[0]),
    }


def bootstrap(X: np.ndarray, bah: np.ndarray, ids: list[str], n_boot: int,
              block: int, rng: np.random.Generator,
              k: np.ndarray | None = None) -> dict[str, np.ndarray]:
    S, M = X.shape
    assert len(ids) == M
    keys = ["mean_dw", "sharpe", "cvar10", "d_mean", "d_sharpe", "d_cvar10"]
    if k is not None:
        keys += ["dm_mean", "dm_cvar10"]
    out = {key: np.empty(n_boot) for key in keys}
    for b in range(n_boot):
        s_idx = rng.integers(0, S, S) if S > 1 else np.zeros(1, dtype=int)
        m_idx = block_indices(ids, block, rng)
        Xb = X[s_idx][:, m_idx]
        bb = bah[m_idx][None, :]
        mean_dw = Xb.mean()
        sh = sharpe_rows(Xb).mean()
        cv = cvar_rows(Xb).mean()
        bah_mean = bb.mean()
        bah_cv = cvar_rows(bb)[0]
        out["mean_dw"][b] = mean_dw
        out["sharpe"][b] = sh
        out["cvar10"][b] = cv
        out["d_mean"][b] = mean_dw - bah_mean
        out["d_sharpe"][b] = sh - sharpe_rows(bb)[0]
        out["d_cvar10"][b] = cv - bah_cv
        if k is not None:
            kb = k[s_idx].mean()
            out["dm_mean"][b] = mean_dw - kb * bah_mean
            out["dm_cvar10"][b] = cv - kb * bah_cv
    return out


def ci(a: np.ndarray) -> list[float]:
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def p_two_sided(boot: np.ndarray, observed: float) -> float:
    """Bootstrap p for H0: difference = 0, by shifting the draws to the null."""
    shifted = boot - boot.mean()
    return float(np.mean(np.abs(shifted) >= abs(observed)))


def analyse_real(X: np.ndarray, bah: np.ndarray, ids: list[str],
                 n_boot: int, rng: np.random.Generator,
                 k: np.ndarray | None = None) -> dict:
    pt = point_stats(X, bah)
    res: dict = {"n_seeds": int(X.shape[0]), "n_months": int(X.shape[1]),
                 "point": pt}
    if k is not None:
        res["exposure_ratio_k"] = [float(v) for v in k]
        pt["matched_bah_mean_dw"] = float(k.mean() * pt["bah_mean_dw"])
        pt["matched_bah_cvar10"] = float(k.mean() * pt["bah_cvar10"])
        pt["seeds_cvar_better_than_matched"] = int(np.sum(
            cvar_rows(X) > k * pt["bah_cvar10"]))
    for label, block in (("block6", 6), ("iid", 1)):
        bs = bootstrap(X, bah, ids, n_boot, block, rng, k)
        if k is not None:
            dm_mean = pt["mean_dw"] - pt["matched_bah_mean_dw"]
            dm_cvar = pt["cvar10"] - pt["matched_bah_cvar10"]
            res.setdefault("vs_matched_bah", {})[label] = {
                "mean_dw": dm_mean, "mean_dw_ci": ci(bs["dm_mean"]),
                "mean_dw_p": p_two_sided(bs["dm_mean"], dm_mean),
                "cvar10": dm_cvar, "cvar10_ci": ci(bs["dm_cvar10"]),
                "cvar10_p": p_two_sided(bs["dm_cvar10"], dm_cvar),
            }
        d_mean = pt["mean_dw"] - pt["bah_mean_dw"]
        d_sharpe = pt["sharpe"] - pt["bah_sharpe"]
        d_cvar = pt["cvar10"] - pt["bah_cvar10"]
        res[label] = {
            "mean_dw_ci": ci(bs["mean_dw"]),
            "sharpe_ci": ci(bs["sharpe"]),
            "cvar10_ci": ci(bs["cvar10"]),
            "diff_vs_bah": {
                "mean_dw": d_mean, "mean_dw_ci": ci(bs["d_mean"]),
                "mean_dw_p": p_two_sided(bs["d_mean"], d_mean),
                "sharpe": d_sharpe, "sharpe_ci": ci(bs["d_sharpe"]),
                "sharpe_p": p_two_sided(bs["d_sharpe"], d_sharpe),
                "cvar10": d_cvar, "cvar10_ci": ci(bs["d_cvar10"]),
                "cvar10_p": p_two_sided(bs["d_cvar10"], d_cvar),
            },
        }
    seed_avg = X.mean(axis=0)
    diff = seed_avg - bah
    nz = diff[np.abs(diff) > 1e-9]
    res["wilcoxon_seedavg_vs_bah"] = (
        {"statistic": float(sps.wilcoxon(nz).statistic),
         "p": float(sps.wilcoxon(nz).pvalue), "n_nonzero": int(nz.size)}
        if nz.size >= 10 else None)
    res["months_beating_bah"] = float(np.mean(seed_avg > bah))
    seed_means = X.mean(axis=1)
    res["per_seed_mean_dw"] = [float(v) for v in seed_means]
    res["per_seed_sharpe"] = [float(v) for v in sharpe_rows(X)]
    res["n_seeds_profitable"] = int(np.sum(seed_means > 0))
    band = [i for i, m in enumerate(ids) if CRISIS_FIRST <= m <= CRISIS_LAST]
    if band:
        res["crisis_band"] = {
            "months": [ids[band[0]], ids[band[-1]]],
            "sum_dw": float(X[:, band].sum(axis=1).mean()),
            "bah_sum_dw": float(bah[band].sum()),
        }
    return res


def spy_real(sim: dict) -> tuple[dict[str, np.ndarray], np.ndarray, list[str],
                                  dict[str, np.ndarray], float]:
    base_all = sim["baselines_real"]["B1b_true_bah"]
    base = base_all["per_month"]
    ids = [m["id"] for m in base]
    bah = np.array([m["delta_w"] for m in base])
    bah_expo = float(base_all["summary"]["mean_exposure"])
    mats, ks = {}, {}
    for al in ALGOS:
        rows, expo = [], []
        for s in sim["results"][al]["per_seed"]:
            pm = s["real_per_month"]
            assert [m["id"] for m in pm] == ids, f"{al}: month ids misaligned"
            rows.append([m["delta_w"] for m in pm])
            expo.append(s["real_transfer"]["mean_exposure"])
        mats[al] = np.array(rows)
        ks[al] = np.array(expo) / bah_expo
    return mats, bah, ids, ks, bah_expo


def other_real(asset: dict) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    ids = list(asset["ids"])
    bah = np.array([m["delta_w"] for m in asset["bah_per_month"]])
    assert [m["id"] for m in asset["bah_per_month"]] == ids
    mats = {}
    for al in ALGOS:
        rows = []
        for seed_rows in asset["algos"][al]:
            assert [m["id"] for m in seed_rows] == ids, f"{al}: ids misaligned"
            rows.append([m["delta_w"] for m in seed_rows])
        mats[al] = np.array(rows)
    return mats, bah, ids


def t_interval(v: np.ndarray) -> dict:
    n = v.size
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = float(sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": m, "sd": sd, "ci95": [m - h, m + h]}


def analyse_sim(sim: dict) -> dict:
    bah = sim["baselines_sim"]["B1b_true_bah"]
    out: dict = {"unit": "seed (n=6); per-episode rows were not saved",
                 "bah": {"overall": bah["summary"]["mean_delta_w"],
                         **{r: bah["per_regime"][r]["mean_delta_w"]
                            for r in ("up", "down", "flat")}}}
    seed_means = {}
    for al in ALGOS:
        ps = sim["results"][al]["per_seed"]
        v = np.array([s["summary"]["mean_delta_w"] for s in ps])
        seed_means[al] = v
        row = {"overall": t_interval(v)}
        for r in ("up", "down", "flat"):
            row[r] = t_interval(np.array(
                [s["per_regime"][r]["mean_delta_w"] for s in ps]))
        row["vs_zero_p"] = float(sps.ttest_1samp(v, 0.0).pvalue)
        row["vs_bah_p"] = float(sps.ttest_1samp(
            v, out["bah"]["overall"]).pvalue)
        out[al] = row
    out["dqn_vs"] = {
        al: {"welch_p": float(sps.ttest_ind(seed_means["dqn"], seed_means[al],
                                            equal_var=False).pvalue),
             "mannwhitney_p": float(sps.mannwhitneyu(
                 seed_means["dqn"], seed_means[al],
                 alternative="two-sided").pvalue)}
        for al in ("reinforce", "ppo")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260928)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    sim = json.loads((RESULTS / "sim_results.json").read_text())
    stop = json.loads((RESULTS / "stop_loss.json").read_text())
    other = json.loads((RESULTS / "other_assets.json").read_text())

    real: dict = {}
    mats, bah, ids, ks, bah_expo = spy_real(sim)
    real["SPY"] = {al: analyse_real(mats[al], bah, ids, args.n_boot, rng,
                                    ks[al])
                   for al in ALGOS}
    sl = stop["stop_loss"]["transfer_per_month"]
    assert [m["id"] for m in sl] == ids, "stop-loss months misaligned"
    sl_k = np.array([stop["stop_loss"]["transfer"]["mean_exposure"] / bah_expo])
    real["SPY"]["stop_loss_trailing5"] = analyse_real(
        np.array([[m["delta_w"] for m in sl]]), bah, ids, args.n_boot, rng,
        sl_k)
    for asset, payload in other["assets"].items():
        m2, b2, i2 = other_real(payload)
        real[asset] = {al: analyse_real(m2[al], b2, i2, args.n_boot, rng)
                       for al in ALGOS}

    payload = {
        "study": "post-viva statistics on saved results (not in dissertation)",
        "method": {
            "real": "two-level bootstrap: seeds with replacement, then months "
                    "(circular blocks of 6; iid as sensitivity); buy-and-hold "
                    "paired on the same month draw",
            "sharpe": "per-seed monthly Sharpe averaged across seeds",
            "cvar10": "mean of the worst 10% of months, dollars",
            "p_values": "bootstrap two-sided, draws shifted to the null",
            "sim": "Student t over the 6 per-seed means",
            "n_boot": args.n_boot, "rng_seed": args.seed,
        },
        "real": real,
        "sim_test": analyse_sim(sim),
        "provenance": provenance(),
    }
    write_results(OUT, payload)

    print("\nSPY vs exposure-matched buy-and-hold (block-6)")
    for name, r in real["SPY"].items():
        p, m = r["point"], r["vs_matched_bah"]["block6"]
        print(f"  {name:22s} mean {p['mean_dw']:7.2f} vs {p['matched_bah_mean_dw']:7.2f}"
              f" p={m['mean_dw_p']:.3f} | CVaR10 {p['cvar10']:8.2f} vs"
              f" {p['matched_bah_cvar10']:8.2f} d={m['cvar10']:+7.2f}"
              f" [{m['cvar10_ci'][0]:+7.2f},{m['cvar10_ci'][1]:+7.2f}]"
              f" p={m['cvar10_p']:.3f}"
              f" seeds better {p['seeds_cvar_better_than_matched']}/{r['n_seeds']}")
    for asset, rows in real.items():
        print(f"\n{asset}  (block-6 bootstrap, 95% CI)")
        for name, r in rows.items():
            p, d = r["point"], r["block6"]["diff_vs_bah"]
            print(f"  {name:22s} mean {p['mean_dw']:8.2f} "
                  f"[{r['block6']['mean_dw_ci'][0]:7.2f},{r['block6']['mean_dw_ci'][1]:7.2f}]"
                  f"  vs BAH {d['mean_dw']:+8.2f} p={d['mean_dw_p']:.3f}"
                  f" | Sharpe {p['sharpe']:.3f} vs {p['bah_sharpe']:.3f}"
                  f" d={d['sharpe']:+.3f} [{d['sharpe_ci'][0]:+.3f},{d['sharpe_ci'][1]:+.3f}]"
                  f" p={d['sharpe_p']:.3f}"
                  f" | CVaR10 {p['cvar10']:8.2f} vs {p['bah_cvar10']:8.2f}"
                  f" p={d['cvar10_p']:.3f}")
    s = payload["sim_test"]
    print("\nSim test (seed = unit, n=6)")
    for al in ALGOS:
        o = s[al]["overall"]
        print(f"  {al:10s} {o['mean']:8.2f} [{o['ci95'][0]:8.2f},{o['ci95'][1]:8.2f}]"
              f"  p(vs 0)={s[al]['vs_zero_p']:.4f}  p(vs BAH)={s[al]['vs_bah_p']:.2e}")
    print("  dqn vs:", s["dqn_vs"])


if __name__ == "__main__":
    main()
