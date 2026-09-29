#!/usr/bin/env python3
"""Gate 2 and Gate 4 measurements for the post-viva experiments.

STUDY: post-viva — no training. Writes results/gate_audit.json. Supports
notes/post-viva/2026-09-29-gate-*.md.

Every number the gate documents quote comes from this script, so each one
can be regenerated. It measures six things.

1. Volume against capacity: trainable parameters per algorithm, distinct
   training transitions, and how many gradient updates each algorithm gets
   from the same 300,000-step budget.
2. Balance: what constant strategies earn, and the ceiling a policy reaches
   if it knows the regime but trades only at the first bar.
3. Signal content: whether the market features at bar t identify the
   month's regime (linear discriminant) and predict the rest of the month's
   return (out-of-sample R squared), in simulation and in real SPY.
4. Realism: stylised facts of daily returns in the generator against real
   SPY (fat tails, volatility clustering, regime persistence across months).
5. Coverage: which (bar, exposure) states a uniform random policy visits.
6. Seeds against months: how much of the real-market interval width comes
   from seed variation, and what 20 seeds would buy.

Run:
  ../../venv/bin/python gate_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

import agents
import data as real_data
import real_transfer
import sim_data
from env import make_env_factory
from features import MARKET_FEATURES, resolve_rung
from harness import INITIAL_CASH, SLICE_FRAC, FEE

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "gate_audit.json"
SIM_RESULTS = HERE / "results" / "sim_results.json"

S3 = resolve_rung("S3")
MKT_S3 = [f for f in S3 if f in MARKET_FEATURES]
MKT_COLS = [list(MARKET_FEATURES).index(f) for f in MKT_S3]
BARS = (0, 5, 10, 15)
TIMESTEPS = 300_000
RNG_SEED = 20260929


# ------------------------------------------------------------ 1. capacity

def n_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def capacity(train_eps: list[dict]) -> dict:
    obs_dim = len(S3)
    q = agents.QNet(obs_dim)
    pi = agents.PolicyNet(obs_dim)

    from sb3_contrib import MaskablePPO
    from month_sampler import MonthSampler
    factory = make_env_factory(S3, initial_cash=INITIAL_CASH, fee=FEE,
                               slice_frac=SLICE_FRAC)
    ppo = MaskablePPO("MlpPolicy", MonthSampler(factory, train_eps[:3], seed=0),
                      seed=0, verbose=0,
                      policy_kwargs={"net_arch": list(agents.HIDDEN_DEFAULT)})
    n_ppo = n_params(ppo.policy)

    steps_per_ep = len(train_eps[0]["prices"]) - 1
    distinct = len(train_eps) * steps_per_ep
    n_rollouts = -(-TIMESTEPS // ppo.n_steps)
    ppo_minibatch_steps = n_rollouts * ppo.n_epochs * (ppo.n_steps // ppo.batch_size)
    dqn_steps = TIMESTEPS - 1_000
    return {
        "obs_dim": obs_dim,
        "params": {"dqn": n_params(q), "reinforce": n_params(pi), "ppo": n_ppo},
        "train_episodes": len(train_eps),
        "steps_per_episode": steps_per_ep,
        "distinct_transitions": distinct,
        "transitions_per_param": {
            "dqn": distinct / n_params(q),
            "reinforce": distinct / n_params(pi),
            "ppo": distinct / n_ppo},
        "episodes_per_param": {
            "dqn": len(train_eps) / n_params(q),
            "ppo": len(train_eps) / n_ppo},
        "budget_timesteps": TIMESTEPS,
        "passes_over_train_set": TIMESTEPS / distinct,
        "gradient_updates": {
            "dqn": dqn_steps,
            "reinforce": TIMESTEPS // steps_per_ep,
            "ppo": ppo_minibatch_steps},
        "samples_through_gradient": {
            "dqn": dqn_steps * 64,
            "reinforce": TIMESTEPS,
            "ppo": n_rollouts * ppo.n_epochs * ppo.n_steps},
        "ppo_defaults_used": {
            "learning_rate": ppo.learning_rate, "n_steps": ppo.n_steps,
            "batch_size": ppo.batch_size, "n_epochs": ppo.n_epochs,
            "gamma": ppo.gamma, "gae_lambda": ppo.gae_lambda,
            "ent_coef": ppo.ent_coef, "vf_coef": ppo.vf_coef,
            "clip_range": 0.2, "normalize_advantage": ppo.normalize_advantage,
            "reward_normalisation": "none (raw dollars into the value loss)"},
    }


# ------------------------------------------------------------- 2. balance

def balance(sim: dict) -> dict:
    per = {name: {g: v["mean_delta_w"] for g, v in b["per_regime"].items()}
           for name, b in sim["baselines_sim"].items()}
    bah, cash = per["B1b_true_bah"], per["B0_do_nothing"]
    oracle = {g: max(bah[g], cash[g]) for g in sim_data.REGIMES}
    dqn = sim["results"]["dqn"]["across_seeds"]
    return {
        "sim_test_constant": {name: b["summary"]["mean_delta_w"]
                              for name, b in sim["baselines_sim"].items()},
        "sim_test_per_regime": per,
        "regime_known_constant_ceiling": {
            "per_regime": oracle,
            "mean": float(np.mean(list(oracle.values())))},
        "dqn_sim_mean": dqn["mean_delta_w"]["mean"],
        "dqn_regime_means": {g: dqn[f"regime_{g}_delta_w"]["mean"]
                             for g in sim_data.REGIMES},
        "real_constant": {name: b["summary"]["mean_delta_w"]
                          for name, b in sim["baselines_real"].items()},
    }


# ------------------------------------------------------ 3. signal content

def xy(eps: list[dict], bar: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features at `bar`, log return over the rest of the month, realised label.

    The realised label buckets the whole month's return (the same rule that
    labelled the calibration months). From bar 1 on, the features already
    contain part of that return, so accuracy at later bars is not a forecast.
    """
    X = np.array([e["market"][bar, MKT_COLS] for e in eps])
    rest = np.array([np.log(e["prices"][-1] / e["prices"][bar]) for e in eps])
    lab = np.array([sim_data.REGIMES.index(sim_data.label_month(e["prices"]))
                    for e in eps])
    return X, rest, lab


def latent(eps: list[dict]) -> np.ndarray:
    """The generating regime of a simulated month, read from its id."""
    return np.array([sim_data.REGIMES.index(sim_data.regime_of(e["id"])) for e in eps])


def majority_rate(train_labels: np.ndarray, test_labels: np.ndarray) -> float:
    """Accuracy of always predicting the training set's most common label."""
    return float(np.mean(test_labels == np.bincount(train_labels).argmax()))


def lda_fit(X: np.ndarray, y: np.ndarray):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    classes = np.unique(y)
    means = np.array([Z[y == c].mean(0) for c in classes])
    resid = np.concatenate([Z[y == c] - means[i] for i, c in enumerate(classes)])
    cov = np.cov(resid.T) + 1e-6 * np.eye(Z.shape[1])
    inv = np.linalg.inv(cov)
    prior = np.log(np.array([np.mean(y == c) for c in classes]))

    def predict(Xn: np.ndarray) -> np.ndarray:
        Zn = (Xn - mu) / sd
        score = Zn @ inv @ means.T - 0.5 * np.sum(means @ inv * means, 1) + prior
        return classes[np.argmax(score, 1)]
    return predict


def ols_fit(X: np.ndarray, y: np.ndarray):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ beta, float(y.mean())


def oos_r2(pred: np.ndarray, y: np.ndarray, bench: float) -> float:
    """Out-of-sample R^2 against the training-mean forecast (Campbell-Thompson)."""
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - bench) ** 2))


def signal(sim_parts: dict, real_train: list[dict], real_test: list[dict]) -> dict:
    out: dict = {}
    ids_q = [e["id"] for e in real_test]
    for bar in BARS:
        Xs, rs, ls = xy(sim_parts["train"], bar)
        Xt, rt, lt = xy(sim_parts["test"], bar)
        Xr, rr, lr = xy(real_train, bar)
        Xq, rq, lq = xy(real_test, bar)

        zs, zt = latent(sim_parts["train"]), latent(sim_parts["test"])
        lda_sim = lda_fit(Xs, ls)
        lda_latent = lda_fit(Xs, zs)
        lda_real = lda_fit(Xr, lr)
        f_sim, m_sim = ols_fit(Xs, rs)
        f_real, m_real = ols_fit(Xr, rr)
        out[f"bar_{bar}"] = {
            "generating_regime_accuracy_sim": {
                "accuracy": float(np.mean(lda_latent(Xt) == zt)),
                "majority_baseline": majority_rate(zs, zt)},
            "realised_label_accuracy": {
                "sim_to_sim": float(np.mean(lda_sim(Xt) == lt)),
                "sim_majority_baseline": majority_rate(ls, lt),
                "sim_to_real": float(np.mean(lda_sim(Xq) == lq)),
                "real_to_real": float(np.mean(lda_real(Xq) == lq)),
                "real_majority_baseline": majority_rate(lr, lq)},
            "rest_of_month_oos_r2": {
                "sim_to_sim": oos_r2(f_sim(Xt), rt, m_sim),
                "sim_to_real": oos_r2(f_sim(Xq), rq, m_real),
                "real_to_real": oos_r2(f_real(Xq), rq, m_real),
                "real_to_real_ci95": r2_interval(f_real(Xq), rq, m_real, ids_q),
                "benchmark": "training-mean forecast; the real calibration mean "
                             "for both real rows"},
        }
    return out


def r2_interval(pred: np.ndarray, y: np.ndarray, bench: float, ids: list[str],
                n_boot: int = 5000, block: int = 6) -> list[float]:
    """Block-bootstrap interval for out-of-sample R^2 over the test months."""
    rng = np.random.default_rng(RNG_SEED)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = window_blocks(ids, block, rng)
        vals[b] = oos_r2(pred[idx], y[idx], bench)
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


# -------------------------------------------------------------- 4. realism

def daily_returns(eps: list[dict]) -> list[np.ndarray]:
    return [np.diff(np.log(e["prices"])) for e in eps]


def lag1_acf(series: list[np.ndarray]) -> float:
    """Lag-1 autocorrelation from pairs inside each month, about the pooled mean.

    Demeaning each 20-day month separately biases the estimate by about -1/19
    and removes clustering at the monthly scale, so one pooled mean is used.
    """
    m = np.concatenate(series).mean()
    a = np.concatenate([s[:-1] - m for s in series])
    b = np.concatenate([s[1:] - m for s in series])
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def half_sign_flip(eps: list[dict]) -> float:
    """Share of months whose first-half and second-half returns differ in sign."""
    flips = []
    for e in eps:
        p = e["prices"]
        h = len(p) // 2
        flips.append(np.sign(p[h] / p[0] - 1) != np.sign(p[-1] / p[h] - 1))
    return float(np.mean(flips))


def realism(sim_parts: dict, real_early: list[dict], real_all: list[dict]) -> dict:
    def facts(eps: list[dict]) -> dict:
        r = daily_returns(eps)
        flat = np.concatenate(r)
        return {
            "n_daily": int(len(flat)),
            "excess_kurtosis": float(sps.kurtosis(flat)),
            "acf1_return": lag1_acf(r),
            "acf1_abs_return": lag1_acf([np.abs(x) for x in r]),
            "half_month_sign_flip": half_sign_flip(eps),
        }

    out = {"sim_train": facts(sim_parts["train"]),
           "real_transfer_180": facts(real_all)}

    # Across consecutive real months (the early window is 144 unbroken months).
    vol = np.array([np.std(np.diff(np.log(e["prices"])), ddof=1) for e in real_early])
    labels = [sim_data.label_month(e["prices"]) for e in real_early]
    trans = {a: {b: 0 for b in sim_data.REGIMES} for a in sim_data.REGIMES}
    for a, b in zip(labels[:-1], labels[1:]):
        trans[a][b] += 1
    trans_p = {a: {b: trans[a][b] / max(1, sum(trans[a].values()))
                   for b in sim_data.REGIMES} for a in sim_data.REGIMES}
    out["real_consecutive_months"] = {
        "n_months": len(real_early),
        "monthly_vol_autocorr": float(np.corrcoef(vol[:-1], vol[1:])[0, 1]),
        "label_transition_prob": trans_p,
        "label_counts": {g: labels.count(g) for g in sim_data.REGIMES},
        "sim_equivalent": "0 by construction: every episode is an independent draw",
    }
    return out


# ------------------------------------------------------------- 5. coverage

def coverage(train_eps: list[dict], n_eps: int = 3000) -> dict:
    factory = make_env_factory(S3, initial_cash=INITIAL_CASH, fee=FEE,
                               slice_frac=SLICE_FRAC)
    rng = np.random.default_rng(RNG_SEED)
    levels = 5  # exposure 0, 1/4, 2/4, 3/4, 1 of capital with a quarter slice
    counts = np.zeros((len(train_eps[0]["prices"]) - 1, levels), dtype=int)
    for i in range(n_eps):
        env = factory(train_eps[i % len(train_eps)])
        obs, _ = env.reset()
        done = False
        while not done:
            expo = env.units * env._price() / INITIAL_CASH
            counts[env.t, min(levels - 1, int(round(expo / SLICE_FRAC)))] += 1
            legal = np.flatnonzero(env.action_masks())
            obs, _, term, trunc, _ = env.step(int(rng.choice(legal)))
            done = term or trunc
    reachable = np.array([[lvl <= t for lvl in range(levels)]
                          for t in range(counts.shape[0])])
    visited = counts[reachable]
    return {
        "policy": "uniform over legal actions",
        "episodes": n_eps,
        "cells_reachable": int(reachable.sum()),
        "cells_visited_30_plus": int(np.sum(visited >= 30)),
        "min_visits_reachable": int(visited.min()),
        "full_exposure_share_by_last_bar": float(counts[-1, -1] / counts[-1].sum()),
    }


# ------------------------------------------------------ 6. seeds v months

def window_blocks(ids: list[str], block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block indices drawn inside each transfer window.

    The 180 months are two separate windows (2006-17 and 2023-25), so a block
    must never run from December 2017 into January 2023.
    """
    out = []
    for window in real_transfer.WINDOWS:
        pos = np.array([i for i, m in enumerate(ids)
                        if real_transfer.window_of(m) == window])
        n = len(pos)
        starts = rng.integers(0, n, -(-n // block))
        local = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        out.append(pos[local])
    return np.concatenate(out)


def seeds_vs_months(sim: dict, n_boot: int = 5000, block: int = 6) -> dict:
    """Interval width for the mean gap to buy-and-hold, by number of seeds.

    Each replicate draws `n` seeds with replacement from the trained ones and
    then months in blocks, and recomputes the whole mean. Drawing 20 from six
    trained seeds understates the true seed diversity, so the 20-seed width is
    a projection, not a measurement. The months-only width holds the six seeds
    fixed and is the floor that more seeds cannot go below.
    """
    ids = [m["id"] for m in sim["baselines_real"]["B1b_true_bah"]["per_month"]]
    bah = np.array([m["delta_w"] for m in
                    sim["baselines_real"]["B1b_true_bah"]["per_month"]])
    out = {}
    rng = np.random.default_rng(RNG_SEED)
    for algo, entry in sim["results"].items():
        X = np.array([[m["delta_w"] for m in s["real_per_month"]]
                      for s in entry["per_seed"]])
        assert [m["id"] for m in entry["per_seed"][0]["real_per_month"]] == ids
        D = X - bah[None, :]
        S = D.shape[0]

        def half_width(n_seeds: int | None) -> float:
            vals = np.empty(n_boot)
            for b in range(n_boot):
                m_idx = window_blocks(ids, block, rng)
                rows = (np.arange(S) if n_seeds is None
                        else rng.integers(0, S, n_seeds))
                vals[b] = D[np.ix_(rows, m_idx)].mean()
            lo, hi = np.percentile(vals, [2.5, 97.5])
            return float((hi - lo) / 2)

        widths = {"6_seeds": half_width(6), "20_seeds_projected": half_width(20),
                  "months_only": half_width(None)}
        out[algo] = {
            "mean_diff_vs_bah": float(D.mean()),
            "ci_half_width": widths,
            "months_only_share_of_6_seed_width": widths["months_only"] / widths["6_seeds"],
        }
    return out


def main() -> None:
    sim = json.loads(SIM_RESULTS.read_text())
    _, sim_parts = sim_data.load()
    _, real_parts = real_data.load()
    _, tparts = real_transfer.load()
    real_all = real_transfer.all_episodes(tparts)
    real_early = sorted(tparts["early"], key=lambda e: e["id"])

    payload = {
        "study": "post-viva gate audit (no training)",
        "features": list(S3),
        "capacity": capacity(sim_parts["train"]),
        "balance": balance(sim),
        "signal": signal(sim_parts, real_parts["train"], real_all),
        "realism": realism(sim_parts, real_early, real_all),
        "coverage": coverage(sim_parts["train"]),
        "seeds_vs_months": seeds_vs_months(sim),
        "notes": {
            "signal_real_train": "the 60 calibration months 2018-2022",
            "signal_real_test": "the 180 transfer months",
            "regime_labels": "monthly return > +2% up, < -2% down, else flat",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
