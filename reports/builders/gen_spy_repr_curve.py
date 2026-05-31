"""Regenerate the representative SPY curve used by the dissertation figures.

This produces ``experiments/results/spy_repr_curve_phase2.csv``: a single
representative SPY run over the 2022--2025 test window with aligned per-day
baseline equity, probabilistic (uncertainty-guarded) equity, and the aleatoric
and epistemic uncertainty scores, at the Phase-2 budget (50,000 PPO time-steps,
seed 42).

Unlike the offline chart builder ``plot_phase2_charts.py``, this script trains
PPO agents and downloads SPY prices from Yahoo Finance, so it requires the
project virtual-env and network access. Its output CSV is committed so that the
chart builder stays fully offline and deterministic; this script only needs to
be re-run if the representative curve itself must be regenerated.

Run from the repository root::

    venv/bin/python reports/builders/gen_spy_repr_curve.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "runners"))

from common import (  # noqa: E402
    EnvConfig,
    StockEnv,
    close_1d,
    fetch_close_frame,
    load_protocol,
    set_global_seed,
)
from run_probabilistic_agent import estimate_uncertainty  # noqa: E402

SEED = 42
TIMESTEPS = 50000
TICKER = "SPY"

protocol = load_protocol(ROOT / "experiments" / "configs" / "dissertation_protocol.json")
test_start, test_end = protocol["splits"]["test"]
q_stop = protocol["probabilistic_agent"]["uncertainty_quantile_stop"]
floor = protocol["probabilistic_agent"]["position_scale_floor"]

print(f"Fetching {TICKER} {test_start}..{test_end}", flush=True)
price_df = fetch_close_frame(TICKER, test_start, test_end)
close = close_1d(price_df)
prices = close.to_numpy(dtype="float32")
dates = pd.to_datetime(price_df.index)
print(f"Got {len(prices)} trading days", flush=True)

print("Estimating aleatoric uncertainty", flush=True)
unc_aleatoric = estimate_uncertainty(prices, mode="aleatoric")
print("Estimating epistemic uncertainty", flush=True)
unc_epistemic = estimate_uncertainty(prices, mode="epistemic")


def train_eval(uncertainty, prob):
    set_global_seed(SEED)
    if prob:
        cfg = EnvConfig(
            uncertainty_stop_quantile=q_stop,
            min_trade_scale=floor,
            initial_balance=1_000_000.0,
        )
        make = lambda: StockEnv(prices=prices, uncertainty=uncertainty, cfg=cfg)
    else:
        cfg = EnvConfig(initial_balance=1_000_000.0)
        make = lambda: StockEnv(prices=prices, cfg=cfg)
    env = DummyVecEnv([make])
    model = PPO(
        "MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=64,
        n_epochs=5, seed=SEED, device="cpu", verbose=0,
    )
    model.learn(total_timesteps=TIMESTEPS)
    eval_env = StockEnv(prices=prices, uncertainty=uncertainty, cfg=cfg) if prob \
        else StockEnv(prices=prices, cfg=cfg)
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, _, _ = eval_env.step(action)
    return np.asarray(eval_env.portfolio_values, dtype=float), eval_env.cfg.lookback


print("Training probabilistic PPO (aleatoric, 50k)", flush=True)
prob_pv, lookback = train_eval(unc_aleatoric, prob=True)
print("Training baseline PPO (50k)", flush=True)
base_pv, _ = train_eval(None, prob=False)

n = min(len(prob_pv), len(base_pv))
prob_pv = prob_pv[:n]
base_pv = base_pv[:n]
seg_dates = dates[lookback : lookback + n]
seg_alea = unc_aleatoric[lookback : lookback + n]
seg_epis = unc_epistemic[lookback : lookback + n]

out = pd.DataFrame({
    "date": seg_dates,
    "baseline_value": base_pv,
    "prob_value": prob_pv,
    "uncertainty_aleatoric": seg_alea,
    "uncertainty_epistemic": seg_epis,
})
out_path = ROOT / "experiments" / "results" / "spy_repr_curve_phase2.csv"
out.to_csv(out_path, index=False)
print(f"WROTE {out_path} rows={len(out)}", flush=True)
print(f"baseline final={base_pv[-1]:,.0f} prob final={prob_pv[-1]:,.0f}", flush=True)
corr = np.corrcoef(seg_alea, seg_epis)[0, 1]
print(f"aleatoric-epistemic per-day corr={corr:.4f}", flush=True)
