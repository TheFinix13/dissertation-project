"""Probabilistic-PPO runner — single fold (legacy: train + evaluate on the test window).

Loops over all (ticker, seed) combinations resolved from CLI flags; falls back
to the original single-ticker, three-seed behaviour when invoked with no flags.
Optional --bootstrap-paths concatenates Politis-Romano synthetic price paths to
the real series before training, expanding the effective training set without
leaving the empirical return distribution.

Uncertainty modes (--uncertainty-mode):
    aleatoric (default): the LSTM outputs a predicted spread per step via
        Gaussian negative log-likelihood. Captures noise in the data only.
    epistemic: adds MC Dropout sampling on top of the same network. Captures
        both aleatoric noise AND the model's own uncertainty about its
        predictions (Gal & Ghahramani, 2016).

Run examples:
    venv/bin/python experiments/runners/run_probabilistic_agent.py
    venv/bin/python experiments/runners/run_probabilistic_agent.py --tickers basket
    venv/bin/python experiments/runners/run_probabilistic_agent.py --tickers SPY --seeds extended --timesteps 50000 --bootstrap-paths 16 --tag full
    venv/bin/python experiments/runners/run_probabilistic_agent.py --uncertainty-mode epistemic --tag epistemic
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    EnvConfig,
    StockEnv,
    add_common_cli,
    close_1d,
    compute_metrics,
    fetch_close_frame,
    load_protocol,
    make_run_id,
    maybe_bootstrap_training_prices,
    resolve_initial_balance,
    resolve_seeds,
    resolve_tickers,
    set_global_seed,
)


class ProbabilisticLSTM(nn.Module):
    """LSTM forecaster with Gaussian NLL head.

    A dropout layer between the LSTM and the heads enables MC Dropout
    (Gal & Ghahramani, 2016) for estimating epistemic uncertainty. When
    dropout=0 the network behaves identically to the original
    aleatoric-only forecaster, preserving backwards compatibility.
    """

    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.0):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=lstm_dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = self.drop(out[:, -1, :])
        mean = self.fc_mean(last)
        log_var = self.fc_logvar(last)
        return mean, log_var


def gaussian_nll(y_true, mean, log_var):
    var = torch.exp(log_var) + 1e-6
    return 0.5 * (torch.log(var) + ((y_true - mean) ** 2) / var).mean()


def build_sequences(data: np.ndarray, seq_len: int):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    x = np.asarray(x, dtype=np.float32)[:, :, None]
    y = np.asarray(y, dtype=np.float32)[:, None]
    return x, y


def estimate_uncertainty(
    prices: np.ndarray,
    seq_len: int = 20,
    epochs: int = 20,
    mode: str = "aleatoric",
    mc_passes: int = 20,
    dropout: float = 0.1,
) -> np.ndarray:
    """Train the LSTM forecaster on log-returns and return a 0-1 uncertainty score per step.

    Parameters
    ----------
    mode:
        "aleatoric" (default) — single deterministic forward pass after training.
        The score is the predicted Gaussian standard deviation (data noise).
        "epistemic" — runs ``mc_passes`` forward passes with dropout active.
        The score combines the mean predicted std (aleatoric) with the
        cross-pass standard deviation of predictions (epistemic).
    mc_passes:
        Number of stochastic forward passes used when mode='epistemic'.
    dropout:
        Dropout rate. Only takes effect when mode='epistemic'; aleatoric mode
        forces dropout=0 so the original behaviour is reproduced exactly.
    """
    if mode not in {"aleatoric", "epistemic"}:
        raise ValueError(f"unknown uncertainty mode: {mode!r}")

    flat_prices = np.asarray(prices, dtype=np.float32).reshape(-1)
    returns = np.diff(np.log(np.maximum(flat_prices, 1e-8))).astype(np.float32)
    x, y = build_sequences(returns, seq_len=seq_len)
    xt = torch.tensor(x)
    yt = torch.tensor(y)

    effective_dropout = dropout if mode == "epistemic" else 0.0
    model = ProbabilisticLSTM(dropout=effective_dropout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        mean, log_var = model(xt)
        loss = gaussian_nll(yt, mean, log_var)
        loss.backward()
        opt.step()

    if mode == "aleatoric":
        model.eval()
        with torch.no_grad():
            _, log_var = model(xt)
            std = torch.exp(0.5 * log_var).squeeze(-1).numpy()
    else:
        # MC Dropout: keep dropout active and average over multiple passes.
        # Aleatoric component = mean predicted std across passes.
        # Epistemic component = std of predicted means across passes.
        model.train()
        means_per_pass = []
        stds_per_pass = []
        with torch.no_grad():
            for _ in range(mc_passes):
                mean_pred, log_var = model(xt)
                means_per_pass.append(mean_pred.squeeze(-1).numpy())
                stds_per_pass.append(torch.exp(0.5 * log_var).squeeze(-1).numpy())
        means_arr = np.asarray(means_per_pass)        # (mc_passes, N)
        stds_arr = np.asarray(stds_per_pass)          # (mc_passes, N)
        aleatoric = stds_arr.mean(axis=0)             # average data noise
        epistemic = means_arr.std(axis=0)             # disagreement across passes
        std = aleatoric + epistemic                   # total predictive uncertainty

    padded = np.zeros(len(prices), dtype=np.float32)
    values = np.clip(std, 1e-6, None)
    values = (values - values.min()) / (values.max() - values.min() + 1e-8)
    start = seq_len + 1
    padded[start : start + len(values)] = values
    if start > 0:
        padded[:start] = values[0]
    return padded


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_cli(parser)
    parser.add_argument(
        "--uncertainty-mode",
        choices=["aleatoric", "epistemic"],
        default=None,
        help="Forecaster uncertainty mode. 'aleatoric' (default) uses the original "
             "single-pass Gaussian NLL. 'epistemic' adds MC Dropout sampling to also "
             "capture model uncertainty (Gal & Ghahramani, 2016).",
    )
    parser.add_argument(
        "--mc-passes", type=int, default=20,
        help="Number of MC Dropout forward passes when --uncertainty-mode=epistemic (default: 20).",
    )
    parser.add_argument(
        "--mc-dropout", type=float, default=0.1,
        help="Dropout rate used by MC Dropout when --uncertainty-mode=epistemic (default: 0.1).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    protocol = load_protocol(root / "configs" / "dissertation_protocol.json")
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(args.tag)

    tickers = resolve_tickers(args.tickers, protocol)
    seeds = resolve_seeds(args.seeds, protocol)
    timesteps = (
        args.timesteps if args.timesteps is not None
        else protocol["probabilistic_agent"]["timesteps"]
    )
    initial_balance = resolve_initial_balance(args, protocol)
    bootstrap_paths = int(args.bootstrap_paths)
    uncertainty_mode = (
        args.uncertainty_mode
        or protocol["probabilistic_agent"].get("uncertainty_mode", "aleatoric")
    )

    test_start, test_end = protocol["splits"]["test"]
    model_name = protocol["probabilistic_agent"]["model_name"]
    if uncertainty_mode == "epistemic":
        model_name = f"{model_name}_epistemic"
    cfg_overrides = dict(
        uncertainty_stop_quantile=protocol["probabilistic_agent"]["uncertainty_quantile_stop"],
        min_trade_scale=protocol["probabilistic_agent"]["position_scale_floor"],
        initial_balance=initial_balance,
    )

    rows = []
    for ticker in tickers:
        try:
            price_df = fetch_close_frame(ticker, test_start, test_end)
        except ValueError as e:
            print(f"[WARN] {ticker}: {e}")
            continue
        close = close_1d(price_df)
        prices = close.to_numpy(dtype="float32")
        uncertainty = estimate_uncertainty(
            prices,
            mode=uncertainty_mode,
            mc_passes=args.mc_passes,
            dropout=args.mc_dropout,
        )

        for seed in seeds:
            set_global_seed(seed)
            env_cfg = EnvConfig(**cfg_overrides)

            train_prices = maybe_bootstrap_training_prices(
                prices, num_paths=bootstrap_paths, protocol=protocol, seed=seed,
            )
            train_uncertainty = (
                estimate_uncertainty(
                    train_prices,
                    mode=uncertainty_mode,
                    mc_passes=args.mc_passes,
                    dropout=args.mc_dropout,
                )
                if bootstrap_paths > 0
                else uncertainty
            )

            def _make_env(prices=train_prices, uncertainty=train_uncertainty, env_cfg=env_cfg):
                return StockEnv(prices=prices, uncertainty=uncertainty, cfg=env_cfg)

            env = DummyVecEnv([_make_env])
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=5,
                seed=seed,
                verbose=0,
            )
            model.learn(total_timesteps=timesteps)

            eval_env = StockEnv(prices=prices, uncertainty=uncertainty, cfg=env_cfg)
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, done, _, _ = eval_env.step(action)

            portfolio_values = eval_env.portfolio_values
            metrics = compute_metrics(portfolio_values)
            metrics["seed"] = seed
            metrics["ticker"] = ticker
            metrics["fold_id"] = "test_legacy"
            metrics["timesteps"] = timesteps
            metrics["bootstrap_paths"] = bootstrap_paths
            metrics["agent"] = model_name
            metrics["uncertainty_mode"] = uncertainty_mode
            if uncertainty_mode == "epistemic":
                metrics["mc_passes"] = args.mc_passes
                metrics["mc_dropout"] = args.mc_dropout
            rows.append(metrics)
            print(
                f"{ticker:<5} seed={seed:>3} ts={timesteps} bs={bootstrap_paths} unc={uncertainty_mode}: "
                f"final={metrics['final_portfolio_value']:.2f}, "
                f"sharpe={metrics['sharpe_ratio']:.4f}, "
                f"max_dd={metrics['max_drawdown']:.4f}, "
                f"preservation={metrics['capital_preservation_rate_95pct_hwm']:.4f}"
            )

    if not rows:
        print("[ERROR] no results were produced; check tickers / network access.")
        return

    json_path = out_dir / f"probabilistic_{run_id}.json"
    csv_path = out_dir / f"probabilistic_{run_id}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote probabilistic results:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
