"""Two-piece ablation of Equation 3.9.

Splits the contribution into its two components and trains a separate PPO
agent for each:

  Arm D --- state-only:
    The uncertainty score u_t is appended to the policy state but the
    environment's trade-scaling factor (1 - u_t) and the binary risk-on
    guard are both disabled. The policy can condition on u_t but the
    environment does not enforce anything.

  Arm E --- guard-only:
    The binary risk-on guard is active (high-uncertainty buys are blocked)
    but the continuous trade-scaling factor is disabled AND the
    uncertainty coordinate in the policy's state is zero-masked. The
    policy is therefore blind to u_t in its observation, while the
    environment quietly suppresses high-uncertainty buys.

  Arm F --- scaling-only:
    The continuous trade-scaling factor is active but the binary guard
    is disabled and the state is masked.

The aim is to attribute the headline gain to the two pieces separately.
Each cell still uses 50,000 PPO time-steps and 10 seeds per ticker, the
same budget as the canonical Phase-2 grid in `experiments/results/`.

Run on the Colab GPU (T4 / A100). Approximate wall time per arm on a T4:
2--3 hours for the 70-ticker x 10-seed grid.

Run:
    venv/bin/python experiments/runners/run_ablation.py --ablation state_only \\
        --tickers market_sample --seeds extended --timesteps 50000 \\
        --tag ablation_state_only
    venv/bin/python experiments/runners/run_ablation.py --ablation guard_only \\
        --tickers market_sample --seeds extended --timesteps 50000 \\
        --tag ablation_guard_only
    venv/bin/python experiments/runners/run_ablation.py --ablation scaling_only \\
        --tickers market_sample --seeds extended --timesteps 50000 \\
        --tag ablation_scaling_only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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
    per_cell_path,
    resolve_initial_balance,
    resolve_seeds,
    resolve_tickers,
    set_global_seed,
)

# Import the forecaster utilities from the sibling runner.
from run_probabilistic_agent import estimate_uncertainty


ABLATION_FLAGS = {
    # Arm D --- state has real u_t; env does not enforce.
    "state_only": dict(
        enable_trade_scaling=False,
        enable_risk_on_guard=False,
        mask_uncertainty_in_state=False,
    ),
    # Arm E --- env enforces the binary guard only; state is masked.
    "guard_only": dict(
        enable_trade_scaling=False,
        enable_risk_on_guard=True,
        mask_uncertainty_in_state=True,
    ),
    # Arm F --- env enforces the continuous scaling only; state is masked.
    "scaling_only": dict(
        enable_trade_scaling=True,
        enable_risk_on_guard=False,
        mask_uncertainty_in_state=True,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_cli(parser)
    parser.add_argument(
        "--ablation",
        choices=sorted(ABLATION_FLAGS.keys()),
        required=True,
        help="Which ablation arm to run.",
    )
    parser.add_argument(
        "--uncertainty-mode",
        choices=["aleatoric", "epistemic"],
        default="aleatoric",
        help="Forecaster mode (default aleatoric, matching the headline grid).",
    )
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--mc-dropout", type=float, default=0.1)
    args = parser.parse_args()

    experiments_root = Path(__file__).resolve().parent.parent
    protocol = load_protocol(experiments_root / "configs" / "dissertation_protocol.json")
    out_dir = experiments_root / "results"
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
    ablation_kwargs = dict(ABLATION_FLAGS[args.ablation])

    test_start, test_end = protocol["splits"]["test"]
    agent_name = f"ppo_ablation_{args.ablation}"

    rows = []
    failed_cells = []
    for ticker in tickers:
        try:
            price_df = fetch_close_frame(ticker, test_start, test_end)
        except ValueError as exc:
            print(f"[WARN] {ticker}: {exc}")
            continue
        close = close_1d(price_df)
        prices = close.to_numpy(dtype="float32")
        uncertainty = estimate_uncertainty(
            prices,
            mode=args.uncertainty_mode,
            mc_passes=args.mc_passes,
            dropout=args.mc_dropout,
        )

        for seed in seeds:
            cell_file = per_cell_path(
                out_dir,
                f"ablation_{args.ablation}",
                args.tag,
                ticker,
                seed,
            )
            failed_marker = cell_file.with_suffix(".failed.json")
            if cell_file.exists() and not args.no_skip:
                with open(cell_file, "r", encoding="utf-8") as f:
                    rows.append(json.load(f))
                continue
            if failed_marker.exists() and not args.no_skip:
                continue

            try:
                set_global_seed(seed)
                env_cfg = EnvConfig(
                    initial_balance=initial_balance,
                    uncertainty_stop_quantile=protocol["probabilistic_agent"][
                        "uncertainty_quantile_stop"
                    ],
                    min_trade_scale=protocol["probabilistic_agent"][
                        "position_scale_floor"
                    ],
                    **ablation_kwargs,
                )

                train_prices = maybe_bootstrap_training_prices(
                    prices, num_paths=bootstrap_paths, protocol=protocol, seed=seed,
                )
                train_uncertainty = (
                    estimate_uncertainty(
                        train_prices,
                        mode=args.uncertainty_mode,
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
                    device=args.device,
                    verbose=0,
                )
                model.learn(total_timesteps=timesteps)

                eval_env = StockEnv(prices=prices, uncertainty=uncertainty, cfg=env_cfg)
                obs, _ = eval_env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, _, done, _, _ = eval_env.step(action)

                metrics = compute_metrics(eval_env.portfolio_values)
                metrics["seed"] = seed
                metrics["ticker"] = ticker
                metrics["fold_id"] = "test_legacy"
                metrics["timesteps"] = timesteps
                metrics["bootstrap_paths"] = bootstrap_paths
                metrics["agent"] = agent_name
                metrics["ablation"] = args.ablation
                metrics["uncertainty_mode"] = args.uncertainty_mode
                metrics["enable_trade_scaling"] = env_cfg.enable_trade_scaling
                metrics["enable_risk_on_guard"] = env_cfg.enable_risk_on_guard
                metrics["mask_uncertainty_in_state"] = env_cfg.mask_uncertainty_in_state

                with open(cell_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                rows.append(metrics)
                print(
                    f"{ticker:<5} seed={seed:>3} ablation={args.ablation}: "
                    f"final={metrics['final_portfolio_value']:.2f}, "
                    f"sharpe={metrics['sharpe_ratio']:.4f}, "
                    f"max_dd={metrics['max_drawdown']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001
                err = {
                    "ticker": ticker,
                    "seed": seed,
                    "ablation": args.ablation,
                    "error_type": type(exc).__name__,
                    "error_msg": str(exc)[:500],
                }
                failed_cells.append(err)
                with open(failed_marker, "w", encoding="utf-8") as f:
                    json.dump(err, f, indent=2)
                print(f"[FAIL] {ticker} seed={seed}: {type(exc).__name__}: {str(exc)[:120]}")

    if failed_cells:
        print(f"\n[!] {len(failed_cells)} cell(s) failed.")

    if not rows:
        print("[ERROR] no rows produced.")
        return

    json_path = out_dir / f"ablation_{args.ablation}_{run_id}.json"
    csv_path = out_dir / f"ablation_{args.ablation}_{run_id}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote: {json_path}\n       {csv_path}")


if __name__ == "__main__":
    main()
