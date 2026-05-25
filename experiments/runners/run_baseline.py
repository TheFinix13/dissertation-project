"""Baseline PPO runner — single fold (legacy: train + evaluate on the test window).

Loops over all (ticker, seed) combinations resolved from CLI flags; falls back
to the original single-ticker, three-seed behaviour when invoked with no flags
so existing dissertation numbers stay reproducible.

Run examples:
    venv/bin/python experiments/runners/run_baseline.py
    venv/bin/python experiments/runners/run_baseline.py --tickers basket
    venv/bin/python experiments/runners/run_baseline.py --tickers SPY,QQQ --seeds extended --timesteps 50000 --tag full
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from common import (
    EnvConfig,
    StockEnv,
    add_common_cli,
    close_1d,
    compute_metrics,
    fetch_close_frame,
    load_protocol,
    make_run_id,
    per_cell_path,
    resolve_initial_balance,
    resolve_seeds,
    resolve_tickers,
    set_global_seed,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_cli(parser)
    args = parser.parse_args()

    experiments_root = Path(__file__).resolve().parent.parent
    protocol = load_protocol(experiments_root / "configs" / "dissertation_protocol.json")
    out_dir = experiments_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(args.tag)

    tickers = resolve_tickers(args.tickers, protocol)
    seeds = resolve_seeds(args.seeds, protocol)
    timesteps = args.timesteps if args.timesteps is not None else protocol["baseline"]["timesteps"]
    initial_balance = resolve_initial_balance(args, protocol)

    test_start, test_end = protocol["splits"]["test"]
    model_name = protocol["baseline"]["model_name"]

    rows = []
    failed_cells = []
    for ticker in tickers:
        prices = None  # lazily loaded so we can skip wholly-cached tickers
        for seed in seeds:
            cell_file = per_cell_path(out_dir, "baseline", args.tag, ticker, seed)
            failed_marker = cell_file.with_suffix(".failed.json")

            if cell_file.exists() and not args.no_skip:
                with open(cell_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                rows.append(cached)
                print(f"{ticker:<5} seed={seed:>3}: skipped (cached at {cell_file.name})")
                continue

            if failed_marker.exists() and not args.no_skip:
                print(f"{ticker:<5} seed={seed:>3}: skipped (previously failed; "
                      f"delete {failed_marker.name} to retry)")
                continue

            try:
                if prices is None:
                    try:
                        price_df = fetch_close_frame(ticker, test_start, test_end)
                    except ValueError as e:
                        print(f"[WARN] {ticker}: {e}")
                        break
                    close = close_1d(price_df)
                    prices = close.to_numpy(dtype="float32")

                set_global_seed(seed)
                env_cfg = EnvConfig(initial_balance=initial_balance)

                def _make_env(prices=prices, env_cfg=env_cfg):
                    return StockEnv(prices=prices, cfg=env_cfg)

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

                eval_env = StockEnv(prices=prices, cfg=env_cfg)
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
                metrics["agent"] = model_name
                metrics["device"] = args.device

                with open(cell_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                rows.append(metrics)
                print(
                    f"{ticker:<5} seed={seed:>3} ts={timesteps}: "
                    f"final={metrics['final_portfolio_value']:.2f}, "
                    f"sharpe={metrics['sharpe_ratio']:.4f}, "
                    f"max_dd={metrics['max_drawdown']:.4f}, "
                    f"preservation={metrics['capital_preservation_rate_95pct_hwm']:.4f} "
                    f"-> {cell_file.name}"
                )
            except Exception as exc:  # noqa: BLE001
                err = {
                    "ticker": ticker,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error_msg": str(exc)[:500],
                    "tag": args.tag,
                }
                failed_cells.append(err)
                with open(failed_marker, "w", encoding="utf-8") as f:
                    json.dump(err, f, indent=2)
                print(
                    f"[FAIL] {ticker:<5} seed={seed:>3}: {type(exc).__name__}: "
                    f"{str(exc)[:120]} -> {failed_marker.name}"
                )

    if failed_cells:
        print(f"\n[!] {len(failed_cells)} cell(s) failed during this run:")
        for f in failed_cells:
            print(f"    {f['ticker']:<5} seed={f['seed']:>3}: {f['error_type']}")
        print("    (.failed.json markers written; resume will skip them.)")

    if not rows:
        print("[ERROR] no results were produced; check tickers / network access.")
        return

    json_path = out_dir / f"baseline_{run_id}.json"
    csv_path = out_dir / f"baseline_{run_id}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote baseline results:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
