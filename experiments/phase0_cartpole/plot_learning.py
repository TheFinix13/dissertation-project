#!/usr/bin/env python3
"""Plot Phase 0 CartPole learning curve for the Nguyen meeting pack."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "phase0_cartpole" / "results"
CHARTS = ROOT / "reports" / "generated" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)


def moving_average(x: np.ndarray, w: int = 20) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main() -> None:
    metrics = json.loads((RES / "metrics.json").read_text())
    rets = np.asarray(metrics["episode_returns"], dtype=float)
    steps = np.asarray(metrics["episode_timesteps"], dtype=float)
    if len(rets) == 0:
        raise SystemExit("No episode returns in metrics.json — run train_ppo.py first.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    ax.scatter(steps, rets, s=8, alpha=0.25, color="#4a6fa5", label="episode return")
    ma = moving_average(rets, 20)
    ax.plot(steps[len(steps) - len(ma) :], ma, color="#0b2c4a", lw=2.2, label="moving avg (20)")
    ax.axhline(
        metrics["random_mean_return_30ep"],
        color="#b33a3a",
        ls="--",
        lw=1.5,
        label=f"random eval mean ({metrics['random_mean_return_30ep']:.0f})",
    )
    ax.axhline(
        metrics["ppo_mean_return_30ep"],
        color="#1a7a4c",
        ls="--",
        lw=1.5,
        label=f"PPO eval mean ({metrics['ppo_mean_return_30ep']:.0f})",
    )
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Episode return")
    ax.set_title("Phase 0 — CartPole-v1 PPO learning curve\n(policy-based RL correctness sandbox)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = CHARTS / "phase0_cartpole_learning_curve.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(RES / "learning_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
