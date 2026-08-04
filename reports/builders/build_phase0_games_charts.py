#!/usr/bin/env python3
"""Phase 0 charts: per-env algorithm comparison bars + learning curves.

Reads suite_metrics.json (+ qlearning_metrics.json for CartPole) and emits
Word-ready PNGs (+ PDFs) under reports/generated/charts/.

Run:
  .venv311/bin/python reports/builders/build_phase0_games_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GAMES = ROOT / "experiments" / "phase0_games" / "results"
CHARTS = ROOT / "reports" / "generated" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

ALGO_ORDER = ["random", "qlearning", "dqn", "reinforce", "a2c", "ppo"]
ALGO_LABEL = {
    "random": "Random",
    "qlearning": "Tabular\nQ-learning",
    "dqn": "DQN",
    "reinforce": "REINFORCE\n(scratch)",
    "a2c": "A2C",
    "ppo": "PPO",
}
ALGO_COLOR = {
    "random": "#8a8a8a",
    "qlearning": "#b06a3b",
    "dqn": "#c4952c",
    "reinforce": "#4a6fa5",
    "a2c": "#3b8a8a",
    "ppo": "#1a7a4c",
}
CURVE_ALGOS = ["reinforce", "a2c", "ppo", "dqn"]


def load_env(dirname: str) -> dict | None:
    p = GAMES / dirname / "suite_metrics.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    ql = GAMES / dirname / "qlearning_metrics.json"
    if ql.exists():
        qd = json.loads(ql.read_text())
        data["results"]["qlearning"] = {
            "algorithm": "tabular_q_learning",
            "mean_eval_return": qd["mean_eval_return"],
            "episode_returns": [],
            "episode_timesteps": [],
        }
    return data


def smooth(x: np.ndarray, k: int = 25) -> np.ndarray:
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def bar_chart(env_name: str, data: dict, fname: str, solved: float | None = None) -> None:
    res = data["results"]
    keys = [k for k in ALGO_ORDER if k in res]
    means = [res[k]["mean_eval_return"] for k in keys]
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=160)
    x = np.arange(len(keys))
    ax.bar(x, means, color=[ALGO_COLOR[k] for k in keys])
    for xi, m in zip(x, means):
        ax.text(xi, m + (abs(max(means)) * 0.02 or 1), f"{m:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    if solved is not None:
        ax.axhline(solved, color="#aa3333", lw=1.4, ls="--")
        ax.text(len(keys) - 0.5, solved, f"  solved ≈ {solved:.0f}",
                color="#aa3333", fontsize=9.5, va="bottom", ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABEL[k] for k in keys])
    ax.set_ylabel("Mean evaluation return (30 episodes)")
    ax.set_title(f"Phase 0 — {env_name}: algorithm comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / f"{fname}.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {CHARTS / (fname + '.png')}")


def curve_chart(env_name: str, data: dict, fname: str, solved: float | None = None) -> None:
    res = data["results"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=160)
    for k in CURVE_ALGOS:
        if k not in res or not res[k]["episode_returns"]:
            continue
        r = np.asarray(res[k]["episode_returns"], dtype=float)
        t = np.asarray(res[k]["episode_timesteps"], dtype=float)
        rs = smooth(r)
        ts = t[len(t) - len(rs):]
        ax.plot(ts, rs, lw=1.8, label=ALGO_LABEL[k].replace("\n", " "),
                color=ALGO_COLOR[k])
    if solved is not None:
        ax.axhline(solved, color="#aa3333", lw=1.2, ls="--", label=f"solved ≈ {solved:.0f}")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Episode return (25-episode moving average)")
    ax.set_title(f"Phase 0 — {env_name}: learning curves")
    ax.legend(loc="best", fontsize=9.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / f"{fname}.png", dpi=220, bbox_inches="tight")
    fig.savefig(CHARTS / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {CHARTS / (fname + '.png')}")


def main() -> None:
    specs = [
        ("cartpole", "CartPole-v1", 475.0),
        ("flappy", "FlappyBird-v0", None),
        ("lunarlander", "LunarLander-v3", 200.0),
    ]
    for dirname, env_name, solved in specs:
        data = load_env(dirname)
        if data is None:
            print(f"skip {dirname}: no suite_metrics.json yet")
            continue
        bar_chart(env_name, data, f"phase0_{dirname}_comparison", solved)
        curve_chart(env_name, data, f"phase0_{dirname}_curves", solved)


if __name__ == "__main__":
    main()
