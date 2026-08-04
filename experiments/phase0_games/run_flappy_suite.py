#!/usr/bin/env python3
"""Phase 0 — Flappy Bird multi-algorithm suite.

Run:
  .venv311/bin/python experiments/phase0_games/run_flappy_suite.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from suite import run_suite  # noqa: E402


def main() -> None:
    run_suite(
        "FlappyBird-v0",
        ROOT / "experiments" / "phase0_games" / "results" / "flappy",
        seed=42,
        timesteps=150_000,
        reinforce_episodes=1_500,
        meta={
            "state_meaning": "12-feature vector: pipe positions, bird position/velocity (use_lidar=False)",
            "actions": {0: "do nothing", 1: "flap"},
            "reward": "+0.1 per frame alive, +1 per pipe passed, -1 on death",
            "why_flappy": (
                "Nguyen's suggested side-scroller: sparse pipe rewards + survival, "
                "harder exploration than CartPole; discrete 2 actions like Hold/Trade."
            ),
        },
    )


if __name__ == "__main__":
    main()
