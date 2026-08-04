#!/usr/bin/env python3
"""Phase 0 — LunarLander multi-algorithm suite (third game, Box2D physics).

Run:
  .venv311/bin/python experiments/phase0_games/run_lunarlander_suite.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from suite import run_suite  # noqa: E402


def main() -> None:
    run_suite(
        "LunarLander-v3",
        ROOT / "experiments" / "phase0_games" / "results" / "lunarlander",
        seed=42,
        timesteps=150_000,
        reinforce_episodes=1_000,
        meta={
            "state_meaning": "8-D: x/y position, x/y velocity, angle, angular velocity, 2 leg-contact flags",
            "actions": {0: "no-op", 1: "fire left engine", 2: "fire main engine", 3: "fire right engine"},
            "reward": "shaped landing reward; +100 safe landing, -100 crash; solved ≈ 200",
            "solved_threshold": 200,
            "why_lunarlander": (
                "Harder continuous-state control with 4 discrete actions and a "
                "shaped reward — closest game analogue to managing a position "
                "with multiple discrete trade actions."
            ),
        },
    )


if __name__ == "__main__":
    main()
