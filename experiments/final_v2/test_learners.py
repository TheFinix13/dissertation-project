"""Known-answer tests for the learners (post-viva Gate 3).

The world has two kinds of month. Rising months gain 1% a day and falling
months lose 1% a day, and the momentum feature says which kind the month is.
The right policy buys in rising months and holds cash in falling ones. A
learner that cannot find that policy cannot be compared fairly on the real
problem, so these tests run before any tuning.

Run:
  ../../venv/bin/python -m pytest -q test_learners.py
"""
from __future__ import annotations

import numpy as np
import pytest

from features import MARKET_FEATURES, resolve_rung
from harness import INITIAL_CASH, Config, ScaledReward, train_one
from rollout import evaluate

S3 = resolve_rung("S3")
MOM = list(MARKET_FEATURES).index("mom_k")
STEPS = 40_000


def month(i: int, up: bool) -> dict:
    drift = 0.01 if up else -0.01
    prices = 100.0 * np.exp(np.cumsum(np.r_[0.0, np.full(20, drift)]))
    market = np.zeros((21, len(MARKET_FEATURES)))
    market[:, MOM] = 0.05 if up else -0.05
    return {"id": f"{'up' if up else 'down'}_{i}", "prices": prices, "market": market}


EPISODES = [month(i, i % 2 == 0) for i in range(200)]
CFG = Config("known_answer", S3, rung="S3")


def score(policy) -> tuple[float, float]:
    rows = evaluate(CFG.factory(), EPISODES[:20], policy)
    up = np.mean([r["delta_w"] for r in rows if r["episode_id"].startswith("up")])
    down = np.mean([r["delta_w"] for r in rows if r["episode_id"].startswith("down")])
    return float(up), float(down)


def solved(up: float, down: float) -> bool:
    return up > 1_500.0 and down > -100.0


def test_scaled_reward_leaves_dollar_accounting_unchanged():
    env = ScaledReward(CFG.factory()(EPISODES[0]), 1.0 / INITIAL_CASH)
    env.reset()
    _, reward, _, _, info = env.step(1)
    env.step(0)
    _, reward, _, _, info = env.step(0)
    assert reward == pytest.approx(info["dw"] / INITIAL_CASH)
    assert env.action_masks().shape == (3,)


def test_dqn_solves_the_known_answer_world():
    policy, _ = train_one("dqn", CFG, EPISODES, seed=1, total_timesteps=STEPS)
    assert solved(*score(policy))


def test_ppo_solves_the_known_answer_world_with_scaled_reward():
    policy, _ = train_one("ppo", CFG, EPISODES, seed=0, total_timesteps=STEPS,
                          reward_scale=1.0 / INITIAL_CASH)
    assert solved(*score(policy))


def test_reinforce_is_invariant_to_reward_scale():
    """REINFORCE standardises returns per month, so the tuning need not search scale."""
    import agents
    kw = dict(obs_dim=len(S3), total_timesteps=5_000, seed=3)
    a = agents.train_reinforce(CFG.factory(), EPISODES, **kw)
    b = agents.train_reinforce(CFG.factory(), EPISODES, reward_scale=1.0 / INITIAL_CASH, **kw)
    assert a.episode_delta_w == b.episode_delta_w


def test_ppo_on_raw_dollars_ignores_the_signal():
    """Documents the dissertation's PPO fault: it buys in falling months too."""
    policy, _ = train_one("ppo", CFG, EPISODES, seed=0, total_timesteps=STEPS)
    up, down = score(policy)
    assert up > 1_500.0 and down < -1_000.0
