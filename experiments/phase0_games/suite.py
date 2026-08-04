"""Generic Phase 0 suite: Random / REINFORCE / A2C / PPO / DQN on one env."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import EpisodeReturnCallback, evaluate_random, evaluate_sb3, make_env  # noqa: E402
from reinforce import evaluate_reinforce, train_reinforce  # noqa: E402


def train_sb3(algo_cls, name: str, env_id: str, out_dir: Path, seed: int,
              timesteps: int, eval_eps: int, **kwargs) -> dict:
    env = Monitor(make_env(env_id))
    model = algo_cls("MlpPolicy", env, verbose=0, seed=seed, **kwargs)
    cb = EpisodeReturnCallback()
    model.learn(total_timesteps=timesteps, callback=cb)
    path = out_dir / f"{name}_{out_dir.name}.zip"
    model.save(str(path))
    mean = evaluate_sb3(model, env_id, n_episodes=eval_eps, seed=seed)
    env.close()
    return {
        "algorithm": name,
        "mean_eval_return": mean,
        "episode_returns": cb.returns,
        "episode_timesteps": cb.timesteps,
        "model": str(path.relative_to(ROOT)),
    }


def run_suite(env_id: str, out_dir: Path, *, seed: int = 42,
              timesteps: int = 50_000, reinforce_episodes: int = 800,
              eval_eps: int = 30, meta: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    print("=== Random ===")
    results["random"] = {
        "algorithm": "random",
        "family": "baseline",
        "mean_eval_return": evaluate_random(env_id, n_episodes=eval_eps, seed=seed),
        "episode_returns": [],
        "episode_timesteps": [],
    }
    print(f"  eval mean: {results['random']['mean_eval_return']:.1f}")

    print("=== REINFORCE (scratch policy gradient) ===")
    import torch

    rf = train_reinforce(env_id, total_episodes=reinforce_episodes, seed=seed)
    rf_mean = evaluate_reinforce(rf.policy, env_id, n_episodes=eval_eps, seed=seed)
    pt = out_dir / f"reinforce_{out_dir.name}.pt"
    torch.save(rf.policy.state_dict(), pt)
    results["reinforce"] = {
        "algorithm": "REINFORCE",
        "family": "policy_gradient (scratch)",
        "mean_eval_return": rf_mean,
        "episode_returns": rf.episode_returns,
        "episode_timesteps": rf.episode_timesteps,
        "model": str(pt.relative_to(ROOT)),
        "note": "Vanilla Monte-Carlo policy gradient; explicit loss = -log π(a|s) * G_t",
    }
    print(f"  eval mean: {rf_mean:.1f}")

    print("=== A2C (actor–critic) ===")
    results["a2c"] = train_sb3(A2C, "a2c", env_id, out_dir, seed, timesteps, eval_eps,
                               learning_rate=7e-4, n_steps=5, gamma=0.99)
    results["a2c"]["family"] = "actor_critic"
    print(f"  eval mean: {results['a2c']['mean_eval_return']:.1f}")

    print("=== PPO ===")
    results["ppo"] = train_sb3(PPO, "ppo", env_id, out_dir, seed, timesteps, eval_eps,
                               learning_rate=3e-4, n_steps=1024, batch_size=64,
                               n_epochs=10, gamma=0.99)
    results["ppo"]["family"] = "policy_gradient (clipped)"
    print(f"  eval mean: {results['ppo']['mean_eval_return']:.1f}")

    print("=== DQN (value-based comparator) ===")
    results["dqn"] = train_sb3(DQN, "dqn", env_id, out_dir, seed, timesteps, eval_eps,
                               learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000,
                               batch_size=64, gamma=0.99, target_update_interval=500,
                               exploration_fraction=0.3)
    results["dqn"]["family"] = "value_based (Q-learning / DQN)"
    print(f"  eval mean: {results['dqn']['mean_eval_return']:.1f}")

    payload = {"env": env_id, "seed": seed, "timesteps_budget_sb3": timesteps,
               **(meta or {}), "results": results}
    out = out_dir / "suite_metrics.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    for k, v in results.items():
        print(f"  {k:12s}  mean_eval={v['mean_eval_return']:.1f}")
    return payload
