"""Shared plumbing for the experiment scripts.

One module owns training, evaluation and reporting so that `run_ladder.py` and
`run_final.py` cannot differ in any way they do not explicitly declare. Every
difference between two reported rows therefore comes from the configuration
dictionary attached to that row, which is what makes the comparisons in
Chapter 5 attributable.

Three conventions are enforced here.

* **Baselines are re-evaluated inside every configuration.** A baseline's
  numbers depend on the fee and the action model, so quoting one figure across
  configurations that differ in either would be wrong. Each configuration
  reports its own reference strategies.

* **Win rates are computed month by month against the aligned reference**, not
  against a mean. "Beat buy-and-hold" means beat it in that month.

* **Every row carries its diagnostics.** The accounting-identity gap and the
  illegal-action count travel with the result rather than being checked once in
  a test and then assumed.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import agents
import data as data_mod
from baselines import BASELINES
from env import make_env_factory
from rollout import (evaluate, identical_months, mask_determined, summarize,
                     win_rate)

#: Six seeds rather than three. The first ladder run showed that the learned
#: policies are bimodal across seeds, landing either on a fully-committed or on a
#: do-nothing corner, so a three-seed mean is a summary of a distribution it
#: cannot describe and its spread is dominated by which corner happened to come
#: up. Six is still small, but it is enough to report how often each mode occurs
#: rather than only an average between them.
SEEDS = (42, 43, 44, 45, 46, 47)

INITIAL_CASH = 10_000.0
FEE = 0.0005

#: Trade slice, chosen by `calibrate_slice.py` on the validation split. A tenth
#: of capital, used earlier in the project, needs ten steps to reach full
#: exposure and so caps mean exposure near 0.69 across a 21-bar month, against
#: buy-and-hold's 0.91. Under that setting no agent can beat buy-and-hold however
#: well it trades, and reporting that it failed to would report arithmetic as a
#: result about learning.
SLICE_FRAC = 0.25


# --------------------------------------------------------------- provenance

def provenance() -> dict:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=Path(__file__).resolve().parents[2],
                text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    import torch
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ------------------------------------------------------------------ config

class Config:
    """One experimental cell: a state, a reward, an action model.

    `rung` names the ladder rung, `risk_lambda` selects the reward, and
    `action_model` selects how a Buy is sized. Constructing the environment
    raises if the state and reward are not mutually admissible, so an invalid
    cell fails at setup rather than producing numbers that look usable.
    """

    def __init__(
        self,
        name: str,
        feature_names: Sequence[str],
        *,
        rung: str | None = None,
        risk_lambda: float = 0.0,
        action_model: str = "slice",
        slice_frac: float = SLICE_FRAC,
        fee: float = FEE,
    ):
        self.name = name
        self.feature_names = tuple(feature_names)
        self.rung = rung
        self.risk_lambda = float(risk_lambda)
        self.action_model = action_model
        self.slice_frac = float(slice_frac)
        self.fee = float(fee)

    def factory(self, **overrides):
        kwargs = dict(
            initial_cash=INITIAL_CASH, fee=self.fee,
            action_model=self.action_model, slice_frac=self.slice_frac,
            risk_lambda=self.risk_lambda,
        )
        kwargs.update(overrides)
        return make_env_factory(self.feature_names, **kwargs)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "rung": self.rung,
            "obs_dim": len(self.feature_names),
            "features": list(self.feature_names),
            "risk_lambda": self.risk_lambda,
            "reward": "risk_aware" if self.risk_lambda > 0 else "wealth_change",
            "action_model": self.action_model,
            "slice_frac": self.slice_frac,
            "fee": self.fee,
        }


# --------------------------------------------------------------- baselines

def run_baselines(cfg: Config, episodes: Sequence[dict]) -> dict[str, dict]:
    """Evaluate every reference strategy under this configuration.

    Baselines are always scored on the pure wealth change, never on a shaped
    reward: a passive benchmark should mean the same thing whichever reward the
    agent was trained on. The environment override `risk_lambda=0` makes that
    explicit rather than incidental.
    """
    out: dict[str, dict] = {}
    for name, spec in BASELINES.items():
        overrides = dict(spec["env_overrides"])
        overrides["risk_lambda"] = 0.0
        rows = evaluate(cfg.factory(**overrides), episodes, spec["policy"]())
        out[name] = {
            "summary": summarize(rows, INITIAL_CASH),
            "description": spec["description"],
            "per_month": [
                {"id": r["episode_id"], "delta_w": r["delta_w"],
                 "n_trades": r["n_trades"], "intra_dd": r["intra_dd"],
                 "mean_exposure": r["mean_exposure"]}
                for r in rows
            ],
            "_rows": rows,
        }
    return out


# ---------------------------------------------------------------- training

def train_one(
    algo: str,
    cfg: Config,
    train_episodes: Sequence[dict],
    *,
    seed: int,
    total_timesteps: int,
    reward_scale: float = 1.0,
) -> tuple[Callable, dict]:
    """Train one agent. Returns (greedy policy, training trace)."""
    factory = cfg.factory()
    obs_dim = len(cfg.feature_names)
    t0 = time.time()

    if algo == "reinforce":
        res = agents.train_reinforce(factory, train_episodes, obs_dim=obs_dim,
                                     total_timesteps=total_timesteps, seed=seed,
                                     reward_scale=reward_scale)
        policy = agents.make_reinforce_policy(res.model)
        trace = {"episode_delta_w": res.episode_delta_w,
                 "episode_returns": res.episode_returns,
                 "episode_timesteps": res.episode_timesteps}
    elif algo == "dqn":
        res = agents.train_dqn(factory, train_episodes, obs_dim=obs_dim,
                               total_timesteps=total_timesteps, seed=seed,
                               reward_scale=reward_scale)
        policy = agents.make_dqn_policy(res.model)
        trace = {"episode_delta_w": res.episode_delta_w,
                 "episode_returns": res.episode_returns,
                 "episode_timesteps": res.episode_timesteps}
    elif algo == "ppo":
        from sb3_contrib import MaskablePPO
        from month_sampler import MonthSampler

        venv = MonthSampler(factory, train_episodes, seed=seed)
        model = MaskablePPO("MlpPolicy", venv, seed=seed, verbose=0,
                            policy_kwargs={"net_arch": [128, 128]})
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        policy = agents.make_sb3_policy(model)
        trace = {"note": "SB3 does not expose per-episode wealth change directly"}
    else:
        raise ValueError(f"unknown algo {algo!r}")

    trace["wall_seconds"] = round(time.time() - t0, 1)
    return policy, trace


def evaluate_agent(
    algo: str,
    cfg: Config,
    train_episodes: Sequence[dict],
    eval_episodes: Sequence[dict],
    reference_rows: Sequence[dict],
    *,
    seeds: Sequence[int] = SEEDS,
    total_timesteps: int,
    reward_scale: float = 1.0,
    log=print,
) -> dict:
    """Train `algo` once per seed and evaluate each on `eval_episodes`."""
    per_seed = []
    for seed in seeds:
        policy, trace = train_one(algo, cfg, train_episodes, seed=seed,
                                  total_timesteps=total_timesteps,
                                  reward_scale=reward_scale)
        # Evaluation always scores the wealth change, so a risk-trained agent
        # and a wealth-trained agent are compared on the same quantity.
        eval_factory = cfg.factory(risk_lambda=0.0)
        rows = evaluate(eval_factory, eval_episodes, policy)
        summary = summarize(rows, INITIAL_CASH)
        summary["win_rate_vs_ref"] = win_rate(rows, reference_rows)
        summary["months_identical_to_ref"] = identical_months(rows, reference_rows)
        conditioning = mask_determined(eval_factory, eval_episodes, policy)
        summary["state_dependent"] = conditioning["state_dependent"]
        per_seed.append({
            "seed": seed,
            "summary": summary,
            "conditioning": conditioning,
            "train": trace,
            "per_month": [
                {"id": r["episode_id"], "delta_w": r["delta_w"],
                 "n_trades": r["n_trades"], "intra_dd": r["intra_dd"],
                 "mean_exposure": r["mean_exposure"]}
                for r in rows
            ],
        })
        log(f"      seed {seed}: dW={summary['mean_delta_w']:+8.2f}  "
            f"trades={summary['mean_trades']:5.1f}  "
            f"expo={summary['mean_exposure']:.2f}  "
            f"ddI={summary['mdd_intra_mean']:.3f}  "
            f"cond={'state' if conditioning['state_dependent'] else 'MASK-ONLY'}  "
            f"({trace['wall_seconds']}s)")

    return {"per_seed": per_seed, "across_seeds": aggregate(per_seed)}


def aggregate(per_seed: Sequence[dict]) -> dict:
    """Mean and spread across seeds for the metrics the stopping rule uses."""
    keys = ("mean_delta_w", "std_delta_w", "mean_trades", "mean_exposure",
            "sharpe_monthly", "mdd_intra_mean", "mdd_intra_max",
            "mdd_monthly_pnl", "win_rate_vs_ref")
    out: dict = {}
    for k in keys:
        vals = np.array([s["summary"][k] for s in per_seed], dtype=float)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "spread": float(vals.max() - vals.min()),
        }
    out["n_state_dependent"] = sum(1 for s in per_seed
                                   if s["summary"].get("state_dependent"))
    out["n_seeds"] = len(per_seed)
    out["accounting_ok"] = all(s["summary"]["accounting_ok"] for s in per_seed)
    out["max_accounting_gap"] = max(s["summary"]["max_accounting_gap"] for s in per_seed)
    out["illegal_rate"] = max(s["summary"]["illegal_rate"] for s in per_seed)
    return out


# ------------------------------------------------------------------- output

def strip_rows(obj):
    """Remove the raw rollout rows before serialising."""
    if isinstance(obj, dict):
        return {k: strip_rows(v) for k, v in obj.items() if k != "_rows"}
    if isinstance(obj, list):
        return [strip_rows(v) for v in obj]
    return obj


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(strip_rows(payload), indent=2))
    print(f"\nwrote {path}")


def load_data(log=print):
    meta, parts = data_mod.load()
    log(f"data: train={len(parts['train'])} val={len(parts['val'])} "
        f"test={len(parts['test'])} months "
        f"({meta['ids']['train'][0]}..{meta['ids']['test'][-1]})")
    return meta, parts
