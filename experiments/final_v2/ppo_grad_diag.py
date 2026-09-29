#!/usr/bin/env python3
"""Measure why PPO on dollar rewards does not learn.

STUDY: post-viva, Gate 3 diagnostic — writes results/ppo_grad_diag.json.
Not in the dissertation.

SB3's MaskablePPO adds the policy loss and half the value loss, clips the
joint gradient to norm 0.5, and steps one Adam optimiser. Adam removes a
constant gradient scale, so clipping alone should not stop learning. It
does stop learning if the clipped policy gradient falls below Adam's
epsilon (1e-5 in SB3). This script trains PPO in chunks of one rollout and,
after each chunk, measures on that rollout:

* the gradient norm of the policy loss over the actor's weights,
* the gradient norm of the weighted value loss over the critic's weights,
* the clip factor min(1, 0.5 / joint norm),
* the median clipped policy-gradient element against Adam's epsilon,
* how far the actor's weights moved during the chunk.

At the end it reports the stochastic policy's action probabilities at the
first bar of rising and falling months. It runs on the known-answer world of
`test_learners.py` and on the main study's simulated training months, with
dollar and capital-scaled rewards.

Run:
  ../../venv/bin/python ppo_grad_diag.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from sb3_contrib import MaskablePPO

import agents
import sim_data
from features import resolve_rung
from harness import INITIAL_CASH, Config, ScaledReward
from month_sampler import MonthSampler
from test_learners import CFG as TOY_CFG
from test_learners import EPISODES as TOY_EPISODES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "ppo_grad_diag.json"


def actor_critic_params(policy):
    actor = (list(policy.mlp_extractor.policy_net.parameters())
             + list(policy.action_net.parameters()))
    critic = (list(policy.mlp_extractor.value_net.parameters())
              + list(policy.value_net.parameters()))
    return actor, critic


def flat(params) -> th.Tensor:
    return th.cat([p.detach().reshape(-1).clone() for p in params])


def grad_stats(model) -> dict:
    policy = model.policy
    actor, critic = actor_critic_params(policy)
    data = next(iter(model.rollout_buffer.get(model.batch_size)))
    actions = data.actions.long().flatten()
    values, log_prob, _ = policy.evaluate_actions(
        data.observations, actions, action_masks=data.action_masks)
    adv = data.advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = th.exp(log_prob - data.old_log_prob)
    clip = model.clip_range(model._current_progress_remaining)
    pg_loss = -th.min(adv * ratio, adv * th.clamp(ratio, 1 - clip, 1 + clip)).mean()
    v_loss = model.vf_coef * F.mse_loss(data.returns, values.flatten())

    g_pi = th.autograd.grad(pg_loss, actor, retain_graph=True)
    g_v = th.autograd.grad(v_loss, critic)
    n_pi = float(th.sqrt(sum((g ** 2).sum() for g in g_pi)))
    n_v = float(th.sqrt(sum((g ** 2).sum() for g in g_v)))
    joint = float(np.hypot(n_pi, n_v))
    factor = min(1.0, model.max_grad_norm / (joint + 1e-6))
    elems = th.cat([g.abs().reshape(-1) for g in g_pi]) * factor
    return {"policy_grad_norm": n_pi, "value_grad_norm": n_v,
            "clip_factor": factor,
            "median_clipped_policy_grad": float(elems.median()),
            "value_loss": float(v_loss / model.vf_coef)}


def first_bar_probs(model, episodes, factory) -> dict:
    out = {}
    for kind in ("up", "down"):
        probs = []
        for ep in [e for e in episodes if sim_data.regime_of(e["id"]) == kind][:50]:
            env = factory(ep)
            obs, _ = env.reset()
            with th.no_grad():
                dist = model.policy.get_distribution(
                    th.as_tensor(obs[None, :]),
                    action_masks=env.action_masks()[None, :])
                probs.append(dist.distribution.probs.numpy()[0])
        out[kind] = {a: float(p) for a, p in zip(("hold", "buy", "sell"),
                                                  np.mean(probs, 0))}
    return out


def run(episodes, cfg, scale: float, chunks: int, seed: int = 0) -> dict:
    base = cfg.factory()
    factory = base if scale == 1.0 else (lambda ep: ScaledReward(base(ep), scale))
    model = MaskablePPO("MlpPolicy", MonthSampler(factory, episodes, seed=seed),
                        seed=seed, verbose=0,
                        policy_kwargs={"net_arch": list(agents.HIDDEN_DEFAULT)})
    adam_eps = model.policy.optimizer.defaults["eps"]
    actor, _ = actor_critic_params(model.policy)
    trace = []
    for i in range(chunks):
        before = flat(actor)
        model.learn(total_timesteps=model.n_steps, reset_num_timesteps=(i == 0))
        row = grad_stats(model)
        row["actor_weight_change"] = float(th.norm(flat(actor) - before))
        trace.append(row)
    return {"reward_scale": scale, "adam_eps": adam_eps, "chunks": chunks,
            "steps": chunks * model.n_steps, "trace": trace,
            "first_bar_action_probs": first_bar_probs(model, episodes, base)}


def summary(r: dict) -> dict:
    t = r["trace"]
    return {k: float(np.median([row[k] for row in t])) for k in t[0]}


def main() -> None:
    _, sim_parts = sim_data.load()
    sim_cfg = Config("SIM_S3_wealth", resolve_rung("S3"), rung="S3")
    worlds = {"known_answer": (TOY_EPISODES, TOY_CFG, 20),
              "simulated_train": (sim_parts["train"], sim_cfg, 30)}
    payload: dict = {"study": "PPO gradient diagnostic", "worlds": {}}
    for name, (eps, cfg, chunks) in worlds.items():
        payload["worlds"][name] = {}
        for label, scale in (("dollars", 1.0), ("fraction_of_capital", 1.0 / INITIAL_CASH)):
            r = run(eps, cfg, scale, chunks)
            r["median_over_chunks"] = summary(r)
            payload["worlds"][name][label] = r
            m = r["median_over_chunks"]
            print(f"{name:16s} {label:20s} |g_pi|={m['policy_grad_norm']:.3g} "
                  f"|g_v|={m['value_grad_norm']:.3g} clip={m['clip_factor']:.3g} "
                  f"median clipped g_pi={m['median_clipped_policy_grad']:.3g} "
                  f"(adam eps {r['adam_eps']:g}) actor move={m['actor_weight_change']:.3g}",
                  flush=True)
            print(f"{'':16s} first-bar probs {r['first_bar_action_probs']}", flush=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
