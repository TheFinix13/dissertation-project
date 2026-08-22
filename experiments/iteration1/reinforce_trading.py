"""From-scratch REINFORCE for the masked Discrete(3) trading MDP.

Same Monte-Carlo policy gradient as experiments/phase0_games/reinforce.py,
adapted for the trading environment's feasibility mask:

  loss  =  - sum_t  log pi_theta(a_t | s_t)  *  G_t

Masking: illegal actions get logit = -inf BEFORE sampling, so the policy
never proposes an infeasible Buy/Sell during training or greedy evaluation.
This mirrors how the SB3 agents are evaluated (mask → forced Hold), but is
cleaner: probability mass is renormalised over legal actions only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def masked_dist(self, obs: np.ndarray, mask: np.ndarray) -> torch.distributions.Categorical:
        logits = self.forward(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
        neg_inf = torch.tensor(-1e9, dtype=logits.dtype)
        masked = torch.where(torch.as_tensor(mask, dtype=torch.bool), logits, neg_inf)
        return torch.distributions.Categorical(logits=masked)


@dataclass
class TradingReinforceResult:
    policy: PolicyNet
    episode_returns: list[float] = field(default_factory=list)
    episode_timesteps: list[int] = field(default_factory=list)
    total_timesteps: int = 0


def train_reinforce_trading(
    make_env: Callable[[np.ndarray], object],
    train_prices: Sequence[np.ndarray],
    *,
    obs_dim: int,
    total_timesteps: int = 80_000,
    seed: int = 42,
    gamma: float = 0.99,
    lr: float = 1e-3,
) -> TradingReinforceResult:
    """Train until the summed env steps reach `total_timesteps`
    (matched budget with the SB3 agents). Each reset samples a train month.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    policy = PolicyNet(obs_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    result = TradingReinforceResult(policy=policy)

    global_t = 0
    ep = 0
    while global_t < total_timesteps:
        prices = train_prices[int(rng.integers(0, len(train_prices)))]
        env = make_env(prices)
        obs, info = env.reset(seed=seed + ep)
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False
        while not done:
            dist = policy.masked_dist(obs, env.action_masks())
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, r, term, trunc, info = env.step(int(action.item()))
            rewards.append(float(r))
            global_t += 1
            done = term or trunc

        # Discounted returns G_t, normalised for variance reduction
        returns = np.zeros(len(rewards), dtype=np.float64)
        g = 0.0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + gamma * g
            returns[t] = g
        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        if len(returns) > 1 and returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = -(torch.stack(log_probs) * returns_t).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        result.episode_returns.append(float(np.sum(rewards)))
        result.episode_timesteps.append(global_t)
        ep += 1

    result.total_timesteps = global_t
    return result


@torch.no_grad()
def make_reinforce_policy(policy: PolicyNet):
    """Greedy masked policy(obs, mask) -> action, matching the eval protocol."""

    def _p(obs, mask) -> int:
        logits = policy(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
        masked = torch.where(
            torch.as_tensor(np.asarray(mask), dtype=torch.bool),
            logits,
            torch.tensor(-1e9, dtype=logits.dtype),
        )
        return int(torch.argmax(masked).item())

    return _p
