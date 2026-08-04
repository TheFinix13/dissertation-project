"""From-scratch REINFORCE (Monte Carlo policy gradient) — no SB3 black box.

Loss per episode:  L(θ) = -Σ_t log π_θ(a_t|s_t) · G_t
where G_t is the discounted return from step t onward.
This is the explicit "policy gradient" Nguyen asked us to be able to derive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from common import make_env


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
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

    def dist(self, obs: np.ndarray) -> torch.distributions.Categorical:
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        return torch.distributions.Categorical(logits=logits)


@dataclass
class ReinforceResult:
    policy: PolicyNet
    episode_returns: list[float] = field(default_factory=list)
    episode_timesteps: list[int] = field(default_factory=list)


def train_reinforce(
    env_id: str,
    total_episodes: int = 800,
    seed: int = 42,
    gamma: float = 0.99,
    lr: float = 1e-3,
) -> ReinforceResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    policy = PolicyNet(obs_dim, n_actions)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    result = ReinforceResult(policy=policy)
    global_t = 0

    for ep in range(total_episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False
        while not done:
            dist = policy.dist(np.asarray(obs, dtype=np.float32).ravel())
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, r, term, trunc, _ = env.step(int(action.item()))
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
        if len(returns) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = -(torch.stack(log_probs) * returns_t).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        result.episode_returns.append(float(np.sum(rewards)))
        result.episode_timesteps.append(global_t)

    env.close()
    return result


@torch.no_grad()
def evaluate_reinforce(policy: PolicyNet, env_id: str, n_episodes: int = 30, seed: int = 0) -> float:
    env = make_env(env_id)
    totals = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            logits = policy(torch.as_tensor(np.asarray(obs, dtype=np.float32).ravel()))
            action = int(torch.argmax(logits).item())  # greedy at eval
            obs, r, term, trunc, _ = env.step(action)
            total += float(r)
            done = term or trunc
        totals.append(total)
    env.close()
    return float(np.mean(totals))
