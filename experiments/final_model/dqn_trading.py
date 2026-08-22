"""From-scratch Deep Q-learning for the masked Discrete(3) trading MDP.

Written to pair with reinforce_trading.py: same network size, same masking
convention (illegal actions get value -inf before argmax), no SB3.

Update rule (Chapter 3, Alg 3.2):

    y_t   = r_t + gamma * max_{a' legal} Q_target(s_{t+1}, a')   (0 if done)
    loss  = mean( (Q(s_t, a_t) - y_t)^2 )

with an experience-replay buffer and a periodically synced target network.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

NEG_INF = -1e9


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TradingDQNResult:
    qnet: QNet
    episode_returns: list[float] = field(default_factory=list)
    episode_timesteps: list[int] = field(default_factory=list)
    total_timesteps: int = 0


def train_dqn_trading(
    make_env: Callable[[np.ndarray], object],
    train_prices: Sequence[np.ndarray],
    *,
    obs_dim: int,
    total_timesteps: int = 80_000,
    seed: int = 42,
    gamma: float = 0.99,
    lr: float = 1e-3,
    buffer_size: int = 50_000,
    batch_size: int = 64,
    target_sync: int = 500,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.5,
    learn_start: int = 1_000,
) -> TradingDQNResult:
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    qnet = QNet(obs_dim)
    target = QNet(obs_dim)
    target.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    buffer: deque = deque(maxlen=buffer_size)
    result = TradingDQNResult(qnet=qnet)

    eps_decay_steps = max(1, int(eps_decay_frac * total_timesteps))

    def epsilon(step: int) -> float:
        frac = min(1.0, step / eps_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    def act(obs: np.ndarray, mask: np.ndarray, eps: float) -> int:
        legal = np.flatnonzero(mask)
        if rng.random() < eps:
            return int(rng.choice(legal))
        with torch.no_grad():
            q = qnet(torch.as_tensor(obs, dtype=torch.float32))
        q = q.numpy().copy()
        q[~mask] = NEG_INF
        return int(np.argmax(q))

    global_t = 0
    ep = 0
    while global_t < total_timesteps:
        prices = train_prices[int(rng.integers(0, len(train_prices)))]
        env = make_env(prices)
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done and global_t < total_timesteps:
            mask = env.action_masks()
            action = act(obs, mask, epsilon(global_t))
            next_obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            next_mask = info["mask"]
            buffer.append((obs, action, float(r), next_obs, next_mask, float(done)))
            obs = next_obs
            ep_reward += float(r)
            global_t += 1

            if len(buffer) >= max(batch_size, learn_start):
                batch = random.sample(buffer, batch_size)
                b_obs = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                b_act = torch.as_tensor([b[1] for b in batch], dtype=torch.int64)
                b_rew = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
                b_next = torch.as_tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
                b_nmask = torch.as_tensor(np.array([b[4] for b in batch]), dtype=torch.bool)
                b_done = torch.as_tensor([b[5] for b in batch], dtype=torch.float32)

                q_sa = qnet(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(b_next)
                    q_next = torch.where(b_nmask, q_next, torch.tensor(NEG_INF))
                    max_next = q_next.max(dim=1).values
                    y = b_rew + gamma * (1.0 - b_done) * max_next
                loss = nn.functional.mse_loss(q_sa, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if global_t % target_sync == 0:
                target.load_state_dict(qnet.state_dict())

        result.episode_returns.append(ep_reward)
        result.episode_timesteps.append(global_t)
        ep += 1

    result.total_timesteps = global_t
    return result


def make_dqn_policy(qnet: QNet):
    """Greedy masked policy(obs, mask) -> action for evaluation."""

    def _p(obs, mask) -> int:
        with torch.no_grad():
            q = qnet(torch.as_tensor(np.asarray(obs, dtype=np.float32))).numpy().copy()
        q[~np.asarray(mask, dtype=bool)] = NEG_INF
        return int(np.argmax(q))

    return _p
