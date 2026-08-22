"""From-scratch REINFORCE and Deep Q-learning for the trading MDP.

Carried over from the earlier implementation with three corrections.

* **Month sampling is driven by the run seed.** Previously the scratch methods
  drew training months from a seeded generator while the SB3 agent used a
  sampler seeded to a constant, so its seed-to-seed spread measured only network
  initialisation and understated the variance the other methods showed.

* **Observation width is taken from the feature set**, not a module constant, so
  every ladder rung runs through the same code.

* **Episode bookkeeping records the undiscounted wealth change** alongside the
  shaped return, so training curves remain comparable when the reward changes.

The update rules themselves are unchanged, because they are what Chapter 3
derives:

    REINFORCE   L(theta) = - sum_t log pi_theta(a_t | s_t) * G_t
    Deep Q      y_t = r_t + gamma * (1 - d_t) * max_{a' legal} Q_target(s_{t+1}, a')
                L(phi) = mean( (Q_phi(s_t, a_t) - y_t)^2 )
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


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def masked_dist(self, obs, mask) -> torch.distributions.Categorical:
        logits = self.forward(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
        neg_inf = torch.full_like(logits, NEG_INF)
        masked = torch.where(torch.as_tensor(np.asarray(mask), dtype=torch.bool),
                             logits, neg_inf)
        return torch.distributions.Categorical(logits=masked)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainResult:
    model: nn.Module
    episode_returns: list[float] = field(default_factory=list)
    episode_delta_w: list[float] = field(default_factory=list)
    episode_timesteps: list[int] = field(default_factory=list)
    total_timesteps: int = 0


def _sample_episode(rng: np.random.Generator, episodes: Sequence[dict]) -> dict:
    return episodes[int(rng.integers(0, len(episodes)))]


def train_reinforce(
    factory: Callable[[dict], object],
    episodes: Sequence[dict],
    *,
    obs_dim: int,
    total_timesteps: int = 80_000,
    seed: int = 42,
    gamma: float = 0.99,
    lr: float = 1e-3,
    hidden: int = 128,
) -> TrainResult:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    policy = PolicyNet(obs_dim, hidden=hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    result = TrainResult(model=policy)

    global_t = 0
    while global_t < total_timesteps:
        env = factory(_sample_episode(rng, episodes))
        obs, _ = env.reset()
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        dws: list[float] = []

        done = False
        while not done:
            dist = policy.masked_dist(obs, env.action_masks())
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, reward, term, trunc, info = env.step(int(action.item()))
            rewards.append(float(reward))
            dws.append(float(info["dw"]))
            global_t += 1
            done = term or trunc

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
        result.episode_delta_w.append(float(np.sum(dws)))
        result.episode_timesteps.append(global_t)

    result.total_timesteps = global_t
    return result


def train_dqn(
    factory: Callable[[dict], object],
    episodes: Sequence[dict],
    *,
    obs_dim: int,
    total_timesteps: int = 80_000,
    seed: int = 42,
    gamma: float = 0.99,
    lr: float = 1e-3,
    hidden: int = 128,
    buffer_size: int = 50_000,
    batch_size: int = 64,
    target_sync: int = 500,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.5,
    learn_start: int = 1_000,
) -> TrainResult:
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    qnet = QNet(obs_dim, hidden=hidden)
    target = QNet(obs_dim, hidden=hidden)
    target.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    buffer: deque = deque(maxlen=buffer_size)
    result = TrainResult(model=qnet)

    eps_decay_steps = max(1, int(eps_decay_frac * total_timesteps))

    def epsilon(step: int) -> float:
        frac = min(1.0, step / eps_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    def act(obs, mask, eps: float) -> int:
        legal = np.flatnonzero(mask)
        if rng.random() < eps:
            return int(rng.choice(legal))
        with torch.no_grad():
            q = qnet(torch.as_tensor(np.asarray(obs, dtype=np.float32))).numpy().copy()
        q[~np.asarray(mask, dtype=bool)] = NEG_INF
        return int(np.argmax(q))

    global_t = 0
    while global_t < total_timesteps:
        env = factory(_sample_episode(rng, episodes))
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_dw = 0.0

        done = False
        while not done and global_t < total_timesteps:
            mask = env.action_masks()
            action = act(obs, mask, epsilon(global_t))
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            buffer.append((obs, action, float(reward), next_obs,
                           np.asarray(info["mask"], dtype=bool), float(done)))
            obs = next_obs
            ep_reward += float(reward)
            ep_dw += float(info["dw"])
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
                    q_next = torch.where(b_nmask, q_next, torch.full_like(q_next, NEG_INF))
                    y = b_rew + gamma * (1.0 - b_done) * q_next.max(dim=1).values
                loss = nn.functional.mse_loss(q_sa, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if global_t % target_sync == 0:
                target.load_state_dict(qnet.state_dict())

        result.episode_returns.append(ep_reward)
        result.episode_delta_w.append(ep_dw)
        result.episode_timesteps.append(global_t)

    result.total_timesteps = global_t
    return result


# ------------------------------------------------------------- evaluation

def make_reinforce_policy(policy: PolicyNet):
    """Greedy masked policy for evaluation."""
    def _p(obs, mask) -> int:
        with torch.no_grad():
            logits = policy(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
            masked = torch.where(torch.as_tensor(np.asarray(mask), dtype=torch.bool),
                                 logits, torch.full_like(logits, NEG_INF))
        return int(torch.argmax(masked).item())
    return _p


def make_dqn_policy(qnet: QNet):
    def _p(obs, mask) -> int:
        with torch.no_grad():
            q = qnet(torch.as_tensor(np.asarray(obs, dtype=np.float32))).numpy().copy()
        q[~np.asarray(mask, dtype=bool)] = NEG_INF
        return int(np.argmax(q))
    return _p


def make_sb3_policy(model):
    """Deterministic SB3 policy. Passes the mask when the model supports it."""
    def _p(obs, mask) -> int:
        try:
            a, _ = model.predict(obs, action_masks=np.asarray(mask, dtype=bool),
                                 deterministic=True)
        except TypeError:  # unmasked algorithms (plain PPO)
            a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        return a if mask[a] else 0
    return _p
