# kokao_engine_v10_0/kokao_rl.py
"""KokaoRL — Онлайн-обучение с подкреплением."""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kokao-v9")
)

import torch
import torch.distributions as dist
import torch.nn as nn

from .core import KokaoCoreV9


class KokaoPolicy(nn.Module):
    """Kokao Policy: компактное представление политики.

    Алгоритмы: PPO, DQN, SAC - future
    Преимущества: робототехника, игры, автономные агенты
    """

    def __init__(self, n_features: int, n_actions: int):
        super().__init__()
        self.kokao = KokaoCoreV9(n_features=n_features)
        self.action_head = nn.Linear(1, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.kokao.signal(x)
        return self.action_head(torch.tensor([s]))

    def get_action(self, state: torch.Tensor) -> int:
        """Выбрать действие через sampling."""
        action_probs = torch.softmax(self.forward(state), dim=-1)
        distribution = dist.Categorical(action_probs)
        return distribution.sample().item()


class KokaoRLAgent:
    """Reinforcement Learning агент на основе Kokao.

    Компоненты:
    - KokaoPolicy
    - Алгоритмы (PPO, DQN, SAC)
    """

    def __init__(self, n_features: int, n_actions: int, lr: float = 0.001):
        self.policy = KokaoPolicy(n_features, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, state: torch.Tensor) -> int:
        """Выбрать действие."""
        return self.policy.get_action(state)

    def train_step(self, state: torch.Tensor, action: int, reward: float) -> float:
        """Один шаг обучения (REINFORCE-style)."""
        self.optimizer.zero_grad()

        action_probs = torch.softmax(self.policy.forward(state), dim=-1)
        log_prob = torch.log(action_probs[action] + 1e-8)

        loss = -log_prob * reward
        loss.backward()
        self.optimizer.step()

        return loss.item()
