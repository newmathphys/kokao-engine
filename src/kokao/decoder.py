# kokao_engine_v10_0/kokao_decoder.py
"""Kokao Decoder — Генерация входов по целевому S."""

import os
import sys

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kokao-v9")
)

from .core import KokaoCoreV9


class KokaoDecoder:
    """Генерация входов по целевому значению S.

    Использует gradient ascent на входном векторе x
    для достижения target_S.
    """

    def __init__(self, core: KokaoCoreV9):
        self.core = core

    def generate(
        self,
        target_S: float,
        steps: int = 500,
        lr: float = 1.0,
        x_init: torch.Tensor = None,
        regularization: float = 0.01,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Сгенерировать вход x для достижения target_S."""
        if device is None:
            device = self.core.device
        if x_init is not None:
            # ✅ Создаём Parameter (leaf tensor) для optimizer
            x = torch.nn.Parameter(x_init.clone().to(device).detach())
        else:
            # ✅ Инициализация с малым шумом для лучшей сходимости
            x = torch.nn.Parameter(torch.randn(len(self.core.w), device=device) * 0.01)

        optimizer = optim.Adam([x], lr=lr)
        target_tensor = torch.tensor(target_S, device=device)

        for step in range(steps):
            optimizer.zero_grad()
            # ✅ Используем signal_tensor для backprop
            s = self.core.signal_tensor(x)
            # Loss = MSE к target + L2 regularization
            loss = torch.nn.functional.mse_loss(s, target_tensor) + regularization * x.pow(2).sum()
            loss.backward()
            optimizer.step()

        return x.detach()

    def generate_batch(
        self, target_S_batch: list, steps: int = 100, lr: float = 0.01, device: torch.device = None
    ) -> torch.Tensor:
        """Сгенерировать batch входов для разных target_S."""
        if device is None:
            device = self.core.device
        x_batch = []
        for target_S in target_S_batch:
            x = self.generate(target_S, steps=steps, lr=lr, device=device)
            x_batch.append(x)
        return torch.stack(x_batch)

    def interpolate(self, x_start: torch.Tensor, x_end: torch.Tensor, n_steps: int = 10) -> list:
        """Интерполяция между двумя входами в пространстве S.

        Returns:
            trajectory: Список (x, S) пар

        """
        trajectory = []
        for alpha in torch.linspace(0, 1, n_steps):
            x = (1 - alpha) * x_start + alpha * x_end
            s = self.core.signal(x)
            trajectory.append((x, s))
        return trajectory
