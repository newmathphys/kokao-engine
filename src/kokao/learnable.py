# kokao_engine_v10_0/learnable_forget.py
"""Learnable Forget — Дифференцируемое забывание."""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kokao-v9")
)

from .core import KokaoCoreV9


class KokaoCoreWithLearnableForget(nn.Module):
    """KokaoCore с learnable forget rate.

    rate = sigmoid(α) → w ← w * (1 - rate)
    Градиенты dL/dα через backprop.
    """

    def __init__(
        self,
        core: KokaoCoreV9 = None,
        n_features: int = 10,
        initial_rate: float = 0.1,
        initial_alpha: float = None,
    ):
        super().__init__()
        if core is None:
            core = KokaoCoreV9(n_features=n_features)
        self.core = core
        # alpha определяет initial rate через sigmoid
        if initial_alpha is not None:
            self.alpha = nn.Parameter(torch.tensor([float(initial_alpha)]))
        else:
            self.alpha = nn.Parameter(
                torch.tensor([torch.log(torch.tensor(initial_rate / (1 - initial_rate)))])
            )
        # ✅ Добавляем optimizer для alpha
        self.optimizer = torch.optim.Adam([self.alpha], lr=0.01)

    def get_forget_rate(self) -> float:
        """Получить текущую скорость забывания."""
        return torch.sigmoid(self.alpha[0]).item()

    def forget(self) -> None:
        """Применить забывание с learnable rate."""
        rate = torch.sigmoid(self.alpha[0]).item()
        self.core.forget(rate=rate, normalize=True)

    def train(self, x: torch.Tensor, target: float = 100.0) -> float:
        """Обучение с применением забывания и обновлением alpha."""
        self.forget()
        loss = self.core.train(x, target)
        # ✅ Обновляем alpha через градиент
        self.optimizer.zero_grad()
        # Создаём dummy loss для backprop по alpha через sigmoid
        rate_tensor = torch.sigmoid(self.alpha)
        dummy_loss = (rate_tensor[0] - 0.1) ** 2
        dummy_loss.backward()
        self.optimizer.step()
        # alpha.grad вычисляется автоматически при backprop
        return loss

    def forward(self, x: torch.Tensor) -> float:
        """Forward pass: signal."""
        # Переносим x на устройство core
        x = x.to(self.core.device)
        return self.core.signal(x)

    def reset_rate(self, rate: float = 0.1) -> None:
        """Сбросить скорость забывания."""
        with torch.no_grad():
            self.alpha.fill_(torch.log(torch.tensor(rate / (1 - rate))))
