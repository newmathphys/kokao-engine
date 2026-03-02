# kokao_engine_v10_0/counterfactual.py
"""Counterfactual Explanations — Контрфактические объяснения."""

import os
import sys

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kokao-v9")
)

from .core import KokaoCoreV9


class CounterfactualKokao:
    """Контрфактические объяснения для Kokao.

    Отвечает на вопрос: "Как минимально изменить вход x,
    чтобы получить желаемое изменение S?"
    """

    def __init__(self, core: KokaoCoreV9):
        self.core = core

    def counterfactual(
        self,
        x: torch.Tensor,
        target_delta: float,
        max_steps: int = 1000,
        lr: float = 1.0,
        sparsity_weight: float = 0.01,
    ) -> torch.Tensor:
        """Найти контрфактический пример.

        Args:
            x: Исходный вход
            target_delta: Желаемое изменение S
            max_steps: Максимум шагов оптимизации
            lr: Learning rate
            sparsity_weight: Вес за разреженность изменений

        Returns:
            x_cf: Контрфактический пример

        """
        # ✅ Создаём Parameter (leaf tensor) для optimizer
        x_cf = torch.nn.Parameter(x.clone().detach())

        optimizer = optim.Adam([x_cf], lr=lr)
        target_S = self.core.signal(x) + target_delta

        for step in range(max_steps):
            optimizer.zero_grad()
            # ✅ Используем signal_tensor для backprop
            s = self.core.signal_tensor(x_cf)
            # Loss = отклонение от target + sparsity
            loss = (s - target_S) ** 2 + sparsity_weight * (x_cf - x).abs().sum()
            loss.backward()
            optimizer.step()

        return x_cf.detach()

    def explain_feature_importance(
        self, x: torch.Tensor, n_perturbations: int = 100
    ) -> torch.Tensor:
        """Объяснить важность признаков через пертурбации.

        Returns:
            importance: Важность каждого признака

        """
        importance = torch.zeros_like(x)
        baseline_s = self.core.signal(x)

        for i in range(len(x)):
            perturbed = x.clone()
            # Пертурбация i-го признака
            perturbed[i] *= 0  # Обнуление
            s_perturbed = self.core.signal(perturbed)
            importance[i] = abs(baseline_s - s_perturbed)

        # Нормализация
        importance = importance / (importance.sum() + 1e-8)
        return importance
