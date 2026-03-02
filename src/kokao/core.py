"""KokaoCoreV9 — базовое ядро."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class KokaoCoreV9:
    """Базовое ядро v9.0."""

    def __init__(
        self,
        n_features: int = 10,
        weights: Optional[torch.Tensor] = None,
        target_sum: float = 100.0,
        device: Optional[str] = None,
    ):
        # ✅ Валидация n_features
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")

        self.n_features = n_features
        self.target_sum = target_sum
        # ✅ device="cpu" по умолчанию (исправлено для тестов)
        self.device = device or "cpu"

        if weights is not None:
            self.w = weights.clone().to(self.device)
        else:
            self.w = torch.randn(n_features, device=self.device)

        self._normalize()
        self.history: List[torch.Tensor] = [self.w.clone().cpu()]

    def _normalize(self, target_sum: Optional[float] = None) -> None:
        """Нормализация: Σ|w| = target_sum."""
        if target_sum is None:
            target_sum = self.target_sum
        with torch.no_grad():
            total = self.w.abs().sum()
            if total.item() < 1e-8:
                self.w = torch.abs(self.w) + 1e-6
                total = self.w.abs().sum()
            self.w = self.w / total * target_sum

    def signal(self, x: torch.Tensor) -> float:
        """S = dot(w, x)."""
        # ✅ Валидация NaN/Inf
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN")
        if torch.isinf(x).any():
            raise ValueError("Input contains Inf")
        return torch.dot(x, self.w).item()

    def signal_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """S = dot(w, x) — возвращает tensor для backprop."""
        # ✅ Валидация NaN/Inf
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN")
        if torch.isinf(x).any():
            raise ValueError("Input contains Inf")
        return torch.dot(x, self.w)

    def train(self, x: torch.Tensor, target: float = 100.0, lr: float = 0.01) -> float:
        """Обучение. Возвращает loss."""
        # ✅ Валидация NaN/Inf
        import math

        if math.isnan(target):
            raise ValueError("Target is NaN")
        if math.isinf(target):
            raise ValueError("Target is Inf")

        s = self.signal(x)
        loss = (s - target) ** 2
        delta = (s - target) * x
        self.w.sub_(delta * lr)
        self._normalize()
        return loss

    def forget(self, rate: float = 0.1, normalize: bool = True) -> None:
        """Забывание: w ← w * (1 - rate)."""
        # ✅ Валидация rate
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"forget rate must be in [0, 1], got {rate}")
        with torch.no_grad():
            self.w.mul_(1.0 - rate)
            if normalize:
                self._normalize()

    def save(self, path: str) -> None:
        """Сохранение в JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "w": self.w.cpu().tolist(),
            "target_sum": self.target_sum,
            "n_features": self.n_features,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "KokaoCoreV9":
        """Загрузка из JSON."""
        with open(path) as f:
            state = json.load(f)
        device = device or "cpu"
        core = cls(
            n_features=state["n_features"],
            weights=torch.tensor(state["w"], device=device),
            target_sum=state["target_sum"],
            device=device,
        )
        return core

    def get_weight_trajectory(self) -> List[List[float]]:
        """История весов."""
        return [t.tolist() for t in self.history]


class KOKAOEngine:
    """Полный движок."""

    def __init__(
        self, n_features: int = 10, target_sum: float = 100.0, device: Optional[str] = None
    ):
        self.core = KokaoCoreV9(n_features=n_features, target_sum=target_sum, device=device)
        self.n_features = n_features
        self.device = self.core.device

    def solve_level1(self, x_plus: torch.Tensor, x_minus: torch.Tensor) -> Dict[str, Any]:
        """L1: S = s⁺/s⁻."""
        # ✅ Переносим x на device ядра
        x_plus = x_plus.to(self.device)
        x_minus = x_minus.to(self.device)
        s_plus = self.core.signal(x_plus)
        s_minus = self.core.signal(x_minus)
        S = s_plus / (s_minus + 1e-8)
        return {"S": S, "level": "L1", "reasoning": "S = s⁺/s⁻"}

    def solve_level2(self, x: torch.Tensor, target: float = 100.0) -> Dict[str, Any]:
        """L2: train + forget."""
        # ✅ Переносим x на device ядра
        x = x.to(self.device)
        s_before = self.core.signal(x)
        self.core.train(x, target)
        self.core.forget(0.1)
        return {"S": s_before, "level": "L2", "weights": self.core.w.tolist()}

    def save(self, path: str) -> None:
        self.core.save(path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "KOKAOEngine":
        core = KokaoCoreV9.load(path, device=device)
        engine = cls(n_features=core.n_features, device=device)
        engine.core = core
        return engine
