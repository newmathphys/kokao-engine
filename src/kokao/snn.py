# kokao_engine_v10_0/snn.py
"""KokaoSNN — Spike Neural Networks + Neuromorphic Computing."""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kokao-v9")
)

try:
    import snntorch as snn

    HAS_SNN = True
except ImportError:
    HAS_SNN = False

import torch
import torch.nn as nn

from .core import KokaoCoreV9


class KokaoSpikingLayer(nn.Module):
    """Kokao + Spiking Neurons (LIF - Leaky Integrate-and-Fire).

    Энергоэффективность: μW vs W (нейроморфные чипы)
    Биологическая правдоподобность: spike-based computation
    """

    def __init__(self, n_features: int, n_hidden: int = 64, beta: float = 0.9):
        super().__init__()
        self.fc = nn.Linear(n_features, n_hidden)
        if HAS_SNN:
            self.lif = snn.Leaky(beta=beta, linear_form=True)
        else:
            # Fallback без snntorch
            self.lif = None
        self.n_hidden = n_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lif is None:
            # Fallback: просто линейный слой + ReLU
            out = torch.relu(self.fc(x))
            return out  # ✅ Возвращаем тензор, не скаляр

        mem = self.lif.init_leaky()
        spk_out = []
        for _ in range(10):  # 10 timesteps
            cur = self.fc(x)
            spk, mem = self.lif(cur, mem)
            spk_out.append(spk)
        return torch.stack(spk_out).mean(dim=0)


class KokaoSNNCore(KokaoCoreV9):
    """Комбинированное ядро: классика + SNN.

    Интеграция: KokaoCoreV9 → KokaoSNNCore
    """

    def __init__(self, n_features: int, n_hidden: int = 64):
        super().__init__(n_features)
        self.snn = KokaoSpikingLayer(n_features, n_hidden)
        self.classical_head = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        snn_out = self.snn(x)
        return self.classical_head(snn_out).squeeze(-1)

    def train(self, x: torch.Tensor, target: float = 100.0) -> float:
        """Обучение SNN + classical head."""
        output = self.forward(x)
        loss = (output - target) ** 2
        return loss.item()
