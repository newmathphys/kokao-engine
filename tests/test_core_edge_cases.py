"""Core Edge Cases — 12 тестов."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.core import KokaoCoreV9


class TestCoreEdgeCases:
    """Edge cases для KokaoCoreV9."""

    def test_signal_zero_vector(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.zeros(5)
        S = core.signal(x)
        assert isinstance(S, float)
        assert S == 0.0

    def test_signal_large_values(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.tensor([1e5] * 5)
        S = core.signal(x)
        assert isinstance(S, float)

    def test_signal_small_values(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.tensor([1e-5] * 5)
        S = core.signal(x)
        assert isinstance(S, float)

    def test_train_zero_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.randn(5, device="cpu")
        loss = core.train(x, target=0.0)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_negative_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.randn(5, device="cpu")
        loss = core.train(x, target=-100.0)
        assert isinstance(loss, float)

    def test_train_extreme_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.randn(5, device="cpu")
        loss = core.train(x, target=1e6)
        assert isinstance(loss, float)

    def test_train_tiny_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.randn(5, device="cpu")
        loss = core.train(x, target=1e-6)
        assert isinstance(loss, float)

    def test_forget_rate_one(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        initial_w = core.w.clone()
        core.forget(rate=0.99, normalize=False)
        assert torch.norm(core.w) < torch.norm(initial_w)

    def test_forget_rate_tiny(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.forget(rate=0.001)
        assert abs(core.w.abs().sum().item() - 100.0) < 1.0

    def test_normalize_extreme_weights(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.w = torch.tensor([1e10, -1e10, 1e10, -1e10, 1e10])
        core._normalize()
        assert abs(core.w.abs().sum().item() - 100.0) < 1.0

    def test_normalize_tiny_weights(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.w = torch.tensor([1e-10] * 5)
        core._normalize()
        assert abs(core.w.abs().sum().item() - 100.0) < 1.0

    def test_single_feature(self):
        core = KokaoCoreV9(n_features=1, device="cpu")
        x = torch.randn(1, device="cpu")
        S = core.signal(x)
        assert isinstance(S, float)
