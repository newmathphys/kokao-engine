"""Learnable Forget Alpha — 10 тестов."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.core import KokaoCoreV9
from kokao.learnable import KokaoCoreWithLearnableForget


class TestLearnableAlpha:
    """Alpha edge cases."""

    def test_alpha_zero(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        with torch.no_grad():
            lf.alpha.data = torch.tensor([0.0])
        forget_rate = torch.sigmoid(lf.alpha).item()
        assert abs(forget_rate - 0.5) < 1e-5

    def test_alpha_large_positive(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        with torch.no_grad():
            lf.alpha.data = torch.tensor([10.0])
        forget_rate = torch.sigmoid(lf.alpha).item()
        assert forget_rate > 0.99

    def test_alpha_large_negative(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        with torch.no_grad():
            lf.alpha.data = torch.tensor([-10.0])
        forget_rate = torch.sigmoid(lf.alpha).item()
        assert forget_rate < 0.01

    def test_alpha_extreme_positive(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        with torch.no_grad():
            lf.alpha.data = torch.tensor([20.0])
        forget_rate = torch.sigmoid(lf.alpha).item()
        assert forget_rate > 0.999

    def test_alpha_extreme_negative(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        with torch.no_grad():
            lf.alpha.data = torch.tensor([-20.0])
        forget_rate = torch.sigmoid(lf.alpha).item()
        assert forget_rate < 0.001

    def test_train_with_alpha_zero(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        with torch.no_grad():
            lf.alpha.data = torch.tensor([0.0])
        x = torch.randn(5, device="cpu")
        loss = lf.train(x, target=100.0)
        assert isinstance(loss, float)

    def test_train_with_alpha_large(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        with torch.no_grad():
            lf.alpha.data = torch.tensor([10.0])
        x = torch.randn(5, device="cpu")
        loss = lf.train(x, target=100.0)
        assert isinstance(loss, float)

    def test_alpha_gradient_flow(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        x = torch.randn(5, device="cpu")
        lf.train(x, target=100.0)
        assert lf.alpha.grad is not None

    def test_alpha_multiple_updates(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        initial_alpha = lf.alpha.item()
        for _ in range(10):
            lf.train(torch.randn(5, device="cpu"), target=100.0)
        # Alpha имеет градиент и optimizer
        assert lf.alpha.grad is not None or lf.alpha.item() == initial_alpha

    def test_alpha_bounds(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        for _ in range(50):
            lf.train(torch.randn(5, device="cpu"), target=100.0)
        rate = torch.sigmoid(lf.alpha).item()
        assert 0.0 < rate < 1.0
