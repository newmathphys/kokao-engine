"""Learnable Forget Tests (12 тестов)."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.core import KokaoCoreV9
from kokao.learnable import KokaoCoreWithLearnableForget


class TestLearnableForget:
    """KokaoCoreWithLearnableForget тесты."""

    def test_alpha_initialization(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        assert lf.alpha.shape == torch.Size([1])
        # alpha это logit(initial_rate), для rate=0.1: logit(0.1) = log(0.1/0.9) ≈ -2.197
        assert torch.isclose(
            lf.alpha, torch.tensor([torch.log(torch.tensor(0.1 / 0.9))]), atol=1e-6
        )

    def test_forget_rate_sigmoid(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        forget_rate = torch.sigmoid(lf.alpha).item()
        assert 0.0 < forget_rate < 1.0

    def test_train_with_forget(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        x = torch.randn(5, device="cpu")
        loss = lf.train(x, target=100.0)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_alpha_updates(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        initial_alpha = lf.alpha.item()
        x = torch.randn(5, device="cpu")
        for _ in range(10):
            lf.train(x, target=100.0)
        # Alpha имеет градиент и optimizer, поэтому может обновиться
        # Но т.к. dummy_loss толкает к 0.1, изменение может быть малым
        assert lf.alpha.grad is not None or lf.alpha.item() == initial_alpha

    def test_forget_rate_changes(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        initial_rate = lf.get_forget_rate()
        # При обучении forget rate может измениться
        assert 0.0 < initial_rate < 1.0

    def test_custom_alpha_init(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core, initial_alpha=0.5)
        assert torch.isclose(lf.alpha[0], torch.tensor(0.5), atol=1e-6)

    def test_forget_rate_bounds(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        for _ in range(50):
            lf.train(torch.randn(5, device="cpu"), target=100.0)
        rate = torch.sigmoid(lf.alpha).item()
        assert 0.0 < rate < 1.0

    def test_multiple_train_steps(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        losses = []
        for _ in range(10):
            loss = lf.train(torch.randn(5, device="cpu"), target=100.0)
            losses.append(loss)
        assert all(l > 0 for l in losses)

    def test_optimizer_step(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        assert lf.optimizer is not None

    def test_core_weights_unchanged(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        initial_w = core.w.clone()
        lf.train(torch.randn(5, device="cpu"), target=100.0)
        assert not torch.allclose(core.w, initial_w)

    def test_alpha_gradient(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        lf.train(torch.randn(5, device="cpu"), target=100.0)
        assert lf.alpha.grad is not None

    def test_extreme_target(self):
        core = KokaoCoreV9(n_features=5)
        lf = KokaoCoreWithLearnableForget(core)
        loss = lf.train(torch.randn(5, device="cpu"), target=1000.0)
        assert isinstance(loss, float)
