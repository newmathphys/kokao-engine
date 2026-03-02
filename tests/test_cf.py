"""Counterfactual Tests (10 тестов)."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.cf import CounterfactualKokao
from kokao.core import KokaoCoreV9


class TestCounterfactual:
    """CounterfactualKokao тесты."""

    def test_init(self):
        core = KokaoCoreV9(n_features=5)
        cf = CounterfactualKokao(core)
        assert cf.core is core

    def test_counterfactual_basic(self):
        core = KokaoCoreV9(n_features=5)
        cf = CounterfactualKokao(core)
        x = torch.randn(5, device="cpu")
        x_cf = cf.counterfactual(x, target_delta=10.0)
        assert x_cf.shape == x.shape

    def test_counterfactual_delta_positive(self):
        core = KokaoCoreV9(n_features=5)
        cf = CounterfactualKokao(core)
        x = torch.randn(5, device="cpu")
        s_before = core.signal(x)
        x_cf = cf.counterfactual(x, target_delta=50.0, max_steps=100)
        s_after = core.signal(x_cf)
        assert s_after > s_before

    def test_counterfactual_delta_negative(self):
        core = KokaoCoreV9(n_features=5)
        cf = CounterfactualKokao(core)
        x = torch.randn(5, device="cpu")
        s_before = core.signal(x)
        x_cf = cf.counterfactual(x, target_delta=-50.0, max_steps=100)
        s_after = core.signal(x_cf)
        assert s_after < s_before

    def test_counterfactual_sparsity(self):
        core = KokaoCoreV9(n_features=5)
        cf = CounterfactualKokao(core)
        x = torch.randn(5, device="cpu")
        x_cf = cf.counterfactual(x, target_delta=10.0, sparsity_weight=1.0)
        diff = (x_cf - x).abs().sum()
        assert diff < 10.0

    def test_counterfactual_zero_delta(self):
        """При delta=0 counterfactual может не менять x — это корректно."""
        core = KokaoCoreV9(n_features=5)
        cf = CounterfactualKokao(core)
        x = torch.randn(5, device="cpu")
        x_cf = cf.counterfactual(x, target_delta=0.0)
        # Проверяем только что результат корректной формы
        assert x_cf.shape == x.shape

    def test_counterfactual_large_delta(self):
        core = KokaoCoreV9(n_features=5)
        cf = CounterfactualKokao(core)
        x = torch.randn(5, device="cpu")
        x_cf = cf.counterfactual(x, target_delta=500.0, max_steps=200)
        assert x_cf.shape == x.shape

    def test_counterfactual_different_lr(self):
        core = KokaoCoreV9(n_features=5)
        cf = CounterfactualKokao(core)
        x = torch.randn(5, device="cpu")
        x_cf1 = cf.counterfactual(x, target_delta=10.0, lr=0.001)
        x_cf2 = cf.counterfactual(x, target_delta=10.0, lr=0.1)
        assert not torch.allclose(x_cf1, x_cf2)

    def test_feature_importance(self):
        core = KokaoCoreV9(n_features=5)
        cf = CounterfactualKokao(core)
        x = torch.randn(5, device="cpu")
        importance = cf.explain_feature_importance(x)
        assert importance.shape == x.shape
        assert importance.sum() > 0.9

    def test_counterfactual_1d(self):
        core = KokaoCoreV9(n_features=1)
        cf = CounterfactualKokao(core)
        x = torch.randn(1, device="cpu")
        x_cf = cf.counterfactual(x, target_delta=10.0)
        assert x_cf.shape == x.shape
