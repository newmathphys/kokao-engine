"""Counterfactual Logic — 8 тестов."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.cf import CounterfactualKokao
from kokao.core import KokaoCoreV9


class TestCFLogic:
    """Counterfactual логика."""

    def test_cf_same_vector(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cf_x = cf.counterfactual(x, target_delta=10.0, max_steps=100)
        assert cf_x.shape == x.shape

    def test_cf_different_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cf_x1 = cf.counterfactual(x, target_delta=10.0, max_steps=100)
        cf_x2 = cf.counterfactual(x, target_delta=100.0, max_steps=100)
        assert not torch.allclose(cf_x1, cf_x2)

    def test_cf_large_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cf_x = cf.counterfactual(x, target_delta=1e6, max_steps=100)
        assert cf_x.shape == x.shape

    def test_cf_negative_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cf_x = cf.counterfactual(x, target_delta=-100.0, max_steps=100)
        assert cf_x.shape == x.shape

    def test_cf_zero_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cf_x = cf.counterfactual(x, target_delta=0.0, max_steps=100)
        assert cf_x.shape == x.shape

    def test_cf_tiny_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cf_x = cf.counterfactual(x, target_delta=1e-6, max_steps=100)
        assert cf_x.shape == x.shape

    def test_cf_sparsity(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cf_x_low = cf.counterfactual(x, target_delta=10.0, sparsity_weight=0.0, max_steps=500)
        cf_x_high = cf.counterfactual(x, target_delta=10.0, sparsity_weight=10.0, max_steps=500)
        diff_low = (cf_x_low - x).abs().sum()
        diff_high = (cf_x_high - x).abs().sum()
        assert diff_high < diff_low

    def test_cf_direction(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)
        x = torch.ones(5)
        s_before = core.signal(x)
        cf_x_pos = cf.counterfactual(x, target_delta=50.0, max_steps=500)
        cf_x_neg = cf.counterfactual(x, target_delta=-50.0, max_steps=500)
        s_pos = core.signal(cf_x_pos)
        s_neg = core.signal(cf_x_neg)
        assert s_pos > s_before
        assert s_neg < s_before
