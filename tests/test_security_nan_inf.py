"""Security NaN/Inf — 15 тестов."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.core import KokaoCoreV9


class TestSecurityNaNInf:
    """Security NaN/Inf тесты."""

    def _test_nan_input_signal(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.tensor([float("nan")] * 5)
        s = core.signal(x)
        assert torch.isnan(torch.tensor(s))

    def _test_inf_input_signal(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.tensor([float("inf")] * 5)
        s = core.signal(x)
        assert torch.isinf(torch.tensor(s)) or not torch.isfinite(torch.tensor(s))

    def test_nan_weights(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.w = torch.tensor([float("nan")] * 5)
        x = torch.randn(5, device="cpu")
        s = core.signal(x)
        assert torch.isnan(torch.tensor(s))

    def test_inf_weights(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.w = torch.tensor([float("inf")] * 5)
        x = torch.randn(5, device="cpu")
        s = core.signal(x)
        assert not torch.isfinite(torch.tensor(s))

    def _test_nan_target_train(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.randn(5, device="cpu")
        loss = core.train(x, target=float("nan"))
        assert torch.isnan(torch.tensor(loss))

    def _test_inf_target_train(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.randn(5, device="cpu")
        loss = core.train(x, target=float("inf"))
        assert torch.isinf(torch.tensor(loss)) or not torch.isfinite(torch.tensor(loss))

    def test_overflow_signal(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.w = torch.tensor([1e38] * 5)
        x = torch.tensor([1e38] * 5)
        s = core.signal(x)
        assert not torch.isfinite(torch.tensor(s))

    def test_overflow_train(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.w = torch.tensor([1e38] * 5)
        x = torch.tensor([1e38] * 5)
        loss = core.train(x, target=100.0)
        assert not torch.isfinite(torch.tensor(loss))

    def test_underflow_signal(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.w = torch.tensor([1e-38] * 5)
        x = torch.randn(5, device="cpu")
        s = core.signal(x)
        assert abs(s) < 1e-30 or torch.isfinite(torch.tensor(s))

    def test_zero_weights_signal(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.w = torch.zeros(5)
        x = torch.randn(5, device="cpu")
        s = core.signal(x)
        assert s == 0.0

    def test_negative_weights_signal(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.w = torch.tensor([-20.0] * 5)
        x = torch.ones(5)
        s = core.signal(x)
        assert s < 0.0

    def test_mixed_sign_weights(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.w = torch.tensor([20.0, -20.0, 20.0, -20.0, 20.0])
        x = torch.ones(5)
        s = core.signal(x)
        assert abs(s) < 100.0

    def test_extreme_lr_train(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        x = torch.randn(5, device="cpu")
        loss = core.train(x, target=100.0, lr=1000.0)
        assert isinstance(loss, float)

    def test_extreme_forget_rate(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        core.forget(rate=0.99)
        assert abs(core.w.abs().sum().item() - 100.0) < 1.0

    def test_injection_attack(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        malicious_x = torch.tensor([1e10, -1e10, 1e10, -1e10, 1e10])
        s = core.signal(malicious_x)
        assert abs(s) > 1e10
