"""Core & v1.0.0 Tests (15 тестов)."""

import os
import sys
import tempfile

import pytest
import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.core import KokaoCoreV9, KOKAOEngine

DEVICE = "cpu"


class TestKokaoCoreV9:
    """KokaoCoreV9 базовые тесты."""

    def test_init_default(self):
        core = KokaoCoreV9(n_features=5, device=DEVICE)
        assert core.n_features == 5
        assert core.w.shape == (5,)
        assert abs(core.w.abs().sum().item() - 100.0) < 1.0

    def test_init_custom_weights(self):
        w = torch.tensor([20.0, 20.0, 20.0, 20.0, 20.0])
        core = KokaoCoreV9(weights=w, device=DEVICE)
        assert torch.allclose(core.w, w)

    def test_init_custom_target_sum(self):
        core = KokaoCoreV9(n_features=5, target_sum=200.0, device=DEVICE)
        assert abs(core.w.abs().sum().item() - 200.0) < 1.0

    def test_signal_basic(self):
        core = KokaoCoreV9(n_features=5, device=DEVICE)
        x = torch.ones(5)
        S = core.signal(x)
        assert isinstance(S, float)
        # assert abs(S - 100.0) < 1.0

    def test_signal_random(self):
        core = KokaoCoreV9(n_features=5, device=DEVICE)
        x = torch.randn(5, device="cpu")
        S = core.signal(x)
        assert isinstance(S, float)

    def test_train_basic(self):
        core = KokaoCoreV9(n_features=5, device=DEVICE)
        x = torch.randn(5, device="cpu")
        loss = core.train(x, target=100.0)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_reduces_loss(self):
        core = KokaoCoreV9(n_features=5, device=DEVICE)
        x = torch.randn(5, device="cpu")
        loss1 = core.train(x, target=100.0)
        loss2 = core.train(x, target=100.0)
        assert loss2 <= loss1

    def test_forget_basic(self):
        core = KokaoCoreV9(n_features=5, device=DEVICE)
        initial_norm = torch.norm(core.w)
        core.forget(rate=0.5, normalize=False)
        new_norm = torch.norm(core.w)
        assert new_norm.item() < initial_norm.item()

    def test_forget_rate_zero(self):
        core = KokaoCoreV9(n_features=5, device=DEVICE)
        initial_w = core.w.clone()
        core.forget(rate=0.0)
        assert torch.allclose(core.w, initial_w)

    def test_normalize_basic(self):
        core = KokaoCoreV9(n_features=5, device=DEVICE)
        core.w = torch.randn(5, device="cpu") * 10
        core._normalize()
        assert abs(core.w.abs().sum().item() - 100.0) < 1.0

    def test_save_and_load(self):
        core = KokaoCoreV9(n_features=5, device=DEVICE)
        core.train(torch.randn(5, device="cpu"), 100.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            core.save(f.name)
            core2 = KokaoCoreV9.load(f.name)
            assert torch.allclose(core.w, core2.w)
            os.unlink(f.name)

    def test_get_weight_trajectory(self):
        core = KokaoCoreV9(n_features=5, device=DEVICE)
        trajectory = core.get_weight_trajectory()
        assert len(trajectory) >= 1
        assert len(trajectory[0]) == 5

    def test_device_cpu(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        assert core.w.device.type == "cpu"

    def test_invalid_n_features(self):
        with pytest.raises(Exception):
            KokaoCoreV9(n_features=0)


class TestKOKAOEngine:
    """KOKAOEngine тесты."""

    def test_init(self):
        engine = KOKAOEngine(n_features=5)
        assert engine.n_features == 5
        assert engine.core is not None

    def test_solve_level1(self):
        engine = KOKAOEngine(n_features=5)
        x_plus = torch.ones(5)
        x_minus = torch.ones(5) * 2
        result = engine.solve_level1(x_plus, x_minus)
        assert "S" in result
        assert result["level"] == "L1"

    def test_solve_level2(self):
        engine = KOKAOEngine(n_features=5)
        x = torch.randn(5, device="cpu")
        result = engine.solve_level2(x, target=100.0)
        assert "S" in result
        assert result["level"] == "L2"
        assert "weights" in result

    def test_engine_save_load(self):
        engine = KOKAOEngine(n_features=5, device=DEVICE)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            engine.save(f.name)
            engine2 = KOKAOEngine.load(f.name, device=DEVICE)
            assert torch.allclose(engine.core.w, engine2.core.w)
            os.unlink(f.name)
