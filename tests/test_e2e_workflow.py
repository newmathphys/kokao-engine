"""E2E Workflow — 10 тестов."""

import os
import sys
import tempfile

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.cf import CounterfactualKokao
from kokao.core import KokaoCoreV9
from kokao.decoder import KokaoDecoder
from kokao.evolve import EvolveKokao
from kokao.learnable import KokaoCoreWithLearnableForget


class TestE2EWorkflow:
    """E2E workflow тесты."""

    def test_e2e_training(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        for _ in range(10):
            x = torch.randn(5, device="cpu")
            core.train(x, target=100.0)
        assert abs(core.w.abs().sum().item() - 100.0) < 1.0

    def test_e2e_save_load(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        for _ in range(5):
            core.train(torch.randn(5, device="cpu"), target=100.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            core.save(f.name)
            core2 = KokaoCoreV9.load(f.name)
            assert torch.allclose(core.w, core2.w)
            os.unlink(f.name)

    def test_e2e_learnable(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        for _ in range(10):
            lf.train(torch.randn(5, device="cpu"), target=100.0)
        assert 0.0 < torch.sigmoid(lf.alpha).item() < 1.0

    def test_e2e_decoder(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=100.0, steps=200)
        s = core.signal(x)
        assert isinstance(s, float)

    def test_e2e_evolution(self):
        ev = EvolveKokao(n_features=5, population_size=20, generations=10)
        ev.evolve(verbose=False)
        best_core = ev.get_best_core()
        assert abs(best_core.w.abs().sum().item() - 100.0) < 1.0

    def test_e2e_full_pipeline(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        decoder = KokaoDecoder(core)
        ev = EvolveKokao(n_features=5, population_size=10, generations=5)

        for _ in range(10):
            lf.train(torch.randn(5, device="cpu"), target=100.0)

        x_gen = decoder.generate(target_S=100.0, steps=50)
        ev.evolve(verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            core.save(f.name)
            core2 = KokaoCoreV9.load(f.name)
            s = core2.signal(x_gen)
            os.unlink(f.name)

        assert isinstance(s, float)

    def test_e2e_counterfactual(self):
        from kokao.cf import CounterfactualKokao

        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)

        torch.manual_seed(42)
        x = torch.randn(5, device="cpu")
        s_before = core.signal(x)
        x_cf = cf.counterfactual(x, target_delta=50.0, max_steps=500, lr=0.5)
        s_after = core.signal(x_cf)

        # Counterfactual должен изменить вход
        assert not torch.allclose(x, x_cf, atol=1e-3)
        # Сигнал должен измениться
        assert s_after != s_before

    def test_e2e_multiple_saves(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        files = []

        for i in range(3):
            core.train(torch.randn(5, device="cpu"), target=100.0)
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                core.save(f.name)
                files.append(f.name)

        for f_name in files:
            core2 = KokaoCoreV9.load(f_name)
            assert core2.w.shape == (5,)
            os.unlink(f_name)

    def test_e2e_batch_training(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        batch_size = 10

        for _ in range(batch_size):
            x = torch.randn(5, device="cpu")
            core.train(x, target=100.0)

        assert abs(core.w.abs().sum().item() - 100.0) < 1.0

    def test_e2e_workflow_complete(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        cf = CounterfactualKokao(core)
        decoder = KokaoDecoder(core)

        torch.manual_seed(42)
        x_cf = None
        for _ in range(5):
            x = torch.randn(5, device="cpu")
            lf.train(x, target=100.0)
            x_cf = cf.counterfactual(x, target_delta=50.0, max_steps=500, lr=0.5)

        x_gen = decoder.generate(target_S=100.0, steps=500)

        assert lf.alpha.grad is not None
        assert x_cf is not None
        assert x_gen.shape == x_cf.shape
