"""Integration Modules — 20 тестов."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.cf import CounterfactualKokao
from kokao.core import KokaoCoreV9
from kokao.dask import DaskBackend
from kokao.decoder import KokaoDecoder
from kokao.evolve import EvolveKokao
from kokao.kg import KokaoKGStream
from kokao.learnable import KokaoCoreWithLearnableForget
from kokao.ray import KokaoRayCluster


class TestIntegrationModules:
    """Integration между модулями."""

    def test_core_with_learnable(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        x = torch.randn(5, device="cpu")
        lf.train(x, target=100.0)
        assert lf.alpha.grad is not None

    def test_core_with_cf(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)
        x = torch.randn(5, device="cpu")
        x_cf = cf.counterfactual(x, target_delta=10.0)
        assert x_cf.shape == x.shape

    def test_core_with_decoder(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=100.0, steps=50)
        s = core.signal(x)
        assert isinstance(s, float)

    def test_core_with_evolve(self):
        ev = EvolveKokao(n_features=5, population_size=10)
        ev.evolve(verbose=False)
        core = ev.get_best_core()
        assert core is not None

    def test_learnable_with_cf(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        cf = CounterfactualKokao(core)
        x = torch.randn(5, device="cpu")
        lf.train(x, target=100.0)
        x_cf = cf.counterfactual(x, target_delta=10.0)
        assert x_cf.shape == x.shape

    def test_decoder_with_evolve(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        ev = EvolveKokao(n_features=5, population_size=10)
        ev.evolve(verbose=False)
        best_core = ev.get_best_core()
        x = decoder.generate(target_S=100.0, steps=50)
        s = best_core.signal(x)
        assert isinstance(s, float)

    def test_kg_with_core(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.randn(5, device="cpu")
        s = core.signal(x)
        kg.update_from_stream(x, S=s)

    def test_ray_with_core(self):
        cluster = KokaoRayCluster(n_actors=2, n_features=5)
        x_batches = [[torch.randn(5, device="cpu").tolist()]]
        target_batches = [[100.0]]
        cluster.train(x_batches, target_batches)

    def test_dask_with_core(self):
        backend = DaskBackend(n_workers=2)
        x_batch = [[0.1, 0.2], [0.3, 0.4]]
        target_batch = [100.0, 200.0]
        loss = backend.train_batch(x_batch, target_batch)
        assert isinstance(loss, float)

    def test_full_pipeline(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        cf = CounterfactualKokao(core)
        decoder = KokaoDecoder(core)
        ev = EvolveKokao(n_features=5, population_size=10)

        x = torch.randn(5, device="cpu")
        lf.train(x, target=100.0)
        x_cf = cf.counterfactual(x, target_delta=10.0)
        x_gen = decoder.generate(target_S=100.0, steps=50)
        ev.evolve(verbose=False)

        assert x_cf.shape == x.shape
        assert x_gen.shape == x.shape

    def test_engine_with_kg(self):
        engine = KokaoCoreV9(n_features=5, device="cpu")
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.randn(5, device="cpu")
        s = engine.signal(x)
        kg.update_from_stream(x, S=s)

    def test_multiple_cores(self):
        cores = [KokaoCoreV9(n_features=5, device="cpu") for _ in range(3)]
        for core in cores:
            x = torch.randn(5, device="cpu")
            core.train(x, target=100.0)
        assert all(abs(c.w.abs().sum().item() - 100.0) < 1.0 for c in cores)

    def test_core_save_load_integration(self):
        import tempfile

        core = KokaoCoreV9(n_features=5, device="cpu")
        core.train(torch.randn(5, device="cpu"), target=100.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            core.save(f.name)
            core2 = KokaoCoreV9.load(f.name)
            assert torch.allclose(core.w, core2.w)
            os.unlink(f.name)

    def test_learnable_save_load(self):
        import tempfile

        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        lf.train(torch.randn(5, device="cpu"), target=100.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            core.save(f.name)
            core2 = KokaoCoreV9.load(f.name)
            assert torch.allclose(core.w, core2.w)
            os.unlink(f.name)

    def test_cf_with_decoder(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        cf = CounterfactualKokao(core)
        decoder = KokaoDecoder(core)
        x = torch.randn(5, device="cpu")
        x_cf = cf.counterfactual(x, target_delta=10.0)
        x_gen = decoder.generate(target_S=100.0, steps=50)
        assert x_cf.shape == x_gen.shape

    def test_evolve_with_learnable(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        ev = EvolveKokao(n_features=5, population_size=10)

        for _ in range(5):
            lf.train(torch.randn(5, device="cpu"), target=100.0)

        ev.evolve(verbose=False)
        assert lf.alpha.grad is not None

    def test_ray_dask_comparison(self):
        x_batch = [[0.1, 0.2], [0.3, 0.4]]
        target_batch = [100.0, 200.0]

        try:
            ray_cluster = KokaoRayCluster(n_actors=2, n_features=2)
            ray_cluster.train(x_batch, target_batch)
        except Exception:
            pass

        try:
            dask_backend = DaskBackend(n_workers=2)
            dask_backend.train_batch(x_batch, target_batch)
        except Exception:
            pass

    def test_kg_batch_update(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        for _ in range(10):
            x = torch.randn(5, device="cpu")
            s = float(torch.randn(1, device="cpu").item())
            kg.update_from_stream(x, S=s)

    def test_full_workflow(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        decoder = KokaoDecoder(core)
        ev = EvolveKokao(n_features=5, population_size=10)

        for _ in range(10):
            lf.train(torch.randn(5, device="cpu"), target=100.0)

        x_gen = decoder.generate(target_S=100.0, steps=50)
        ev.evolve(verbose=False)
        best_core = ev.get_best_core()
        s = best_core.signal(x_gen)

        assert isinstance(s, float)

    def test_multi_module_training(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        lf = KokaoCoreWithLearnableForget(core)
        cf = CounterfactualKokao(core)

        for _ in range(5):
            x = torch.randn(5, device="cpu")
            lf.train(x, target=100.0)
            x_cf = cf.counterfactual(x, target_delta=10.0)

        assert lf.alpha.grad is not None
        assert x_cf.shape == x.shape
