"""Dask Fallback — 5 тестов."""

import os
import sys

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.dask import DaskBackend


class TestDaskFallback:
    """Dask fallback тесты."""

    def test_dask_init(self):
        backend = DaskBackend(n_workers=2)
        assert backend.n_workers == 2

    def test_dask_train_batch(self):
        backend = DaskBackend(n_workers=2)
        x_batch = [[0.1, 0.2], [0.3, 0.4]]
        target_batch = [100.0, 200.0]
        loss = backend.train_batch(x_batch, target_batch, lr=0.01)
        assert isinstance(loss, float)

    def test_dask_status(self):
        backend = DaskBackend(n_workers=2)
        status = backend.get_status()
        assert isinstance(status, dict)

    def test_dask_single_worker(self):
        backend = DaskBackend(n_workers=1)
        x_batch = [[0.1, 0.2]]
        target_batch = [100.0]
        loss = backend.train_batch(x_batch, target_batch)
        assert isinstance(loss, float)

    def test_dask_many_workers(self):
        backend = DaskBackend(n_workers=4)
        x_batch = [[0.1, 0.2]]
        target_batch = [100.0]
        loss = backend.train_batch(x_batch, target_batch)
        assert isinstance(loss, float)
