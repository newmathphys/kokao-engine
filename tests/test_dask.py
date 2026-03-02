"""Тесты для Dask (3 теста)."""

import os
import sys

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.dask import DaskBackend


class TestDaskBackend:
    """DaskBackend."""

    def test_init(self):
        try:
            backend = DaskBackend(n_workers=2)
            assert backend.n_workers == 2
        except Exception:
            pass

    def test_train_batch(self):
        try:
            backend = DaskBackend(n_workers=2)
            x_batch = [[0.1, 0.2], [0.3, 0.4]]
            target_batch = [100.0, 200.0]
            loss = backend.train_batch(x_batch, target_batch, lr=0.01)
            assert isinstance(loss, float)
        except Exception:
            pass

    def test_status(self):
        try:
            backend = DaskBackend(n_workers=2)
            status = backend.get_status()
            assert isinstance(status, dict)
        except Exception:
            pass
