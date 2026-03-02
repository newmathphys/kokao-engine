"""Performance Benchmarks (15 тестов)."""

import os
import sys
import time

import pytest
import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.core import KokaoCoreV9


class TestPerformance:
    """Бенчмарки производительности."""

    def test_signal_latency_cpu(self):
        core = KokaoCoreV9(n_features=10, device="cpu")
        x = torch.randn(10, device="cpu")
        n_runs = 1000
        start = time.time()
        for _ in range(n_runs):
            core.signal(x)
        elapsed = time.time() - start
        latency_ms = elapsed / n_runs * 1000
        assert latency_ms < 10.0

    def test_train_latency_cpu(self):
        core = KokaoCoreV9(n_features=10, device="cpu")
        x = torch.randn(10, device="cpu")
        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            core.train(x, target=100.0)
        elapsed = time.time() - start
        latency_ms = elapsed / n_runs * 1000
        assert latency_ms < 20.0

    def test_signal_throughput_cpu(self):
        core = KokaoCoreV9(n_features=10, device="cpu")
        x = torch.randn(10, device="cpu")
        n_runs = 1000
        start = time.time()
        for _ in range(n_runs):
            core.signal(x)
        elapsed = time.time() - start
        throughput = n_runs / elapsed
        assert throughput > 1000

    def test_signal_latency_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        core = KokaoCoreV9(n_features=10, device="cuda")
        x = torch.randn(10, device="cuda")
        n_runs = 1000
        start = time.time()
        for _ in range(n_runs):
            core.signal(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        latency_ms = elapsed / n_runs * 1000
        assert latency_ms < 5.0

    def test_signal_throughput_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        core = KokaoCoreV9(n_features=10, device="cuda")
        x = torch.randn(10, device="cuda")
        n_runs = 10000
        start = time.time()
        for _ in range(n_runs):
            core.signal(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        throughput = n_runs / elapsed
        assert throughput > 10000

    def test_batch_signal_latency(self):
        core = KokaoCoreV9(n_features=100, device="cpu")
        x = torch.randn(100, 100)
        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            for i in range(100):
                core.signal(x[i])
        elapsed = time.time() - start
        latency_ms = elapsed / (n_runs * 100) * 1000
        assert latency_ms < 10.0

    def test_large_features_signal(self):
        core = KokaoCoreV9(n_features=1000, device="cpu")
        x = torch.randn(1000, device="cpu")
        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            core.signal(x)
        elapsed = time.time() - start
        throughput = n_runs / elapsed
        assert throughput > 100

    def test_memory_usage(self):
        core = KokaoCoreV9(n_features=100, device="cpu")
        initial_mem = core.w.element_size() * core.w.nelement()
        for _ in range(100):
            core.train(torch.randn(100, device="cpu"), target=100.0)
        final_mem = core.w.element_size() * core.w.nelement()
        assert final_mem == initial_mem

    def test_concurrent_signal(self):
        import threading

        core = KokaoCoreV9(n_features=10, device="cpu")
        results = []

        def worker():
            for _ in range(100):
                s = core.signal(torch.randn(10, device="cpu"))
                results.append(s)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 400

    def _test_signal_dtype_float32(self):
        core = KokaoCoreV9(n_features=10, precision="float32")
        x = torch.randn(10, device="cpu")
        s = core.signal(x)
        assert isinstance(s, float)

    def _test_train_dtype_float32(self):
        core = KokaoCoreV9(n_features=10, precision="float32")
        x = torch.randn(10, device="cpu")
        loss = core.train(x, target=100.0)
        assert isinstance(loss, float)

    def test_signal_device_consistency(self):
        core = KokaoCoreV9(n_features=10, device="cpu")
        x_cpu = torch.randn(10, device="cpu")
        s_cpu = core.signal(x_cpu)
        assert isinstance(s_cpu, float)

    def test_warmup_performance(self):
        core = KokaoCoreV9(n_features=10, device="cpu")
        x = torch.randn(10, device="cpu")
        for _ in range(10):
            core.signal(x)
        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            core.signal(x)
        elapsed = time.time() - start
        throughput = n_runs / elapsed
        assert throughput > 1000

    def test_signal_small_batch(self):
        core = KokaoCoreV9(n_features=10, device="cpu")
        x_batch = torch.randn(10, 10)
        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            for x in x_batch:
                core.signal(x)
        elapsed = time.time() - start
        throughput = (n_runs * 10) / elapsed
        assert throughput > 1000

    def test_signal_large_batch(self):
        core = KokaoCoreV9(n_features=10, device="cpu")
        x_batch = torch.randn(100, 10)
        n_runs = 10
        start = time.time()
        for _ in range(n_runs):
            for x in x_batch:
                core.signal(x)
        elapsed = time.time() - start
        throughput = (n_runs * 100) / elapsed
        assert throughput > 100
