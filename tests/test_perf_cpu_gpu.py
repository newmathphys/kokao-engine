"""Performance CPU/GPU — 10 тестов."""

import os
import sys
import time

import pytest
import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.core import KokaoCoreV9


class TestPerfCPUGPU:
    """Performance CPU/GPU."""

    def test_cpu_signal_latency(self):
        core = KokaoCoreV9(n_features=10, device="cpu")
        x = torch.randn(10, device="cpu")
        n_runs = 1000
        start = time.time()
        for _ in range(n_runs):
            core.signal(x)
        elapsed = time.time() - start
        latency_ms = elapsed / n_runs * 1000
        assert latency_ms < 10.0

    def test_cpu_train_latency(self):
        core = KokaoCoreV9(n_features=10, device="cpu")
        x = torch.randn(10, device="cpu")
        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            core.train(x, target=100.0)
        elapsed = time.time() - start
        latency_ms = elapsed / n_runs * 1000
        assert latency_ms < 20.0

    def test_cpu_throughput(self):
        core = KokaoCoreV9(n_features=10, device="cpu")
        x = torch.randn(10, device="cpu")
        n_runs = 1000
        start = time.time()
        for _ in range(n_runs):
            core.signal(x)
        elapsed = time.time() - start
        throughput = n_runs / elapsed
        assert throughput > 100

    def test_gpu_signal_latency(self):
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

    def test_gpu_throughput(self):
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
        assert throughput > 1000

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
        assert throughput > 10

    def test_memory_usage(self):
        core = KokaoCoreV9(n_features=100, device="cpu")
        initial_mem = core.w.element_size() * core.w.nelement()
        for _ in range(100):
            core.train(torch.randn(100, device="cpu"), target=100.0)
        final_mem = core.w.element_size() * core.w.nelement()
        assert final_mem == initial_mem

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
        assert throughput > 100

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
        assert throughput > 100
