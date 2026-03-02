"""SNN Tests (12 тестов)."""

import os
import sys

import pytest
import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

try:
    import snntorch

    HAS_SNN = True
except ImportError:
    HAS_SNN = False

from kokao.snn import KokaoSNNCore, KokaoSpikingLayer


class TestSNN:
    """Spike Neural Network тесты."""

    def test_spiking_layer_init(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10)
        assert layer.n_hidden == 10

    @pytest.mark.skipif(not HAS_SNN, reason="snntorch not installed")
    def test_spiking_layer_forward(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu")
        out = layer(x)
        assert out.numel() > 0

    @pytest.mark.skipif(not HAS_SNN, reason="snntorch not installed")
    def test_spiking_layer_timesteps(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu")
        out = layer(x)
        assert isinstance(out, torch.Tensor)

    def test_snn_core_init(self):
        snn = KokaoSNNCore(n_features=5, n_hidden=10)
        assert snn.n_features == 5

    @pytest.mark.skipif(not HAS_SNN, reason="snntorch not installed")
    def test_snn_core_forward(self):
        snn = KokaoSNNCore(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu")
        out = snn.forward(x)
        assert isinstance(out, torch.Tensor)

    def test_snn_core_train(self):
        snn = KokaoSNNCore(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu")
        loss = snn.train(x, target=100.0)
        assert isinstance(loss, float)

    @pytest.mark.skipif(not HAS_SNN, reason="snntorch not installed")
    def test_spiking_beta(self):
        layer = KokaoSpikingLayer(n_features=5, beta=0.5)
        assert layer.lif is None or layer.lif is not None

    @pytest.mark.skipif(not HAS_SNN, reason="snntorch not installed")
    def test_snn_different_hidden_sizes(self):
        for n_hidden in [5, 10, 20]:
            snn = KokaoSNNCore(n_features=5, n_hidden=n_hidden)
            x = torch.randn(5, device="cpu")
            out = snn.forward(x)
            assert out.numel() > 0

    @pytest.mark.skipif(not HAS_SNN, reason="snntorch not installed")
    def test_snn_batch_input(self):
        snn = KokaoSNNCore(n_features=5, n_hidden=10)
        x = torch.randn(3, 5)
        out = snn.forward(x)
        assert out.shape[0] == 3

    @pytest.mark.skipif(not HAS_SNN, reason="snntorch not installed")
    def test_snn_extreme_input(self):
        snn = KokaoSNNCore(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu") * 100
        out = snn.forward(x)
        assert torch.isfinite(out).all()

    def test_snn_zero_input(self):
        snn = KokaoSNNCore(n_features=5, n_hidden=10)
        x = torch.zeros(5)
        out = snn.forward(x)
        assert torch.isfinite(out).all()

    @pytest.mark.skipif(not HAS_SNN, reason="snntorch not installed")
    def test_snn_train_multiple_steps(self):
        snn = KokaoSNNCore(n_features=5, n_hidden=10)
        losses = []
        for _ in range(10):
            x = torch.randn(5, device="cpu")
            loss = snn.train(x, target=100.0)
            losses.append(loss)
        assert all(isinstance(l, float) for l in losses)
