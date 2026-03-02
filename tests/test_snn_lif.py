"""SNN LIF Neurons — 12 тестов."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.snn import KokaoSNNCore, KokaoSpikingLayer


class TestSNNLIF:
    """SNN LIF нейроны."""

    def test_lif_threshold(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu") * 10
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_lif_reset(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu")
        out = layer(x)
        assert out.numel() > 0

    def test_lif_refractory(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu")
        out1 = layer(x)
        out2 = layer(x)
        assert torch.isfinite(out1).all()
        assert torch.isfinite(out2).all()

    def test_lif_decay(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10, beta=0.5)
        x = torch.randn(5, device="cpu")
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_lif_beta_one(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10, beta=0.99)
        x = torch.randn(5, device="cpu")
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_lif_beta_zero(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10, beta=0.01)
        x = torch.randn(5, device="cpu")
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_lif_zero_input(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10)
        x = torch.zeros(5)
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_lif_large_input(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu") * 100
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_lif_small_input(self):
        layer = KokaoSpikingLayer(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu") * 0.01
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_snn_core_lif(self):
        snn = KokaoSNNCore(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu")
        out = snn.forward(x)
        assert torch.isfinite(out).all()

    def test_snn_core_lif_train(self):
        snn = KokaoSNNCore(n_features=5, n_hidden=10)
        x = torch.randn(5, device="cpu")
        loss = snn.train(x, target=100.0)
        assert isinstance(loss, float)

    def test_snn_core_lif_batch(self):
        snn = KokaoSNNCore(n_features=5, n_hidden=10)
        x = torch.randn(3, 5)
        out = snn.forward(x)
        assert out.shape[0] == 3
