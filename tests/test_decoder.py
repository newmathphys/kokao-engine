"""Decoder Tests (10 тестов)."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.core import KokaoCoreV9
from kokao.decoder import KokaoDecoder


class TestDecoder:
    """KokaoDecoder тесты."""

    def test_init(self):
        core = KokaoCoreV9(n_features=5)
        decoder = KokaoDecoder(core)
        assert decoder.core is core

    def test_generate_basic(self):
        core = KokaoCoreV9(n_features=5)
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=100.0, steps=200)
        assert x.shape == (5,)

    def test_generate_achieves_target(self):
        core = KokaoCoreV9(n_features=5)
        decoder = KokaoDecoder(core)
        target_S = 100.0
        x = decoder.generate(target_S=target_S, steps=500)
        s = core.signal(x)
        assert abs(s - target_S) < 50.0

    def test_generate_custom_init(self):
        core = KokaoCoreV9(n_features=5)
        decoder = KokaoDecoder(core)
        x_init = torch.ones(5)
        x = decoder.generate(target_S=100.0, x_init=x_init, steps=200)
        assert x.shape == (5,)

    def test_generate_regularization(self):
        core = KokaoCoreV9(n_features=5)
        decoder = KokaoDecoder(core)
        x1 = decoder.generate(target_S=100.0, regularization=0.0, steps=200)
        x2 = decoder.generate(target_S=100.0, regularization=1.0, steps=200)
        assert torch.norm(x1) > torch.norm(x2)

    def test_generate_batch(self):
        core = KokaoCoreV9(n_features=5)
        decoder = KokaoDecoder(core)
        targets = [50.0, 100.0, 150.0]
        x_batch = decoder.generate_batch(targets, steps=200)
        assert x_batch.shape == (3, 5)

    def test_generate_different_targets(self):
        core = KokaoCoreV9(n_features=5)
        decoder = KokaoDecoder(core)
        x1 = decoder.generate(target_S=50.0, steps=200)
        x2 = decoder.generate(target_S=150.0, steps=200)
        s1 = core.signal(x1)
        s2 = core.signal(x2)
        assert s2 > s1

    def test_generate_interpolate(self):
        core = KokaoCoreV9(n_features=5)
        decoder = KokaoDecoder(core)
        x_start = torch.randn(5, device="cpu")
        x_end = torch.randn(5, device="cpu")
        trajectory = decoder.interpolate(x_start, x_end, n_steps=10)
        assert len(trajectory) == 10
        for x, s in trajectory:
            assert x.shape == (5,)
            assert isinstance(s, float)

    def test_generate_zero_target(self):
        core = KokaoCoreV9(n_features=5)
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=0.0, steps=200)
        assert x.shape == (5,)

    def test_generate_negative_target(self):
        core = KokaoCoreV9(n_features=5)
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=-100.0, steps=200)
        assert x.shape == (5,)
