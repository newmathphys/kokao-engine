"""Decoder Generation — 8 тестов."""

import os
import sys

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.core import KokaoCoreV9
from kokao.decoder import KokaoDecoder


class TestDecoderGen:
    """Decoder генерация."""

    def test_decoder_zero_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=0.0, steps=50)
        assert x.shape == (5,)

    def test_decoder_large_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=1000.0, steps=50)
        assert x.shape == (5,)

    def test_decoder_negative_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=-100.0, steps=50)
        assert x.shape == (5,)

    def test_decoder_tiny_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=1e-6, steps=50)
        assert x.shape == (5,)

    def test_decoder_extreme_target(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=1e6, steps=50)
        assert x.shape == (5,)

    def test_decoder_zero_steps(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=100.0, steps=0)
        assert x.shape == (5,)

    def test_decoder_one_step(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=100.0, steps=1)
        assert x.shape == (5,)

    def test_decoder_many_steps(self):
        core = KokaoCoreV9(n_features=5, device="cpu")
        decoder = KokaoDecoder(core)
        x = decoder.generate(target_S=100.0, steps=500)
        assert x.shape == (5,)
