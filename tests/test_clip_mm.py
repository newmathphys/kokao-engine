"""CLIP Multimodal — 10 тестов."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.clip import KokaoMultimodal


class TestCLIPMM:
    """CLIP мультимодальные тесты."""

    def test_clip_emb_dim_128(self):
        model = KokaoMultimodal(n_latent=128)
        assert model.n_latent == 128

    def test_clip_emb_dim_256(self):
        model = KokaoMultimodal(n_latent=256)
        assert model.n_latent == 256

    def test_clip_emb_dim_512(self):
        model = KokaoMultimodal(n_latent=512)
        assert model.n_latent == 512

    def test_clip_emb_dim_1024(self):
        model = KokaoMultimodal(n_latent=1024)
        assert model.n_latent == 1024

    def test_clip_text_only(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, *args, **kwargs):
                return {"last_hidden_state": torch.randn(1, 5, 768)}

        model.text_encoder = MockEncoder()
        s = model(text="test")
        assert isinstance(s, float)

    def test_clip_vision_only(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, *args, **kwargs):
                return {"pooler_output": torch.randn(1, 768)}

        model.vision_encoder = MockEncoder()
        s = model(image=None)
        assert isinstance(s, float)

    def test_clip_both_modalities(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, *args, **kwargs):
                return {"last_hidden_state": torch.randn(1, 5, 768)}

        model.text_encoder = MockEncoder()
        model.vision_encoder = MockEncoder()
        s = model(text="test", image=None)
        assert isinstance(s, float)

    def test_clip_different_texts(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, text, **kwargs):
                return {"last_hidden_state": torch.randn(1, 5, 768)}

        model.text_encoder = MockEncoder()
        s1 = model(text="hello")
        s2 = model(text="world")
        assert isinstance(s1, float)
        assert isinstance(s2, float)

    def test_clip_different_images(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, image, **kwargs):
                return {"pooler_output": torch.randn(1, 768)}

        model.vision_encoder = MockEncoder()
        s1 = model(image=None)
        s2 = model(image=None)
        assert isinstance(s1, float)
        assert isinstance(s2, float)

    def test_clip_latent_projection(self):
        model = KokaoMultimodal(n_latent=512)
        assert model.latent_layer is not None
        assert model.latent_layer.out_features == 512
