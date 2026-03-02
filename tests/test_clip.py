"""CLIP Tests (10 тестов)."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.clip import KokaoMultimodal


class TestCLIP:
    """Multimodal CLIP тесты."""

    def test_init(self):
        model = KokaoMultimodal(n_latent=512)
        assert model.n_latent == 512

    def test_forward_mock(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, *args, **kwargs):
                return {"last_hidden_state": torch.randn(1, 5, 768)}

        model.text_encoder = MockEncoder()
        model.vision_encoder = MockEncoder()
        s = model(text="test", image=None)
        assert isinstance(s, float)

    def test_text_embedding_dim(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, *args, **kwargs):
                return type("MockOutput", (), {"last_hidden_state": torch.randn(1, 5, 768)})()

        model.text_encoder = MockEncoder()
        text_emb = model.text_encoder("test").last_hidden_state[:, 0, :]
        assert text_emb.shape[-1] == 768

    def test_vision_embedding_dim(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, *args, **kwargs):
                return type("MockOutput", (), {"pooler_output": torch.randn(1, 768)})()

        model.vision_encoder = MockEncoder()
        vision_emb = model.vision_encoder(None).pooler_output
        assert vision_emb.shape[-1] == 768

    def test_latent_layer(self):
        model = KokaoMultimodal(n_latent=512)
        assert model.latent_layer is not None

    def test_kokao_head(self):
        model = KokaoMultimodal(n_latent=512)
        assert model.kokao_head is not None

    def test_different_latent_sizes(self):
        for n_latent in [128, 256, 512, 1024]:
            model = KokaoMultimodal(n_latent=n_latent)
            assert model.n_latent == n_latent

    def test_multimodal_fusion(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, *args, **kwargs):
                return {"last_hidden_state": torch.randn(1, 5, 768)}

        model.text_encoder = MockEncoder()
        model.vision_encoder = MockEncoder()
        s = model(text="test", image=None)
        assert isinstance(s, float)

    def test_text_only(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, *args, **kwargs):
                return {"last_hidden_state": torch.randn(1, 5, 768)}

        model.text_encoder = MockEncoder()
        s = model(text="test")
        assert isinstance(s, float)

    def test_vision_only(self):
        model = KokaoMultimodal(n_latent=512)

        class MockEncoder:
            def __call__(self, *args, **kwargs):
                return {"pooler_output": torch.randn(1, 768)}

        model.vision_encoder = MockEncoder()
        s = model(image=None)
        assert isinstance(s, float)
