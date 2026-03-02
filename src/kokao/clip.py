# kokao_engine_v10_0/kokao_clip.py
"""KokaoCLIP — Мультимодальные фундаментальные модели."""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kokao-v9")
)

try:
    from transformers import BertTokenizer, CLIPProcessor

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import torch
import torch.nn as nn

from .core import KokaoCoreV9


class KokaoMultimodal(nn.Module):
    """KokaoCLIP: мультимодальный эмбеддер + KokaoLayer.

    Эмбеддеры:
    - BERT (текст)
    - CLIP (изображение)
    - Wav2Vec (звук) - future

    Преимущества: Zero-shot learning, кроссплатформенность
    """

    def __init__(self, n_latent: int = 512):
        super().__init__()
        self.n_latent = n_latent

        if HAS_TRANSFORMERS:
            self.text_encoder = BertTokenizer.from_pretrained("bert-base-uncased")
            self.vision_encoder = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.text_encoder = None
            self.vision_encoder = None

        self.latent_layer = nn.Linear(768 * 2, n_latent)
        self.kokao_head = KokaoCoreV9(n_features=n_latent)

    def forward(self, text: str = None, image=None) -> float:
        """Forward pass: текст + изображение → S."""
        # Mock embeddings если transformers не доступен
        if self.text_encoder is not None and text:
            text_outputs = self.text_encoder(text, return_tensors="pt")
            text_emb = text_outputs["last_hidden_state"][:, 0, :]
        else:
            text_emb = torch.randn(1, 768)

        if self.vision_encoder is not None and image:
            vision_outputs = self.vision_encoder(image, return_tensors="pt")
            vision_emb = vision_outputs["last_hidden_state"][:, 0, :]
        else:
            vision_emb = torch.randn(1, 768)

        emb = torch.cat([text_emb, vision_emb], dim=-1)
        latent = self.latent_layer(emb)
        return self.kokao_head.signal(latent.squeeze(0))
