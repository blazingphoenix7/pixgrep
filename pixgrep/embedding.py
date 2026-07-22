from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from transformers import AutoModel, AutoProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Embedder:
    """SigLIP 2 wrapper producing L2-normalized float32 embeddings."""

    def __init__(self, model_id: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id, token=False).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, token=False)

    @torch.no_grad()
    def embed_images(self, images: list) -> np.ndarray:
        pil = [self._to_pil(i) for i in images]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return self._normalize(feats)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding="max_length"
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return self._normalize(feats)

    @staticmethod
    def _to_pil(x) -> Image.Image:
        if isinstance(x, (str, Path)):
            return Image.open(x).convert("RGB")
        return x.convert("RGB")

    @staticmethod
    def _normalize(feats) -> np.ndarray:
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        v = feats.detach().cpu().float().numpy()
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (v / norms).astype(np.float32)
