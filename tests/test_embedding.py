import numpy as np
import pytest
from PIL import Image

from pixgrep.config import DEFAULT_MODEL_ID
from pixgrep.embedding import Embedder


@pytest.fixture(scope="module")
def embedder():
    # NOTE: first run downloads the model (~400 MB). This is intentional — P0
    # is proving the real model works.
    return Embedder(DEFAULT_MODEL_ID)


def test_image_embedding_shape_and_norm(embedder, tmp_path):
    p = tmp_path / "red.jpg"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(p)
    v = embedder.embed_images([p])
    assert v.ndim == 2 and v.shape[0] == 1
    assert v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_text_embedding_shape_and_norm(embedder):
    v = embedder.embed_texts(["a red square", "a blue circle"])
    assert v.shape[0] == 2
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_image_dim_matches_text_dim(embedder, tmp_path):
    p = tmp_path / "g.jpg"
    Image.new("RGB", (32, 32), (10, 220, 10)).save(p)
    iv = embedder.embed_images([p])
    tv = embedder.embed_texts(["green"])
    assert iv.shape[1] == tv.shape[1]
