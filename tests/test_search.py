import numpy as np
import pytest

from pixgrep.search import SearchEngine
from pixgrep.store import save_index


class FakeEmbedder:
    """Maps known strings/images to fixed unit vectors for deterministic tests."""

    def __init__(self, dim=4):
        self.dim = dim

    def _vec(self, seed: int) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        v[seed % self.dim] = 1.0
        return v

    def embed_texts(self, texts):
        return np.vstack([self._vec(len(t)) for t in texts])

    def embed_images(self, images):
        # use image width as the seed
        return np.vstack([self._vec(im.size[0]) for im in images])


@pytest.fixture()
def engine(tmp_path):
    # rows 0..3 with embeddings on distinct axes; rows 0 and 1 share an axis-0-ish direction
    emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    paths = ["a/one.jpg", "a/two.jpg", "b/three.jpg", "c/four.jpg"]
    groups = ["one", "two", "three", "four"]
    save_index(tmp_path, paths, groups, emb)
    return SearchEngine(tmp_path, FakeEmbedder())


def test_count_and_path_for(engine):
    assert engine.count == 4
    assert engine.path_for(2) == "b/three.jpg"
    with pytest.raises(IndexError):
        engine.path_for(99)


def test_text_search_returns_ranked_results(engine):
    # query of length 4 -> axis 0 -> rows 0 and 1 rank first
    results = engine.text_search("abcd", k=2)
    assert [r["row"] for r in results] == [0, 1]
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["name"] == "one.jpg"
    assert results[0]["folder"] == "a"


def test_image_search_uses_image_embedding(engine):
    from PIL import Image

    img = Image.new("RGB", (5, 5))  # width 5 -> axis 1 -> row 2 first
    results = engine.image_search(img, k=1)
    assert results[0]["row"] == 2


def test_similar_excludes_self(engine):
    results = engine.similar(0, k=3)
    rows = [r["row"] for r in results]
    assert 0 not in rows
    assert rows[0] == 1  # nearest neighbor of row 0 is row 1


def test_k_larger_than_count_is_safe(engine):
    # min_ratio=0 disables the relevance cutoff -> all rows come back
    assert len(engine.text_search("abcd", k=100, min_ratio=0.0)) == 4


def test_relevance_cutoff_drops_weak_results(engine):
    # axis-0 query: rows 0 (sim 1.0) and 1 (sim ~0.99) pass the 0.6 ratio;
    # rows 2 and 3 (sim 0.0) are dropped even though k allows them
    results = engine.text_search("abcd", k=100)
    assert [r["row"] for r in results] == [0, 1]


def test_relevance_cutoff_ratio_is_tunable(engine):
    # a ratio above ~0.995 keeps only the single best hit
    results = engine.text_search("abcd", k=100, min_ratio=0.999)
    assert [r["row"] for r in results] == [0]
