"""Tests for pixgrep/junk.py — scoring math, save/load, _rank masking."""
from __future__ import annotations

import numpy as np
import pytest

from pixgrep.junk import (
    _DEFAULT_GOOD_PROMPTS,
    _DEFAULT_JUNK_PROMPTS,
    junk_scores,
    load_junk_scores,
    save_junk_scores,
)
from pixgrep.search import SearchEngine
from pixgrep.store import save_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


class _FixedEmbedder:
    """Returns fixed unit vectors for the given text → vec mapping."""

    def __init__(self, mapping: dict[str, np.ndarray]):
        self._m = mapping

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._m[t] for t in texts])


class _AxisEmbedder:
    """Maps text to axis (len(t) % dim); used for _rank tests."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            v[len(t) % self.dim] = 1.0
            out.append(v)
        return np.vstack(out)

    def embed_images(self, images):
        return self.embed_texts(["x"] * len(images))


def _build_engine(
    tmp_path,
    junk_threshold: float = 0.0,
    junk_arr: np.ndarray | None = None,
) -> SearchEngine:
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
    save_index(
        tmp_path,
        ["a/one.jpg", "a/two.jpg", "b/three.jpg", "c/four.jpg"],
        ["g0", "g1", "g2", "g3"],
        emb,
    )
    if junk_arr is not None:
        save_junk_scores(tmp_path, junk_arr)
    return SearchEngine(tmp_path, _AxisEmbedder(), junk_threshold=junk_threshold)


# ---------------------------------------------------------------------------
# junk_scores math
# ---------------------------------------------------------------------------


def test_junk_scores_shape_and_dtype():
    emb = np.eye(3, 2, dtype=np.float32)
    embedder = _FixedEmbedder(
        {"j": _unit([1.0, 0.0]), "g": _unit([0.0, 1.0])}
    )
    s = junk_scores(emb, embedder, junk_prompts=["j"], good_prompts=["g"])
    assert s.shape == (3,)
    assert s.dtype == np.float32


def test_junk_scores_junk_row_positive():
    """Row aligned with junk prompt gets positive score."""
    junk_vec = _unit([1.0, 0.0])
    good_vec = _unit([0.0, 1.0])
    embedder = _FixedEmbedder({"junk": junk_vec, "good": good_vec})
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    s = junk_scores(emb, embedder, junk_prompts=["junk"], good_prompts=["good"])
    assert s[0] > 0.0    # junk-like
    assert s[1] < 0.0    # good-like


def test_junk_scores_multi_prompt_uses_max():
    """max_cos is taken across all junk / good prompts."""
    # Two junk prompts on axes 0 and 1; one good prompt on axis 2
    j0 = _unit([1.0, 0.0, 0.0])
    j1 = _unit([0.0, 1.0, 0.0])
    g0 = _unit([0.0, 0.0, 1.0])
    embedder = _FixedEmbedder({"j0": j0, "j1": j1, "g0": g0})
    # Row aligned with j1 → max_cos(junk) via j1 = 1, max_cos(good) = 0 → score 1
    emb = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    s = junk_scores(emb, embedder, junk_prompts=["j0", "j1"], good_prompts=["g0"])
    assert s[0] == pytest.approx(1.0, abs=1e-5)


def test_default_prompt_lists_nonempty_and_disjoint():
    """Default junk/good prompt lists must both be populated and share no entries."""
    assert len(_DEFAULT_JUNK_PROMPTS) > 0
    assert len(_DEFAULT_GOOD_PROMPTS) > 0
    assert set(_DEFAULT_JUNK_PROMPTS).isdisjoint(set(_DEFAULT_GOOD_PROMPTS))


def test_junk_scores_default_prompts_shape():
    """Default prompts produce a (N,) float32 result."""
    dim = 8
    emb = np.eye(dim, dtype=np.float32)

    class _NeutralEmbedder:
        def embed_texts(self, texts):
            n = np.ones((len(texts), dim), dtype=np.float32)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            return n

    s = junk_scores(emb, _NeutralEmbedder())
    assert s.shape == (dim,)
    assert s.dtype == np.float32


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    save_index(
        tmp_path, ["a.jpg", "b.jpg", "c.jpg"], ["g0", "g1", "g2"],
        np.eye(3, dtype=np.float32),
    )
    scores = np.array([0.12, -0.05, 0.30], dtype=np.float32)
    save_junk_scores(tmp_path, scores)
    loaded = load_junk_scores(tmp_path, 3)
    assert loaded is not None
    np.testing.assert_allclose(loaded, scores, atol=1e-6)


def test_save_overwrites_previous(tmp_path):
    save_index(tmp_path, ["a.jpg"], ["g0"], np.eye(1, dtype=np.float32))
    save_junk_scores(tmp_path, np.array([0.5], dtype=np.float32))
    save_junk_scores(tmp_path, np.array([0.1], dtype=np.float32))
    loaded = load_junk_scores(tmp_path, 1)
    assert loaded is not None
    assert loaded[0] == pytest.approx(0.1, abs=1e-6)


def test_load_missing_table_returns_none(tmp_path):
    save_index(tmp_path, ["a.jpg"], ["g0"], np.eye(1, dtype=np.float32))
    assert load_junk_scores(tmp_path, 1) is None


def test_load_missing_db_returns_none(tmp_path):
    assert load_junk_scores(tmp_path / "noexist", 5) is None


# ---------------------------------------------------------------------------
# _rank masking integration
# ---------------------------------------------------------------------------


def test_rank_junk_masks_row_above_threshold(tmp_path):
    """Row 2 (axis 1, len-1 query) is masked when its junk score >= threshold."""
    # "a" has len=1, 1%4=1 → axis 1 → emb[2]=[0,1,0,0] matches perfectly
    junk_arr = np.array([0.0, 0.0, 0.2, 0.0], dtype=np.float32)
    engine = _build_engine(tmp_path, junk_threshold=0.15, junk_arr=junk_arr)
    results = engine.text_search("a", k=4, min_ratio=0.0, min_score=0.0)
    assert 2 not in {r["row"] for r in results}


def test_rank_junk_threshold_zero_disables_masking(tmp_path):
    """junk_threshold=0 means disabled — all rows remain eligible."""
    junk_arr = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    engine = _build_engine(tmp_path, junk_threshold=0.0, junk_arr=junk_arr)
    results = engine.text_search("a", k=4, min_ratio=0.0, min_score=0.0)
    assert len(results) == 4


def test_rank_missing_junk_table_no_effect(tmp_path):
    """No junk_scores table → engine works as before, nothing masked."""
    engine = _build_engine(tmp_path, junk_threshold=0.1, junk_arr=None)
    results = engine.text_search("a", k=4, min_ratio=0.0, min_score=0.0)
    assert len(results) == 4


def test_rank_junk_exact_threshold_boundary(tmp_path):
    """Row with score == threshold is masked (>=)."""
    junk_arr = np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float32)
    engine = _build_engine(tmp_path, junk_threshold=0.1, junk_arr=junk_arr)
    results = engine.text_search("a", k=4, min_ratio=0.0, min_score=0.0)
    assert 2 not in {r["row"] for r in results}
