"""Tests for feedback-marked-result suppression in pixgrep/search.py.

Covers SearchEngine's optional FeedbackStore wiring: suppression applies to
text_search only (query-scoped), never to image_search/similar, and marks
key on sha1 so they survive a simulated index rebuild with reordered rows.
All data is synthetic; no real filenames, SKUs, or company data.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pixgrep.feedback import FeedbackStore, normalize_query_key
from pixgrep.search import SearchEngine
from pixgrep.store import open_db, save_index


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
        return np.vstack([self._vec(im.size[0]) for im in images])


def _with_sha1s(index_dir, sha1s: list[str]) -> None:
    con = open_db(index_dir)
    for row, sha1 in enumerate(sha1s):
        con.execute("UPDATE images SET sha1=? WHERE row=?", (sha1, row))
    con.commit()
    con.close()


@pytest.fixture()
def feedback_engine(tmp_path):
    # rows 0,1 close together on axis0 (both pass the default 0.6 ratio floor
    # against each other); rows 2,3 on distinct axes.
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
    index_dir = tmp_path / "index"
    save_index(index_dir, paths, groups, emb)
    _with_sha1s(index_dir, ["sha-one", "sha-two", "sha-three", "sha-four"])
    store = FeedbackStore(tmp_path / "feedback.sqlite")
    engine = SearchEngine(index_dir, FakeEmbedder(), feedback=store)
    return engine, store


# --- sha1_for_row / rows_for_sha1s ---

def test_sha1_for_row(feedback_engine):
    engine, _ = feedback_engine
    assert engine.sha1_for_row(0) == "sha-one"
    assert engine.sha1_for_row(3) == "sha-four"


def test_sha1_for_row_unknown_returns_none(feedback_engine):
    engine, _ = feedback_engine
    assert engine.sha1_for_row(99) is None


def test_rows_for_sha1s(feedback_engine):
    engine, _ = feedback_engine
    assert engine.rows_for_sha1s({"sha-one", "sha-three"}) == {0, 2}
    assert engine.rows_for_sha1s(set()) == set()


# --- suppression: query-scoped, text_search only ---

def test_marked_row_suppressed_from_text_search(feedback_engine):
    engine, store = feedback_engine
    query_key = normalize_query_key("abcd")  # -> axis0 query
    store.toggle("alice", query_key, "sha-one", 0, engine.path_for(0))

    results = engine.text_search("abcd", k=100)
    rows = [r["row"] for r in results]
    assert 0 not in rows
    assert 1 in rows  # row1's own score now anchors the floor, so it survives


def test_unmark_restores_visibility(feedback_engine):
    engine, store = feedback_engine
    query_key = normalize_query_key("abcd")
    store.toggle("alice", query_key, "sha-one", 0, engine.path_for(0))
    assert 0 not in [r["row"] for r in engine.text_search("abcd", k=100)]

    store.toggle("alice", query_key, "sha-one", 0, engine.path_for(0))  # unmark
    assert 0 in [r["row"] for r in engine.text_search("abcd", k=100)]


def test_suppression_is_query_scoped(feedback_engine):
    """A mark made under one query doesn't suppress the same row under another."""
    engine, store = feedback_engine
    other_query_key = normalize_query_key("xyz")
    store.toggle("alice", other_query_key, "sha-one", 0, engine.path_for(0))

    assert 0 in [r["row"] for r in engine.text_search("abcd", k=100)]


def test_marked_row_still_appears_in_image_search(feedback_engine):
    engine, store = feedback_engine
    store.toggle("alice", normalize_query_key("abcd"), "sha-one", 0, engine.path_for(0))

    img = Image.new("RGB", (4, 4))  # width 4 -> axis0 -> row0 ranks first
    results = engine.image_search(img, k=1)
    assert results[0]["row"] == 0


def test_marked_row_still_appears_in_similar(feedback_engine):
    engine, store = feedback_engine
    store.toggle("alice", normalize_query_key("abcd"), "sha-two", 1, engine.path_for(1))

    results = engine.similar(0, k=3)
    assert 1 in [r["row"] for r in results]


def test_shorthand_and_plain_phrasing_share_marks(feedback_engine):
    """Marking under jewelry-trade shorthand suppresses the plain phrasing too."""
    engine, store = feedback_engine
    mark_key = normalize_query_key("YG band")  # normalizes to "yellow gold band"
    store.toggle("alice", mark_key, "sha-one", 0, engine.path_for(0))

    results = engine.text_search("yellow gold band", k=100)
    assert 0 not in [r["row"] for r in results]


# --- no feedback store: behavior unchanged ---

def test_no_feedback_store_leaves_text_search_unchanged(tmp_path):
    emb = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    paths = ["a.jpg", "b.jpg", "c.jpg"]
    save_index(tmp_path, paths, ["g0", "g1", "g2"], emb)

    without_kw = SearchEngine(tmp_path, FakeEmbedder())
    explicit_none = SearchEngine(tmp_path, FakeEmbedder(), feedback=None)

    r1 = without_kw.text_search("abcd", k=100)
    r2 = explicit_none.text_search("abcd", k=100)
    assert [r["row"] for r in r1] == [r["row"] for r in r2] == [0, 1]


# --- sha1 keying survives a simulated index rebuild ---

def test_sha1_keying_survives_rebuild_with_reordered_rows(tmp_path):
    """A rebuild that reshuffles row numbers must not lose the mark: it is
    keyed on content (sha1), not row position."""
    store = FeedbackStore(tmp_path / "feedback.sqlite")

    # content -> fixed embedding, independent of row placement
    vec_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)   # marked content
    vec_a2 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)  # different content, same family
    vec_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    vec_c = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # Original index: row0=A, row1=A2, row2=B, row3=C
    index_a = tmp_path / "index_a"
    save_index(
        index_a,
        ["pA.jpg", "pA2.jpg", "pB.jpg", "pC.jpg"],
        ["gA", "gA2", "gB", "gC"],
        np.vstack([vec_a, vec_a2, vec_b, vec_c]),
    )
    _with_sha1s(index_a, ["sha-A", "sha-A2", "sha-B", "sha-C"])
    engine_a = SearchEngine(index_a, FakeEmbedder(), feedback=store)
    query_key = normalize_query_key("abcd")
    store.toggle("alice", query_key, "sha-A", 0, engine_a.path_for(0))

    # Rebuilt index: reordered — row0=C, row1=A, row2=B, row3=A2
    index_b = tmp_path / "index_b"
    save_index(
        index_b,
        ["pC.jpg", "pA.jpg", "pB.jpg", "pA2.jpg"],
        ["gC", "gA", "gB", "gA2"],
        np.vstack([vec_c, vec_a, vec_b, vec_a2]),
    )
    _with_sha1s(index_b, ["sha-C", "sha-A", "sha-B", "sha-A2"])
    engine_b = SearchEngine(index_b, FakeEmbedder(), feedback=store)

    results = engine_b.text_search("abcd", k=100)
    rows = [r["row"] for r in results]
    assert 1 not in rows  # content A, now at row1, is still suppressed
    assert 3 in rows      # content A2 (different sha1) is unaffected
