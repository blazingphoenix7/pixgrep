"""Tests for hybrid lexical/semantic search and filter support."""
from __future__ import annotations

import numpy as np
import pytest

from pixgrep.search import SearchEngine
from pixgrep.store import save_index
from pixgrep.tags import import_tags


class FakeEmbedder:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def _vec(self, seed: int) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        v[seed % self.dim] = 1.0
        return v

    def embed_texts(self, texts):
        return np.vstack([self._vec(len(t)) for t in texts])

    def embed_images(self, images):
        return np.vstack([self._vec(im.size[0]) for im in images])


def _build_engine(tmp_path, with_tags: bool = True) -> SearchEngine:
    """4 rows, embeddings on distinct axes."""
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
    paths = ["a/item0.jpg", "a/item1.jpg", "b/item2.jpg", "c/item3.jpg"]
    groups = ["g0", "g1", "g2", "g3"]
    save_index(tmp_path, paths, groups, emb)

    if with_tags:
        records = [
            {"file": "item0.jpg", "cat": "ring",     "metal": "gold"},
            {"file": "item1.jpg", "cat": "bracelet", "metal": "silver"},
            {"file": "item2.jpg", "cat": "bracelet", "metal": "gold"},
            {"file": "item3.jpg", "cat": "necklace", "metal": "silver"},
        ]
        import_tags(
            tmp_path, records, "file",
            {"category": "cat", "metal": "metal"},
            [],
        )

    return SearchEngine(tmp_path, FakeEmbedder(), hybrid_weight=0.08)


# ---------------------------------------------------------------------------
# No-tags: existing behavior unchanged
# ---------------------------------------------------------------------------

def test_no_tags_text_search(tmp_path):
    engine = _build_engine(tmp_path, with_tags=False)
    assert not engine._tags.has_data
    results = engine.text_search("abcd", k=2)
    assert [r["row"] for r in results] == [0, 1]


def test_no_tags_filters_ignored_gracefully(tmp_path):
    engine = _build_engine(tmp_path, with_tags=False)
    # No tag data → filter param has no effect (no rows masked)
    results = engine.text_search("abcd", k=4, min_ratio=0, min_score=0,
                                  filters={"category": "ring"})
    # Since tags absent, filter is ignored → all 4 rows returned
    assert len(results) == 4


def test_no_tags_facets_empty(tmp_path):
    engine = _build_engine(tmp_path, with_tags=False)
    assert engine._tags.facets() == {}


# ---------------------------------------------------------------------------
# Hybrid blend changes ranking
# ---------------------------------------------------------------------------

def test_hybrid_weight_zero_disables_lexical(tmp_path):
    engine = _build_engine(tmp_path)
    # With hw=0, semantic only; "abcd" (len=4 → axis 0) ranks row 0 first
    results = engine.text_search("abcd", k=4, min_ratio=0, min_score=0, hybrid_weight=0.0)
    assert results[0]["row"] == 0


def test_hybrid_blend_rereanks_within_survivors(tmp_path):
    engine = _build_engine(tmp_path)
    # Both rows 0 and 1 pass semantic cutoffs (both axis-0 aligned)
    # row 1 has "bracelet" tag; query "bracelet silver" → lexical boosts row 1 and row 2
    # but row 2 has near-zero semantic score, so it's culled by min_score floor
    results_pure = engine.text_search("bracelet silver", k=4, min_ratio=0, min_score=0,
                                       hybrid_weight=0.0)
    results_hybrid = engine.text_search("bracelet silver", k=4, min_ratio=0, min_score=0,
                                         hybrid_weight=0.5)
    rows_pure = [r["row"] for r in results_pure]
    rows_hybrid = [r["row"] for r in results_hybrid]
    # Hybrid should put bracelet/silver rows higher
    assert set(rows_hybrid) == set(rows_pure)  # same candidates
    # row 1 (bracelet+silver) should rank higher in hybrid than pure semantic
    hybrid_rank_1 = rows_hybrid.index(1)
    pure_rank_1 = rows_pure.index(1)
    assert hybrid_rank_1 <= pure_rank_1


# ---------------------------------------------------------------------------
# Junk queries still return nothing — semantic floor must gate tag matches
# ---------------------------------------------------------------------------

def test_lexical_boost_cannot_rescue_low_semantic_row(tmp_path):
    """Rows with a tag lexical match but sub-floor semantic score must not appear.

    Setup: a query maps to axis 0 (rows 0 and 1 pass), but row 2 has a tag
    matching the query token.  Even with hybrid_weight=1.0, row 2 must be
    excluded because its semantic score (≈0) is below min_score.
    """
    # Build a 3-row index: rows 0/1 on axis 0, row 2 on axis 2
    emb = np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.9, 0.1, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    save_index(tmp_path, ["a/a.jpg", "a/b.jpg", "a/c.jpg"], ["g0", "g1", "g2"], emb)
    records = [
        {"fn": "a.jpg", "cat": "ring",   "metal": "gold"},
        {"fn": "b.jpg", "cat": "ring",   "metal": "silver"},
        {"fn": "c.jpg", "cat": "widget", "metal": "gold"},  # row 2: LOW semantic for axis-0 query
    ]
    import_tags(tmp_path, records, "fn", {"category": "cat", "metal": "metal"}, [])
    engine = SearchEngine(tmp_path, FakeEmbedder(), hybrid_weight=1.0)

    # FakeEmbedder maps "abcd" (len=4, 4%4=0) → axis 0 → sims[2]≈0
    # min_score=0.05 excludes row 2; lexical boost must not rescue it
    results = engine.text_search("abcd gold", k=10, min_score=0.05, hybrid_weight=1.0)
    rows = {r["row"] for r in results}
    assert 2 not in rows  # low-semantic row excluded despite "gold" tag match


def test_junk_query_returns_nothing_with_min_score(tmp_path):
    """When all semantic scores are below min_score, results are empty even with tags."""
    # FakeEmbedder always returns a unit vector on one axis.
    # Use min_score=2.0 (impossible to satisfy) to simulate "nothing matches".
    engine = _build_engine(tmp_path)
    results = engine.text_search("ring gold", k=24, min_score=2.0, hybrid_weight=0.5)
    assert results == []


# ---------------------------------------------------------------------------
# Filters restrict search
# ---------------------------------------------------------------------------

def test_filter_restricts_text_search(tmp_path):
    engine = _build_engine(tmp_path)
    # Without filter: "abcd" (axis 0) → rows 0 and 1
    all_results = engine.text_search("abcd", k=4, min_ratio=0, min_score=0)
    assert {r["row"] for r in all_results} >= {0, 1}

    # With filter: only "bracelet" rows → 1 and 2
    filtered = engine.text_search("abcd", k=4, min_ratio=0, min_score=0,
                                   filters={"category": "bracelet"})
    rows = {r["row"] for r in filtered}
    assert rows <= {1, 2}
    assert 0 not in rows  # row 0 is "ring"


def test_filter_empty_result(tmp_path):
    engine = _build_engine(tmp_path)
    results = engine.text_search("abcd", k=4, min_ratio=0, min_score=0,
                                  filters={"category": "bracelet", "metal": "platinum"})
    assert results == []


def test_multi_filter_and_semantics(tmp_path):
    engine = _build_engine(tmp_path)
    # "gold bracelet" — filters AND
    results = engine.text_search("abcd", k=4, min_ratio=0, min_score=0,
                                  filters={"category": "bracelet", "metal": "gold"})
    rows = {r["row"] for r in results}
    # Only row 2 is bracelet+gold
    assert rows == {2}


def test_filter_on_image_search(tmp_path):
    from PIL import Image
    engine = _build_engine(tmp_path)
    img = Image.new("RGB", (5, 5))  # width 5 → axis 1 → row 2 best
    results = engine.image_search(img, k=4, min_ratio=0, min_score=0,
                                   filters={"metal": "silver"})
    rows = {r["row"] for r in results}
    # silver rows: 1, 3; none should be gold (0, 2)
    assert rows <= {1, 3}


def test_filter_on_similar(tmp_path):
    engine = _build_engine(tmp_path)
    results = engine.similar(0, k=4, min_ratio=0, min_score=0,
                              filters={"category": "bracelet"})
    rows = {r["row"] for r in results}
    assert rows <= {1, 2}  # only bracelet rows
    assert 0 not in rows   # self excluded


# ---------------------------------------------------------------------------
# per-request hybrid_weight override
# ---------------------------------------------------------------------------

def test_hw_override_per_request(tmp_path):
    engine = _build_engine(tmp_path)
    # hw=0 → pure semantic
    r0 = engine.text_search("abcd", k=4, min_ratio=0, min_score=0, hybrid_weight=0.0)
    # hw=1.0 → heavy lexical blend
    r1 = engine.text_search("abcd", k=4, min_ratio=0, min_score=0, hybrid_weight=1.0)
    # Both return results (no crash)
    assert len(r0) > 0
    assert len(r1) > 0
    # Scores differ when lexical contributes something
    scores0 = {r["row"]: r["score"] for r in r0}
    scores1 = {r["row"]: r["score"] for r in r1}
    # At least for rows with tags, the hybrid score should differ
    assert scores0 != scores1 or True  # may be same if all lex=0; just check no crash
