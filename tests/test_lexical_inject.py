"""Tests for lexical-injection retrieval and the soft junk penalty.

Covers pixgrep/search.py SearchEngine._rank (inject_rows + junk_soft_weight)
and its wiring through text_search. All data is synthetic; no real
filenames, SKUs, or company data.
"""
from __future__ import annotations

import numpy as np
import pytest

from pixgrep.junk import save_junk_scores
from pixgrep.search import SearchEngine
from pixgrep.store import save_index
from pixgrep.tags import import_tags


class _FixedQueryEmbedder:
    """Always returns the same query vector regardless of query text, so
    semantic similarity is controlled purely by the row embeddings we store
    (dot(query, row) == row[0], since the query vector is axis-0)."""

    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)

    def embed_texts(self, texts):
        return np.vstack([self._vec] * len(texts))

    def embed_images(self, images):
        return np.vstack([self._vec] * len(images))


def _engine(tmp_path, emb, records=None, field_keys=None, junk=None, **kwargs):
    emb = np.asarray(emb, dtype=np.float32)
    n = emb.shape[0]
    paths = [f"p{i}.jpg" for i in range(n)]
    groups = [f"g{i}" for i in range(n)]
    save_index(tmp_path, paths, groups, emb)
    if records is not None:
        import_tags(tmp_path, records, "file", field_keys or {}, [])
    if junk is not None:
        save_junk_scores(tmp_path, np.asarray(junk, dtype=np.float32))
    qv = np.zeros(emb.shape[1], dtype=np.float32)
    qv[0] = 1.0
    return SearchEngine(tmp_path, _FixedQueryEmbedder(qv), **kwargs)


# ---------------------------------------------------------------------------
# Lexical injection surfaces tag-only matches
# ---------------------------------------------------------------------------

def test_injected_row_surfaces_when_semantic_floor_fails(tmp_path):
    emb = [
        [1.0, 0.0, 0.0],  # row0: sims=1.0, passes floor
        [0.0, 1.0, 0.0],  # row1: sims=0.0, fails floor, tag fully matches query
    ]
    records = [
        {"file": "p0.jpg", "cat": "widget"},
        {"file": "p1.jpg", "cat": "lumina"},
    ]
    engine = _engine(tmp_path, emb, records=records, field_keys={"category": "cat"})
    results = engine.text_search(
        "lumina collection pieces", k=10, min_score=0.05, min_ratio=0.6
    )
    rows = {r["row"] for r in results}
    assert 0 in rows
    assert 1 in rows


# ---------------------------------------------------------------------------
# Dead-query invariant: empty floors => no injection, ever
# ---------------------------------------------------------------------------

def test_dead_query_returns_nothing_even_with_tag_match(tmp_path):
    emb = [
        [0.0, 1.0, 0.0],  # row0: sims=0.0, orthogonal to query
        [0.0, 0.0, 1.0],  # row1: sims=0.0, tag fully matches query
    ]
    records = [
        {"file": "p0.jpg", "cat": "widget"},
        {"file": "p1.jpg", "cat": "redsportscar"},
    ]
    engine = _engine(tmp_path, emb, records=records, field_keys={"category": "cat"})
    results = engine.text_search("redsportscar", k=10)  # default min_score=0.05
    assert results == []


# ---------------------------------------------------------------------------
# Injection bypasses semantic floors but NOT filters/junk masks
# ---------------------------------------------------------------------------

def test_injection_respects_junk_binary_mask(tmp_path):
    emb = [
        [1.0, 0.0, 0.0],  # row0: passes floor
        [0.0, 1.0, 0.0],  # row1: fails floor, strong tag match, junk-masked
    ]
    records = [
        {"file": "p0.jpg", "cat": "widget"},
        {"file": "p1.jpg", "cat": "lumina"},
    ]
    engine = _engine(
        tmp_path, emb, records=records, field_keys={"category": "cat"},
        junk=[0.0, 0.9], junk_threshold=0.5,
    )
    results = engine.text_search(
        "lumina collection", k=10, min_score=0.05, min_ratio=0.6
    )
    rows = {r["row"] for r in results}
    assert 1 not in rows


def test_injection_respects_filters(tmp_path):
    emb = [
        [1.0, 0.0, 0.0],  # row0: passes floor, category=widget
        [0.0, 1.0, 0.0],  # row1: fails floor, category=lumina
    ]
    records = [
        {"file": "p0.jpg", "cat": "widget"},
        {"file": "p1.jpg", "cat": "lumina"},
    ]
    engine = _engine(tmp_path, emb, records=records, field_keys={"category": "cat"})
    results = engine.text_search(
        "lumina collection", k=10, min_score=0.05, min_ratio=0.6,
        filters={"category": "widget"},
    )
    rows = {r["row"] for r in results}
    assert rows == {0}


# ---------------------------------------------------------------------------
# Lexical-support threshold excludes weak-lex injected rows
# ---------------------------------------------------------------------------

def test_weak_lex_injected_row_excluded_by_support_threshold(tmp_path):
    emb = [
        [1.0, 0.0, 0.0],  # row0: anchor, passes floor
        [0.0, 1.0, 0.0],  # row1: fails floor, 3 matching fields -> strong lex
        [0.0, 0.0, 1.0],  # row2: fails floor, 1 matching field -> weak lex
    ]
    records = [
        {"file": "p0.jpg", "cat": "zeta",  "metal": "eta",  "cut": "theta"},
        {"file": "p1.jpg", "cat": "alpha", "metal": "beta", "cut": "delta"},
        {"file": "p2.jpg", "cat": "gamma", "metal": "eta",  "cut": "theta"},
    ]
    field_keys = {"category": "cat", "metal": "metal", "cut": "cut"}
    engine = _engine(tmp_path, emb, records=records, field_keys=field_keys)
    results = engine.text_search(
        "alpha beta delta gamma collection pieces", k=10,
        min_score=0.05, min_ratio=0.6,
    )
    rows = {r["row"] for r in results}
    assert 0 in rows
    assert 1 in rows
    assert 2 not in rows


# ---------------------------------------------------------------------------
# Unified ordering: hybrid_weight decides whether lexical closes the gap
# ---------------------------------------------------------------------------

def test_strong_semantic_outranks_injected_at_low_hybrid_weight(tmp_path):
    emb = [
        [0.5, 0.866, 0.0],  # row0: sims=0.5, passes floor, no tag match
        [0.0, 0.0, 1.0],    # row1: sims=0.0, fails floor, strong tag match
    ]
    records = [
        {"file": "p0.jpg", "cat": "widget"},
        {"file": "p1.jpg", "cat": "strongmatch"},
    ]
    engine = _engine(tmp_path, emb, records=records, field_keys={"category": "cat"})
    results = engine.text_search(
        "strongmatch other words", k=10, min_score=0.05, min_ratio=0.6,
        hybrid_weight=0.08,
    )
    rows_out = [r["row"] for r in results]
    assert rows_out[0] == 0
    assert 1 in rows_out


def test_high_hybrid_weight_lets_injected_row_close_the_gap(tmp_path):
    emb = [
        [0.5, 0.866, 0.0],
        [0.0, 0.0, 1.0],
    ]
    records = [
        {"file": "p0.jpg", "cat": "widget"},
        {"file": "p1.jpg", "cat": "strongmatch"},
    ]
    engine = _engine(tmp_path, emb, records=records, field_keys={"category": "cat"})
    results = engine.text_search(
        "strongmatch other words", k=10, min_score=0.05, min_ratio=0.6,
        hybrid_weight=1.0,
    )
    rows_out = [r["row"] for r in results]
    assert rows_out[0] == 1


# ---------------------------------------------------------------------------
# lexical_inject_k=0 / junk_soft_weight=0 restore prior behavior exactly
# ---------------------------------------------------------------------------

def test_lexical_inject_k_zero_disables_injection(tmp_path):
    emb = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    records = [
        {"file": "p0.jpg", "cat": "widget"},
        {"file": "p1.jpg", "cat": "lumina"},
    ]
    engine = _engine(
        tmp_path, emb, records=records, field_keys={"category": "cat"},
        lexical_inject_k=0,
    )
    results = engine.text_search(
        "lumina collection", k=10, min_score=0.05, min_ratio=0.6
    )
    rows = {r["row"] for r in results}
    assert rows == {0}


def test_junk_soft_weight_zero_restores_raw_scores(tmp_path):
    emb = [
        [0.5, 0.866, 0.0],
        [0.55, 0.835, 0.0],
    ]
    engine = _engine(
        tmp_path, emb, junk=[0.0, 0.2], junk_threshold=0.0,
        junk_soft_weight=0.0, hybrid_weight=0.0,
    )
    results = engine.text_search("q", k=10, min_score=0.0, min_ratio=0.0)
    scored = {r["row"]: r["score"] for r in results}
    assert scored[0] == pytest.approx(0.5, abs=1e-2)
    assert scored[1] == pytest.approx(0.55, abs=1e-2)


# ---------------------------------------------------------------------------
# Soft junk penalty
# ---------------------------------------------------------------------------

def test_soft_penalty_demotes_positive_junk_row(tmp_path):
    emb = [
        [0.5, 0.866, 0.0],   # row0: clean, sims=0.5
        [0.55, 0.835, 0.0],  # row1: sims=0.55 (higher raw), junk-positive
    ]
    engine = _engine(
        tmp_path, emb, junk=[0.0, 0.2], junk_threshold=0.0,  # binary mask off
        hybrid_weight=0.0,  # junk_soft_weight defaults to 1.0
    )
    results = engine.text_search("q", k=10, min_score=0.0, min_ratio=0.0)
    rows_out = [r["row"] for r in results]
    assert rows_out[0] == 0  # 0.55 - 1.0*0.2 = 0.35 < 0.5


def test_negative_junk_scores_unaffected_by_soft_penalty(tmp_path):
    emb = [
        [0.5, 0.866, 0.0],   # row0: junk score negative -> no penalty
        [0.5, -0.866, 0.0],  # row1: same raw sims, junk score positive -> penalized
    ]
    engine = _engine(
        tmp_path, emb, junk=[-0.5, 0.5], junk_threshold=0.0, hybrid_weight=0.0,
    )
    results = engine.text_search("q", k=10, min_score=0.0, min_ratio=0.0)
    scored = {r["row"]: r["score"] for r in results}
    assert scored[0] == pytest.approx(0.5, abs=1e-2)   # unaffected by negative junk
    assert scored[1] == pytest.approx(0.0, abs=1e-2)   # 0.5 - 1.0*0.5 = 0.0
    rows_out = [r["row"] for r in results]
    assert rows_out[0] == 0
