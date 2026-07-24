"""Tests for near-duplicate collapse at ranking time (pixgrep/search.py _rank)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pixgrep.search import SearchEngine
from pixgrep.store import save_index
from pixgrep.tags import import_tags


class _FixedQueryEmbedder:
    """Always returns an axis-0 unit vector, so sims[i] == emb[i][0]."""

    def __init__(self, dim: int):
        self.dim = dim

    def embed_texts(self, texts):
        v = np.zeros(self.dim, dtype=np.float32)
        v[0] = 1.0
        return np.vstack([v] * len(texts))

    def embed_images(self, images):
        return self.embed_texts(["x"] * len(images))


def _angle_vec(angle: float, dim: int = 2) -> list[float]:
    v = [0.0] * dim
    v[0] = math.cos(angle)
    v[1] = math.sin(angle)
    return v


def _build(tmp_path, rows, near_dupe_cos: float = 0.985) -> SearchEngine:
    """rows: list of (angle, group) — vectors are 2D unit vectors on the
    unit circle, so dot(v_i, v_j) == cos(angle_i - angle_j), and the query
    (axis-0 unit vector from _FixedQueryEmbedder) scores each row by
    cos(angle) too (its own first coordinate).
    """
    emb = np.array([_angle_vec(a) for a, _ in rows], dtype=np.float32)
    paths = [f"p{i}.jpg" for i in range(len(rows))]
    groups = [g for _, g in rows]
    save_index(tmp_path, paths, groups, emb)
    return SearchEngine(tmp_path, _FixedQueryEmbedder(dim=2), near_dupe_cos=near_dupe_cos)


# cos(theta) values used across tests
_ABOVE = math.acos(0.99)   # dot ~0.99 > default 0.985 threshold
_BELOW = math.acos(0.98)   # dot ~0.98 < default 0.985 threshold


def test_near_identical_pair_same_group_only_higher_survives(tmp_path):
    # row0: angle 0 -> sims=1.0 (best); row1: angle _ABOVE -> sims=0.99, cos(row0,row1)=0.99
    rows = [(0.0, "G1"), (_ABOVE, "G1")]
    engine = _build(tmp_path, rows)
    results = engine.text_search("q", k=2, min_ratio=0.0, min_score=0.0)
    rows_out = [r["row"] for r in results]
    assert rows_out == [0]


def test_near_identical_pair_different_groups_both_stay(tmp_path):
    rows = [(0.0, "G1"), (_ABOVE, "G2")]
    engine = _build(tmp_path, rows)
    results = engine.text_search("q", k=2, min_ratio=0.0, min_score=0.0)
    rows_out = [r["row"] for r in results]
    assert rows_out == [0, 1]


def test_cos_below_threshold_same_group_both_stay(tmp_path):
    rows = [(0.0, "G1"), (_BELOW, "G1")]
    engine = _build(tmp_path, rows)
    results = engine.text_search("q", k=2, min_ratio=0.0, min_score=0.0)
    rows_out = [r["row"] for r in results]
    assert rows_out == [0, 1]


def test_near_dupe_cos_zero_disables_collapse(tmp_path):
    rows = [(0.0, "G1"), (_ABOVE, "G1")]
    engine = _build(tmp_path, rows, near_dupe_cos=0.0)
    results = engine.text_search("q", k=2, min_ratio=0.0, min_score=0.0)
    rows_out = [r["row"] for r in results]
    assert rows_out == [0, 1]


def test_overfetch_backfills_after_collapse(tmp_path):
    # row0 (best) and row1 are a near-dupe pair, same group -> row1 collapses.
    # rows 2 and 3 are distinct (different groups, no near-dupe with anything).
    # k=3 should still return 3 results: 0, 2, 3 (row1 dropped, backfilled).
    rows = [
        (0.0, "G1"),                 # sims=1.00
        (_ABOVE, "G1"),              # sims=0.99, dupe of row0 -> dropped
        (math.radians(40), "G2"),    # sims=cos(40deg)~0.766
        (math.radians(50), "G3"),    # sims=cos(50deg)~0.643
    ]
    engine = _build(tmp_path, rows)
    results = engine.text_search("q", k=3, min_ratio=0.0, min_score=0.0)
    rows_out = [r["row"] for r in results]
    assert rows_out == [0, 2, 3]


def test_floors_respected_in_overfetched_pool(tmp_path):
    # row1 is a near-dupe of row0 (dropped by collapse); row2 has a much
    # lower semantic score that fails min_score even though the pool is
    # over-fetched to make room for backfill.
    rows = [
        (0.0, "G1"),                  # sims=1.0
        (_ABOVE, "G1"),               # sims=0.99, dupe -> dropped
        (math.radians(89), "G2"),     # sims=cos(89deg)~0.017 -> below floor
    ]
    engine = _build(tmp_path, rows)
    results = engine.text_search("q", k=3, min_ratio=0.0, min_score=0.5)
    rows_out = [r["row"] for r in results]
    assert rows_out == [0]  # row1 collapsed, row2 fails min_score floor


def test_exclude_interaction_unchanged(tmp_path):
    rows = [(0.0, "G1"), (_ABOVE, "G1"), (math.radians(40), "G2")]
    engine = _build(tmp_path, rows)
    results = engine.similar(0, k=3, min_ratio=0.0, min_score=0.0)
    rows_out = [r["row"] for r in results]
    assert 0 not in rows_out
    # row1 is a near-dupe of row0, but row0 (self) is excluded, so row1 is
    # the top-ranked survivor and nothing outranks it to collapse against.
    assert rows_out == [1, 2]


def test_filters_interaction_unchanged(tmp_path):
    emb = np.array(
        [_angle_vec(0.0), _angle_vec(_ABOVE), _angle_vec(math.radians(40))],
        dtype=np.float32,
    )
    paths = ["a.jpg", "b.jpg", "c.jpg"]
    groups = ["G1", "G1", "G2"]
    save_index(tmp_path, paths, groups, emb)
    records = [
        {"fn": "a.jpg", "cat": "ring"},
        {"fn": "b.jpg", "cat": "bracelet"},
        {"fn": "c.jpg", "cat": "ring"},
    ]
    import_tags(tmp_path, records, "fn", {"category": "cat"}, [])
    engine = SearchEngine(tmp_path, _FixedQueryEmbedder(dim=2), near_dupe_cos=0.985)

    results = engine.text_search(
        "q", k=3, min_ratio=0.0, min_score=0.0, filters={"category": "bracelet"}
    )
    rows_out = [r["row"] for r in results]
    assert rows_out == [1]  # only row1 matches filter; collapse doesn't remove it


def test_junk_mask_interaction_unchanged(tmp_path):
    from pixgrep.junk import save_junk_scores

    rows = [(0.0, "G1"), (_ABOVE, "G1"), (math.radians(40), "G2")]
    emb = np.array([_angle_vec(a) for a, _ in rows], dtype=np.float32)
    paths = [f"p{i}.jpg" for i in range(len(rows))]
    groups = [g for _, g in rows]
    save_index(tmp_path, paths, groups, emb)
    save_junk_scores(tmp_path, np.array([0.9, 0.0, 0.0], dtype=np.float32))
    engine = SearchEngine(
        tmp_path, _FixedQueryEmbedder(dim=2), near_dupe_cos=0.985, junk_threshold=0.5
    )
    results = engine.text_search("q", k=3, min_ratio=0.0, min_score=0.0)
    rows_out = [r["row"] for r in results]
    # row0 masked as junk -> row1 becomes top-ranked survivor, nothing above
    # it to collapse against, row2 stays too.
    assert rows_out == [1, 2]
