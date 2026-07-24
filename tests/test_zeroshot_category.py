"""Tests for scripts/zeroshot_category.py — scoring, thresholds, apply path."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixgrep.store import save_index
from pixgrep.tags import import_tags
from scripts.zeroshot_category import (
    CATEGORY_PROMPTS,
    apply_predictions,
    build_report,
    embed_prompts,
    load_category_tags,
    recommend_threshold,
    score_rows,
    sweep_margin_thresholds,
    tag_value_to_canonical,
    top_confusions,
    untagged_coverage_at_threshold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


class _FixedEmbedder:
    """Returns fixed unit vectors for a given text -> vec mapping."""

    def __init__(self, mapping: dict[str, np.ndarray]):
        self._m = mapping

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._m[t] for t in texts])


# ---------------------------------------------------------------------------
# embed_prompts
# ---------------------------------------------------------------------------


def test_embed_prompts_normalizes_and_orders_by_label():
    mapping = {
        CATEGORY_PROMPTS["ring"]: np.array([3.0, 0.0], dtype=np.float32),
        CATEGORY_PROMPTS["chain"]: np.array([0.0, 4.0], dtype=np.float32),
    }
    embedder = _FixedEmbedder(mapping)
    vecs = embed_prompts(embedder, ["ring", "chain"])
    assert vecs.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(vecs[0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(vecs[1], [0.0, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# score_rows
# ---------------------------------------------------------------------------


def test_score_rows_picks_highest_cosine():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pred_idx, top1, margin = score_rows(emb, text_vecs)
    assert list(pred_idx) == [0, 1]
    np.testing.assert_allclose(top1, [1.0, 1.0], atol=1e-6)


def test_score_rows_margin_is_top1_minus_top2():
    # Row aligned exactly with category 0, close-ish to category 1.
    v = _unit([1.0, 0.3])
    emb = np.array([v], dtype=np.float32)
    text_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pred_idx, top1, margin = score_rows(emb, text_vecs)
    cos0 = float(v[0])
    cos1 = float(v[1])
    assert pred_idx[0] == 0
    assert margin[0] == pytest.approx(cos0 - cos1, abs=1e-5)


def test_score_rows_single_category_margin_equals_top1():
    emb = np.array([[1.0, 0.0]], dtype=np.float32)
    text_vecs = np.array([[1.0, 0.0]], dtype=np.float32)
    pred_idx, top1, margin = score_rows(emb, text_vecs)
    assert margin[0] == pytest.approx(top1[0], abs=1e-6)


def test_score_rows_defensively_normalizes_unnormalized_input():
    emb = np.array([[5.0, 0.0]], dtype=np.float32)  # not unit-length
    text_vecs = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    pred_idx, top1, margin = score_rows(emb, text_vecs)
    assert pred_idx[0] == 0
    assert top1[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# tag_value_to_canonical
# ---------------------------------------------------------------------------


def test_tag_value_to_canonical_aliases_fold_to_ring():
    m = tag_value_to_canonical()
    assert m["ring"] == "ring"
    assert m["fashion ring"] == "ring"
    assert m["gents ring's"] == "ring"
    assert m["earring"] == "earring"
    assert "bridal-set" not in m  # deliberately unmapped


# ---------------------------------------------------------------------------
# sweep_margin_thresholds / recommend_threshold
# ---------------------------------------------------------------------------


def test_sweep_accuracy_and_coverage():
    pred = ["ring", "ring", "earring", "ring"]
    true = ["ring", "earring", "earring", "ring"]
    margin = np.array([0.10, 0.01, 0.10, 0.02], dtype=np.float32)
    sweep = sweep_margin_thresholds(pred, true, margin, thresholds=(0.0, 0.02, 0.05))

    row0 = sweep[0]
    assert row0["threshold"] == 0.0
    assert row0["n"] == 4
    assert row0["coverage"] == pytest.approx(1.0)
    assert row0["accuracy"] == pytest.approx(3 / 4)

    row1 = sweep[1]  # threshold 0.02 drops the 0.01-margin wrong row
    assert row1["n"] == 3
    assert row1["accuracy"] == pytest.approx(1.0)

    row2 = sweep[2]  # threshold 0.05 keeps only the two 0.10-margin rows
    assert row2["n"] == 2
    assert row2["accuracy"] == pytest.approx(1.0)


def test_sweep_empty_validation_set_is_safe():
    sweep = sweep_margin_thresholds([], [], np.array([]), thresholds=(0.0, 0.05))
    assert all(row["n"] == 0 and row["coverage"] == 0.0 and row["accuracy"] == 0.0 for row in sweep)


def test_recommend_threshold_picks_smallest_qualifying():
    sweep = [
        {"threshold": 0.0, "n": 10, "coverage": 1.0, "accuracy": 0.80},
        {"threshold": 0.02, "n": 8, "coverage": 0.8, "accuracy": 0.96},
        {"threshold": 0.05, "n": 5, "coverage": 0.5, "accuracy": 1.00},
    ]
    assert recommend_threshold(sweep, min_accuracy=0.95) == 0.02


def test_recommend_threshold_none_when_unreachable():
    sweep = [
        {"threshold": 0.0, "n": 10, "coverage": 1.0, "accuracy": 0.5},
        {"threshold": 0.05, "n": 2, "coverage": 0.2, "accuracy": 0.6},
    ]
    assert recommend_threshold(sweep, min_accuracy=0.95) is None


# ---------------------------------------------------------------------------
# top_confusions
# ---------------------------------------------------------------------------


def test_top_confusions_counts_and_orders_mismatches():
    pred = ["ring", "earring", "ring", "ring", "bracelet"]
    true = ["ring", "ring", "ring", "earring", "bracelet"]
    confusions = top_confusions(pred, true)
    assert confusions[0] == (("ring", "earring"), 1) or confusions[0] == (("earring", "ring"), 1)
    pairs = dict(confusions)
    assert pairs[("ring", "earring")] == 1
    assert pairs[("earring", "ring")] == 1
    assert ("bracelet", "bracelet") not in pairs  # correct predictions excluded


# ---------------------------------------------------------------------------
# load_category_tags
# ---------------------------------------------------------------------------


def test_load_category_tags_reads_field(tmp_path):
    save_index(tmp_path, ["a.jpg", "b.jpg", "c.jpg"], ["a", "b", "c"], np.eye(3, dtype=np.float32))
    import_tags(
        tmp_path,
        [{"fn": "a.jpg", "cat": "ring"}, {"fn": "b.jpg", "cat": "earring"}],
        "fn",
        {"category": "cat"},
        [],
    )
    tags = load_category_tags(tmp_path)
    assert tags == {0: "ring", 1: "earring"}


def test_load_category_tags_missing_table_returns_empty(tmp_path):
    save_index(tmp_path, ["a.jpg"], ["a"], np.eye(1, dtype=np.float32))
    assert load_category_tags(tmp_path) == {}


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


def test_build_report_excludes_unmapped_and_lists_untagged():
    labels = list(CATEGORY_PROMPTS.keys())
    # 4 rows: 0 tagged "ring" (correct pred), 1 tagged "bridal-set" (unmapped),
    # 2 untagged, 3 tagged "earring" (wrong pred -> confusion).
    pred_canonical = ["ring", "ring", "ring", "ring"]
    margin = np.array([0.10, 0.10, 0.10, 0.10], dtype=np.float32)
    category_tags = {0: "ring", 1: "bridal-set", 3: "earring"}

    report = build_report(labels, pred_canonical, margin, category_tags)

    assert report["n_total"] == 4
    assert report["n_with_tag"] == 3
    assert report["n_mapped"] == 2  # rows 0 and 3
    assert report["n_unmapped"] == 1  # row 1
    assert report["unmapped_counts"]["bridal-set"] == 1
    assert report["untagged_rows"] == [2]
    assert report["overall_agreement"] == pytest.approx(1 / 2)  # 1 correct of 2 mapped
    assert (("earring", "ring"), 1) in report["confusions"]


# ---------------------------------------------------------------------------
# untagged_coverage_at_threshold
# ---------------------------------------------------------------------------


def test_untagged_coverage_splits_vocab_and_non_vocab():
    pred_canonical = ["ring", "watch", "earring", "ring"]
    margin = np.array([0.10, 0.10, 0.01, 0.10], dtype=np.float32)
    untagged_rows = [0, 1, 2, 3]
    cov = untagged_coverage_at_threshold(untagged_rows, pred_canonical, margin, threshold=0.03)
    # row 2 dropped by threshold; row 1 predicts "watch" (non-vocab) -> skipped
    assert cov["n_untagged"] == 4
    assert cov["writable"] == 2  # rows 0 and 3 (ring)
    assert cov["skipped_non_vocab"] == 1  # row 1 (watch)


# ---------------------------------------------------------------------------
# apply_predictions (integration, synthetic index only)
# ---------------------------------------------------------------------------


def _build_index(tmp_path, filenames, tagged: dict[int, str]):
    n = len(filenames)
    emb = np.eye(max(n, 2), dtype=np.float32)[:n]
    save_index(tmp_path, filenames, [f"g{i}" for i in range(n)], emb)
    if tagged:
        records = [{"fn": filenames[r], "cat": v} for r, v in tagged.items()]
        import_tags(tmp_path, records, "fn", {"category": "cat"}, [])
    return filenames


def test_apply_writes_only_untagged_rows_above_threshold(tmp_path):
    filenames = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    _build_index(tmp_path, filenames, tagged={0: "necklace"})  # row 0 already tagged

    pred_canonical = ["ring", "ring", "earring", "watch"]
    margin = np.array([0.10, 0.01, 0.10, 0.10], dtype=np.float32)
    untagged_rows = [1, 2, 3]  # row 0 excluded by caller (already tagged)

    result = apply_predictions(
        tmp_path, filenames, untagged_rows, pred_canonical, margin, threshold=0.03
    )

    # row 1 dropped (margin below threshold), row 3 dropped (non-vocab "watch")
    # only row 2 ("earring") qualifies
    assert result["attempted"] == 1

    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    rows = con.execute("SELECT row, value FROM tags WHERE field='category'").fetchall()
    con.close()
    assert (0, "necklace") in rows  # untouched
    assert (2, "earring") in rows
    assert (1, "ring") not in rows
    assert not any(r == 3 for r, _ in rows)


def test_apply_never_writes_non_vocab_label(tmp_path):
    filenames = ["a.jpg", "b.jpg"]
    _build_index(tmp_path, filenames, tagged={})

    pred_canonical = ["watch", "gemstone"]
    margin = np.array([0.5, 0.5], dtype=np.float32)
    result = apply_predictions(tmp_path, filenames, [0, 1], pred_canonical, margin, threshold=0.0)

    assert result["attempted"] == 0
    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    count = con.execute("SELECT COUNT(*) FROM tags").fetchone()[0] if "tags" in tables else 0
    con.close()
    assert count == 0


def test_apply_is_idempotent(tmp_path):
    filenames = ["a.jpg", "b.jpg"]
    _build_index(tmp_path, filenames, tagged={})
    pred_canonical = ["ring", "ring"]
    margin = np.array([0.5, 0.5], dtype=np.float32)

    apply_predictions(tmp_path, filenames, [0, 1], pred_canonical, margin, threshold=0.0)
    apply_predictions(tmp_path, filenames, [0, 1], pred_canonical, margin, threshold=0.0)

    con = sqlite3.connect(str(tmp_path / "pixgrep.sqlite"))
    n = con.execute("SELECT COUNT(*) FROM tags WHERE field='category'").fetchone()[0]
    con.close()
    assert n == 2  # not duplicated
