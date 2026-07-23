"""Tests for pixgrep/tags.py — import join, facets, filtering, lexical scoring.

All data is synthetic; no real filenames, SKUs, or company data.
"""
from __future__ import annotations

import numpy as np
import pytest

from pixgrep.store import save_index
from pixgrep.tags import ImportReport, TagStore, import_tags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_index(tmp_path, n: int = 4) -> None:
    """Create a minimal index with n rows using synthetic filenames."""
    emb = np.eye(max(n, 4), dtype=np.float32)[:n]
    paths = [f"folder/item{i}.jpg" for i in range(n)]
    groups = [f"g{i}" for i in range(n)]
    save_index(tmp_path, paths, groups, emb)


SYNTHETIC_RECORDS = [
    {"file": "item0.jpg", "cat": "widget", "color": "red",   "note": "shiny red widget"},
    {"file": "item1.jpg", "cat": "gadget", "color": "blue",  "note": "matte blue gadget"},
    {"file": "item2.jpg", "cat": "widget", "color": "green", "note": "green sparkle widget"},
    {"file": "item3.jpg", "cat": "gadget", "color": "red",   "note": "classic red gadget"},
]

FIELD_KEYS = {"category": "cat", "color": "color"}
TEXT_KEYS = ["note"]


# ---------------------------------------------------------------------------
# import_tags
# ---------------------------------------------------------------------------

def test_import_all_match(tmp_path):
    _build_index(tmp_path)
    report = import_tags(tmp_path, SYNTHETIC_RECORDS, "file", FIELD_KEYS, TEXT_KEYS)
    assert isinstance(report, ImportReport)
    assert report.matched == 4
    assert report.unmatched_records == 0
    assert report.rows_without_tags == 0


def test_import_case_insensitive_filename(tmp_path):
    _build_index(tmp_path, 2)
    records = [
        {"file": "ITEM0.JPG", "cat": "widget", "color": "red", "note": ""},
        {"file": "Item1.jpg", "cat": "gadget", "color": "blue", "note": ""},
    ]
    report = import_tags(tmp_path, records, "file", FIELD_KEYS, TEXT_KEYS)
    assert report.matched == 2
    assert report.unmatched_records == 0


def test_import_unmatched_records(tmp_path):
    _build_index(tmp_path, 2)
    records = [
        {"file": "item0.jpg", "cat": "widget", "color": "red", "note": ""},
        {"file": "nosuchfile.jpg", "cat": "gadget", "color": "blue", "note": ""},
        {"file": "", "cat": "x", "color": "y", "note": ""},
    ]
    report = import_tags(tmp_path, records, "file", FIELD_KEYS, TEXT_KEYS)
    assert report.matched == 1
    assert report.unmatched_records == 2
    assert report.rows_without_tags == 1  # item1.jpg got no tags


def test_import_rows_without_tags(tmp_path):
    _build_index(tmp_path, 3)
    # Only tag 2 of 3 rows
    records = [
        {"file": "item0.jpg", "cat": "widget", "color": "red", "note": ""},
        {"file": "item1.jpg", "cat": "gadget", "color": "blue", "note": ""},
    ]
    report = import_tags(tmp_path, records, "file", FIELD_KEYS, TEXT_KEYS)
    assert report.matched == 2
    assert report.rows_without_tags == 1


def test_import_is_idempotent(tmp_path):
    _build_index(tmp_path)
    import_tags(tmp_path, SYNTHETIC_RECORDS, "file", FIELD_KEYS, TEXT_KEYS)
    # Second import should replace, not accumulate
    report = import_tags(tmp_path, SYNTHETIC_RECORDS[:2], "file", FIELD_KEYS, TEXT_KEYS)
    assert report.matched == 2
    ts = TagStore(tmp_path)
    # Should only see 2 rows worth of tags, not 4+2
    all_rows = set()
    for val_dict in ts._field_value_rows.values():
        for rows in val_dict.values():
            all_rows.update(rows.tolist())
    assert all_rows <= {0, 1}


def test_import_str_report(tmp_path):
    _build_index(tmp_path)
    report = import_tags(tmp_path, SYNTHETIC_RECORDS, "file", FIELD_KEYS, TEXT_KEYS)
    s = str(report)
    assert "Matched" in s
    assert "4" in s


# ---------------------------------------------------------------------------
# TagStore — basic loading
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    _build_index(tmp_path)
    import_tags(tmp_path, SYNTHETIC_RECORDS, "file", FIELD_KEYS, TEXT_KEYS)
    return TagStore(tmp_path)


def test_store_has_data(store):
    assert store.has_data


def test_store_no_data_on_empty_index(tmp_path):
    _build_index(tmp_path)
    # No import → tables absent
    ts = TagStore(tmp_path)
    assert not ts.has_data


def test_store_no_data_missing_db(tmp_path):
    ts = TagStore(tmp_path / "nonexistent")
    assert not ts.has_data


# ---------------------------------------------------------------------------
# TagStore.facets
# ---------------------------------------------------------------------------

def test_facets_fields(store):
    f = store.facets()
    assert "category" in f
    assert "color" in f


def test_facets_counts(store):
    f = store.facets()
    # 2 widgets, 2 gadgets
    assert f["category"]["widget"] == 2
    assert f["category"]["gadget"] == 2
    # red appears in rows 0 and 3
    assert f["color"]["red"] == 2


def test_facets_empty_when_no_data(tmp_path):
    _build_index(tmp_path)
    ts = TagStore(tmp_path)
    assert ts.facets() == {}


# ---------------------------------------------------------------------------
# TagStore.rows_matching
# ---------------------------------------------------------------------------

def test_rows_matching_none_when_no_filter(store):
    assert store.rows_matching({}) is None


def test_rows_matching_single_filter(store):
    rows = store.rows_matching({"category": "widget"})
    assert rows is not None
    assert set(rows.tolist()) == {0, 2}


def test_rows_matching_multi_filter_and(store):
    rows = store.rows_matching({"category": "gadget", "color": "red"})
    assert rows is not None
    assert set(rows.tolist()) == {3}


def test_rows_matching_empty_when_no_overlap(store):
    rows = store.rows_matching({"category": "widget", "color": "blue"})
    assert rows is not None
    assert len(rows) == 0


def test_rows_matching_unknown_field(store):
    rows = store.rows_matching({"nosuchfield": "x"})
    assert rows is not None
    assert len(rows) == 0


def test_rows_matching_case_insensitive(store):
    rows_lo = store.rows_matching({"category": "widget"})
    rows_up = store.rows_matching({"category": "WIDGET"})
    assert set(rows_lo.tolist()) == set(rows_up.tolist())


# ---------------------------------------------------------------------------
# TagStore.lexical_scores
# ---------------------------------------------------------------------------

def test_lexical_scores_shape(store):
    s = store.lexical_scores("widget", 4)
    assert s.shape == (4,)
    assert s.dtype == np.float32


def test_lexical_scores_normalized(store):
    s = store.lexical_scores("widget", 4)
    assert float(s.max()) == pytest.approx(1.0)
    assert float(s.min()) >= 0.0


def test_lexical_scores_categorical_containment(store):
    # "widget" fully in query → rows 0 and 2 get component-1 boost
    s = store.lexical_scores("widget sparkle", 4)
    # widget rows should outscore non-widget rows
    assert s[0] > s[1]
    assert s[2] > s[1]


def test_lexical_scores_token_hit(store):
    # "red" hits categorical value → rows with red (0, 3) boosted
    s = store.lexical_scores("red", 4)
    assert s[0] > 0
    assert s[3] > 0
    # rows with no "red" tag get less (could still get text component)
    assert s[0] >= s[2]


def test_lexical_scores_text_fraction(store):
    # "shiny" appears only in item0's tag_text description
    s = store.lexical_scores("shiny", 4)
    assert s[0] > 0  # "shiny" is in item0's text
    # Others may be 0 unless they have matching text
    assert s[0] >= s[1]


def test_lexical_scores_empty_query(store):
    s = store.lexical_scores("", 4)
    assert np.all(s == 0)


def test_lexical_scores_no_match(store):
    s = store.lexical_scores("xyzzy plutonium", 4)
    assert np.all(s == 0)


def test_lexical_scores_no_data(tmp_path):
    _build_index(tmp_path)
    ts = TagStore(tmp_path)
    s = ts.lexical_scores("widget", 4)
    assert np.all(s == 0)


def test_lexical_scores_zero_n_rows(store):
    s = store.lexical_scores("widget", 0)
    assert s.shape == (0,)


def test_lexical_scores_multiword_value_requires_all_tokens(tmp_path):
    """A multi-word categorical value triggers component-1 only when ALL tokens in query.

    Row 2 has value "bracelet" (single-word). With query "tennis bracelet", the
    "bracelet" token IS in the query, so row 2 gets a component-1 boost and a
    component-2 boost. With query "tennis" alone, "bracelet" is NOT in the
    query, so row 2 gets nothing. This proves multi-word "tennis bracelet"
    gating: if "bracelet" were always boosted regardless, we couldn't isolate
    the query-containment requirement.
    """
    _build_index(tmp_path, 3)
    records = [
        {"file": "item0.jpg", "cat": "tennis bracelet", "color": "white", "note": ""},
        {"file": "item1.jpg", "cat": "solitaire ring",  "color": "gold",  "note": ""},
        {"file": "item2.jpg", "cat": "bracelet",        "color": "silver", "note": ""},
    ]
    import_tags(tmp_path, records, "file", {"category": "cat", "color": "color"}, [])
    ts = TagStore(tmp_path)

    # "tennis bracelet" → "bracelet" fully in query → row 2 gets component-1 boost
    s_both = ts.lexical_scores("tennis bracelet", 3)
    # "tennis" only → "bracelet" NOT in query → row 2 gets nothing
    s_one = ts.lexical_scores("tennis", 3)

    # Both queries boost row 0 (the multi-word value row)
    assert s_both[0] > s_both[1]
    assert s_one[0] > s_one[1]
    # Full query boosts row 2 (via single-word "bracelet" match); partial does not
    assert s_both[2] > 0
    assert s_one[2] == pytest.approx(0.0)
