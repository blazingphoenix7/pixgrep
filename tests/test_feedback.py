"""Tests for pixgrep/feedback.py: FeedbackStore and its normalization key.

All data is synthetic; no real filenames, SKUs, or company data.
"""
from __future__ import annotations

from pixgrep.feedback import FeedbackStore, normalize_query_key


def _store(tmp_path) -> FeedbackStore:
    return FeedbackStore(tmp_path / "feedback.sqlite")


# --- normalize_query_key ---

def test_normalize_query_key_lowercases_and_strips():
    assert normalize_query_key("  Yellow Gold Band  ") == "yellow gold band"


def test_normalize_query_key_expands_shorthand():
    # "YG" is jewelry-trade shorthand for "yellow gold" (see query_norm.py)
    assert normalize_query_key("YG band") == normalize_query_key("yellow gold band")
    assert normalize_query_key("YG band") == "yellow gold band"


# --- toggle ---

def test_toggle_marks_then_unmarks(tmp_path):
    store = _store(tmp_path)
    assert store.toggle("alice", "ring", "sha-1", 0, "a.jpg") is True
    assert store.toggle("alice", "ring", "sha-1", 0, "a.jpg") is False


def test_toggle_is_per_query(tmp_path):
    """Marking a sha1 bad for one query doesn't mark it for another query."""
    store = _store(tmp_path)
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")
    assert store.suppressed_sha1s("ring") == {"sha-1"}
    assert store.suppressed_sha1s("necklace") == set()


def test_toggle_unique_constraint_survives_repeated_mark_unmark(tmp_path):
    """Re-marking after unmarking must succeed (no leftover unique-constraint
    row blocking a fresh insert)."""
    store = _store(tmp_path)
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")  # unmark
    assert store.toggle("alice", "ring", "sha-1", 0, "a.jpg") is True  # mark again
    assert store.suppressed_sha1s("ring") == {"sha-1"}


# --- suppressed_sha1s: shared across users ---

def test_marks_are_shared_across_users(tmp_path):
    store = _store(tmp_path)
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")
    # bob's search for the same query is suppressed by alice's mark too
    assert store.suppressed_sha1s("ring") == {"sha-1"}


def test_suppressed_sha1s_unions_multiple_users(tmp_path):
    store = _store(tmp_path)
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")
    store.toggle("bob", "ring", "sha-2", 1, "b.jpg")
    assert store.suppressed_sha1s("ring") == {"sha-1", "sha-2"}


# --- marks_for_query ---

def test_marks_for_query_returns_matching_records(tmp_path):
    store = _store(tmp_path)
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")
    store.toggle("alice", "necklace", "sha-2", 1, "b.jpg")
    marks = store.marks_for_query("ring")
    assert len(marks) == 1
    m = marks[0]
    assert m["user"] == "alice"
    assert m["query"] == "ring"
    assert m["sha1"] == "sha-1"
    assert m["path"] == "a.jpg"
    assert "id" in m and "created" in m


# --- list_all ---

def test_list_all_newest_first(tmp_path):
    store = _store(tmp_path)
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")
    store.toggle("bob", "necklace", "sha-2", 1, "b.jpg")
    marks = store.list_all()
    assert [m["sha1"] for m in marks] == ["sha-2", "sha-1"]


def test_list_all_includes_all_users_and_queries(tmp_path):
    store = _store(tmp_path)
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")
    store.toggle("bob", "necklace", "sha-2", 1, "b.jpg")
    assert len(store.list_all()) == 2


# --- delete by id ---

def test_delete_by_id(tmp_path):
    store = _store(tmp_path)
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")
    mark_id = store.list_all()[0]["id"]
    assert store.delete(mark_id) is True
    assert store.suppressed_sha1s("ring") == set()


def test_delete_unknown_id_returns_false(tmp_path):
    store = _store(tmp_path)
    assert store.delete(999) is False


# --- bulk deletes ---

def test_delete_user_removes_only_that_users_marks(tmp_path):
    store = _store(tmp_path)
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")
    store.toggle("alice", "necklace", "sha-2", 1, "b.jpg")
    store.toggle("bob", "ring", "sha-3", 2, "c.jpg")
    assert store.delete_user("alice") == 2
    remaining = store.list_all()
    assert len(remaining) == 1
    assert remaining[0]["user"] == "bob"


def test_delete_query_removes_only_that_querys_marks(tmp_path):
    store = _store(tmp_path)
    store.toggle("alice", "ring", "sha-1", 0, "a.jpg")
    store.toggle("bob", "ring", "sha-2", 1, "b.jpg")
    store.toggle("bob", "necklace", "sha-3", 2, "c.jpg")
    assert store.delete_query("ring") == 2
    remaining = store.list_all()
    assert len(remaining) == 1
    assert remaining[0]["query"] == "necklace"


# --- fresh-connection reads (no caching) ---

def test_reads_see_writes_from_a_different_store_instance(tmp_path):
    """Every read opens a fresh connection: marks apply instantly, with no
    in-process caching to go stale."""
    db_path = tmp_path / "feedback.sqlite"
    writer = FeedbackStore(db_path)
    writer.toggle("alice", "ring", "sha-1", 0, "a.jpg")

    reader = FeedbackStore(db_path)
    assert reader.suppressed_sha1s("ring") == {"sha-1"}
