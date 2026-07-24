"""Tests for the /api/feedback/* endpoints in pixgrep/server.py.

Exercises the exact API contract another worker's UI codes against:
POST toggle, GET marks, GET list, DELETE by id / by user / by query.
All data is synthetic; no real filenames, SKUs, or company data.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pixgrep.feedback import FeedbackStore, normalize_query_key
from pixgrep.search import SearchEngine
from pixgrep.server import create_app
from pixgrep.store import open_db, save_index


class FakeEmbedder:
    def embed_texts(self, texts):
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def embed_images(self, images):
        return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)


@pytest.fixture()
def feedback_client(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    index_dir = tmp_path / "index"
    rel_paths = []
    emb = np.eye(4, dtype=np.float32)[:3]
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        p = img_dir / f"img{i}.jpg"
        Image.new("RGB", (8, 8), color).save(p)
        rel_paths.append(str(p))
    save_index(index_dir, rel_paths, ["g0", "g1", "g2"], emb)
    con = open_db(index_dir)
    for i, sha in enumerate(["sha-0", "sha-1", "sha-2"]):
        con.execute("UPDATE images SET sha1=? WHERE row=?", (sha, i))
    con.commit()
    con.close()

    store = FeedbackStore(tmp_path / "feedback.sqlite")
    engine = SearchEngine(index_dir, FakeEmbedder(), feedback=store)
    app = create_app(engine, store)
    return TestClient(app), store, engine


# --- POST /api/feedback/toggle ---

def test_toggle_marks_and_returns_marked_true(feedback_client):
    client, store, _ = feedback_client
    r = client.post("/api/feedback/toggle", json={"user": "alice", "query": "ring", "row": 0})
    assert r.status_code == 200
    assert r.json() == {"marked": True}
    assert store.suppressed_sha1s(normalize_query_key("ring")) == {"sha-0"}


def test_toggle_twice_unmarks(feedback_client):
    client, _, _ = feedback_client
    client.post("/api/feedback/toggle", json={"user": "alice", "query": "ring", "row": 0})
    r = client.post("/api/feedback/toggle", json={"user": "alice", "query": "ring", "row": 0})
    assert r.json() == {"marked": False}


def test_toggle_empty_user_400(feedback_client):
    client, _, _ = feedback_client
    r = client.post("/api/feedback/toggle", json={"user": "", "query": "ring", "row": 0})
    assert r.status_code == 400
    r2 = client.post("/api/feedback/toggle", json={"user": "  ", "query": "ring", "row": 0})
    assert r2.status_code == 400


def test_toggle_empty_query_400(feedback_client):
    client, _, _ = feedback_client
    r = client.post("/api/feedback/toggle", json={"user": "alice", "query": "", "row": 0})
    assert r.status_code == 400
    r2 = client.post("/api/feedback/toggle", json={"user": "alice", "query": "  ", "row": 0})
    assert r2.status_code == 400


def test_toggle_bad_row_404(feedback_client):
    client, _, _ = feedback_client
    r = client.post("/api/feedback/toggle", json={"user": "alice", "query": "ring", "row": 999})
    assert r.status_code == 404


# --- GET /api/feedback/marks ---

def test_marks_endpoint_returns_current_rows(feedback_client):
    client, _, _ = feedback_client
    client.post("/api/feedback/toggle", json={"user": "alice", "query": "ring", "row": 0})
    client.post("/api/feedback/toggle", json={"user": "bob", "query": "ring", "row": 2})
    r = client.get("/api/feedback/marks", params={"query": "ring"})
    assert r.status_code == 200
    assert sorted(r.json()["rows"]) == [0, 2]


def test_marks_endpoint_empty_when_no_marks(feedback_client):
    client, _, _ = feedback_client
    r = client.get("/api/feedback/marks", params={"query": "nonexistent"})
    assert r.json() == {"rows": []}


def test_marks_endpoint_shares_shorthand_normalization(feedback_client):
    client, _, _ = feedback_client
    client.post("/api/feedback/toggle", json={"user": "alice", "query": "YG band", "row": 0})
    r = client.get("/api/feedback/marks", params={"query": "yellow gold band"})
    assert r.json()["rows"] == [0]


# --- GET /api/feedback/list ---

def test_list_endpoint_shape_and_ordering(feedback_client):
    client, _, _ = feedback_client
    client.post("/api/feedback/toggle", json={"user": "alice", "query": "ring", "row": 0})
    client.post("/api/feedback/toggle", json={"user": "bob", "query": "necklace", "row": 1})
    r = client.get("/api/feedback/list")
    assert r.status_code == 200
    marks = r.json()["marks"]
    assert len(marks) == 2
    assert marks[0]["sha1"] == "sha-1"  # newest first
    assert set(marks[0].keys()) == {"id", "user", "query", "sha1", "path", "created"}


# --- DELETE /api/feedback/{id} ---

def test_delete_by_id(feedback_client):
    client, store, _ = feedback_client
    client.post("/api/feedback/toggle", json={"user": "alice", "query": "ring", "row": 0})
    mark_id = store.list_all()[0]["id"]
    r = client.delete(f"/api/feedback/{mark_id}")
    assert r.status_code == 200
    assert r.json() == {"deleted": True}
    assert store.list_all() == []


def test_delete_unknown_id_returns_false(feedback_client):
    client, _, _ = feedback_client
    r = client.delete("/api/feedback/999")
    assert r.status_code == 200
    assert r.json() == {"deleted": False}


# --- DELETE /api/feedback/user/{user} and /api/feedback/query ---

def test_delete_user_bulk(feedback_client):
    client, store, _ = feedback_client
    client.post("/api/feedback/toggle", json={"user": "alice", "query": "ring", "row": 0})
    client.post("/api/feedback/toggle", json={"user": "alice", "query": "necklace", "row": 1})
    client.post("/api/feedback/toggle", json={"user": "bob", "query": "ring", "row": 2})
    r = client.delete("/api/feedback/user/alice")
    assert r.status_code == 200
    assert r.json() == {"deleted": 2}
    remaining = store.list_all()
    assert len(remaining) == 1 and remaining[0]["user"] == "bob"


def test_delete_query_bulk(feedback_client):
    client, store, _ = feedback_client
    client.post("/api/feedback/toggle", json={"user": "alice", "query": "ring", "row": 0})
    client.post("/api/feedback/toggle", json={"user": "bob", "query": "ring", "row": 1})
    client.post("/api/feedback/toggle", json={"user": "bob", "query": "necklace", "row": 2})
    r = client.delete("/api/feedback/query", params={"query": "ring"})
    assert r.status_code == 200
    assert r.json() == {"deleted": 2}
    remaining = store.list_all()
    assert len(remaining) == 1 and remaining[0]["query"] == "necklace"


# --- end-to-end suppression through /api/search ---

def test_marked_row_suppressed_in_search_endpoint(feedback_client):
    client, _, _ = feedback_client
    r = client.get("/api/search", params={"q": "anything", "k": 2, "min_ratio": 0, "min_score": 0})
    rows_before = [x["row"] for x in r.json()["results"]]
    assert 0 in rows_before

    client.post("/api/feedback/toggle", json={"user": "alice", "query": "anything", "row": 0})
    r2 = client.get("/api/search", params={"q": "anything", "k": 2, "min_ratio": 0, "min_score": 0})
    rows_after = [x["row"] for x in r2.json()["results"]]
    assert 0 not in rows_after


# --- create_app without a feedback store: existing behavior preserved ---

def test_create_app_without_feedback_store_still_serves_search(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    index_dir = tmp_path / "index"
    p = img_dir / "img0.jpg"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(p)
    save_index(index_dir, [str(p)], ["g0"], np.eye(4, dtype=np.float32)[:1])
    engine = SearchEngine(index_dir, FakeEmbedder())
    client = TestClient(create_app(engine))
    assert client.get("/api/meta").status_code == 200
    assert client.get("/api/feedback/list").json() == {"marks": []}
