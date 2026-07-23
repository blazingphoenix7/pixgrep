"""Server tests for /api/facets and filter params."""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pixgrep.search import SearchEngine
from pixgrep.server import create_app
from pixgrep.store import save_index
from pixgrep.tags import import_tags


class FakeEmbedder:
    def embed_texts(self, texts):
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def embed_images(self, images):
        return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)


@pytest.fixture()
def tagged_client(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    emb = np.eye(4, dtype=np.float32)[:3]
    paths = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        p = img_dir / f"item{i}.jpg"
        Image.new("RGB", (8, 8), color).save(p)
        paths.append(str(p))
    idx = tmp_path / "index"
    save_index(idx, paths, ["g0", "g1", "g2"], emb)

    records = [
        {"fn": "item0.jpg", "cat": "ring",     "metal": "gold"},
        {"fn": "item1.jpg", "cat": "bracelet", "metal": "silver"},
        {"fn": "item2.jpg", "cat": "bracelet", "metal": "gold"},
    ]
    import_tags(idx, records, "fn", {"category": "cat", "metal": "metal"}, [])
    engine = SearchEngine(idx, FakeEmbedder())
    return TestClient(create_app(engine))


@pytest.fixture()
def plain_client(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    emb = np.eye(4, dtype=np.float32)[:3]
    paths = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        p = img_dir / f"item{i}.jpg"
        Image.new("RGB", (8, 8), color).save(p)
        paths.append(str(p))
    idx = tmp_path / "index"
    save_index(idx, paths, ["g0", "g1", "g2"], emb)
    engine = SearchEngine(idx, FakeEmbedder())
    return TestClient(create_app(engine))


# ---------------------------------------------------------------------------
# /api/facets
# ---------------------------------------------------------------------------

def test_facets_empty_without_tags(plain_client):
    r = plain_client.get("/api/facets")
    assert r.status_code == 200
    assert r.json() == {}


def test_facets_with_tags(tagged_client):
    r = tagged_client.get("/api/facets")
    assert r.status_code == 200
    data = r.json()
    assert "category" in data
    assert "metal" in data
    cats = {x["value"] for x in data["category"]}
    assert "ring" in cats
    assert "bracelet" in cats


def test_facets_sorted_by_count_desc(tagged_client):
    r = tagged_client.get("/api/facets")
    cats = r.json()["category"]
    counts = [x["count"] for x in cats]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# /api/search with f= filters
# ---------------------------------------------------------------------------

def test_search_with_f_param(tagged_client):
    r = tagged_client.get("/api/search", params=[("q", "anything"), ("f", "category:bracelet"),
                                                  ("min_ratio", "0"), ("min_score", "0")])
    assert r.status_code == 200
    rows = {x["row"] for x in r.json()["results"]}
    # rows 1 and 2 are bracelet; row 0 (ring) must be excluded
    assert 0 not in rows


def test_search_multi_f_params(tagged_client):
    r = tagged_client.get("/api/search",
                          params=[("q", "anything"), ("f", "category:bracelet"),
                                  ("f", "metal:gold"), ("min_ratio", "0"), ("min_score", "0")])
    assert r.status_code == 200
    rows = {x["row"] for x in r.json()["results"]}
    # Only row 2 is bracelet+gold
    assert rows == {2}


def test_search_f_no_match_returns_empty(tagged_client):
    r = tagged_client.get("/api/search",
                          params=[("q", "anything"), ("f", "metal:platinum"),
                                  ("min_ratio", "0"), ("min_score", "0")])
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_search_without_f_returns_normal(tagged_client):
    r = tagged_client.get("/api/search", params={"q": "anything", "min_score": "0", "min_ratio": "0"})
    assert r.status_code == 200
    assert len(r.json()["results"]) > 0


# ---------------------------------------------------------------------------
# /api/search with hw= param
# ---------------------------------------------------------------------------

def test_search_hw_zero_no_crash(tagged_client):
    r = tagged_client.get("/api/search", params={"q": "ring gold", "hw": "0",
                                                  "min_ratio": "0", "min_score": "0"})
    assert r.status_code == 200


def test_search_hw_one_no_crash(tagged_client):
    r = tagged_client.get("/api/search", params={"q": "ring gold", "hw": "1.0",
                                                  "min_ratio": "0", "min_score": "0"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /api/search/image with f= filters
# ---------------------------------------------------------------------------

def test_image_search_with_f_param(tagged_client):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (9, 9, 9)).save(buf, format="JPEG")
    r = tagged_client.post(
        "/api/search/image",
        files={"file": ("q.jpg", buf.getvalue(), "image/jpeg")},
        params=[("f", "category:ring"), ("min_ratio", "0"), ("min_score", "0")],
    )
    assert r.status_code == 200
    rows = {x["row"] for x in r.json()["results"]}
    assert rows <= {0}  # only ring


# ---------------------------------------------------------------------------
# /api/similar/{row} with f= filters
# ---------------------------------------------------------------------------

def test_similar_with_f_param(tagged_client):
    r = tagged_client.get("/api/similar/0",
                          params=[("f", "category:bracelet"), ("min_ratio", "0"), ("min_score", "0")])
    assert r.status_code == 200
    rows = {x["row"] for x in r.json()["results"]}
    assert 0 not in rows
    assert rows <= {1, 2}
