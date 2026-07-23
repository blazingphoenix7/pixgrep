import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pixgrep.search import SearchEngine
from pixgrep.server import create_app
from pixgrep.store import save_index


class FakeEmbedder:
    def embed_texts(self, texts):
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def embed_images(self, images):
        return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)


@pytest.fixture()
def client(tmp_path):
    # real image files on disk so /api/image can serve them
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rel_paths = []
    emb = np.eye(4, dtype=np.float32)[:3]
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        p = img_dir / f"img{i}.jpg"
        Image.new("RGB", (8, 8), color).save(p)
        rel_paths.append(str(p))
    save_index(tmp_path / "index", rel_paths, ["g0", "g1", "g2"], emb)
    engine = SearchEngine(tmp_path / "index", FakeEmbedder())
    app = create_app(engine)
    return TestClient(app)


def test_meta(client):
    r = client.get("/api/meta")
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_text_search(client):
    # default relevance cutoff: only the strong hit (row 0, sim 1.0) survives;
    # the zero-scoring rows are dropped even though k allows them
    r = client.get("/api/search", params={"q": "anything", "k": 2})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["row"] == 0  # axis-0 query -> row 0 first

    # both cutoffs disabled -> raw top-k comes back
    r = client.get(
        "/api/search",
        params={"q": "anything", "k": 2, "min_ratio": 0, "min_score": 0},
    )
    assert len(r.json()["results"]) == 2


def test_text_search_requires_query(client):
    assert client.get("/api/search").status_code in (400, 422)
    assert client.get("/api/search", params={"q": "  "}).status_code == 400


def test_image_search_roundtrip(client):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (9, 9, 9)).save(buf, format="JPEG")
    r = client.post(
        "/api/search/image",
        files={"file": ("q.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert r.status_code == 200
    assert r.json()["results"][0]["row"] == 1  # axis-1 image embedding -> row 1


def test_image_search_rejects_garbage(client):
    r = client.post(
        "/api/search/image",
        files={"file": ("q.jpg", b"not an image", "image/jpeg")},
    )
    assert r.status_code == 400


def test_similar(client):
    r = client.get("/api/similar/0", params={"k": 2})
    assert r.status_code == 200
    rows = [x["row"] for x in r.json()["results"]]
    assert 0 not in rows
    assert client.get("/api/similar/999").status_code == 404


def test_image_serving(client):
    r = client.get("/api/image/1")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
    assert client.get("/api/image/999").status_code == 404


def test_root_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "pixgrep" in r.text.lower()


def test_static_assets_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


@pytest.fixture()
def thumb_client(tmp_path):
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

    # Create a thumbnail only for row 0; row 1 has none (to test fallback)
    thumbs_dir = index_dir / "thumbs"
    thumbs_dir.mkdir()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(thumbs_dir / "0.jpg", "JPEG")

    engine = SearchEngine(index_dir, FakeEmbedder())
    app = create_app(engine)
    return TestClient(app)


def test_thumb_serves_thumb(thumb_client):
    r = thumb_client.get("/api/thumb/0")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_thumb_falls_back_to_image(thumb_client):
    # row 1 has no thumb file → falls back to original image
    r = thumb_client.get("/api/thumb/1")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_thumb_unknown_row_returns_404(thumb_client):
    assert thumb_client.get("/api/thumb/999").status_code == 404
