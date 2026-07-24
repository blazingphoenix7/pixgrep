import numpy as np
import pytest

from pixgrep.search import SearchEngine
from pixgrep.store import save_index


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
        # use image width as the seed
        return np.vstack([self._vec(im.size[0]) for im in images])


@pytest.fixture()
def engine(tmp_path):
    # rows 0..3 with embeddings on distinct axes; rows 0 and 1 share an axis-0-ish direction
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
    save_index(tmp_path, paths, groups, emb)
    return SearchEngine(tmp_path, FakeEmbedder())


def test_count_and_path_for(engine):
    assert engine.count == 4
    assert engine.path_for(2) == "b/three.jpg"
    with pytest.raises(IndexError):
        engine.path_for(99)


def test_text_search_returns_ranked_results(engine):
    # query of length 4 -> axis 0 -> rows 0 and 1 rank first
    results = engine.text_search("abcd", k=2)
    assert [r["row"] for r in results] == [0, 1]
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["name"] == "one.jpg"
    assert results[0]["folder"] == "a"


def test_image_search_uses_image_embedding(engine):
    from PIL import Image

    img = Image.new("RGB", (5, 5))  # width 5 -> axis 1 -> row 2 first
    results = engine.image_search(img, k=1)
    assert results[0]["row"] == 2


def test_similar_excludes_self(engine):
    results = engine.similar(0, k=3)
    rows = [r["row"] for r in results]
    assert 0 not in rows
    assert rows[0] == 1  # nearest neighbor of row 0 is row 1


def test_k_larger_than_count_is_safe(engine):
    # both cutoffs disabled -> all rows come back
    assert len(engine.text_search("abcd", k=100, min_ratio=0.0, min_score=0.0)) == 4


def test_absolute_floor_drops_no_match_rows(engine):
    # even with the relative test off, rows scoring below min_score are dropped
    results = engine.text_search("abcd", k=100, min_ratio=0.0, min_score=0.5)
    assert [r["row"] for r in results] == [0, 1]


def test_relevance_cutoff_drops_weak_results(engine):
    # axis-0 query: rows 0 (sim 1.0) and 1 (sim ~0.99) pass the 0.6 ratio;
    # rows 2 and 3 (sim 0.0) are dropped even though k allows them
    results = engine.text_search("abcd", k=100)
    assert [r["row"] for r in results] == [0, 1]


def test_relevance_cutoff_ratio_is_tunable(engine):
    # a ratio above ~0.995 keeps only the single best hit
    results = engine.text_search("abcd", k=100, min_ratio=0.999)
    assert [r["row"] for r in results] == [0]


# --- group_members tests ---

@pytest.fixture()
def grouped_engine(tmp_path):
    """6 rows: rows 0,1,2 share group 'A'; rows 3,4 share 'B'; row 5 is singleton 'C'."""
    emb = np.eye(6, dtype=np.float32)
    paths = [f"p{i}.jpg" for i in range(6)]
    groups = ["A", "A", "A", "B", "B", "C"]
    save_index(tmp_path, paths, groups, emb)
    return SearchEngine(tmp_path, FakeEmbedder())


def test_group_members_includes_self(grouped_engine):
    for row in range(6):
        rows = [m["row"] for m in grouped_engine.group_members(row)]
        assert row in rows


def test_group_members_grouping(grouped_engine):
    assert [m["row"] for m in grouped_engine.group_members(1)] == [0, 1, 2]
    assert [m["row"] for m in grouped_engine.group_members(3)] == [3, 4]


def test_group_members_ordering(grouped_engine):
    rows = [m["row"] for m in grouped_engine.group_members(2)]
    assert rows == sorted(rows)


def test_group_members_singleton(grouped_engine):
    assert [m["row"] for m in grouped_engine.group_members(5)] == [5]


def test_group_members_cap(tmp_path):
    n = 70
    emb = np.eye(n, dtype=np.float32)
    paths = [f"p{i}.jpg" for i in range(n)]
    groups = ["X"] * n
    save_index(tmp_path, paths, groups, emb)
    eng = SearchEngine(tmp_path, FakeEmbedder())
    assert len(eng.group_members(0)) == 60


def test_group_members_result_shape(grouped_engine):
    m = grouped_engine.group_members(0)[0]
    for key in ("row", "path", "name", "group", "score"):
        assert key in m


def test_group_members_index_error(engine):
    with pytest.raises(IndexError):
        engine.group_members(99)


def test_group_members_includes_duplicate_aliases(tmp_path):
    """A row whose discarded exact-duplicate copy carries another style's
    filename joins that style's group — in both lookup directions."""
    from pixgrep.store import open_db

    emb = np.eye(3, dtype=np.float32)
    # row 2 was indexed under a junk name; its SKU-named copy became a dupe
    paths = ["ab100w.jpg", "ab100y.jpg", "final_approved.jpg"]
    groups = ["ab100", "ab100", "final_approved"]
    save_index(tmp_path, paths, groups, emb)
    con = open_db(tmp_path)
    con.execute(
        "INSERT INTO duplicates (path, size, mtime, sha1, duplicate_of) "
        "VALUES (?, ?, ?, ?, ?)",
        ("somewhere/AB100R.jpg", 1, 1.0, "x", 2),
    )
    con.commit()
    con.close()

    pattern = r"(?<=\d)[a-z]+\d*[a-z]*$"
    eng = SearchEngine(tmp_path, FakeEmbedder(), group_strip_pattern=pattern)

    assert [m["row"] for m in eng.group_members(0)] == [0, 1, 2]  # sibling side
    assert [m["row"] for m in eng.group_members(2)] == [0, 1, 2]  # orphan side

    # without a pattern the alias grouping is off (old behavior)
    eng_off = SearchEngine(tmp_path, FakeEmbedder())
    assert [m["row"] for m in eng_off.group_members(2)] == [2]


def test_k_zero_returns_all_floor_passing(tmp_path):
    """k=0 removes the count cap: every row passing the relevance floors
    comes back, not a fixed-size page."""
    n = 40
    emb = np.zeros((n, 3), dtype=np.float32)
    emb[:, 0] = 1.0  # all rows identical -> all pass any relative floor
    paths = [f"p{i}.jpg" for i in range(n)]
    save_index(tmp_path, paths, [f"g{i}" for i in range(n)], emb)
    eng = SearchEngine(tmp_path, FakeEmbedder())

    qv = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    capped = eng._rank(qv, 24, min_ratio=0, min_score=0)
    uncapped = eng._rank(qv, 0, min_ratio=0, min_score=0)
    assert len(capped) == 24
    assert len(uncapped) == n

    # floors still apply when uncapped: nothing relevant -> nothing returned
    qv_off = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    assert eng._rank(qv_off, 0, min_ratio=0.6, min_score=0.05) == []
