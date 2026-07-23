import sqlite3

import numpy as np
from PIL import Image

from pixgrep.config import Config
from pixgrep.indexer import build_index, find_images, load_rgb
from pixgrep.store import load_index


class FakeEmbedder:
    """Deterministic embeddings: encodes each image's mean color → unit vector."""

    def embed_images(self, images):
        out = []
        for img in images:
            arr = np.asarray(img.convert("RGB")).reshape(-1, 3).mean(axis=0)
            v = arr.astype(np.float32)
            v = v / (np.linalg.norm(v) or 1.0)
            out.append(v)
        return np.vstack(out)


def _cfg(tmp_path, images_dir):
    return Config(
        image_root=images_dir,
        index_dir=tmp_path / "index",
        model_id="unused",
        group_strip_pattern=r"[-_ ]*\d+$",
        batch_size=2,
    )


def test_find_images_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 8), (4, 5, 6)).save(tmp_path / "sub" / "b.png")
    (tmp_path / "notes.txt").write_text("ignore me")
    found = find_images(tmp_path)
    assert {p.name for p in found} == {"a.jpg", "b.png"}


def test_load_rgb_returns_none_on_garbage(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a real jpeg")
    assert load_rgb(bad) is None


def test_build_index_end_to_end(tmp_path):
    images = tmp_path / "imgs"
    images.mkdir()
    # Three distinct images: two share a group key, one is different
    Image.new("RGB", (16, 16), (200, 10, 10)).save(images / "item-1.jpg")
    Image.new("RGB", (16, 16), (150, 10, 10)).save(images / "item-2.jpg")
    Image.new("RGB", (16, 16), (10, 10, 200)).save(images / "other-1.jpg")

    cfg = _cfg(tmp_path, images)
    result = build_index(cfg, FakeEmbedder())

    assert result["indexed"] == 3
    assert result["skipped"] == 0
    assert result["dupes"] == 0
    assert result["quarantined"] == 0

    paths, groups, emb = load_index(cfg.index_dir)
    assert len(paths) == 3
    assert emb.shape[0] == 3
    # variants share a group key; the different item does not
    by_name = dict(zip((p.split("/")[-1].split("\\")[-1] for p in paths), groups))
    assert by_name["item-1.jpg"] == by_name["item-2.jpg"]
    assert by_name["item-1.jpg"] != by_name["other-1.jpg"]


def test_build_index_skips_unreadable(tmp_path):
    images = tmp_path / "imgs"
    images.mkdir()
    Image.new("RGB", (8, 8), (1, 1, 1)).save(images / "ok-1.jpg")
    (images / "broken-1.jpg").write_bytes(b"garbage")
    cfg = _cfg(tmp_path, images)
    result = build_index(cfg, FakeEmbedder())
    assert result["indexed"] == 1
    assert result["skipped"] == 1


def test_resume_skips_same_stat(tmp_path):
    """Files with unchanged size+mtime are not re-embedded on resume."""
    images = tmp_path / "imgs"
    images.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(images / "a.jpg")
    Image.new("RGB", (8, 8), (40, 50, 60)).save(images / "b.jpg")

    cfg = _cfg(tmp_path, images)
    r1 = build_index(cfg, FakeEmbedder(), resume=False)
    assert r1["indexed"] == 2

    r2 = build_index(cfg, FakeEmbedder(), resume=True)
    # No new indexing — same stat as before
    assert r2["indexed"] == 2
    assert r2["dupes"] == 0
    assert r2["quarantined"] == 0


def test_resume_reindexes_changed_file(tmp_path):
    """A file whose mtime changed is re-indexed (metadata updated in db)."""
    images = tmp_path / "imgs"
    images.mkdir()
    img_path = images / "a.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(img_path)

    cfg = _cfg(tmp_path, images)
    build_index(cfg, FakeEmbedder(), resume=False)

    # Overwrite with different content — this changes mtime
    Image.new("RGB", (8, 8), (200, 100, 50)).save(img_path)

    build_index(cfg, FakeEmbedder(), resume=True)

    con = sqlite3.connect(str(cfg.index_dir / "pixgrep.sqlite"))
    rows = con.execute(
        "SELECT mtime, sha1 FROM images WHERE path=?", (str(img_path),)
    ).fetchall()
    con.close()

    assert len(rows) == 1  # no duplicate path entries
    assert rows[0][0] == img_path.stat().st_mtime


def test_sha1_dedup_goes_to_duplicates(tmp_path):
    """Two files with identical content → one in images, one in duplicates table."""
    images = tmp_path / "imgs"
    images.mkdir()
    # Save same solid-color PNG to guarantee identical bytes
    img = Image.new("RGB", (8, 8), (1, 2, 3))
    img.save(images / "orig.png")
    img.save(images / "copy.png")

    cfg = _cfg(tmp_path, images)
    result = build_index(cfg, FakeEmbedder())

    assert result["indexed"] == 1
    assert result["dupes"] == 1

    con = sqlite3.connect(str(cfg.index_dir / "pixgrep.sqlite"))
    img_count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    dup_count = con.execute("SELECT COUNT(*) FROM duplicates").fetchone()[0]
    con.close()

    assert img_count == 1
    assert dup_count == 1

    paths, _, emb = load_index(cfg.index_dir)
    assert len(paths) == 1
    assert emb.shape[0] == 1


def test_no_resume_wipes_existing_index(tmp_path):
    """--no-resume rebuilds from scratch, discarding any previous index."""
    images = tmp_path / "imgs"
    images.mkdir()
    Image.new("RGB", (8, 8), (1, 1, 1)).save(images / "a.jpg")

    cfg = _cfg(tmp_path, images)
    build_index(cfg, FakeEmbedder(), resume=False)
    r2 = build_index(cfg, FakeEmbedder(), resume=False)

    assert r2["indexed"] == 1  # fresh rebuild, not double-counted


def test_thumbnail_created(tmp_path):
    """Thumbnail files are created at thumbs/<row>.jpg for each indexed image."""
    images = tmp_path / "imgs"
    images.mkdir()
    Image.new("RGB", (200, 200), (1, 2, 3)).save(images / "a.jpg")
    Image.new("RGB", (200, 200), (4, 5, 6)).save(images / "b.jpg")

    cfg = _cfg(tmp_path, images)
    build_index(cfg, FakeEmbedder())

    assert (cfg.index_dir / "thumbs" / "0.jpg").is_file()
    assert (cfg.index_dir / "thumbs" / "1.jpg").is_file()

    # Thumbnail must fit within 384px on its longest side
    thumb = Image.open(cfg.index_dir / "thumbs" / "0.jpg")
    assert max(thumb.size) <= 384
