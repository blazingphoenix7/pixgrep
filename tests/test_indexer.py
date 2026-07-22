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
    # two variants of one item (same color) + one different item
    Image.new("RGB", (16, 16), (200, 10, 10)).save(images / "item-1.jpg")
    Image.new("RGB", (16, 16), (200, 10, 10)).save(images / "item-2.jpg")
    Image.new("RGB", (16, 16), (10, 10, 200)).save(images / "other-1.jpg")

    cfg = _cfg(tmp_path, images)
    result = build_index(cfg, FakeEmbedder())

    assert result["indexed"] == 3
    assert result["skipped"] == 0

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
