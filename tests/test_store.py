import numpy as np

from pixgrep.store import load_index, save_index


def test_save_and_load_roundtrip(tmp_path):
    paths = ["a/1.jpg", "a/2.jpg", "b/3.jpg"]
    groups = ["a1", "a1", "b3"]
    emb = np.random.rand(3, 8).astype(np.float32)

    save_index(tmp_path, paths, groups, emb)
    lp, lg, le = load_index(tmp_path)

    assert lp == paths
    assert lg == groups
    assert le.shape == (3, 8)
    assert le.dtype == np.float32
    # float16 storage → tolerance on values
    assert np.allclose(le, emb, atol=1e-3)


def test_load_preserves_row_order(tmp_path):
    paths = [f"{i}.jpg" for i in range(20)]
    groups = [f"g{i}" for i in range(20)]
    emb = np.random.rand(20, 4).astype(np.float32)
    save_index(tmp_path, paths, groups, emb)
    lp, _, _ = load_index(tmp_path)
    assert lp == paths
