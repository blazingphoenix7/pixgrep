import numpy as np
import pytest

from pixgrep.store import (
    BIN_FILENAME,
    append_vecs,
    load_index,
    open_db,
    overwrite_vec,
    save_index,
    set_meta,
)


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
    assert np.allclose(le, emb, atol=1e-3)


def test_load_preserves_row_order(tmp_path):
    paths = [f"{i}.jpg" for i in range(20)]
    groups = [f"g{i}" for i in range(20)]
    emb = np.random.rand(20, 4).astype(np.float32)
    save_index(tmp_path, paths, groups, emb)
    lp, _, _ = load_index(tmp_path)
    assert lp == paths


def test_npy_backward_compat(tmp_path):
    """Old .npy format loads transparently."""
    paths = ["x.jpg", "y.jpg"]
    groups = ["gx", "gy"]
    emb = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    save_index(tmp_path, paths, groups, emb)

    lp, lg, le = load_index(tmp_path)
    assert lp == paths
    assert lg == groups
    assert le.shape == (2, 2)
    assert np.allclose(le, emb, atol=1e-3)


def test_v2_append_load_roundtrip(tmp_path):
    """open_db + append_vecs + load_index roundtrip for the v2 format."""
    con = open_db(tmp_path)
    dim = 4
    set_meta(con, "embedding_dim", str(dim))
    set_meta(con, "schema_version", "2")

    paths = ["a.jpg", "b.jpg"]
    vecs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    for i, p in enumerate(paths):
        con.execute(
            "INSERT INTO images (row, path, group_key, mtime, size, sha1) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, p, "g", 1.0, 100, f"sha{i}"),
        )
    append_vecs(tmp_path / BIN_FILENAME, vecs)
    con.commit()
    con.close()

    lp, lg, le = load_index(tmp_path)
    assert lp == paths
    assert le.shape == (2, 4)
    assert np.allclose(le, vecs, atol=1e-3)


def test_bin_truncation_recovery(tmp_path):
    """Bin file with extra rows (crash between append and commit) is truncated on load."""
    con = open_db(tmp_path)
    dim = 4
    set_meta(con, "embedding_dim", str(dim))

    for i in range(2):
        con.execute(
            "INSERT INTO images (row, path, group_key, mtime, size, sha1) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, f"{i}.jpg", "g", 1.0, 100, f"sha{i}"),
        )
    con.commit()

    # Write 3 embeddings — simulating crash after append, before commit of 3rd row
    vecs = np.eye(3, 4, dtype=np.float32)
    append_vecs(tmp_path / BIN_FILENAME, vecs)
    con.close()

    lp, _, le = load_index(tmp_path)
    assert len(lp) == 2
    assert le.shape == (2, 4)

    # Bin must be physically truncated to 2 rows
    bin_bytes = (tmp_path / BIN_FILENAME).stat().st_size
    assert bin_bytes == 2 * dim * 2  # float16 = 2 bytes


def test_overwrite_vec_updates_in_place(tmp_path):
    """overwrite_vec replaces the embedding at the given row without touching others."""
    con = open_db(tmp_path)
    dim = 4
    set_meta(con, "embedding_dim", str(dim))

    original = np.eye(3, 4, dtype=np.float32)
    for i in range(3):
        con.execute(
            "INSERT INTO images (row, path, group_key, mtime, size, sha1) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, f"{i}.jpg", "g", 1.0, 100, f"sha{i}"),
        )
    append_vecs(tmp_path / BIN_FILENAME, original)
    con.commit()
    con.close()

    new_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    overwrite_vec(tmp_path / BIN_FILENAME, 1, dim, new_vec)

    _, _, le = load_index(tmp_path)
    assert np.allclose(le[0], original[0], atol=1e-3)
    assert np.allclose(le[1], new_vec, atol=1e-3)
    assert np.allclose(le[2], original[2], atol=1e-3)
