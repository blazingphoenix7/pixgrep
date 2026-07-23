from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np

BIN_FILENAME = "embeddings.f16.bin"
NPY_FILENAME = "embeddings.npy"


def save_index(
    index_dir: Path,
    paths: list[str],
    group_keys: list[str],
    embeddings: np.ndarray,
) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / NPY_FILENAME, embeddings.astype(np.float16))

    con = sqlite3.connect(str(index_dir / "pixgrep.sqlite"))
    try:
        con.execute("DROP TABLE IF EXISTS images")
        con.execute(
            "CREATE TABLE images (row INTEGER PRIMARY KEY, path TEXT, group_key TEXT)"
        )
        con.executemany(
            "INSERT INTO images (row, path, group_key) VALUES (?, ?, ?)",
            [(i, p, g) for i, (p, g) in enumerate(zip(paths, group_keys))],
        )
        con.commit()
    finally:
        con.close()


def load_index(index_dir: Path) -> tuple[list[str], list[str], np.ndarray]:
    index_dir = Path(index_dir)
    con = sqlite3.connect(str(index_dir / "pixgrep.sqlite"))
    try:
        rows = con.execute(
            "SELECT row, path, group_key FROM images ORDER BY row"
        ).fetchall()
        sqlite_count = len(rows)

        bin_file = index_dir / BIN_FILENAME
        npy_file = index_dir / NPY_FILENAME

        if bin_file.exists():
            dim = _meta_int(con, "embedding_dim")
            raw = np.fromfile(str(bin_file), dtype=np.float16)
            if dim is not None and sqlite_count > 0:
                expected = sqlite_count * dim
                if len(raw) > expected:
                    # Crash recovery: bin has orphaned rows — truncate to sqlite count
                    raw = raw[:expected]
                    _truncate_file(bin_file, expected * 2)
                emb = raw.reshape(sqlite_count, dim).astype(np.float32)
            elif sqlite_count == 0:
                emb = np.zeros((0, 1), dtype=np.float32)
            else:
                d = len(raw) // max(sqlite_count, 1)
                emb = raw.reshape(sqlite_count, max(d, 1)).astype(np.float32)
        else:
            emb = np.load(str(npy_file)).astype(np.float32)
    finally:
        con.close()

    paths = [r[1] for r in rows]
    groups = [r[2] for r in rows]
    return paths, groups, emb


def open_db(index_dir: Path) -> sqlite3.Connection:
    """Open or create a v2 database. Applies lightweight migration if needed."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(index_dir / "pixgrep.sqlite"))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS images (
            row INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            group_key TEXT NOT NULL,
            mtime REAL,
            size INTEGER,
            sha1 TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS duplicates (
            path TEXT NOT NULL,
            size INTEGER,
            mtime REAL,
            sha1 TEXT,
            duplicate_of INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            val TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_images_sha1 ON images(sha1)")
    # Migrate old 3-col schema to v2 if needed
    cols = {r[1] for r in con.execute("PRAGMA table_info(images)").fetchall()}
    for col, typedef in [("mtime", "REAL"), ("size", "INTEGER"), ("sha1", "TEXT")]:
        if col not in cols:
            con.execute(f"ALTER TABLE images ADD COLUMN {col} {typedef}")
    con.commit()
    return con


def set_meta(con: sqlite3.Connection, key: str, val: str) -> None:
    con.execute("INSERT OR REPLACE INTO meta (key, val) VALUES (?, ?)", (key, val))


def get_meta(con: sqlite3.Connection, key: str) -> str | None:
    try:
        r = con.execute("SELECT val FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None
    except Exception:
        return None


def get_path_index(con: sqlite3.Connection) -> dict[str, tuple[int, float | None, int | None]]:
    """Return {path: (row, mtime, size)} for all indexed images."""
    rows = con.execute("SELECT path, row, mtime, size FROM images").fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def get_sha1_index(con: sqlite3.Connection) -> dict[str, int]:
    """Return {sha1: row} for all indexed images."""
    rows = con.execute(
        "SELECT sha1, row FROM images WHERE sha1 IS NOT NULL"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def append_vecs(bin_file: Path, vecs: np.ndarray) -> None:
    """Append float16 rows to the bin file (create if absent).

    fsyncs before returning: the crash-recovery contract requires the bin
    file to be durable BEFORE the matching sqlite commit.
    """
    f16 = vecs.astype(np.float16)
    with open(str(bin_file), "ab") as f:
        f.write(f16.tobytes())
        f.flush()
        os.fsync(f.fileno())


def overwrite_vec(bin_file: Path, row: int, dim: int, vec: np.ndarray) -> None:
    """Overwrite the embedding for an existing row in place."""
    offset = row * dim * 2  # float16 = 2 bytes
    f16 = vec.astype(np.float16)
    with open(str(bin_file), "r+b") as f:
        f.seek(offset)
        f.write(f16.tobytes())
        f.flush()
        os.fsync(f.fileno())


def _meta_int(con: sqlite3.Connection, key: str) -> int | None:
    try:
        r = con.execute("SELECT val FROM meta WHERE key=?", (key,)).fetchone()
        return int(r[0]) if r else None
    except Exception:
        return None


def _truncate_file(path: Path, byte_size: int) -> None:
    try:
        with open(str(path), "r+b") as f:
            f.truncate(byte_size)
    except OSError:
        pass
