from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


def save_index(
    index_dir: Path,
    paths: list[str],
    group_keys: list[str],
    embeddings: np.ndarray,
) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings.astype(np.float16))

    con = sqlite3.connect(index_dir / "pixgrep.sqlite")
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
    emb = np.load(index_dir / "embeddings.npy").astype(np.float32)
    con = sqlite3.connect(index_dir / "pixgrep.sqlite")
    try:
        rows = con.execute(
            "SELECT row, path, group_key FROM images ORDER BY row"
        ).fetchall()
    finally:
        con.close()
    paths = [r[1] for r in rows]
    groups = [r[2] for r in rows]
    return paths, groups, emb
