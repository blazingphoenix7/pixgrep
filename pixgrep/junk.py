from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

_DEFAULT_JUNK_PROMPTS = [
    "screenshot of software",
    "scanned text document",
    "spreadsheet",
    "page of text",
    "logo graphic",
    "clip art",
    "diagram",
    "stock photo of a couple",
    "people hugging in a lifestyle scene",
    "romantic greeting-card photograph",
    "advertising banner with large text",
    "photo of a person's face portrait",
    "gift box wrapping scene",
    "couple celebrating a special occasion",
    "marketing lifestyle photograph of people",
    "birthday party celebration scene",
]

_DEFAULT_GOOD_PROMPTS = [
    "professional photograph of a product",
    "photograph of jewelry",
    "studio product photo",
    "photo of an object",
    "close-up of jewelry worn on a hand",
    "close-up of an earring worn on an ear",
    "jewelry modeled on a person, jewelry in focus",
    "close-up of a necklace worn around a neck",
    "close-up product shot of a ring on a finger",
]


def junk_scores(
    emb: np.ndarray,
    embedder,
    junk_prompts: list[str] | None = None,
    good_prompts: list[str] | None = None,
) -> np.ndarray:
    """Score each row for junk-ness: max_cos(junk) - max_cos(good).

    Higher score = more junk-like. emb must be L2-normalised float32 (N, D).
    Returns float32 array of shape (N,).
    """
    if junk_prompts is None:
        junk_prompts = _DEFAULT_JUNK_PROMPTS
    if good_prompts is None:
        good_prompts = _DEFAULT_GOOD_PROMPTS

    junk_vecs = np.array(embedder.embed_texts(junk_prompts), dtype=np.float32)
    good_vecs = np.array(embedder.embed_texts(good_prompts), dtype=np.float32)

    # Defensive normalisation (embedder contract, but emb may be float16-loaded)
    junk_vecs /= np.linalg.norm(junk_vecs, axis=1, keepdims=True).clip(1e-8)
    good_vecs /= np.linalg.norm(good_vecs, axis=1, keepdims=True).clip(1e-8)

    e = emb.astype(np.float32)
    junk_max = (e @ junk_vecs.T).max(axis=1)
    good_max = (e @ good_vecs.T).max(axis=1)
    return (junk_max - good_max).astype(np.float32)


def save_junk_scores(index_dir: Path, scores: np.ndarray) -> None:
    """Persist junk scores to the index sqlite DB (overwrites previous run)."""
    index_dir = Path(index_dir)
    con = sqlite3.connect(str(index_dir / "pixgrep.sqlite"))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS junk_scores "
            "(row INTEGER PRIMARY KEY, score REAL)"
        )
        con.execute("DELETE FROM junk_scores")
        con.executemany(
            "INSERT INTO junk_scores (row, score) VALUES (?, ?)",
            [(int(i), float(s)) for i, s in enumerate(scores)],
        )
        con.commit()
    finally:
        con.close()


def load_junk_scores(index_dir: Path, n_rows: int) -> np.ndarray | None:
    """Load junk scores from the index sqlite DB. Returns None if table absent."""
    index_dir = Path(index_dir)
    db_path = index_dir / "pixgrep.sqlite"
    if not db_path.exists():
        return None
    con = sqlite3.connect(str(db_path))
    try:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "junk_scores" not in tables:
            return None
        rows = con.execute(
            "SELECT row, score FROM junk_scores ORDER BY row"
        ).fetchall()
        if not rows:
            return None
        out = np.zeros(n_rows, dtype=np.float32)
        for db_row, score in rows:
            if db_row < n_rows:
                out[db_row] = float(score)
        return out
    finally:
        con.close()
