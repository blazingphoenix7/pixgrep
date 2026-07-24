from __future__ import annotations

import sqlite3
from pathlib import Path

from .query_norm import normalize_query


def normalize_query_key(query: str) -> str:
    """Canonical key marks are stored/looked-up under.

    Shorthand-expanded (shares marks across "YG band" / "yellow gold band"),
    then lowercased and stripped.
    """
    return normalize_query(query).lower().strip()


class FeedbackStore:
    """Shared "bad result for this query" marks, keyed by (user, query_norm, sha1).

    Owns its own sqlite file, independent of the search index — marks key on
    each image's content hash so they survive index rebuilds regardless of
    row renumbering. Every read opens a fresh connection: marks made by
    another user or process are visible immediately, with no caching.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = self._connect()
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS marks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT NOT NULL,
                    query_norm TEXT NOT NULL,
                    sha1 TEXT NOT NULL,
                    row_at_mark INTEGER,
                    path_at_mark TEXT,
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user, query_norm, sha1)
                )
            """)
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_marks_query_norm ON marks(query_norm)"
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_marks_user ON marks(user)")
            con.commit()
        finally:
            con.close()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.execute("PRAGMA journal_mode=WAL")
        con.row_factory = sqlite3.Row
        return con

    def toggle(self, user: str, query_norm: str, sha1: str, row: int, path: str) -> bool:
        """Insert a mark if absent, else remove it. Returns the marked-now state."""
        con = self._connect()
        try:
            existing = con.execute(
                "SELECT id FROM marks WHERE user=? AND query_norm=? AND sha1=?",
                (user, query_norm, sha1),
            ).fetchone()
            if existing:
                con.execute("DELETE FROM marks WHERE id=?", (existing["id"],))
                con.commit()
                return False
            con.execute(
                "INSERT INTO marks (user, query_norm, sha1, row_at_mark, path_at_mark) "
                "VALUES (?, ?, ?, ?, ?)",
                (user, query_norm, sha1, row, path),
            )
            con.commit()
            return True
        finally:
            con.close()

    def suppressed_sha1s(self, query_norm: str) -> set[str]:
        """All sha1s marked bad for this query, across every user."""
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT DISTINCT sha1 FROM marks WHERE query_norm=?", (query_norm,)
            ).fetchall()
            return {r["sha1"] for r in rows}
        finally:
            con.close()

    def marks_for_query(self, query_norm: str) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT id, user, query_norm, sha1, row_at_mark, path_at_mark, created "
                "FROM marks WHERE query_norm=? ORDER BY id DESC",
                (query_norm,),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            con.close()

    def list_all(self) -> list[dict]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT id, user, query_norm, sha1, row_at_mark, path_at_mark, created "
                "FROM marks ORDER BY id DESC"
            ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            con.close()

    def delete(self, id: int) -> bool:
        con = self._connect()
        try:
            cur = con.execute("DELETE FROM marks WHERE id=?", (id,))
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()

    def delete_user(self, user: str) -> int:
        con = self._connect()
        try:
            cur = con.execute("DELETE FROM marks WHERE user=?", (user,))
            con.commit()
            return cur.rowcount
        finally:
            con.close()

    def delete_query(self, query_norm: str) -> int:
        con = self._connect()
        try:
            cur = con.execute("DELETE FROM marks WHERE query_norm=?", (query_norm,))
            con.commit()
            return cur.rowcount
        finally:
            con.close()


def _row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "user": r["user"],
        "query": r["query_norm"],
        "sha1": r["sha1"],
        "row": r["row_at_mark"],
        "path": r["path_at_mark"],
        "created": r["created"],
    }
