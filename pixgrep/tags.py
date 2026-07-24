from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class ImportReport:
    matched: int
    unmatched_records: int
    rows_without_tags: int

    def __str__(self) -> str:
        return (
            f"Matched:             {self.matched}\n"
            f"Unmatched records:   {self.unmatched_records}\n"
            f"Index rows w/o tags: {self.rows_without_tags}"
        )


def import_tags(
    index_dir: Path,
    records: list[dict],
    filename_key: str,
    field_keys: dict[str, str],
    text_keys: list[str],
    merge: bool = False,
) -> ImportReport:
    """Import per-image tag data into the index DB.

    Creates/replaces sqlite tables:
      tags(row, field, value)   — one row per non-empty categorical field value
      tag_text(row, text)       — lowercased concatenation of text_keys + field values

    Joins records to index rows by case-insensitive basename match on filename_key.

    merge=False (default): drops and recreates both tables before importing.
    merge=True: creates tables if absent; for each matched row, deletes existing
      (row, field) tags for fields being written, then inserts new values;
      tag_text is replaced (INSERT OR REPLACE) when the incoming record has text.

    Returns an ImportReport with match counts.
    """
    index_dir = Path(index_dir)
    con = sqlite3.connect(str(index_dir / "pixgrep.sqlite"))
    try:
        db_rows = con.execute("SELECT row, path FROM images ORDER BY row").fetchall()
        basename_to_row: dict[str, int] = {
            Path(p).name.lower(): r for r, p in db_rows
        }

        if not merge:
            con.execute("DROP TABLE IF EXISTS tags")
            con.execute("DROP TABLE IF EXISTS tag_text")
            con.execute("CREATE TABLE tags (row INTEGER, field TEXT, value TEXT)")
            con.execute("CREATE INDEX idx_tags_row ON tags(row)")
            con.execute("CREATE INDEX idx_tags_field_value ON tags(field, value)")
            con.execute("CREATE TABLE tag_text (row INTEGER PRIMARY KEY, text TEXT)")
        else:
            con.execute(
                "CREATE TABLE IF NOT EXISTS tags (row INTEGER, field TEXT, value TEXT)"
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_tags_row ON tags(row)")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_tags_field_value ON tags(field, value)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS tag_text (row INTEGER PRIMARY KEY, text TEXT)"
            )

        matched = 0
        unmatched = 0
        rows_with_tags: set[int] = set()
        tag_rows_data: list[tuple[int, str, str]] = []
        text_rows_data: list[tuple[int, str]] = []
        # (row, field) pairs whose existing tags should be deleted before insert
        delete_pairs: list[tuple[int, str]] = []

        for rec in records:
            fname = str(rec.get(filename_key, "") or "").strip()
            if not fname:
                unmatched += 1
                continue
            db_row = basename_to_row.get(Path(fname).name.lower())
            if db_row is None:
                unmatched += 1
                continue

            matched += 1
            rows_with_tags.add(db_row)
            field_vals: list[str] = []

            for canonical, col_key in field_keys.items():
                val = str(rec.get(col_key, "") or "").strip().lower()
                if val:
                    if merge:
                        delete_pairs.append((db_row, canonical))
                    tag_rows_data.append((db_row, canonical, val))
                    field_vals.append(val)

            text_parts: list[str] = []
            for tk in text_keys:
                v = str(rec.get(tk, "") or "").strip().lower()
                if v:
                    text_parts.append(v)
            text_parts.extend(field_vals)
            combined = " ".join(text_parts)
            if combined.strip():
                text_rows_data.append((db_row, combined))

        if merge:
            for pair in delete_pairs:
                con.execute("DELETE FROM tags WHERE row=? AND field=?", pair)
            con.executemany(
                "INSERT INTO tags (row, field, value) VALUES (?, ?, ?)", tag_rows_data
            )
            for row_text in text_rows_data:
                con.execute(
                    "INSERT OR REPLACE INTO tag_text (row, text) VALUES (?, ?)",
                    row_text,
                )
        else:
            con.executemany(
                "INSERT INTO tags (row, field, value) VALUES (?, ?, ?)", tag_rows_data
            )
            con.executemany(
                "INSERT INTO tag_text (row, text) VALUES (?, ?)", text_rows_data
            )
        con.commit()

        rows_without_tags = len(db_rows) - len(rows_with_tags)
        return ImportReport(
            matched=matched,
            unmatched_records=unmatched,
            rows_without_tags=rows_without_tags,
        )
    finally:
        con.close()


class TagStore:
    """Loads tag tables from an index DB for fast in-memory scoring.

    Gracefully handles missing tables (has_data stays False, all methods return
    empty/None so the rest of the engine works unchanged without tags).
    """

    def __init__(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        db_path = index_dir / "pixgrep.sqlite"

        self._has_data = False
        self._n_rows = 0
        # field -> value -> sorted numpy array of row indices
        self._field_value_rows: dict[str, dict[str, np.ndarray]] = {}
        # token -> sorted numpy array of row indices (union across all values)
        self._value_token_rows: dict[str, np.ndarray] = {}
        # row -> frozenset of tokens from tag_text
        self._row_text_tokens: dict[int, frozenset[str]] = {}
        # field -> value -> count (for facets)
        self._facets: dict[str, dict[str, int]] = {}

        if not db_path.exists():
            return

        con = sqlite3.connect(str(db_path))
        try:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            if "tags" not in tables or "tag_text" not in tables:
                return

            self._n_rows = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]

            tag_data = con.execute("SELECT row, field, value FROM tags").fetchall()
            if not tag_data:
                return

            self._has_data = True

            # Accumulate into sets first (deduplication)
            fv_sets: dict[str, dict[str, set[int]]] = {}
            vt_sets: dict[str, set[int]] = {}

            for db_row, field, value in tag_data:
                fv_sets.setdefault(field, {}).setdefault(value, set()).add(db_row)
                field_facets = self._facets.setdefault(field, {})
                field_facets[value] = field_facets.get(value, 0) + 1
                for tok in _tokenize(value):
                    vt_sets.setdefault(tok, set()).add(db_row)

            # Convert to sorted numpy arrays for O(rows) scatter
            self._field_value_rows = {
                field: {
                    val: np.array(sorted(rows), dtype=np.int64)
                    for val, rows in val_dict.items()
                }
                for field, val_dict in fv_sets.items()
            }
            self._value_token_rows = {
                tok: np.array(sorted(rows), dtype=np.int64)
                for tok, rows in vt_sets.items()
            }

            text_data = con.execute("SELECT row, text FROM tag_text").fetchall()
            for db_row, text in text_data:
                self._row_text_tokens[db_row] = frozenset(_tokenize(text))

        finally:
            con.close()

    @property
    def has_data(self) -> bool:
        return self._has_data

    def facets(self) -> dict[str, dict[str, int]]:
        return self._facets

    def rows_matching(self, filters: dict[str, str]) -> np.ndarray | None:
        """Return sorted array of row indices matching ALL filters, or None if no filters."""
        if not filters:
            return None
        result: np.ndarray | None = None
        for field, value in filters.items():
            val_lower = value.lower().strip()
            rows = self._field_value_rows.get(field, {}).get(val_lower)
            if rows is None:
                return np.array([], dtype=np.int64)
            result = rows if result is None else np.intersect1d(result, rows)
        return result if result is not None else np.array([], dtype=np.int64)

    def lexical_scores(self, query: str, n_rows: int) -> np.ndarray:
        """Vectorized lexical scoring, result normalized to [0, 1].

        Per row accumulates:
          +1.0 per categorical value fully contained in query tokens
          +0.5 per unique query token that appears in any categorical value
          +0.25 * (fraction of query token set present in tag_text)
        """
        if not self._has_data or n_rows == 0:
            return np.zeros(n_rows, dtype=np.float32)

        scores = np.zeros(n_rows, dtype=np.float32)
        query_tokens = _tokenize(query)
        if not query_tokens:
            return scores

        query_token_set = frozenset(query_tokens)
        n_q = len(query_tokens)

        # Component 1: +1.0 per categorical value fully contained in query
        for field_data in self._field_value_rows.values():
            for value, rows in field_data.items():
                val_toks = _tokenize(value)
                if val_toks and all(t in query_token_set for t in val_toks):
                    valid = rows[rows < n_rows]
                    scores[valid] += 1.0

        # Component 2: +0.5 per query token hitting any categorical value
        for tok in query_token_set:
            rows = self._value_token_rows.get(tok)
            if rows is not None:
                valid = rows[rows < n_rows]
                scores[valid] += 0.5

        # Component 3: +0.25 * fraction of query tokens in tag_text
        for r, text_toks in self._row_text_tokens.items():
            if r < n_rows:
                frac = len(query_token_set & text_toks) / n_q
                scores[r] += 0.25 * frac

        mx = float(scores.max())
        if mx > 0.0:
            scores /= mx

        return scores
