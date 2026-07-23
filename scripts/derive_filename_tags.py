"""Derive metal tags from SKU filenames for rows without spreadsheet metal tags.

Metal is encoded in the suffix letters immediately after the base style number.
Rule (applied to first two letters of the suffix, uppercase):
  - Starts with T, or two-char combo WT/YT, or WY/YW → two-tone
  - First letter W → white
  - First letter Y → yellow
  - First letter R → rose
  - Anything else  → skip (conservative)

Usage:
    python scripts/derive_filename_tags.py --config config.local.json [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixgrep.config import load_config


def _first_alnum_run(stem: str) -> str:
    """Return first contiguous alphanumeric run (stop at space / # / -)."""
    m = re.match(r"[a-zA-Z0-9]+", stem)
    return m.group(0) if m else stem


def _derive_metal(suffix: str) -> str | None:
    """Return metal value from suffix first-two-letters, or None to skip."""
    s = suffix.upper()
    if not s:
        return None
    first = s[0]
    two = s[:2]

    # Two-tone: T at start, or WT/YT/WY/YW two-char combos
    if first == "T":
        return "two-tone"
    if two in ("WT", "YT", "WY", "YW"):
        return "two-tone"
    if first == "W":
        return "white"
    if first == "Y":
        return "yellow"
    if first == "R":
        return "rose"
    return None


def derive_filename_tags(index_dir: Path, dry_run: bool = False) -> dict:
    """Derive and insert metal tags from filenames.

    Reads group_key from the images table (already stripped by the indexer).
    Only processes rows without an existing metal tag.
    Does NOT drop or recreate tags / tag_text tables.

    Returns counts: derived, skipped_ambiguous, already_tagged.
    """
    db_path = Path(index_dir) / "pixgrep.sqlite"
    con = sqlite3.connect(str(db_path))
    try:
        db_rows = con.execute(
            "SELECT row, path, group_key FROM images ORDER BY row"
        ).fetchall()

        # Rows that already have a metal tag
        try:
            metal_rows = {
                r[0]
                for r in con.execute(
                    "SELECT row FROM tags WHERE field='metal'"
                ).fetchall()
            }
        except Exception:
            metal_rows = set()

        # Existing tag_text for append
        try:
            text_rows = {
                r[0]: r[1]
                for r in con.execute("SELECT row, text FROM tag_text").fetchall()
            }
        except Exception:
            text_rows = {}

        derived = 0
        skipped_ambiguous = 0
        already_tagged = 0
        insert_tags: list[tuple] = []
        upsert_text: list[tuple] = []

        for db_row, path, group_key in db_rows:
            if db_row in metal_rows:
                already_tagged += 1
                continue

            stem = Path(path).stem
            clean = _first_alnum_run(stem)
            gk = (group_key or "").lower()
            if gk and clean.lower().startswith(gk):
                suffix = clean[len(gk):]
            else:
                suffix = clean

            metal = _derive_metal(suffix)
            if metal is None:
                skipped_ambiguous += 1
                continue

            derived += 1
            if not dry_run:
                insert_tags.append((db_row, "metal", metal))
                existing = text_rows.get(db_row, "")
                new_text = (existing + " " + metal).strip() if existing else metal
                upsert_text.append((db_row, new_text))

        if not dry_run and insert_tags:
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
            con.executemany(
                "INSERT INTO tags (row, field, value) VALUES (?, ?, ?)", insert_tags
            )
            for db_row, text in upsert_text:
                con.execute(
                    "INSERT INTO tag_text (row, text) VALUES (?, ?)"
                    " ON CONFLICT(row) DO UPDATE SET text=excluded.text",
                    (db_row, text),
                )
            con.commit()

        return {
            "derived": derived,
            "skipped_ambiguous": skipped_ambiguous,
            "already_tagged": already_tagged,
        }
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive metal tags from filenames")
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print counts only, no DB writes"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    counts = derive_filename_tags(cfg.index_dir, dry_run=args.dry_run)

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"{prefix}Derived:           {counts['derived']}")
    print(f"{prefix}Skipped/ambiguous: {counts['skipped_ambiguous']}")
    print(f"{prefix}Already tagged:    {counts['already_tagged']}")


if __name__ == "__main__":
    main()
